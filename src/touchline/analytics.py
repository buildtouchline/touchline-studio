"""Player-level and match-level analytics computation.

Computes player trajectories, visible exposure, speed profiling,
spatial heatmaps, tracking quality indicators, and explicit player associations.

Enforces strict calibration gates: uncalibrated inputs never produce
unverified physical metres or speeds (TL-006, TL-007).
Enforces run-scoped identity and revisioned corrections (TL-004, TL-S05).
"""
from .diagnostics import diagnosed, record_failure
from collections import defaultdict
import math
from pathlib import Path
from typing import Any

from .contracts import Box
from .metrics import Calibration, Point, activity_summary, intensity_zone_summary, movement_summary
from .tracking import reviewed_association


def compute_spatial_heatmap(
    points: list[tuple[float, float]],
    width: float,
    height: float,
    grid_cols: int = 16,
    grid_rows: int = 9,
) -> dict[str, Any]:
    """Compute a normalized 2D spatial occupancy grid from image points."""
    if width <= 0 or height <= 0:
        raise ValueError("positive frame dimensions required")
    if grid_cols <= 0 or grid_rows <= 0:
        raise ValueError("positive grid dimensions required")

    grid = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
    valid_points = 0

    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        col = int((x / width) * grid_cols)
        row = int((y / height) * grid_rows)
        col = max(0, min(grid_cols - 1, col))
        row = max(0, min(grid_rows - 1, row))
        grid[row][col] += 1
        valid_points += 1

    max_count = max((max(row) for row in grid), default=0)
    normalized = [
        [round(cell / max_count, 4) if max_count > 0 else 0.0 for cell in row]
        for row in grid
    ]

    return {
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "total_points": valid_points,
        "max_cell_count": max_count,
        "density_matrix": normalized,
    }


def compute_track_analytics(
    track_id: int,
    observations: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    calibration: Calibration | None = None,
    max_gap_s: float = 0.75,
    peak_window_s: float = 1.0,
    association: Any | None = None,
    events: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate trajectory, exposure, movement, and heatmap for a single track."""
    if not observations:
        raise ValueError("observations cannot be empty")

    sorted_obs = sorted(observations, key=lambda o: o["frame"]["timestamp_s"])
    review = reviewed_association(association, run_id=run_id, local_track_id=track_id)
    timestamps = [o["frame"]["timestamp_s"] for o in sorted_obs]
    first_time = timestamps[0]
    last_time = timestamps[-1]
    visible_duration = round(last_time - first_time, 4)

    identity_reviewed = review is not None
    identity_review_ref = review["review_ref"] if review else None

    # Extract ground contact coordinates (bottom-centre of bounding box)
    ground_contacts = []
    trajectory = []
    ambiguous_count = 0
    detector_scores = []
    points_for_metrics = []

    for obs in sorted_obs:
        bbox = obs["bbox_xyxy"]
        box_obj = Box(*bbox)
        gx, gy = box_obj.ground_contact_point(convention="bottom_center")
        ground_contacts.append((gx, gy))

        t_s = obs["frame"]["timestamp_s"]
        score = obs.get("detector_score", 0.0)
        if score is not None and math.isfinite(score):
            detector_scores.append(score)

        assoc_state = obs.get("association_state", "tentative")
        if assoc_state == "ambiguous":
            ambiguous_count += 1

        trajectory.append({
            "frame_index": obs["frame"]["index"],
            "timestamp_s": t_s,
            "pts": obs["frame"].get("pts"),
            "bbox_xyxy": bbox,
            "ground_contact_px": [round(gx, 2), round(gy, 2)],
            "detector_score": score,
            "association_state": assoc_state,
        })

        points_for_metrics.append(
            Point(
                time_s=t_s,
                x_px=gx,
                y_px=gy,
                segment_id=f"track_{track_id}",
                identity_review_ref=(review["review_ref"] if review and assoc_state != "ambiguous" else None),
                contact_convention="bottom_center",
            )
        )

    # Calculate consecutive temporal gaps
    gaps = []
    pixel_distance = 0.0
    for prev_obs, curr_obs in zip(trajectory, trajectory[1:]):
        dt = curr_obs["timestamp_s"] - prev_obs["timestamp_s"]
        if dt > max_gap_s:
            gaps.append({
                "start_s": prev_obs["timestamp_s"],
                "end_s": curr_obs["timestamp_s"],
                "gap_s": round(dt, 4),
            })
        dx = curr_obs["ground_contact_px"][0] - prev_obs["ground_contact_px"][0]
        dy = curr_obs["ground_contact_px"][1] - prev_obs["ground_contact_px"][1]
        pixel_distance += math.hypot(dx, dy)

    # Movement calculation with calibration safety
    if calibration is not None and calibration.is_valid:
        try:
            calibration.check_geometry_compatibility(frame_width, frame_height)
            m_summary = movement_summary(
                points_for_metrics,
                calibration=calibration,
                max_gap_s=max_gap_s,
                peak_window_s=peak_window_s,
            )
            movement_data = {
                "status": m_summary.get("status", "unavailable"),
                "calibrated": True,
                "identity_reviewed": identity_reviewed,
                "identity_review_ref": identity_review_ref if identity_reviewed else None,
                "distance_m": round(m_summary["distance_m"], 2) if m_summary.get("distance_m") is not None else None,
                "average_speed_mps": round(m_summary["average_speed_mps"], 2) if m_summary.get("average_speed_mps") is not None else None,
                "maximum_windowed_speed_mps": round(m_summary["maximum_windowed_speed_mps"], 2) if m_summary.get("maximum_windowed_speed_mps") is not None else None,
                "eligible_duration_s": round(m_summary.get("eligible_duration_s", 0.0), 3),
                "excluded_segments": m_summary.get("excluded_segments", []),
                "reason": m_summary.get("reason"),
                "units": "metres and metres_per_second",
            }
        except Exception as exc:
            record_failure(exc,"analytics")
            movement_data = {
                "status": "unavailable",
                "calibrated": True,
                "identity_reviewed": identity_reviewed,
                "identity_review_ref": identity_review_ref if identity_reviewed else None,
                "distance_m": None,
                "average_speed_mps": None,
                "maximum_windowed_speed_mps": None,
                "reason": "Calibration geometry/projection failed; inspect the diagnostic trace before using physical metrics.",
                "units": "metres and metres_per_second",
            }
    else:
        avg_px_speed = (pixel_distance / visible_duration) if visible_duration > 0 else 0.0
        movement_data = {
            "status": "uncalibrated",
            "calibrated": False,
            "identity_reviewed": identity_reviewed,
            "identity_review_ref": identity_review_ref if identity_reviewed else None,
            "distance_m": None,
            "average_speed_mps": None,
            "maximum_windowed_speed_mps": None,
            "image_pixel_distance_px": round(pixel_distance, 1),
            "average_pixel_speed_px_per_s": round(avg_px_speed, 1),
            "reason": (
                f"uncalibrated: {calibration.invalidation_reason}"
                if (calibration is not None and not calibration.is_valid)
                else "uncalibrated: pitch calibration not provided; physical distance/speed suppressed per TL-006/TL-007"
            ),
            "units": "image_pixels",
        }

    movement_data["identity_reviewed"] = review is not None
    movement_data["identity_review_ref"] = review["review_ref"] if review else None

    # Heatmap
    heatmap = compute_spatial_heatmap(
        ground_contacts,
        width=float(frame_width),
        height=float(frame_height),
        grid_cols=16,
        grid_rows=9,
    )

    total_obs = len(observations)
    ambiguous_ratio = round(ambiguous_count / total_obs, 3) if total_obs > 0 else 0.0
    if ambiguous_count == 0:
        quality_label = "high"
    elif ambiguous_ratio < 0.25:
        quality_label = "moderate"
    else:
        quality_label = "contested"

    # Player association details (TL-004)
    confirmed_player_id = None
    player_name = None
    is_target = False
    association_revision = None
    reviewed_by = None
    if association is not None:
        if hasattr(association, "player_id"):
            confirmed_player_id = association.player_id
            player_name = getattr(association, "player_name", None)
            is_target = getattr(association, "is_target", False)
            association_revision = getattr(association, "revision", 1)
            reviewed_by = getattr(association, "reviewed_by", None)
        elif isinstance(association, dict):
            confirmed_player_id = association.get("player_id")
            player_name = association.get("player_name")
            is_target = association.get("is_target", False)
            association_revision = association.get("revision", 1)
            reviewed_by = association.get("reviewed_by")
    # Labels and observation fields cannot self-certify a reviewed association.
    confirmed_player_id = review["player_id"] if review else None
    association_revision = review["revision"] if review else None
    reviewed_by = review["reviewed_by"] if review else None
    is_target = bool(is_target and review)

    track_events = [e for e in (events or []) if e.get("local_track_id") == track_id]

    return {
        "local_track_id": track_id,
        "confirmed_player_id": confirmed_player_id,
        "player_name": player_name,
        "is_target": is_target,
        "identity_reviewed": identity_reviewed,
        "identity_review_ref": identity_review_ref if identity_reviewed else None,
        "association_revision": association_revision,
        "reviewed_by": reviewed_by,
        "identity_review": review,
        "observation_count": total_obs,
        "start_time_s": first_time,
        "end_time_s": last_time,
        "visible_duration_s": visible_duration,
        "mean_detector_score": round(sum(detector_scores) / len(detector_scores), 3) if detector_scores else None,
        "ambiguous_observations": ambiguous_count,
        "ambiguous_ratio": ambiguous_ratio,
        "tracking_quality": quality_label,
        "movement": movement_data,
        "activity_classification": activity_summary(points_for_metrics, calibration=calibration, max_gap_s=max_gap_s),
        "intensity_zones": intensity_zone_summary(points_for_metrics, calibration=calibration, max_gap_s=max_gap_s),
        "gaps_exceeding_max_gap": gaps,
        "spatial_heatmap": heatmap,
        "trajectory": trajectory,
        "tracking_events": track_events,
    }


@diagnosed("analytics")
def generate_match_analytics(
    manifest: dict[str, Any],
    observations: list[dict[str, Any]],
    calibration: Calibration | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    max_gap_s: float = 0.75,
    peak_window_s: float = 1.0,
    associations: Any | None = None,
    tracking_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate complete player roster analytics and match summary."""
    tracks_map = defaultdict(list)
    obs_count = 0
    inferred_width = frame_width
    inferred_height = frame_height
    min_ts = None
    max_ts = None

    for obs in observations:
        obs_count += 1
        tid = obs.get("local_track_id")
        if tid is not None:
            tracks_map[tid].append(obs)
        if "frame" in obs:
            f_meta = obs["frame"]
            if "width" in f_meta and (inferred_width is None or f_meta["width"] > inferred_width):
                inferred_width = f_meta["width"]
            if "height" in f_meta and (inferred_height is None or f_meta["height"] > inferred_height):
                inferred_height = f_meta["height"]
            ts = f_meta.get("timestamp_s")
            if ts is not None:
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if max_ts is None or ts > max_ts:
                    max_ts = ts

    frame_width = inferred_width if inferred_width is not None else 1920
    frame_height = inferred_height if inferred_height is not None else 1080

    roster = []
    all_contacts = []
    total_ambiguous = 0

    for tid in sorted(tracks_map.keys()):
        track_obs = tracks_map[tid]
        assoc = None
        if associations is not None:
            if hasattr(associations, "get_association"):
                if associations.run_id == manifest.get("run_id"):
                    assoc = associations.get_association(tid)
            elif isinstance(associations, dict):
                assoc_map = associations.get("associations", associations)
                assoc = assoc_map.get(str(tid)) or assoc_map.get(tid)
                if isinstance(assoc, dict):
                    assoc = {**assoc, "run_id": assoc.get("run_id", associations.get("run_id"))}

        analytics = compute_track_analytics(
            track_id=tid,
            observations=track_obs,
            frame_width=frame_width,
            frame_height=frame_height,
            calibration=calibration,
            max_gap_s=max_gap_s,
            peak_window_s=peak_window_s,
            association=assoc,
            events=tracking_events,
            run_id=manifest.get("run_id"),
        )
        roster.append(analytics)
        total_ambiguous += analytics["ambiguous_observations"]
        for pt in analytics["trajectory"]:
            all_contacts.append(tuple(pt["ground_contact_px"]))

    # Match-level combined spatial occupancy heatmap
    team_heatmap = compute_spatial_heatmap(
        all_contacts,
        width=float(frame_width),
        height=float(frame_height),
        grid_cols=16,
        grid_rows=9,
    )

    start_time = min_ts if min_ts is not None else 0.0
    end_time = max_ts if max_ts is not None else 0.0
    total_duration = round(end_time - start_time, 4)

    is_synthetic = bool(manifest.get("synthetic", False))
    banner = (
        "SYNTHETIC CONTRACT FIXTURE — NOT FIELD EVIDENCE"
        if is_synthetic
        else "EXPERIMENTAL BASELINE — ROSTER IDENTITY UNCONFIRMED"
    )

    confirmed_count = sum(1 for r in roster if r.get("confirmed_player_id") is not None)
    target_roster = [r for r in roster if r.get("is_target")]

    corrections_list = []
    if associations is not None:
        if hasattr(associations, "corrections"):
            corrections_list = list(associations.corrections)
        elif isinstance(associations, dict):
            corrections_list = list(associations.get("corrections", []))

    return {
        "schema_version": "touchline.analytics.v1",
        "run_id": manifest.get("run_id"),
        "synthetic": is_synthetic,
        "banner": banner,
        "session": {
            "source": manifest.get("source"),
            "source_sha256": manifest.get("source_sha256"),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "duration_s": total_duration,
            "decoded_frames": manifest.get("decoded_frames", 0),
            "selected_frames": manifest.get("selected_frames", 0),
            "observation_count": obs_count,
            "sampler": manifest.get("sampler"),
            "detector": manifest.get("detector"),
        },
        "calibration": {
            "calibrated": calibration is not None and calibration.is_valid,
            "units": calibration.units if calibration is not None else None,
            "status": "valid" if (calibration is not None and calibration.is_valid) else ("invalid" if calibration is not None else "unavailable"),
            "invalidation_reason": calibration.invalidation_reason if (calibration is not None and not calibration.is_valid) else None,
        },
        "quality_indicators": {
            "total_tracks": len(tracks_map),
            "total_ambiguous_observations": total_ambiguous,
            "clean_tracks": sum(1 for r in roster if r["ambiguous_observations"] == 0),
            "contested_tracks": sum(1 for r in roster if r["tracking_quality"] == "contested"),
            "confirmed_players_count": confirmed_count,
            "target_tracks_count": len(target_roster),
            "corrections_count": len(corrections_list),
        },
        "target_players": [
            {
                "local_track_id": t["local_track_id"],
                "player_id": t["confirmed_player_id"],
                "player_name": t["player_name"],
                "visible_duration_s": t["visible_duration_s"],
            }
            for t in target_roster
        ],
        "corrections": corrections_list,
        "team_spatial_heatmap": team_heatmap,
        "player_roster": roster,
    }


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize coach-reviewed attempt outcomes with explicit denominators (TL-005, TL-005.2).

    Rules:
    - Conclusive outcomes ('success'/'conclusive_success' and 'error'/'conclusive_error')
      form the conclusive denominator.
    - 'inconclusive' (obscured/uncertain) and 'pending' (unreviewed) are reported
      separately as exclusions and NEVER counted as failures in the denominator.
    - Zero conclusive attempts yields status='unavailable' and success_rate=None.
    """
    if not isinstance(attempts, list):
        raise ValueError("attempts must be a list of dictionaries")

    success_count = 0
    error_count = 0
    inconclusive_count = 0
    pending_count = 0

    for idx, att in enumerate(attempts):
        if not isinstance(att, dict):
            raise ValueError(f"attempts[{idx}] must be a dictionary")
        outcome = str(att.get("outcome", "pending")).strip().lower()
        if outcome in ("success", "conclusive_success"):
            success_count += 1
        elif outcome in ("error", "conclusive_error"):
            error_count += 1
        elif outcome == "inconclusive":
            inconclusive_count += 1
        elif outcome == "pending":
            pending_count += 1
        else:
            raise ValueError(f"unsupported attempt outcome at index {idx}: {outcome!r}")

    conclusive_total = success_count + error_count
    if conclusive_total > 0:
        rate = round(success_count / conclusive_total, 4)
        status = "available"
    else:
        rate = None
        status = "unavailable"

    return {
        "status": status,
        "total_attempts": len(attempts),
        "conclusive_denominator": conclusive_total,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": rate,
        "excluded_from_denominator": {
            "inconclusive_count": inconclusive_count,
            "pending_count": pending_count,
            "total_excluded": inconclusive_count + pending_count,
        },
    }
