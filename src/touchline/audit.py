"""Stage-by-stage audit and miss ledger for TL-030.1.

Compares human-reviewed ground truth player boxes against pipeline run artifacts
(manifest.json, coverage.jsonl, observations.jsonl) without fabricating
unobserved internal states or unsubstantiated tracking success claims.
"""
from .diagnostics import diagnosed
import json
import math
from pathlib import Path
from typing import Any
from .contracts import Box

# COCO pixel-area size bin thresholds (pixels^2)
# small: area < 32^2 (1024)
# medium: 32^2 <= area < 96^2 (1024 <= area < 9216)
# large: area >= 96^2 (area >= 9216)
AREA_SMALL_LIMIT = 32.0 ** 2
AREA_MEDIUM_LIMIT = 96.0 ** 2


def get_pixel_area_bin(width: float, height: float) -> str:
    """Return COCO pixel-area size category."""
    area = width * height
    if area < AREA_SMALL_LIMIT:
        return "small"
    if area < AREA_MEDIUM_LIMIT:
        return "medium"
    return "large"


@diagnosed("audit")
def audit_ground_truth(run_dir: str | Path, ground_truth: list[dict[str, Any]], iou_threshold: float = 0.5, score_threshold: float = 0.5) -> dict[str, Any]:
    """Audit human-reviewed ground truth against persisted pipeline artifacts.

    Supports positive player bounding boxes as well as negative-control frames
    (empty background with bbox_xyxy: null and target_category: 'empty_background_control')
    to verify absence of false positives.

    Args:
        run_dir: Path to output directory containing manifest.json, coverage.jsonl, observations.jsonl.
        ground_truth: List of ground-truth annotations with source_id, bbox_xyxy (or null for empty
            background controls), and pts or timestamp_s.
        iou_threshold: Minimum IoU required to consider a detection matching (default 0.5).

    Returns:
        Structured audit report with overall summary and per-annotation ledger.
        The summary reports:
            - total_ground_truth: Total annotations audited.
            - detected_and_persisted: Number of matched positive detections.
            - sampling_skip: Annotations on decoded frames intentionally skipped by sampler.
            - unobserved_in_pipeline: Positive annotations on sampled frames with no matching detection.
            - unrecorded_frame: Frames absent from coverage ledger.
            - empty_controls_clean: Evaluated negative controls with 0 false-positive observations.
            - false_positive_frames: Negative control frames with >= 1 false-positive observations.
            - false_positive_observations: Total observation count across negative control frames.
            - empty_controls: Detailed breakdown dictionary for negative controls.
            - by_size_bin: Breakdown of positive annotations by COCO pixel-area size bin.
    """
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)) or not math.isfinite(iou_threshold) or not (0.0 < iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be a finite float in (0, 1]")
    if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)) or not math.isfinite(score_threshold) or not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be a finite float in [0, 1]")
    if not isinstance(ground_truth, list):
        raise ValueError("ground_truth must be a list of annotation dictionaries")

    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_path}")

    manifest_file = run_path / "manifest.json"
    coverage_file = run_path / "coverage.jsonl"
    observations_file = run_path / "observations.jsonl"

    for artifact in (manifest_file, coverage_file, observations_file):
        if not artifact.is_file():
            raise FileNotFoundError(f"required run artifact missing: {artifact}")

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"malformed manifest.json: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    run_source = manifest.get("source")
    if run_source is None:
        raise ValueError("manifest.json missing required 'source' field")

    run_source_id = None
    if isinstance(run_source, dict):
        run_source_id = run_source.get("id")
    elif isinstance(run_source, str):
        run_source_id = run_source

    if not run_source_id or not isinstance(run_source_id, str):
        raise ValueError("manifest.json 'source' missing a valid non-empty string identifier")

    # Read coverage records and validate uniqueness of (source_id, pts)
    coverage_by_pts: dict[tuple[str, int], dict[str, Any]] = {}
    coverage_by_time: list[tuple[str, float, dict[str, Any]]] = []
    with coverage_file.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise ValueError(f"malformed JSON on line {line_num} in {coverage_file}: {exc}") from exc

            if not isinstance(rec, dict):
                raise ValueError(f"line {line_num} in {coverage_file} must be a JSON object")

            frame_meta = rec.get("frame")
            if not isinstance(frame_meta, dict) or "source_id" not in frame_meta:
                raise ValueError(f"malformed coverage record on line {line_num} in {coverage_file}: missing frame metadata")

            s_id = frame_meta["source_id"]
            pts = frame_meta.get("pts")
            ts = frame_meta.get("timestamp_s")

            if pts is not None:
                if not isinstance(pts, int) or pts < 0:
                    raise ValueError(f"line {line_num} in {coverage_file} has invalid pts: {pts}")
                if (s_id, pts) in coverage_by_pts:
                    raise ValueError(f"duplicate coverage record found for source '{s_id}' and pts {pts} on line {line_num}")
                coverage_by_pts[(s_id, pts)] = rec

            if ts is not None and not isinstance(ts, bool) and isinstance(ts, (int, float)) and math.isfinite(ts):
                coverage_by_time.append((s_id, float(ts), rec))

    # Read observations records
    observations_by_pts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    observations_by_time: list[tuple[str, float, dict[str, Any]]] = []
    with observations_file.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
            except Exception as exc:
                raise ValueError(f"malformed JSON on line {line_num} in {observations_file}: {exc}") from exc

            if not isinstance(obs, dict):
                raise ValueError(f"line {line_num} in {observations_file} must be a JSON object")

            frame_meta = obs.get("frame")
            if not isinstance(frame_meta, dict) or "source_id" not in frame_meta:
                raise ValueError(f"malformed observation record on line {line_num} in {observations_file}: missing frame metadata")

            s_id = frame_meta["source_id"]
            pts = frame_meta.get("pts")
            ts = frame_meta.get("timestamp_s")

            if pts is not None and isinstance(pts, int):
                observations_by_pts.setdefault((s_id, pts), []).append(obs)
            if ts is not None and not isinstance(ts, bool) and isinstance(ts, (int, float)) and math.isfinite(ts):
                observations_by_time.append((s_id, float(ts), obs))

    # Read optional raw_proposals records if present (TL-030.1 / TL-030.3 stage diagnostics)
    raw_proposals_file = run_path / "raw_proposals.jsonl"
    proposals_by_pts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    proposals_by_time: list[tuple[str, float, dict[str, Any]]] = []
    has_raw_proposals = raw_proposals_file.is_file()
    if has_raw_proposals:
        with raw_proposals_file.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    prop = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"malformed JSON on line {line_num} in {raw_proposals_file}: {exc}") from exc
                if isinstance(prop, dict):
                    frame_meta = prop.get("frame", {})
                    s_id = frame_meta.get("source_id", run_source_id)
                    pts = frame_meta.get("pts")
                    ts = frame_meta.get("timestamp_s")
                    if pts is not None and isinstance(pts, int):
                        proposals_by_pts.setdefault((s_id, pts), []).append(prop)
                    if ts is not None and not isinstance(ts, bool) and isinstance(ts, (int, float)) and math.isfinite(ts):
                        proposals_by_time.append((s_id, float(ts), prop))

    # Validate ground truth inputs and validate source matching
    frame_annotation_types: dict[tuple[str, Any], str] = {}
    for i, gt in enumerate(ground_truth):
        if not isinstance(gt, dict):
            raise ValueError(f"ground_truth[{i}] must be a dictionary")
        if "source_id" not in gt or not isinstance(gt["source_id"], str) or not gt["source_id"]:
            raise ValueError(f"ground_truth[{i}] missing non-empty string 'source_id'")
        if gt["source_id"] != run_source_id:
            raise ValueError(
                f"ground_truth[{i}] source_id '{gt['source_id']}' does not match run source_id '{run_source_id}'"
            )

        pts = gt.get("pts")
        ts = gt.get("timestamp_s")
        if pts is None and ts is None:
            raise ValueError(f"ground_truth[{i}] requires at least 'pts' or 'timestamp_s'")
        if pts is not None and (isinstance(pts, bool) or not isinstance(pts, int) or pts < 0):
            raise ValueError(f"ground_truth[{i}] 'pts' must be a non-negative integer")
        if ts is not None and (isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts) or ts < 0):
            raise ValueError(f"ground_truth[{i}] 'timestamp_s' must be a finite non-negative number")

        if "bbox_xyxy" not in gt:
            raise ValueError(f"ground_truth[{i}] missing required 'bbox_xyxy' field (must be null for empty_background_control or 4 numbers for positive boxes)")

        raw_box = gt.get("bbox_xyxy")

        # Determine target_category from top-level or annotation_metadata
        target_cat = gt.get("target_category")
        if target_cat is None and isinstance(gt.get("annotation_metadata"), dict):
            target_cat = gt["annotation_metadata"].get("target_category")

        frame_key = (gt["source_id"], pts if pts is not None else ts)
        if raw_box is None:
            if target_cat != "empty_background_control":
                raise ValueError(
                    f"ground_truth[{i}] has bbox_xyxy null but target_category is not 'empty_background_control'"
                )
            if frame_key in frame_annotation_types:
                prev_type = frame_annotation_types[frame_key]
                if prev_type == "empty_background_control":
                    raise ValueError(f"ground_truth[{i}] has duplicate empty_background_control for frame {frame_key}")
                else:
                    raise ValueError(f"ground_truth[{i}] has contradictory annotations: empty_background_control on frame {frame_key} which also has positive boxes")
            frame_annotation_types[frame_key] = "empty_background_control"
        else:
            if target_cat == "empty_background_control":
                raise ValueError(
                    f"ground_truth[{i}] cannot specify bbox_xyxy for an empty_background_control"
                )
            if frame_key in frame_annotation_types and frame_annotation_types[frame_key] == "empty_background_control":
                raise ValueError(f"ground_truth[{i}] has contradictory annotations: positive box on frame {frame_key} which is already labelled empty_background_control")
            frame_annotation_types[frame_key] = "positive"

            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw_box):
                raise ValueError(f"ground_truth[{i}] 'bbox_xyxy' must be a 4-element sequence of numbers or null for empty_background_control")
            try:
                Box(*raw_box)
            except ValueError as exc:
                raise ValueError(f"ground_truth[{i}] invalid box geometry: {exc}") from exc

    ledger_entries = []
    summary_counts = {
        "total_ground_truth": len(ground_truth),
        "detected_and_persisted": 0,
        "sampling_skip": 0,
        "unobserved_in_pipeline": 0,
        "detector_raw_miss": 0,
        "confidence_filtered": 0,
        "tracker_unconfirmed": 0,
        "unrecorded_frame": 0,
        "empty_controls_clean": 0,
        "false_positive_frames": 0,
        "false_positive_observations": 0,
        "empty_controls": {
            "total": 0,
            "evaluated": 0,
            "clean": 0,
            "false_positive_frames": 0,
            "false_positive_observations": 0,
            "unsampled": 0,
            "unrecorded": 0,
        },
        "by_size_bin": {
            "small": {"total": 0, "detected": 0, "missed": 0},
            "medium": {"total": 0, "detected": 0, "missed": 0},
            "large": {"total": 0, "detected": 0, "missed": 0},
        },
    }

    for i, gt in enumerate(ground_truth):
        s_id = gt["source_id"]
        pts = gt.get("pts")
        ts = gt.get("timestamp_s")
        raw_box = gt.get("bbox_xyxy")
        is_negative_control = (raw_box is None)

        if is_negative_control:
            gt_box = None
            size_bin = None
            pixel_area = None
            summary_counts["empty_controls"]["total"] += 1
        else:
            gt_box = Box(*raw_box)
            width = gt_box.x2 - gt_box.x1
            height = gt_box.y2 - gt_box.y1
            pixel_area = width * height
            size_bin = get_pixel_area_bin(width, height)
            summary_counts["by_size_bin"][size_bin]["total"] += 1

        # Match frame in coverage
        matched_coverage = None
        if pts is not None and (s_id, pts) in coverage_by_pts:
            matched_coverage = coverage_by_pts[(s_id, pts)]
            # If timestamp is also specified, verify they do not conflict
            if ts is not None and matched_coverage.get("frame", {}).get("timestamp_s") is not None:
                cov_ts = matched_coverage["frame"]["timestamp_s"]
                if not math.isclose(cov_ts, ts, abs_tol=1e-3):
                    raise ValueError(
                        f"ground_truth[{i}] has conflicting pts ({pts}) and timestamp_s ({ts}s): "
                        f"matched frame has timestamp_s {cov_ts}s"
                    )
        elif ts is not None:
            # Documented timestamp fallback with 1ms tolerance
            candidates = [rec for (cand_sid, cand_ts, rec) in coverage_by_time if cand_sid == s_id and math.isclose(cand_ts, ts, abs_tol=1e-3)]
            if len(candidates) == 1:
                matched_coverage = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(f"ambiguous timestamp match for ground_truth[{i}] at {ts}s: multiple coverage frames found")

        gt_id = gt.get("label_id", f"gt_{i+1}")
        frame_id_record = {
            "source_id": s_id,
            "pts": pts if pts is not None else (matched_coverage["frame"]["pts"] if matched_coverage else None),
            "timestamp_s": ts if ts is not None else (matched_coverage["frame"]["timestamp_s"] if matched_coverage else None),
            "index": gt.get("frame_index", matched_coverage["frame"]["index"] if matched_coverage else None),
        }

        # Case 1: Frame not found in coverage ledger
        # We cannot conclude the decoder crashed or failed; we only know the frame was not recorded.
        if matched_coverage is None:
            summary_counts["unrecorded_frame"] += 1
            if is_negative_control:
                summary_counts["empty_controls"]["unrecorded"] += 1
                ledger_entries.append({
                    "gt_id": gt_id,
                    "target_category": "empty_background_control",
                    "frame": frame_id_record,
                    "gt_bbox_xyxy": None,
                    "size_bin": None,
                    "pixel_area": None,
                    "diagnosis": "unrecorded_frame",
                    "evidence": {
                        "coverage_record_found": False,
                        "decode_status": "unknown_absent_from_coverage_ledger",
                        "sampling_state": "unknown",
                        "observations_count": 0,
                        "observations": [],
                        "matched_observation": None,
                    },
                    "stage_breakdown": {
                        "decode": "unknown_absent_from_coverage_ledger",
                        "sampling": "unknown",
                        "detection": "unknown",
                        "tracking": "unknown",
                        "export": "unknown",
                    },
                    "notes": "Negative-control frame was absent from coverage ledger; decode and sampling status are unknown.",
                })
            else:
                summary_counts["by_size_bin"][size_bin]["missed"] += 1
                ledger_entries.append({
                    "gt_id": gt_id,
                    "frame": frame_id_record,
                    "gt_bbox_xyxy": gt_box.values(),
                    "size_bin": size_bin,
                    "pixel_area": pixel_area,
                    "diagnosis": "unrecorded_frame",
                    "evidence": {
                        "coverage_record_found": False,
                        "decode_status": "unknown_absent_from_coverage_ledger",
                        "sampling_state": "unknown",
                        "matched_observation": None,
                    },
                    "stage_breakdown": {
                        "decode": "unknown_absent_from_coverage_ledger",
                        "sampling": "unknown",
                        "detection": "unknown",
                        "tracking": "unknown",
                        "export": "unknown",
                    },
                })
            continue

        sampling_state = matched_coverage.get("state")

        # Case 2: Frame intentionally skipped by sampler
        if sampling_state == "unsampled_by_sampler":
            summary_counts["sampling_skip"] += 1
            if is_negative_control:
                summary_counts["empty_controls"]["unsampled"] += 1
                ledger_entries.append({
                    "gt_id": gt_id,
                    "target_category": "empty_background_control",
                    "frame": frame_id_record,
                    "gt_bbox_xyxy": None,
                    "size_bin": None,
                    "pixel_area": None,
                    "diagnosis": "sampling_skip",
                    "evidence": {
                        "coverage_record_found": True,
                        "decode_status": "decoded",
                        "sampling_state": "unsampled_by_sampler",
                        "evaluation_status": "not_evaluated_skipped_by_sampler",
                        "observations_count": 0,
                        "observations": [],
                        "matched_observation": None,
                    },
                    "stage_breakdown": {
                        "decode": "success",
                        "sampling": "skipped_by_sampler",
                        "detection": "not_run_by_design",
                        "tracking": "not_run_by_design",
                        "export": "not_run_by_design",
                    },
                    "notes": "Negative-control frame was decoded but skipped by sampler; frame was not evaluated for detections.",
                })
            else:
                summary_counts["by_size_bin"][size_bin]["missed"] += 1
                ledger_entries.append({
                    "gt_id": gt_id,
                    "frame": frame_id_record,
                    "gt_bbox_xyxy": gt_box.values(),
                    "size_bin": size_bin,
                    "pixel_area": pixel_area,
                    "diagnosis": "sampling_skip",
                    "evidence": {
                        "coverage_record_found": True,
                        "decode_status": "decoded",
                        "sampling_state": "unsampled_by_sampler",
                        "matched_observation": None,
                    },
                    "stage_breakdown": {
                        "decode": "success",
                        "sampling": "skipped_by_sampler",
                        "detection": "not_run_by_design",
                        "tracking": "not_run_by_design",
                        "export": "not_run_by_design",
                    },
                })
            continue

        # Case 3 & 4: Frame was sampled - check observations
        frame_observations: list[dict[str, Any]] = []
        if pts is not None and (s_id, pts) in observations_by_pts:
            frame_observations = observations_by_pts[(s_id, pts)]
        elif matched_coverage.get("frame", {}).get("pts") is not None:
            cov_pts = matched_coverage["frame"]["pts"]
            frame_observations = observations_by_pts.get((s_id, cov_pts), [])
        elif ts is not None:
            frame_observations = [obs for (cand_sid, cand_ts, obs) in observations_by_time if cand_sid == s_id and math.isclose(cand_ts, ts, abs_tol=1e-3)]

        if is_negative_control:
            summary_counts["empty_controls"]["evaluated"] += 1
            if len(frame_observations) == 0:
                summary_counts["empty_controls"]["clean"] += 1
                summary_counts["empty_controls_clean"] += 1
                ledger_entries.append({
                    "gt_id": gt_id,
                    "target_category": "empty_background_control",
                    "frame": frame_id_record,
                    "gt_bbox_xyxy": None,
                    "size_bin": None,
                    "pixel_area": None,
                    "diagnosis": "empty_control_verified_clean",
                    "evidence": {
                        "coverage_record_found": True,
                        "decode_status": "decoded",
                        "sampling_state": sampling_state,
                        "observations_count": 0,
                        "observations": [],
                        "matched_observation": None,
                    },
                    "stage_breakdown": {
                        "decode": "success",
                        "sampling": "selected",
                        "detection": "zero_observations_on_empty_control",
                        "tracking": "not_applicable",
                        "export": "clean",
                    },
                    "notes": "Frame was sampled and verified to contain zero persisted observations.",
                })
            else:
                summary_counts["empty_controls"]["false_positive_frames"] += 1
                summary_counts["false_positive_frames"] += 1
                summary_counts["empty_controls"]["false_positive_observations"] += len(frame_observations)
                summary_counts["false_positive_observations"] += len(frame_observations)
                observed_details = []
                for obs in frame_observations:
                    observed_details.append({
                        "local_track_id": obs.get("local_track_id"),
                        "bbox_xyxy": obs.get("bbox_xyxy"),
                        "detector_score": obs.get("detector_score"),
                        "association_state": obs.get("association_state", "unrecorded"),
                    })
                ledger_entries.append({
                    "gt_id": gt_id,
                    "target_category": "empty_background_control",
                    "frame": frame_id_record,
                    "gt_bbox_xyxy": None,
                    "size_bin": None,
                    "pixel_area": None,
                    "diagnosis": "false_positive_observation",
                    "evidence": {
                        "coverage_record_found": True,
                        "decode_status": "decoded",
                        "sampling_state": sampling_state,
                        "observations_count": len(frame_observations),
                        "observations": observed_details,
                        "matched_observation": None,
                    },
                    "stage_breakdown": {
                        "decode": "success",
                        "sampling": "selected",
                        "detection": "spurious_or_unlabelled_detection",
                        "tracking": "local_track_assigned" if any(obs.get("local_track_id") is not None for obs in frame_observations) else "unassigned",
                        "export": "persisted_in_observations",
                    },
                    "notes": (
                        f"{len(frame_observations)} observation(s) persisted on a frame labelled as containing no people. "
                        "This diagnosis indicates an observation exists on an empty control frame, not that the detection "
                        "is definitively a non-player without human review."
                    ),
                })
            continue

        best_obs = None
        best_iou = 0.0
        for obs in frame_observations:
            obs_box_vals = obs.get("bbox_xyxy")
            if obs_box_vals and len(obs_box_vals) == 4:
                obs_box = Box(*obs_box_vals)
                iou = gt_box.iou(obs_box)
                if iou > best_iou:
                    best_iou = iou
                    best_obs = obs

        if best_iou >= iou_threshold and best_obs is not None:
            summary_counts["detected_and_persisted"] += 1
            summary_counts["by_size_bin"][size_bin]["detected"] += 1
            assoc_state = best_obs.get("association_state", "unrecorded")
            local_tid = best_obs.get("local_track_id")
            ledger_entries.append({
                "gt_id": gt_id,
                "frame": frame_id_record,
                "gt_bbox_xyxy": gt_box.values(),
                "size_bin": size_bin,
                "pixel_area": pixel_area,
                "diagnosis": "detected_and_persisted",
                "evidence": {
                    "coverage_record_found": True,
                    "decode_status": "decoded",
                    "sampling_state": sampling_state,
                    "matched_observation": {
                        "local_track_id": local_tid,
                        "bbox_xyxy": best_obs.get("bbox_xyxy"),
                        "detector_score": best_obs.get("detector_score"),
                        "association_state": assoc_state,
                        "iou": best_iou,
                    },
                },
                "stage_breakdown": {
                    "decode": "success",
                    "sampling": "selected",
                    "detection": "detected",
                    "detection_persistence": "persisted_in_observations",
                    "tracking_status": f"local_track_assigned_{assoc_state}",
                    "export": "persisted",
                },
                "notes": "Detection is matched and persisted. Local track ID and association state are recorded as assigned by the baseline tracker; single-frame matching does not establish cross-frame identity continuity.",
            })
        else:
            summary_counts["unobserved_in_pipeline"] += 1
            summary_counts["by_size_bin"][size_bin]["missed"] += 1

            frame_proposals = []
            if has_raw_proposals:
                if pts is not None:
                    frame_proposals = proposals_by_pts.get((run_source_id, pts), [])
                elif matched_time_record is not None:
                    matched_ts = matched_time_record[1]
                    frame_proposals = [p for s_id, p_ts, p in proposals_by_time if s_id == run_source_id and math.isclose(p_ts, matched_ts, abs_tol=1e-3)]

            best_prop = None
            best_prop_iou = 0.0
            for prop in frame_proposals:
                p_box_vals = prop.get("bbox_xyxy")
                if p_box_vals and len(p_box_vals) == 4:
                    iou = gt_box.iou(Box(*p_box_vals))
                    if iou > best_prop_iou:
                        best_prop_iou = iou
                        best_prop = prop

            if has_raw_proposals:
                if best_prop_iou < iou_threshold or best_prop is None:
                    diag = "detector_raw_miss"
                    summary_counts["detector_raw_miss"] += 1
                    stage_breakdown = {
                        "decode": "success",
                        "sampling": "selected",
                        "detector_proposals": "failed_no_candidate_box",
                        "detector_filters": "not_reached",
                        "tracking": "not_reached",
                        "export": "not_persisted",
                    }
                    notes = "Frame was processed by detector but model generated no candidate proposal matching this person."
                else:
                    prop_score = best_prop.get("score", 0.0)
                    if prop_score < score_threshold:
                        diag = "confidence_filtered"
                        summary_counts["confidence_filtered"] += 1
                        stage_breakdown = {
                            "decode": "success",
                            "sampling": "selected",
                            "detector_proposals": "generated",
                            "detector_filters": f"filtered_sub_threshold_score_{prop_score:.3f}",
                            "tracking": "not_reached",
                            "export": "not_persisted",
                        }
                        notes = f"Detector proposed a matching box (IoU {best_prop_iou:.2f}) but score ({prop_score:.3f}) was below threshold."
                    else:
                        diag = "tracker_unconfirmed"
                        summary_counts["tracker_unconfirmed"] += 1
                        stage_breakdown = {
                            "decode": "success",
                            "sampling": "selected",
                            "detector_proposals": "generated",
                            "detector_filters": "passed",
                            "tracking": "dropped_or_unassociated",
                            "export": "not_persisted",
                        }
                        notes = f"Detection passed confidence filtering (IoU {best_prop_iou:.2f}, score {prop_score:.3f}) but was not confirmed by tracker."
            else:
                diag = "unobserved_in_pipeline"
                stage_breakdown = {
                    "decode": "success",
                    "sampling": "selected",
                    "detector_proposals": "unknown_insufficient_evidence",
                    "detector_filters": "unknown_insufficient_evidence",
                    "tracking": "unknown_insufficient_evidence",
                    "export": "not_persisted",
                }
                notes = (
                    "Frame was selected for inference but no matching observation was persisted. "
                    "Because raw_proposals.jsonl was not supplied, the exact failing stage remains unknown."
                )

            entry = {
                "gt_id": gt_id,
                "frame": frame_id_record,
                "gt_bbox_xyxy": gt_box.values(),
                "size_bin": size_bin,
                "pixel_area": pixel_area,
                "diagnosis": diag,
                "evidence": {
                    "coverage_record_found": True,
                    "decode_status": "decoded",
                    "sampling_state": sampling_state,
                    "detections_on_frame": matched_coverage.get("detections"),
                    "best_observation_iou": best_iou,
                    "matched_observation": None,
                    "best_raw_proposal_iou": best_prop_iou if has_raw_proposals else None,
                    "best_raw_proposal_score": best_prop.get("score") if best_prop else None,
                },
                "stage_breakdown": stage_breakdown,
                "notes": notes,
            }
            if not has_raw_proposals:
                entry["limitation"] = (
                    "Frame was selected for inference but no matching observation was persisted. "
                    "Because existing artifacts do not record raw model proposals, sub-threshold scores, "
                    "or unassociated tracking states, the exact failing stage between detector proposal, "
                    "filtering, tracking, or export remains unknown with insufficient evidence."
                )
            ledger_entries.append(entry)

    return {
        "schema_version": "touchline.audit_ledger.v1",
        "run_id": manifest.get("run_id"),
        "source": run_source,
        "iou_threshold": iou_threshold,
        "summary": summary_counts,
        "ledger": ledger_entries,
    }
