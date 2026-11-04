"""Grounded player report drafts (TL-015) and comparable two-session development review (TL-016).

Enforces TL-S05 rules:
- Every factual claim/number in a report draft resolves to an explicit metric result or approved
  observation and source revision.
- Unsupported physical/skill conclusions cannot be filled with plausible text (`value=None` + reason).
- Editing text, clips, or supporting identity/calibration revisions invalidates prior approval
  (marks the prior version stale and creates a new draft revision requiring renewed review).
- Drafting or approval never initiates sending or public publishing; recipient authorization is
  checked independently.
- Two-session comparison (TL-016) requires a confirmed player ID (raw local track IDs like Track 7
  in two sessions never establish the same player), matching objective/criteria, and compatible
  drill/coverage. Six of eight (6/8) versus nine of twelve (9/12) conclusive attempts is reported
  as equal 75.0% sample rates (0.0 percentage-point descriptive difference) with explicit exclusions,
  never as proof of durable development or lack of development.
"""
from __future__ import annotations
from .diagnostics import diagnosed

import copy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from .analytics import summarize_attempts
from .metrics import compare_zone_summaries
from .tracking import reviewed_association


def _canonical_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _report_content_hash(report):
    """Bind the approved projection, including player, scope and exact revision."""
    keys = ("schema_version", "report_id", "revision", "player", "source_lineage",
            "objective", "claims", "coach_observations", "recipient_scope")
    return _canonical_hash({key: report[key] for key in keys})


@diagnosed("report.draft")
def generate_player_report_draft(
    analytics: dict[str, Any],
    local_track_id: int,
    *,
    objective: dict[str, Any] | None = None,
    attempts: list[dict[str, Any]] | None = None,
    coach_notes: dict[str, str] | None = None,
    recipient_scope: dict[str, Any] | None = None,
    transcript: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a grounded report draft (TL-015) for a specific track in a session analytics payload.

    Every number/claim links to its exact source revision (`run_id`, `association_revision`,
    `calibration_status`, and source intervals `[start_s, end_s]`).
    """
    if not isinstance(analytics, dict):
        raise ValueError("analytics must be a dictionary")

    roster = analytics.get("player_roster", [])
    track = next((r for r in roster if r.get("local_track_id") == local_track_id), None)
    if track is None:
        raise ValueError(f"local_track_id {local_track_id} not found in analytics player_roster")

    run_id = analytics.get("run_id") or "unknown_run"
    is_synthetic = bool(analytics.get("synthetic", False))
    confirmed_player_id = track.get("confirmed_player_id")
    player_name = track.get("player_name")
    assoc_rev = track.get("association_revision")
    cal_info = analytics.get("calibration", {})
    is_calibrated = bool(cal_info.get("calibrated", False))

    trajectory = track.get("trajectory", [])
    start_s = track.get("start_time_s", 0.0)
    end_s = track.get("end_time_s", 0.0)
    first_pts = trajectory[0].get("pts") if trajectory else None
    last_pts = trajectory[-1].get("pts") if trajectory else None

    source_lineage = {
        "run_id": run_id,
        "source": analytics.get("session", {}).get("source"),
        "synthetic": is_synthetic,
        "local_track_id": local_track_id,
        "confirmed_player_id": confirmed_player_id,
        "association_revision": assoc_rev,
        "identity_review": track.get("identity_review"),
        "calibration_status": cal_info.get("status", "unavailable"),
        "calibration_calibrated": is_calibrated,
        "source_interval_s": [start_s, end_s],
        "source_pts_bounds": [first_pts, last_pts],
        "observation_count": track.get("observation_count", 0),
    }

    # Build grounded claims with explicit evidence references
    claims: list[dict[str, Any]] = []

    # 1. Exposure & visibility claim
    claims.append({
        "claim_id": "exposure_visibility",
        "category": "visibility",
        "status": "supported",
        "value": {
            "visible_duration_s": track.get("visible_duration_s"),
            "observation_count": track.get("observation_count"),
            "ambiguous_observations": track.get("ambiguous_observations"),
            "tracking_quality": track.get("tracking_quality"),
            "gaps_exceeding_max_gap": track.get("gaps_exceeding_max_gap", []),
        },
        "units": "seconds_and_frames",
        "source_ref": {
            "run_id": run_id,
            "local_track_id": local_track_id,
            "interval_s": [start_s, end_s],
            "pts_bounds": [first_pts, last_pts],
            "association_revision": assoc_rev,
        },
        "reason": None,
    })

    # 2. Physical movement claim (strictly gated on calibration)
    movement = track.get("movement", {})
    if is_calibrated and movement.get("status") == "estimated":
        claims.append({
            "claim_id": "physical_workload",
            "category": "physical_metrics",
            "status": "estimated_internal",
            "value": {
                "distance_m": movement.get("distance_m"),
                "average_speed_mps": movement.get("average_speed_mps"),
                "maximum_windowed_speed_mps": movement.get("maximum_windowed_speed_mps"),
                "eligible_duration_s": movement.get("eligible_duration_s"),
            },
            "units": "metres_and_metres_per_second",
            "source_ref": {
                "run_id": run_id,
                "local_track_id": local_track_id,
                "interval_s": [start_s, end_s],
                "calibration_status": cal_info.get("status"),
                "excluded_segments": movement.get("excluded_segments", []),
            },
            "reason": "Internal calibrated trajectory estimate; not validated device telemetry.",
        })
    else:
        claims.append({
            "claim_id": "physical_workload",
            "category": "physical_metrics",
            "status": "suppressed",
            "value": None,
            "units": None,
            "source_ref": {
                "run_id": run_id,
                "local_track_id": local_track_id,
                "calibration_status": cal_info.get("status", "unavailable"),
            },
            "reason": movement.get(
                "reason",
                "uncalibrated: physical distance/speed suppressed; cannot fill with plausible text (TL-015/TL-S05)",
            ),
        })

    # 3. Activity & Intensity Zones claim
    iz = track.get("intensity_zones", {})
    ac = track.get("activity_classification", {})
    if is_calibrated and iz.get("status") == "estimated":
        # Collect high-intensity evidence intervals from activity classification
        hi_intervals = [
            iv for iv in ac.get("intervals", [])
            if iv.get("activity") in ("running", "sprinting")
        ]
        claims.append({
            "claim_id": "activity_and_intensity_zones",
            "category": "workload_distribution",
            "status": "estimated_internal",
            "value": {
                "class_durations_s": ac.get("class_durations_s"),
                "unknown_duration_s": ac.get("unknown_duration_s"),
                "high_speed_running_time_s": iz.get("high_speed_running_time_s"),
                "high_speed_running_distance_m": iz.get("high_speed_running_distance_m"),
                "zones": iz.get("zones"),
            },
            "units": "seconds_and_metres",
            "source_ref": {
                "run_id": run_id,
                "local_track_id": local_track_id,
                "policy_version": iz.get("policy_version"),
                "high_intensity_intervals": hi_intervals,
            },
            "reason": None,
        })
    else:
        claims.append({
            "claim_id": "activity_and_intensity_zones",
            "category": "workload_distribution",
            "status": "suppressed",
            "value": None,
            "units": None,
            "source_ref": {
                "run_id": run_id,
                "local_track_id": local_track_id,
            },
            "reason": iz.get("reason", "suppressed: valid pitch calibration required"),
        })

    # 4. Reviewed attempt outcomes claim (TL-005 / TL-015)
    attempt_list = list(attempts) if attempts is not None else []
    attempt_summary = summarize_attempts(attempt_list)
    claims.append({
        "claim_id": "objective_attempts",
        "category": "reviewed_attempts",
        "status": attempt_summary["status"],
        "value": attempt_summary if attempt_summary["status"] == "available" else None,
        "exclusions": attempt_summary["excluded_from_denominator"],
        "total_recorded_attempts": attempt_summary["total_attempts"],
        "units": "conclusive_ratio",
        "source_ref": {
            "run_id": run_id,
            "local_track_id": local_track_id,
            "objective_id": (objective or {}).get("objective_id"),
            "criteria_revision": (objective or {}).get("criteria_revision", "v1"),
            "attempt_intervals": [
                {
                    "attempt_id": a.get("attempt_id", i + 1),
                    "outcome": a.get("outcome"),
                    "timestamp_s": a.get("timestamp_s"),
                    "pts": a.get("pts"),
                }
                for i, a in enumerate(attempt_list)
            ],
        },
        "reason": (
            None
            if attempt_summary["status"] == "available"
            else "No conclusive reviewed attempts available; cannot synthesize skill completion text."
        ),
    })

    # 5. Deferred/unsupported constructs explicitly blocked (TL-S05)
    claims.append({
        "claim_id": "deferred_constructs",
        "category": "unsupported_constructs",
        "status": "unavailable",
        "value": None,
        "units": None,
        "source_ref": {"run_id": run_id},
        "reason": "Fatigue, calories, stamina, power, and composite player ratings are not supported by defensible evidence.",
    })

    notes = coach_notes or {}
    aligned_voice_cues = []
    if transcript and isinstance(transcript.get("segments"), list):
        for seg in transcript["segments"]:
            s_t = seg.get("video_start_s", 0.0)
            if start_s <= s_t <= end_s:
                aligned_voice_cues.append({
                    "segment_id": seg.get("segment_id"),
                    "text": seg.get("text"),
                    "video_start_s": s_t,
                    "video_end_s": seg.get("video_end_s"),
                    "nearest_frame_pts": seg.get("nearest_frame_pts"),
                })
    if transcript and isinstance(transcript.get("manual_cues"), list):
        for cue in transcript["manual_cues"]:
            c_target = cue.get("target_track_id")
            s_t = cue.get("video_start_s", -1.0)
            if (c_target is None or c_target == local_track_id) and start_s <= s_t <= end_s:
                aligned_voice_cues.append({
                    "cue_id": cue.get("cue_id"),
                    "note": cue.get("note"),
                    "video_start_s": s_t,
                    "nearest_frame_pts": cue.get("nearest_frame_pts"),
                })

    coach_observations = {
        "attribution": "coach_subjective_observation",
        "what_went_well": notes.get("what_went_well"),
        "next_focus": notes.get("next_focus"),
        "general_notes": notes.get("general_notes"),
        "aligned_voice_cues": aligned_voice_cues,
    }

    scope = recipient_scope or {
        "team_id": None,
        "authorized_player_id": confirmed_player_id,
        "allowed_recipients": [],
        "parent_sharing_authorized": False,
    }

    report = {
        "schema_version": "touchline.player_report.v1",
        "report_id": f"report-{run_id[:8]}-t{local_track_id}",
        "revision": 1,
        "version_hash": None,
        "publication_state": "draft",
        "review_state": "pending",
        "is_stale": False,
        "stale_reason": None,
        "approval": None,
        "delivery_state": "not_sent",
        "public_published": False,
        "player": {
            "confirmed_player_id": confirmed_player_id,
            "player_name": player_name,
            "local_track_id": local_track_id,
            "identity_confirmed": confirmed_player_id is not None,
        },
        "source_lineage": source_lineage,
        "objective": objective,
        "claims": claims,
        "coach_observations": coach_observations,
        "recipient_scope": scope,
        "history": [],
    }
    report["version_hash"] = _report_content_hash(report)
    return report


@diagnosed("report.approve")
def approve_report_version(
    report: dict[str, Any],
    *,
    reviewer: str,
    expected_version_hash: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Approve an exact report version for sharing (TL-015).

    Rules:
    - Exact `expected_version_hash` must match `report["version_hash"]`.
    - Stale reports or reports without confirmed player identity cannot be approved for sharing.
    - Approval transitions `publication_state` to `'approved_for_sharing'` and `review_state` to `'approved'`,
      but NEVER sends or publicly publishes the report (`delivery_state == 'not_sent'`).
    """
    if not reviewer or not str(reviewer).strip():
        raise ValueError("reviewer identity is required for approval")
    if report.get("is_stale"):
        raise ValueError(f"cannot approve stale report: {report.get('stale_reason')}")
    if (report.get("version_hash") != expected_version_hash
            or _report_content_hash(report) != expected_version_hash):
        raise ValueError(
            f"version_hash mismatch: expected {expected_version_hash}, got {report.get('version_hash')}"
        )
    player = report.get("player", {})
    if (not player.get("identity_confirmed") or not player.get("confirmed_player_id")
            or player.get("confirmed_player_id") != report.get("source_lineage", {}).get("confirmed_player_id")):
        raise ValueError("cannot approve report for sharing without confirmed_player_id (TL-004/TL-015)")
    lineage = report["source_lineage"]
    review = reviewed_association(lineage.get("identity_review"), run_id=lineage.get("run_id"),
                                  local_track_id=player.get("local_track_id"))
    if (review is None or review["player_id"] != player["confirmed_player_id"]
            or review["revision"] != lineage.get("association_revision")):
        raise ValueError("current run-scoped identity review required for approval")

    updated = copy.deepcopy(report)
    updated["publication_state"] = "approved_for_sharing"
    updated["review_state"] = "approved"
    updated["approval"] = {
        "approved_by": str(reviewer).strip(),
        "approved_version_hash": expected_version_hash,
        "approved_revision": updated["revision"],
        "notes": notes,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    updated["delivery_state"] = "not_sent"
    updated["public_published"] = False
    return updated


@diagnosed("report.correct")
def update_report_draft(
    report: dict[str, Any],
    *,
    coach_notes: dict[str, str] | None = None,
    new_association_revision: int | None = None,
    new_calibration_status: str | None = None,
) -> dict[str, Any]:
    """Edit coach text or update underlying source/identity revision (TL-015).

    Any edit to text, clips, or supporting identity/calibration records the prior version in
    `history` (marked stale if previously approved), increments `revision`, recomputes `version_hash`,
    and resets `publication_state='draft'`, `review_state='pending'`, and `approval=None`.
    """
    updated = copy.deepcopy(report)
    prior_snapshot = {
        "revision": updated.get("revision", 1),
        "version_hash": updated.get("version_hash"),
        "publication_state": updated.get("publication_state"),
        "review_state": updated.get("review_state"),
        "approval": updated.get("approval"),
        "marked_stale": True,
    }
    updated.setdefault("history", []).append(prior_snapshot)

    if coach_notes is not None:
        for k in ("what_went_well", "next_focus", "general_notes"):
            if k in coach_notes:
                updated["coach_observations"][k] = coach_notes[k]

    if new_association_revision is not None:
        updated["source_lineage"]["association_revision"] = new_association_revision
        for claim in updated.get("claims", []):
            if "association_revision" in claim.get("source_ref", {}):
                claim["source_ref"]["association_revision"] = new_association_revision

    if new_calibration_status is not None:
        updated["source_lineage"]["calibration_status"] = new_calibration_status

    updated["revision"] = int(updated.get("revision", 1)) + 1
    updated["version_hash"] = _report_content_hash(updated)
    updated["publication_state"] = "draft"
    updated["review_state"] = "pending"
    updated["approval"] = None
    updated["is_stale"] = False
    updated["stale_reason"] = None
    return updated


def check_report_access(
    report: dict[str, Any],
    *,
    requester_id: str,
    requester_role: str,
    target_player_id: str | None = None,
) -> dict[str, Any]:
    """Enforce recipient scope and prevent unauthorized parent access (TL-015 / TL-S05).

    - Coaches on the authorized team may view drafts and approved reports.
    - Parents/guardians may ONLY access reports when:
      1. `publication_state == 'approved_for_sharing'` and `not is_stale`
      2. `recipient_scope['parent_sharing_authorized'] is True`
      3. `requester_id` is in `recipient_scope['allowed_recipients']`
      4. The requested `target_player_id` matches `report['player']['confirmed_player_id']`
         (never exposing another child's report).
    """
    player_id = report.get("player", {}).get("confirmed_player_id")
    scope = report.get("recipient_scope", {})

    if target_player_id is not None and target_player_id != player_id:
        return {
            "authorized": False,
            "reason": "cross_player_access_denied: target_player_id does not match report confirmed_player_id",
        }

    if requester_role == "coach":
        return {"authorized": True, "reason": "coach_access_permitted"}

    if report.get("is_stale"):
        return {"authorized": False, "reason": "stale_report_access_denied"}

    if report.get("publication_state") != "approved_for_sharing" or report.get("approval") is None:
        return {
            "authorized": False,
            "reason": "unapproved_draft_access_denied: report requires exact-version coach approval before sharing",
        }

    if not scope.get("parent_sharing_authorized", False):
        return {
            "authorized": False,
            "reason": "recipient_scope_denied: parent_sharing_authorized is False",
        }

    try:
        current_hash = _report_content_hash(report)
    except (KeyError, TypeError, ValueError):
        return {"authorized": False, "reason": "invalid_report_content"}
    approval = report["approval"]
    if (not isinstance(approval, dict)
            or current_hash != report.get("version_hash")
            or current_hash != approval.get("approved_version_hash")
            or report.get("revision") != approval.get("approved_revision")
            or not approval.get("approved_by")):
        return {"authorized": False, "reason": "approved_content_changed_or_revision_mismatch"}

    allowed = scope.get("allowed_recipients", [])
    if requester_id not in allowed:
        return {
            "authorized": False,
            "reason": f"unauthorized_recipient: {requester_id!r} is not in allowed_recipients",
        }

    return {"authorized": True, "reason": "authorized_recipient_verified"}


@diagnosed("run.export")
def export_approved_player_package(
    report: dict[str, Any],
    output_dir: Any,
    *,
    requester_id: str,
    requester_role: str = "parent",
) -> dict[str, Any]:
    """Export an approved player report package cleanly for parents/coaches (TL-043.1).

    Guards:
    - Verifies access permissions via check_report_access().
    - Stale or mutated reports cannot be exported.
    - Exports exact approved content. Never put secrets in report fields.
    - Exports a safe JSON report, markdown readable summary, and lineage receipt.
    """
    from pathlib import Path
    out_path = Path(output_dir).absolute()
    if any(p.is_symlink() for p in (out_path, *out_path.parents)):
        raise ValueError("Unsafe export directory")

    target_player_id = report.get("player", {}).get("confirmed_player_id")
    access = check_report_access(
        report,
        requester_id=requester_id,
        requester_role=requester_role,
        target_player_id=target_player_id,
    )
    if not access.get("authorized"):
        raise PermissionError(f"Export unauthorized: {access.get('reason')}")

    if (report.get('is_stale') or report.get('publication_state') != 'approved_for_sharing'
            or not isinstance(report.get('approval'), dict)
            or _report_content_hash(report) != report.get('version_hash')
            or report['approval'].get('approved_version_hash') != report.get('version_hash')
            or report['approval'].get('approved_revision') != report.get('revision')):
        raise PermissionError('Exact current approved report required for package export')
    out_path.mkdir(parents=True, exist_ok=False)
    report_file = out_path / "player_report.json"
    summary_file = out_path / "player_summary.md"

    safe_report = copy.deepcopy(report)
    safe_report["delivery_state"] = "exported_for_recipient"
    safe_report["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    safe_report["exported_to"] = {"requester_id": requester_id, "role": requester_role}

    report_file.write_text(json.dumps(safe_report, indent=2) + "\n", encoding="utf-8")

    player_info = safe_report.get("player", {})
    name = player_info.get("player_name") or player_info.get("confirmed_player_id")
    rows = [
        f"# Touchline Player Performance Report — {name}",
        "",
        f"**Player ID:** {player_info.get('confirmed_player_id')} | **Report Revision:** {safe_report.get('revision')} | **Status:** {safe_report.get('publication_state')}",
        f"**Approved by:** {safe_report.get('approval', {}).get('approved_by')} on {safe_report.get('approval', {}).get('approved_at_utc')}",
        "",
        "## Verified Match Observations & Workload",
        "",
    ]
    for claim in safe_report.get("claims", []):
        cid = claim.get("claim_id")
        val = claim.get("value")
        stat = claim.get("status")
        label = cid.replace("_", " ").title()
        if stat == "estimated_internal" and isinstance(val, dict):
            rows.append(f"- **{label}:** {val.get('distance_m')} m covered | {val.get('average_speed_mps')} m/s avg speed | {val.get('maximum_windowed_speed_mps')} m/s peak speed")
        elif stat == "supported" and isinstance(val, dict):
            rows.append(f"- **{label}:** {val.get('visible_duration_s')}s visible duration ({val.get('observation_count')} observations)")
        elif stat in ("suppressed", "unavailable"):
            rows.append(f"- **{label}:** {stat} ({claim.get('reason')})")

    obs = safe_report.get("coach_observations", {})
    rows += [
        "",
        "## Coach Commentary & Tactical Feedback",
        f"- **What went well:** {obs.get('what_went_well') or 'N/A'}",
        f"- **Next focus:** {obs.get('next_focus') or 'N/A'}",
    ]
    voice_cues = obs.get("aligned_voice_cues", [])
    if voice_cues:
        rows += ["", "### Aligned Voice Notes"]
        for vc in voice_cues:
            txt = vc.get("text") or vc.get("note")
            rows.append(f"- `[{vc.get('video_start_s')}s]`: *'{txt}'*")

    summary_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return {
        "status": "exported",
        "report_id": safe_report.get("report_id"),
        "version_hash": safe_report.get("version_hash"),
        "report_file": str(report_file),
        "summary_file": str(summary_file),
    }


@diagnosed("report.compare")
def compare_two_sessions(
    session_a: dict[str, Any] | None,
    session_b: dict[str, Any] | None,
    *,
    min_exposure_s: float = 0.5,
) -> dict[str, Any]:
    """Compare a player's development evidence across two sessions (TL-016 / TL-S05).

    Each session dict expects:
    - `session_id`: str
    - `run_id`: str
    - `local_track_id`: int
    - `confirmed_player_id`: str | None
    - `association_revision`: int | None
    - `objective_id`: str | None
    - `criteria_revision`: str | None
    - `drill_id`: str | None
    - `visible_duration_s`: float
    - `attempts`: list[dict] (optional)
    - `intensity_zones`: dict (optional, from `intensity_zone_summary`)
    - `movement`: dict (optional, from `compute_track_analytics`)
    """
    if session_a is None or session_b is None:
        return {
            "schema_version": "touchline.two_session_comparison.v1",
            "two_session_gate_satisfied": False,
            "comparable": False,
            "status": "incomplete_single_session",
            "incompatibility_reasons": ["two distinct sessions are required; a single-session trial cannot satisfy TL-016"],
            "automatic_improvement_score": None,
            "causal_or_durable_progress_claim": None,
        }

    reasons: list[str] = []

    if any(session_a.get(key) and session_a.get(key) == session_b.get(key)
           for key in ("session_id", "run_id", "source_sha256", "source_video")):
        reasons.append("duplicate_session: Session A and Session B reference the same session/run")
    if any(not session.get("run_id") or not session.get("session_id") for session in (session_a, session_b)):
        reasons.append("missing_session_identity: run_id and session_id required")

    pid_a = session_a.get("confirmed_player_id")
    pid_b = session_b.get("confirmed_player_id")
    if not pid_a or not pid_b:
        reasons.append(
            "unconfirmed_player_identity: confirmed_player_id required in both sessions (matching local_track_id alone never establishes the same player)"
        )
    elif pid_a != pid_b:
        reasons.append(f"player_mismatch: {pid_a!r} vs {pid_b!r}")
    for label, session in (("a", session_a), ("b", session_b)):
        review = reviewed_association(session.get("identity_review"), run_id=session.get("run_id"),
                                      local_track_id=session.get("local_track_id"))
        if (review is None or review["player_id"] != session.get("confirmed_player_id")
                or review["revision"] != session.get("association_revision")):
            reasons.append(f"unreviewed_association_{label}: current run-scoped review required")

    obj_a = session_a.get("objective_id")
    obj_b = session_b.get("objective_id")
    if not obj_a or not obj_b or obj_a != obj_b:
        reasons.append(f"incompatible_objective: {obj_a!r} vs {obj_b!r}")

    crit_a = session_a.get("criteria_revision", "v1")
    crit_b = session_b.get("criteria_revision", "v1")
    if crit_a != crit_b:
        reasons.append(f"incompatible_criteria_revision: {crit_a!r} vs {crit_b!r}")

    drill_a = session_a.get("drill_id")
    drill_b = session_b.get("drill_id")
    if not drill_a or not drill_b or drill_a != drill_b:
        reasons.append(f"incompatible_drill: {drill_a!r} vs {drill_b!r}")

    exp_a = float(session_a.get("visible_duration_s") or 0.0)
    exp_b = float(session_b.get("visible_duration_s") or 0.0)
    if exp_a < min_exposure_s or exp_b < min_exposure_s:
        reasons.append(
            f"insufficient_coverage: visible_duration_s ({exp_a}s, {exp_b}s) below min_exposure_s ({min_exposure_s}s)"
        )

    att_a = summarize_attempts(session_a.get("attempts", []))
    att_b = summarize_attempts(session_b.get("attempts", []))

    rate_a = att_a["success_rate"]
    rate_b = att_b["success_rate"]
    pct_point_diff = (
        round((rate_b - rate_a) * 100.0, 2)
        if (rate_a is not None and rate_b is not None)
        else None
    )

    comparable = len(reasons) == 0

    # Physical zone comparison (only when comparable and both sessions have estimated calibrated zones)
    zones_a = session_a.get("intensity_zones")
    zones_b = session_b.get("intensity_zones")
    zone_comparison = None
    if comparable and isinstance(zones_a, dict) and isinstance(zones_b, dict):
        zone_comparison = compare_zone_summaries(zones_a, zones_b)

    return {
        "schema_version": "touchline.two_session_comparison.v1",
        "two_session_gate_satisfied": comparable,
        "comparable": comparable,
        "status": "comparable_descriptive" if comparable else "incompatible_side_by_side_only",
        "incompatibility_reasons": reasons,
        "automatic_improvement_score": None,
        "causal_or_durable_progress_claim": (
            "Descriptive sample difference only; not proof of durable development or lack of development (TL-016)."
        ),
        "player_id": pid_a if (pid_a and pid_a == pid_b) else None,
        "objective_id": obj_a if (obj_a and obj_a == obj_b) else None,
        "criteria_revision": crit_a if crit_a == crit_b else None,
        "drill_id": drill_a if (drill_a and drill_a == drill_b) else None,
        "session_a_summary": {
            "session_id": session_a.get("session_id"),
            "run_id": session_a.get("run_id"),
            "local_track_id": session_a.get("local_track_id"),
            "confirmed_player_id": pid_a,
            "association_revision": session_a.get("association_revision"),
            "visible_duration_s": exp_a,
            "attempts": att_a,
        },
        "session_b_summary": {
            "session_id": session_b.get("session_id"),
            "run_id": session_b.get("run_id"),
            "local_track_id": session_b.get("local_track_id"),
            "confirmed_player_id": pid_b,
            "association_revision": session_b.get("association_revision"),
            "visible_duration_s": exp_b,
            "attempts": att_b,
        },
        "attempt_rate_comparison": (
            {
                "status": "available" if pct_point_diff is not None else "unavailable",
                "session_a_conclusive_str": f"{att_a['success_count']}/{att_a['conclusive_denominator']}",
                "session_b_conclusive_str": f"{att_b['success_count']}/{att_b['conclusive_denominator']}",
                "session_a_rate_pct": round(rate_a * 100.0, 2) if rate_a is not None else None,
                "session_b_rate_pct": round(rate_b * 100.0, 2) if rate_b is not None else None,
                "descriptive_diff_percentage_points": pct_point_diff if comparable else None,
                "session_a_excluded": att_a["excluded_from_denominator"],
                "session_b_excluded": att_b["excluded_from_denominator"],
                "interpretation_note": (
                    "Equal sample rates (e.g. 6/8 vs 9/12 = 75.0%) reflect descriptive sample parity with stated exclusions, not proof of no development."
                ),
            }
        ),
        "zone_comparison": zone_comparison,
    }



@diagnosed("report.compare")
def compare_two_video_runs(
    run_dir_a: Any,
    run_dir_b: Any,
    *,
    confirmed_player_id: str,
    objective_id: str = "OBJ-DEFAULT",
    criteria_revision: str = "v1",
    drill_id: str = "standard_drill",
    local_track_id_a: int | None = None,
    local_track_id_b: int | None = None,
    attempts_a: list[dict[str, Any]] | None = None,
    attempts_b: list[dict[str, Any]] | None = None,
    min_exposure_s: float = 0.5,
) -> dict[str, Any]:
    """Compare a confirmed player across two analyzed MP4 run directories (TL-016).

    Reads `analytics.json` (and `manifest.json`) from both `run_dir_a` and `run_dir_b`,
    locates the track associated with `confirmed_player_id` (or uses explicit `local_track_id_a`/`b`
    when verified against `confirmed_player_id`), and computes a full two-session comparison
    including exposure, reviewed attempt denominators, movement deltas (distance_m, speed_mps),
    and intensity zone comparisons.
    """
    from pathlib import Path

    dir_a = Path(run_dir_a)
    dir_b = Path(run_dir_b)
    if not (dir_a / "analytics.json").is_file() or not (dir_b / "analytics.json").is_file():
        raise ValueError("Both run_dir_a and run_dir_b must contain analytics.json")

    ana_a = json.loads((dir_a / "analytics.json").read_text(encoding="utf-8"))
    ana_b = json.loads((dir_b / "analytics.json").read_text(encoding="utf-8"))

    def _find_track(ana: dict[str, Any], explicit_tid: int | None) -> dict[str, Any] | None:
        roster = ana.get("player_roster", [])
        if explicit_tid is not None:
            return next((r for r in roster if r.get("local_track_id") == explicit_tid), None)
        return next((r for r in roster if r.get("confirmed_player_id") == confirmed_player_id), None)

    track_a = _find_track(ana_a, local_track_id_a)
    track_b = _find_track(ana_b, local_track_id_b)
    if track_a is None or track_b is None:
        raise ValueError(
            f"Could not find target track for player {confirmed_player_id!r} in one or both runs"
        )

    def _review(folder, ana, track):
        path = folder / "associations.json"
        if not path.is_file():
            return None
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("run_id") != ana.get("run_id"):
            return None
        tid = track.get("local_track_id")
        entry = ledger.get("associations", {}).get(str(tid))
        if not isinstance(entry, dict):
            return None
        review = reviewed_association({**entry, "run_id": entry.get("run_id", ledger["run_id"])},
                                      run_id=ana.get("run_id"), local_track_id=tid)
        if (review is None or review["player_id"] != confirmed_player_id
                or review["player_id"] != track.get("confirmed_player_id")
                or review["revision"] != track.get("association_revision")
                or review != track.get("identity_review")):
            return None
        return review

    review_a = _review(dir_a, ana_a, track_a)
    review_b = _review(dir_b, ana_b, track_b)

    session_a = {
        "session_id": ana_a.get("session", {}).get("session_id") or ana_a.get("run_id"),
        "run_id": ana_a.get("run_id"),
        "source_video": ana_a.get("session", {}).get("source"),
        "source_sha256": ana_a.get("session", {}).get("source_sha256"),
        "local_track_id": track_a.get("local_track_id"),
        "confirmed_player_id": review_a["player_id"] if review_a else None,
        "association_revision": review_a["revision"] if review_a else None,
        "identity_review": review_a,
        "objective_id": objective_id,
        "criteria_revision": criteria_revision,
        "drill_id": drill_id,
        "visible_duration_s": track_a.get("visible_duration_s", 0.0),
        "attempts": attempts_a or [],
        "intensity_zones": track_a.get("intensity_zones"),
        "movement": track_a.get("movement"),
    }
    session_b = {
        "session_id": ana_b.get("session", {}).get("session_id") or ana_b.get("run_id"),
        "run_id": ana_b.get("run_id"),
        "source_video": ana_b.get("session", {}).get("source"),
        "source_sha256": ana_b.get("session", {}).get("source_sha256"),
        "local_track_id": track_b.get("local_track_id"),
        "confirmed_player_id": review_b["player_id"] if review_b else None,
        "association_revision": review_b["revision"] if review_b else None,
        "identity_review": review_b,
        "objective_id": objective_id,
        "criteria_revision": criteria_revision,
        "drill_id": drill_id,
        "visible_duration_s": track_b.get("visible_duration_s", 0.0),
        "attempts": attempts_b or [],
        "intensity_zones": track_b.get("intensity_zones"),
        "movement": track_b.get("movement"),
    }

    result = compare_two_sessions(session_a, session_b, min_exposure_s=min_exposure_s)
    mov_a = track_a.get("movement", {})
    mov_b = track_b.get("movement", {})
    if (
        result["comparable"]
        and mov_a.get("calibrated")
        and mov_b.get("calibrated")
        and mov_a.get("distance_m") is not None
        and mov_b.get("distance_m") is not None
    ):
        result["physical_movement_comparison"] = {
            "status": "estimated_internal",
            "session_a_distance_m": mov_a["distance_m"],
            "session_b_distance_m": mov_b["distance_m"],
            "delta_distance_m": round(mov_b["distance_m"] - mov_a["distance_m"], 2),
            "session_a_avg_speed_mps": mov_a["average_speed_mps"],
            "session_b_avg_speed_mps": mov_b["average_speed_mps"],
            "delta_avg_speed_mps": round(mov_b["average_speed_mps"] - mov_a["average_speed_mps"], 2),
            "session_a_peak_speed_mps": mov_a["maximum_windowed_speed_mps"],
            "session_b_peak_speed_mps": mov_b["maximum_windowed_speed_mps"],
            "delta_peak_speed_mps": (
                round(mov_b["maximum_windowed_speed_mps"] - mov_a["maximum_windowed_speed_mps"], 2)
                if (mov_a.get("maximum_windowed_speed_mps") is not None and mov_b.get("maximum_windowed_speed_mps") is not None)
                else None
            ),
        }
    else:
        result["physical_movement_comparison"] = {
            "status": "suppressed",
            "reason": "Both video runs must be comparable and calibrated to compare physical distance/speed.",
        }
    return result
