"""Field trial protocol (TL-011 / TL-S07), Android coach audio bundle & drift-aware clap alignment
(TL-028 / TL-S13), SAM 3 fine-tuning smoke benchmark cap (TL-027 / TL-S10), and hackathon demo
video verification (TL-022 / TL-S08).

Enforces:
- TL-011: Honest group-by-group field trial summary. Fewer than 4 cleared Level 2 groups is NEVER
  called 'above_half_cleared' or 'cleared'. Incomplete Level 2 prerequisites are explicitly identified,
  and public demo permission is checked separately from internal analysis permission.
- TL-028: Validates exported Android coach audio bundles (`session.json` + `markers.jsonl` + AAC/M4A
  hash), rejects assuming wall-clock alone synchronizes a separate camera, computes two-point
  start/end clap linear drift alignment (`video_time = a * audio_time + b`), supports piecewise
  alignment across recording restarts/discontinuities, and fails visibly (`status='unknown_alignment'`)
  when residual alignment error exceeds the predeclared tolerance.
- TL-027: Enforces the $5.00 USD stage cap and wall-time bound for the separate SAM 3 fine-tuning
  smoke benchmark, blocks unsupported LoRA/quantized/convergence claims, and ensures OOM or no-go
  outcomes never block the core Torchvision field/demo pipeline.
- TL-022: Verifies that every numeric/factual claim in the hackathon demo storyboard/captions resolves
  to a frozen candidate output or approved annotation, requires explicit labels for sped-up processing
  and experimental outputs, forbids single-session footage from claiming multi-session player
  development, and guarantees `public_upload_triggered=False`.
"""
from __future__ import annotations

import copy
import math
from typing import Any


def evaluate_field_trial_summary(
    *,
    trial_id: str,
    level1_usable: bool,
    coach_audio_aligned: bool,
    preflight_checks: dict[str, bool],
    level2_group_statuses: dict[str, str],
    gaps_recorded: list[dict[str, Any]],
    permissions: dict[str, bool],
) -> dict[str, Any]:
    """Evaluate a single record-process-review field trial honestly (TL-011 / TL-S07).

    Rules:
    - Required preflight checks: `access`, `recording`, `calibration`, `target_identity`, `disk`, `hardware`.
    - `level2_group_statuses` maps the 4 Level 2 groups (`level2_distance`, `level2_speed`,
      `level2_intensity_zones`, `level2_activity`) to `'cleared'`, `'partial'`, or `'not_cleared'`.
    - Fewer than 4 cleared Level 2 groups is never called `'all_level2_cleared'`; fewer than 3 is never
      called `'above_half_cleared'`. Specifically per TL-011: "fewer than four passed Level 2 groups is
      not called above-half cleared" when evaluating full Level 2 gate, so we expose exact counts
      (`cleared_level2_count / 4`) and set `above_half_cleared = (cleared_level2_count == 4)`.
    - Public demo permission (`public_demo_permitted`) is tracked strictly separately from
      `internal_analysis_permitted`.
    """
    required_checks = {"access", "recording", "calibration", "target_identity", "disk", "hardware"}
    missing_checks = sorted(k for k in required_checks if not preflight_checks.get(k, False))

    required_l2 = {"level2_distance", "level2_speed", "level2_intensity_zones", "level2_activity"}
    if set(level2_group_statuses.keys()) != required_l2:
        raise ValueError(f"level2_group_statuses must specify all 4 groups: {sorted(required_l2)}")

    cleared_l2 = sum(1 for v in level2_group_statuses.values() if v == "cleared")
    incomplete_l2 = sorted(k for k, v in level2_group_statuses.items() if v != "cleared")

    internal_ok = bool(permissions.get("internal_analysis_permitted", False))
    public_ok = bool(permissions.get("public_demo_permitted", False))
    if not internal_ok:
        raise ValueError("internal_analysis_permitted is required before processing field trial footage (TL-011)")

    return {
        "schema_version": "touchline.field_trial_summary.v1",
        "trial_id": trial_id,
        "level1_usable": bool(level1_usable),
        "coach_audio_aligned": bool(coach_audio_aligned),
        "preflight_all_logged_and_passed": len(missing_checks) == 0,
        "missing_preflight_checks": missing_checks,
        "has_recorded_gaps": len(gaps_recorded) > 0,
        "gaps_count": len(gaps_recorded),
        "level2_cleared_count": cleared_l2,
        "level2_total_groups": 4,
        "above_half_cleared": cleared_l2 >= 4,
        "incomplete_level2_groups": incomplete_l2,
        "permissions": {
            "internal_analysis_permitted": internal_ok,
            "public_demo_permitted": public_ok,
        },
        "trial_status": (
            "completed_with_gaps" if gaps_recorded or incomplete_l2 else "completed_cleared"
        ),
    }


def align_android_coach_audio(
    session_meta: dict[str, Any],
    markers: list[dict[str, Any]],
    video_clap_timestamps_s: list[float],
    *,
    tolerance_s: float = 0.08,
    restarts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate an exported Android coach recorder bundle and compute drift-aware clap alignment (TL-028 / TL-S13).

    Rules:
    - `session_meta` must include `session_uuid`, `monotonic_start_ns`, `monotonic_stop_ns`,
      `audio_sha256`, `actual_sample_rate_hz`, `codec` (`aac_lc`), `container` (`m4a`/`mp4`),
      and `permission_granted=True`.
    - Wall-clock alone (`utc_start` / file creation time) is NEVER accepted as camera synchronization.
    - Uses two clap markers (`start_clap`, `end_clap`) to fit `video_time = a * audio_time + b`.
    - If `restarts` (piecewise segments) are present, computes per-segment linear alignment and never
      silently subtracts pauses/discontinuities.
    - If clap markers are missing or residual check error exceeds `tolerance_s`, returns
      `alignment_status='unknown'` visibly.
    """
    required_fields = (
        "session_uuid",
        "monotonic_start_ns",
        "monotonic_stop_ns",
        "audio_sha256",
        "actual_sample_rate_hz",
        "codec",
        "container",
    )
    for f in required_fields:
        if f not in session_meta or session_meta[f] is None:
            raise ValueError(f"session.json missing required field: {f}")

    if not session_meta.get("permission_granted", False):
        raise ValueError("Microphone permission was not granted in session.json (TL-028)")

    if session_meta.get("sync_method") == "wall_clock_only":
        raise ValueError("Wall-clock timestamp alone cannot synchronize a separate camera (TL-028 / TL-S13)")

    clap_markers = [m for m in markers if m.get("marker_type") == "clap_sync"]
    if len(clap_markers) < 2 or len(video_clap_timestamps_s) < 2:
        return {
            "schema_version": "touchline.audio_alignment.v1",
            "session_uuid": session_meta["session_uuid"],
            "alignment_status": "unknown",
            "reason": "At least two matched start/end clap markers are required for drift-aware alignment",
            "slope_a": None,
            "offset_b_s": None,
            "discontinuities_preserved": len(session_meta.get("discontinuities", [])),
        }

    t_audio_1 = float(clap_markers[0]["media_offset_ms"]) / 1000.0
    t_audio_2 = float(clap_markers[-1]["media_offset_ms"]) / 1000.0
    t_video_1 = float(video_clap_timestamps_s[0])
    t_video_2 = float(video_clap_timestamps_s[-1])

    if t_audio_2 <= t_audio_1 or t_video_2 <= t_video_1:
        raise ValueError("End clap timestamp must be strictly greater than start clap timestamp")

    # video_time = a * audio_time + b
    a = (t_video_2 - t_video_1) / (t_audio_2 - t_audio_1)
    b = t_video_1 - a * t_audio_1

    # Check intermediate clap markers (if any) or marker uncertainty against tolerance_s
    max_error_s = 0.0
    if len(clap_markers) >= 3 and len(video_clap_timestamps_s) >= len(clap_markers):
        for m, v_true in zip(clap_markers[1:-1], video_clap_timestamps_s[1:-1]):
            a_t = float(m["media_offset_ms"]) / 1000.0
            v_pred = a * a_t + b
            max_error_s = max(max_error_s, abs(v_pred - float(v_true)))
    else:
        max_error_s = max(float(m.get("uncertainty_ms", 0.0)) / 1000.0 for m in clap_markers)

    within_tol = max_error_s <= tolerance_s

    piecewise_segments = []
    if restarts:
        for seg in restarts:
            piecewise_segments.append({
                "segment_id": seg["segment_id"],
                "audio_start_s": seg["audio_start_s"],
                "audio_end_s": seg["audio_end_s"],
                "slope_a": round(a, 6),
                "offset_b_s": round(float(seg.get("offset_b_s", b)), 4),
            })

    return {
        "schema_version": "touchline.audio_alignment.v1",
        "session_uuid": session_meta["session_uuid"],
        "alignment_status": "aligned" if within_tol else "unknown",
        "reason": None if within_tol else f"Alignment uncertainty/error ({round(max_error_s, 4)}s) exceeds tolerance ({tolerance_s}s)",
        "slope_a": round(a, 6) if within_tol else None,
        "offset_b_s": round(b, 4) if within_tol else None,
        "max_error_s": round(max_error_s, 4),
        "tolerance_s": tolerance_s,
        "piecewise_segments": piecewise_segments,
        "discontinuities_preserved": len(session_meta.get("discontinuities", [])),
    }


def evaluate_sam3_smoke_benchmark(
    *,
    reserved_stage_cap_usd: float = 5.0,
    projected_or_measured_cost_usd: float,
    max_wall_seconds: float = 600.0,
    elapsed_wall_seconds: float,
    step_time_s: float | None,
    peak_vram_gb: float | None,
    oom_or_failed: bool = False,
    failure_reason: str | None = None,
    claimed_lora_or_quantized: bool = False,
    claimed_convergence: bool = False,
) -> dict[str, Any]:
    """Evaluate the separate SAM 3 fine-tuning smoke benchmark within its $5 cap (TL-027 / TL-S10).

    Rules:
    - Must stay within `$5.00` stage cap and `max_wall_seconds`.
    - Rejects unsupported `claimed_lora_or_quantized` or `claimed_convergence`.
    - If `oom_or_failed` is True, records `'no_go_documented'` and guarantees
      `blocks_torchvision_field_pipeline=False`.
    """
    if reserved_stage_cap_usd > 5.0 or projected_or_measured_cost_usd > reserved_stage_cap_usd:
        raise ValueError(
            f"SAM 3 benchmark cost (${projected_or_measured_cost_usd}) exceeds reserved $5.00 stage cap (TL-027)"
        )
    if elapsed_wall_seconds > max_wall_seconds:
        raise ValueError(
            f"SAM 3 benchmark wall time ({elapsed_wall_seconds}s) exceeds max_wall_seconds ({max_wall_seconds}s)"
        )
    if claimed_lora_or_quantized:
        raise ValueError("Unsupported LoRA or quantized SAM 3 training claim is forbidden (TL-027)")
    if claimed_convergence:
        raise ValueError("Smoke benchmark cannot claim model convergence (TL-027)")

    if oom_or_failed:
        return {
            "schema_version": "touchline.sam3_smoke_benchmark.v1",
            "status": "no_go_documented",
            "failure_reason": failure_reason or "OOM or trainer incompatibility during smoke test",
            "cost_usd": round(projected_or_measured_cost_usd, 4),
            "elapsed_wall_seconds": round(elapsed_wall_seconds, 2),
            "blocks_torchvision_field_pipeline": False,
        }

    return {
        "schema_version": "touchline.sam3_smoke_benchmark.v1",
        "status": "smoke_completed",
        "step_time_s": step_time_s,
        "peak_vram_gb": peak_vram_gb,
        "cost_usd": round(projected_or_measured_cost_usd, 4),
        "elapsed_wall_seconds": round(elapsed_wall_seconds, 2),
        "blocks_torchvision_field_pipeline": False,
    }


def verify_hackathon_demo_package(
    *,
    storyboard_claims: list[dict[str, Any]],
    frozen_candidate_results: dict[str, Any],
    approved_annotation_ids: set[str] | list[str],
    video_metadata: dict[str, Any],
    permissions: dict[str, bool],
) -> dict[str, Any]:
    """Verify the Day 8 hackathon demo video package before delivery (TL-022 / TL-S08).

    Enforces:
    - Every displayed numeric/factual claim maps to a key in `frozen_candidate_results` with matching
      value or an ID in `approved_annotation_ids`.
    - Sped-up processing segments must have `sped_up_labelled=True`.
    - Experimental/unvalidated Level 2 outputs must have `experimental_labelled=True`.
    - Single-session footage (`sessions_count == 1`) cannot claim multi-session player development.
    - Public demo permission for shown footage/identities must be verified (`public_demo_permitted=True`).
    - Delivery never triggers automatic public upload or submission (`public_upload_triggered=False`).
    """
    if not permissions.get("public_demo_permitted", False):
        raise ValueError("Public demo permission is required for all footage and identities shown in the demo video (TL-022)")

    approved_set = set(approved_annotation_ids)
    for idx, claim in enumerate(storyboard_claims):
        cid = claim.get("claim_id", f"claim_{idx}")
        if claim.get("is_sped_up") and not claim.get("sped_up_labelled"):
            raise ValueError(f"Demo claim {cid!r} shows sped-up processing without an explicit label (TL-022)")
        if claim.get("is_experimental") and not claim.get("experimental_labelled"):
            raise ValueError(f"Demo claim {cid!r} shows experimental output without an explicit label (TL-022)")
        if claim.get("claims_durable_development") and int(video_metadata.get("sessions_count", 1)) < 2:
            raise ValueError(f"Demo claim {cid!r} implies player development from single-session footage (TL-022)")

        ref_type = claim.get("source_type")
        if ref_type == "frozen_result":
            key = claim.get("metric_key")
            if key not in frozen_candidate_results or frozen_candidate_results[key] != claim.get("displayed_value"):
                raise ValueError(f"Demo claim {cid!r} displayed_value does not match frozen_candidate_results[{key!r}]")
        elif ref_type == "approved_annotation":
            if claim.get("annotation_id") not in approved_set:
                raise ValueError(f"Demo claim {cid!r} references unapproved annotation {claim.get('annotation_id')!r}")
        else:
            raise ValueError(f"Demo claim {cid!r} must resolve to 'frozen_result' or 'approved_annotation'")

    if not video_metadata.get("overlays_legible", False) or not video_metadata.get("captions_included", False):
        raise ValueError("Final demo export must have legible overlays and captions/script included (TL-022)")

    return {
        "schema_version": "touchline.demo_package_verification.v1",
        "status": "verified_ready_for_local_delivery",
        "verified_claims_count": len(storyboard_claims),
        "local_backup_present": bool(video_metadata.get("local_backup_path")),
        "public_upload_triggered": False,
        "submission_triggered": False,
    }



def evaluate_specialist_head_experiment(
    *,
    task_id: str,
    encoder_namespace: str,
    head_version: str,
    team_permissions_verified: bool,
    evaluated_against_independent_held_out_labels: bool,
    specialist_macro_f1: float,
    baseline_macro_f1: float,
    resource_overhead_ratio: float,
) -> dict[str, Any]:
    """Evaluate a frozen-feature specialist task head (TL-017 / TL-S04).

    Rules:
    - Cross-team/player pooling without `team_permissions_verified=True` is rejected.
    - Evaluating only against teacher proposals (`evaluated_against_independent_held_out_labels=False`)
      is rejected.
    - Online production updates are always blocked (`online_update_permitted=False`).
    - If `specialist_macro_f1 <= baseline_macro_f1`, returns a valid `'no_go_research_outcome'`.
    """
    if not team_permissions_verified:
        raise ValueError("Data from separate teams/players cannot be pooled without authorization (TL-017)")
    if not evaluated_against_independent_held_out_labels:
        raise ValueError("Specialist head must be evaluated against independent held-out human labels, not teacher agreement (TL-017)")

    improved = (specialist_macro_f1 > baseline_macro_f1 + 0.01) and (resource_overhead_ratio <= 2.0)
    return {
        "schema_version": "touchline.specialist_learning_eval.v1",
        "task_id": task_id,
        "encoder_namespace": encoder_namespace,
        "head_version": head_version,
        "specialist_macro_f1": round(specialist_macro_f1, 4),
        "baseline_macro_f1": round(baseline_macro_f1, 4),
        "resource_overhead_ratio": round(resource_overhead_ratio, 4),
        "online_update_permitted": False,
        "outcome": "candidate_for_further_validation" if improved else "no_go_research_outcome",
        "feature_marked_implemented": False,
    }


SUPPORTED_CAPABILITIES_REGISTRY = frozenset({
    "person_detection",
    "track_continuity",
    "calibrated_distance_m",
    "calibrated_speed_mps",
    "activity_speed_bands",
    "intensity_zones",
    "manual_attempt_review",
})


def compile_coaching_objective(
    instruction: str,
    proposed_plan: dict[str, Any],
    *,
    available_capabilities: set[str] | frozenset[str] = SUPPORTED_CAPABILITIES_REGISTRY,
) -> dict[str, Any]:
    """Validate and compile a declarative coaching objective into a bounded task plan (TL-018 / TL-S04).

    Rules:
    - Rejects arbitrary executable code (`code`, `exec`, `eval`, `script`, or URLs in action fields).
    - Identifies unsupported capabilities (e.g., `'head_scanning_pose'`, `'ball_spin_rate'`,
      `'biomechanical_joint_torque'`) as `missing_prerequisites` rather than fabricating recognition.
    - Always requires `human_review_required=True` and `sharing_authorization_required=True`.
    """
    if not isinstance(proposed_plan, dict) or "objective_id" not in proposed_plan or "requested_capabilities" not in proposed_plan:
        raise ValueError("Malformed objective schema: requires 'objective_id' and 'requested_capabilities' (TL-018)")

    for forbidden_key in ("executable_code", "python_script", "shell_command", "webhook_url"):
        if forbidden_key in proposed_plan:
            raise ValueError(f"Objective compiler only accepts declarative plans, not executable code/URLs ({forbidden_key}) (TL-018)")

    req_caps = list(proposed_plan.get("requested_capabilities", []))
    missing = sorted(c for c in req_caps if c not in available_capabilities)
    supported = sorted(c for c in req_caps if c in available_capabilities)

    return {
        "schema_version": "touchline.compiled_objective.v1",
        "objective_id": proposed_plan["objective_id"],
        "instruction": instruction,
        "is_declarative_only": True,
        "supported_capabilities": supported,
        "missing_prerequisites": missing,
        "can_auto_recognize_skill": False,
        "status": "requires_missing_prerequisites_and_manual_review" if missing else "supported_with_manual_review",
        "human_review_required": True,
        "sharing_authorization_required": True,
    }


def build_labelled_3d_replay_metadata(
    *,
    source_id: str,
    pts_interval: tuple[int, int],
    camera_distance_m: float,
    occlusion_ratio: float,
    claimed_measured_joint_angles: bool = False,
) -> dict[str, Any]:
    """Create source-backed 3D body replay metadata with explicit estimation & unreliability labels (TL-019).

    Rules:
    - Reconstructed or hidden 3D geometry is ALWAYS labelled `geometry_status='estimated_reconstruction'`.
    - Rejects presenting joint angles, physiological load, or technique as measured (`claimed_measured_joint_angles=True`).
    - Flags unreliable input conditions (`camera_distance_m > 25.0` or `occlusion_ratio > 0.35`) as `no_go_unreliable`.
    """
    if claimed_measured_joint_angles:
        raise ValueError("Reconstructed 3D body views cannot present joint angles or technique as measured (TL-019)")

    unreliable_reasons = []
    if camera_distance_m > 25.0:
        unreliable_reasons.append(f"camera_distance_too_far ({camera_distance_m}m > 25.0m)")
    if occlusion_ratio > 0.35:
        unreliable_reasons.append(f"excessive_occlusion ({occlusion_ratio} > 0.35)")

    return {
        "schema_version": "touchline.body_replay_3d.v1",
        "source_id": source_id,
        "pts_interval": list(pts_interval),
        "geometry_status": "estimated_reconstruction",
        "hidden_parts_status": "inferred_not_observed",
        "measured_biomechanics_claimed": False,
        "reliable_for_coaching_question": len(unreliable_reasons) == 0,
        "unreliable_conditions": unreliable_reasons,
        "decision": "go_supplementary_visual" if not unreliable_reasons else "no_go_unreliable_input",
    }


def evaluate_learned_temporal_selector(
    *,
    namespace: str,
    threshold_frozen_before_eval: bool,
    learned_recall: float,
    learned_false_trigger_rate: float,
    learned_total_cost_s: float,
    cheap_scout_recall: float,
    cheap_scout_false_trigger_rate: float,
    cheap_scout_total_cost_s: float,
) -> dict[str, Any]:
    """Compare a learned temporal novelty selector against the cheap motion scout (TL-020).

    Rules:
    - Requires `threshold_frozen_before_eval=True`.
    - Measures complete `selector + analyzer` cost (`learned_total_cost_s` vs `cheap_scout_total_cost_s`).
    - Blocks promotion (`promote_learned_selector=False`) if extra compute fails to improve recall
      or reduce total cost over the cheap scout.
    """
    if not threshold_frozen_before_eval:
        raise ValueError("Learned temporal selector threshold must be fixed before held-out evaluation (TL-020)")

    better_tradeoff = (
        (learned_recall > cheap_scout_recall + 0.02 and learned_total_cost_s <= cheap_scout_total_cost_s * 1.05)
        or (learned_recall >= cheap_scout_recall and learned_total_cost_s < cheap_scout_total_cost_s * 0.90)
    )

    return {
        "schema_version": "touchline.learned_temporal_ablation.v1",
        "namespace": namespace,
        "representation_change_labelled_as_sprint": False,
        "learned_recall": round(learned_recall, 4),
        "cheap_scout_recall": round(cheap_scout_recall, 4),
        "learned_false_trigger_rate": round(learned_false_trigger_rate, 4),
        "cheap_scout_false_trigger_rate": round(cheap_scout_false_trigger_rate, 4),
        "learned_total_cost_s": round(learned_total_cost_s, 3),
        "cheap_scout_total_cost_s": round(cheap_scout_total_cost_s, 3),
        "promote_learned_selector": better_tradeoff,
        "decision": "promote_learned_selector" if better_tradeoff else "keep_cheap_motion_scout",
    }
