"""Lightweight scout and adaptive sliding-window controller (TL-012 / TL-S03).

Implements:
- Pass A reduced-frame motion, camera-change, and tracking-quality trigger evaluation
  (`player_motion`, `direction_change`, `track_uncertain`, `occlusion_recovery`,
  `new_entrant`, `camera_change`, `coach_requested`).
- State machine (`BASELINE`, `DETAIL`, `COOLDOWN`) with distinct enter/exit thresholds,
  minimum stable duration, padding, and interval merging that NEVER bridges across a
  corrupt or discontinuous timeline.
- Pass B exact-source refinement with single observation ownership per `(source_frame_pts,
  local_track_id, capability)` so overlapping baseline and detailed windows never duplicate
  observations, attempt counts, or distance increments.
- Bounded window retry on OOM (up to 2 retries with halved window size while preserving
  quality settings).
- Independent selector event recall evaluation vs. detailed analyzer accuracy, plus
  union-of-refined-time fraction (`p`) and go/no-go comparison against fixed-rate baselines.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from .sampling import merge_intervals


VALID_TRIGGER_REASONS = frozenset({
    "player_motion",
    "direction_change",
    "track_uncertain",
    "occlusion_recovery",
    "new_entrant",
    "camera_change",
    "coach_requested",
})

CRITICAL_ESCALATION_REASONS = frozenset({
    "track_uncertain",
    "occlusion_recovery",
    "camera_change",
    "coach_requested",
})


class AdaptiveWindowController:
    """State machine (`BASELINE` -> `DETAIL` -> `COOLDOWN`) and interval scheduler (TL-012 / TL-S03)."""

    def __init__(
        self,
        *,
        enter_motion_threshold: float = 0.25,
        exit_motion_threshold: float = 0.12,
        min_detail_duration_s: float = 0.5,
        cooldown_duration_s: float = 0.5,
        padding_s: float = 1.0,
        max_contiguous_gap_s: float = 2.0,
    ):
        if enter_motion_threshold <= exit_motion_threshold:
            raise ValueError("enter_motion_threshold must be strictly greater than exit_motion_threshold")
        if min_detail_duration_s < 0 or cooldown_duration_s < 0 or padding_s < 0:
            raise ValueError("durations and padding_s must be non-negative")
        self.enter_motion_threshold = enter_motion_threshold
        self.exit_motion_threshold = exit_motion_threshold
        self.min_detail_duration_s = min_detail_duration_s
        self.cooldown_duration_s = cooldown_duration_s
        self.padding_s = padding_s
        self.max_contiguous_gap_s = max_contiguous_gap_s

    def schedule_pass_a(
        self,
        scout_samples: list[dict[str, Any]],
        *,
        source_duration_s: float,
    ) -> dict[str, Any]:
        """Run Pass A state transitions over chronologically ordered scout samples.

        Each sample in `scout_samples` expects:
        - `timestamp_s`: float
        - `pts`: int
        - `motion_score`: float (in [0, 1], measured in expanded player regions)
        - `midpoint_motion_spike`: bool (optional: detects motion between similar anchor endpoints)
        - `reasons`: list[str] (optional trigger reasons such as `track_uncertain`, `camera_change`)
        - `timeline_discontinuity`: bool (optional: true if corrupt/cut interval occurs before this sample)
        """
        if source_duration_s <= 0:
            raise ValueError("source_duration_s must be positive")

        state = "BASELINE"
        state_entered_s = 0.0
        active_start_s: float | None = None
        active_reasons: set[str] = set()
        raw_intervals: list[dict[str, Any]] = []
        state_trace: list[dict[str, Any]] = []
        camera_change_intervals: list[tuple[float, float]] = []

        # Partition samples into contiguous timeline segments so we NEVER merge across discontinuities
        segments: list[list[dict[str, Any]]] = []
        current_seg: list[dict[str, Any]] = []
        prev_t: float | None = None

        for sample in sorted(scout_samples, key=lambda s: float(s["timestamp_s"])):
            t = float(sample["timestamp_s"])
            is_discont = bool(sample.get("timeline_discontinuity", False)) or (
                prev_t is not None and (t - prev_t) > self.max_contiguous_gap_s
            )
            if is_discont and current_seg:
                segments.append(current_seg)
                current_seg = []
            current_seg.append(sample)
            prev_t = t
        if current_seg:
            segments.append(current_seg)

        merged_windows: list[dict[str, Any]] = []

        for seg_idx, seg in enumerate(segments):
            seg_min_t = max(0.0, float(seg[0]["timestamp_s"]) - self.padding_s if seg_idx == 0 else float(seg[0]["timestamp_s"]))
            seg_max_t = min(source_duration_s, float(seg[-1]["timestamp_s"]) + self.padding_s if seg_idx == len(segments) - 1 else float(seg[-1]["timestamp_s"]))

            state = "BASELINE"
            active_start_s = None
            active_reasons = set()
            seg_candidates: list[dict[str, Any]] = []

            for i, s in enumerate(seg):
                t = float(s["timestamp_s"])
                motion = float(s.get("motion_score", 0.0))
                midpoint_spike = bool(s.get("midpoint_motion_spike", False))
                reasons = set(s.get("reasons", []))
                unknown = reasons - VALID_TRIGGER_REASONS
                if unknown:
                    raise ValueError(f"Unsupported trigger reasons: {sorted(unknown)}")

                if midpoint_spike:
                    reasons.add("player_motion")
                if motion >= self.enter_motion_threshold:
                    reasons.add("player_motion")

                # Critical reasons immediately escalate regardless of weighted average motion
                has_critical = bool(reasons & CRITICAL_ESCALATION_REASONS)
                qualifies_enter = has_critical or midpoint_spike or (motion >= self.enter_motion_threshold)
                qualifies_stay = has_critical or midpoint_spike or (motion >= self.exit_motion_threshold)

                if "camera_change" in reasons:
                    camera_change_intervals.append((t, min(source_duration_s, t + 0.5)))

                if state == "BASELINE":
                    if qualifies_enter:
                        state = "DETAIL"
                        state_entered_s = t
                        active_start_s = t
                        active_reasons = set(reasons)
                elif state == "DETAIL":
                    active_reasons |= reasons
                    if not qualifies_stay and (t - state_entered_s) >= self.min_detail_duration_s:
                        state = "COOLDOWN"
                        state_entered_s = t
                elif state == "COOLDOWN":
                    if qualifies_enter:
                        state = "DETAIL"
                        state_entered_s = t
                        active_reasons |= reasons
                    elif (t - state_entered_s) >= self.cooldown_duration_s:
                        end_t = max(t, (active_start_s or t) + 0.04)
                        seg_candidates.append({
                            "start_s": active_start_s if active_start_s is not None else t,
                            "end_s": end_t,
                            "reasons": sorted(active_reasons),
                        })
                        state = "BASELINE"
                        active_start_s = None
                        active_reasons = set()

                state_trace.append({
                    "timestamp_s": t,
                    "pts": s.get("pts"),
                    "state": state,
                    "reasons": sorted(reasons),
                })

            if state in ("DETAIL", "COOLDOWN") and active_start_s is not None:
                end_t = max(float(seg[-1]["timestamp_s"]) + 0.04, active_start_s + 0.04)
                seg_candidates.append({
                    "start_s": active_start_s,
                    "end_s": min(source_duration_s, end_t),
                    "reasons": sorted(active_reasons),
                })

            raw_intervals.extend(seg_candidates)

            # Pad and merge ONLY within this contiguous timeline segment
            if seg_candidates:
                tuples = [(c["start_s"], min(seg_max_t, max(c[" start_s"] if " start_s" in c else c["start_s"] + 0.01, c["end_s"]))) for c in seg_candidates]
                merged_tuples = merge_intervals(tuples, padding_s=self.padding_s, duration_s=seg_max_t)
                for m_start, m_end in merged_tuples:
                    clamped_start = max(seg_min_t, m_start)
                    clamped_end = min(seg_max_t, m_end)
                    if clamped_start < clamped_end:
                        combined_reasons: set[str] = set()
                        for c in seg_candidates:
                            if max(clamped_start, c["start_s"] - self.padding_s) <= min(clamped_end, c["end_s"] + self.padding_s):
                                combined_reasons.update(c["reasons"])
                        merged_windows.append({
                            "segment_index": seg_idx,
                            "start_s": round(clamped_start, 4),
                            "end_s": round(clamped_end, 4),
                            "duration_s": round(clamped_end - clamped_start, 4),
                            "reasons": sorted(combined_reasons),
                            "camera_uncalibrated_warning": "camera_change" in combined_reasons,
                        })

        union_refined_s = round(sum(w["duration_s"] for w in merged_windows), 4)
        refined_fraction_p = round(min(1.0, union_refined_s / source_duration_s), 4)
        estimated_calls_per_s = round(2.0 * (1.0 - refined_fraction_p) + 24.0 * refined_fraction_p, 4)

        return {
            "schema_version": "touchline.adaptive_schedule.v1",
            "source_duration_s": source_duration_s,
            "contiguous_segments_count": len(segments),
            "raw_trigger_count": len(raw_intervals),
            "merged_windows": merged_windows,
            "union_refined_duration_s": union_refined_s,
            "union_refined_fraction_p": refined_fraction_p,
            "illustrative_calls_per_second": estimated_calls_per_s,
            "camera_change_intervals": camera_change_intervals,
            "state_trace": state_trace,
        }


def reconcile_single_ownership_observations(
    baseline_observations: list[dict[str, Any]],
    detailed_observations: list[dict[str, Any]],
    *,
    run_revision: int = 1,
) -> dict[str, Any]:
    """Reconcile baseline and detailed Pass B observations with single ownership (TL-012 / TL-S03).

    Rules:
    - Each emitted production observation has exactly ONE owner per `(pts, local_track_id, capability)`.
    - When both `baseline` and `detail` observations exist for the same `(pts, local_track_id)`,
      `detail` takes ownership (`owner='detail_pass_b'`), while `baseline` is retained in
      `audit_coexistence` only—never duplicated in production `final_observations`.
    - Repeated/overlapping detail windows with identical `(pts, local_track_id)` are deduplicated.
    """
    by_key: dict[tuple[Any, Any, str], dict[str, Any]] = {}
    audit_coexistence: list[dict[str, Any]] = []
    duplicates_prevented = 0

    for obs in baseline_observations:
        pts = obs.get("frame", {}).get("pts", obs.get("pts"))
        tid = obs.get("local_track_id")
        cap = obs.get("capability", "person_detection")
        key = (pts, tid, cap)
        entry = copy.deepcopy(obs)
        entry["owner"] = "baseline_pass_a"
        entry["run_revision"] = run_revision
        if key in by_key:
            duplicates_prevented += 1
        by_key[key] = entry

    for obs in detailed_observations:
        pts = obs.get("frame", {}).get("pts", obs.get("pts"))
        tid = obs.get("local_track_id")
        cap = obs.get("capability", "person_detection")
        key = (pts, tid, cap)
        entry = copy.deepcopy(obs)
        entry["owner"] = "detail_pass_b"
        entry["run_revision"] = run_revision
        if key in by_key:
            prior = by_key[key]
            if prior["owner"] == "baseline_pass_a":
                audit_coexistence.append(prior)
            else:
                duplicates_prevented += 1
        by_key[key] = entry

    final_obs = sorted(
        by_key.values(),
        key=lambda o: (
            float(o.get("frame", {}).get("timestamp_s", o.get("timestamp_s", 0.0))),
            int(o.get("local_track_id", 0)),
        ),
    )

    return {
        "schema_version": "touchline.ownership_reconciliation.v1",
        "run_revision": run_revision,
        "final_observations": final_obs,
        "final_count": len(final_obs),
        "audit_coexistence_count": len(audit_coexistence),
        "duplicates_prevented": duplicates_prevented,
    }


def execute_window_with_oom_retry(
    window_frames: list[Any],
    worker_fn: Any,
    *,
    max_retries: int = 2,
    quality_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a Pass B refinement window with bounded OOM cleanup and sub-window retry (TL-012 / TL-S03).

    On `MemoryError` (or `RuntimeError` containing `'out of memory'`), cleans up and retries smaller
    sub-windows at most `max_retries` times (default 2) while strictly preserving `quality_settings`.
    """
    preserved_quality = copy.deepcopy(quality_settings or {"min_size": 640, "max_size": 1280})
    attempt = 0
    chunk_size = max(1, len(window_frames))

    while attempt <= max_retries:
        try:
            results = []
            for idx in range(0, len(window_frames), chunk_size):
                sub_window = window_frames[idx : idx + chunk_size]
                results.extend(worker_fn(sub_window, preserved_quality))
            return {
                "status": "completed",
                "attempts_used": attempt + 1,
                "oom_retries": attempt,
                "final_chunk_size": chunk_size,
                "quality_settings_preserved": preserved_quality == (quality_settings or {"min_size": 640, "max_size": 1280}),
                "quality_settings": preserved_quality,
                "results": results,
            }
        except (MemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, MemoryError) or ("out of memory" in str(exc).lower())
            if not is_oom or attempt >= max_retries or chunk_size <= 1:
                raise RuntimeError(f"Window refinement failed after {attempt} OOM retries: {exc}") from exc
            attempt += 1
            chunk_size = max(1, chunk_size // 2)

    raise RuntimeError("Unreachable retry state")


def evaluate_adaptive_benchmark(
    *,
    ground_truth_events: list[dict[str, Any]],
    merged_windows: list[dict[str, Any]],
    detailed_detected_event_ids: set[str] | list[str],
    adaptive_runtime_s: float,
    fixed_6fps_runtime_s: float,
    fixed_6fps_recall: float,
    source_duration_s: float,
) -> dict[str, Any]:
    """Measure selector missed events separately from detailed-analyzer accuracy and make a go/no-go decision (TL-012).

    `ground_truth_events`: list of `{"event_id": str, "timestamp_s": float, "category": str}`
    """
    detected_set = set(detailed_detected_event_ids)
    total_events = len(ground_truth_events)
    selected_by_scout = 0
    selector_missed_ids: list[str] = []
    analyzer_missed_inside_window_ids: list[str] = []
    end_to_end_detected = 0

    for ev in ground_truth_events:
        ev_id = ev["event_id"]
        t = float(ev["timestamp_s"])
        in_window = any(float(w["start_s"]) <= t <= float(w["end_s"]) for w in merged_windows)
        if in_window:
            selected_by_scout += 1
            if ev_id in detected_set:
                end_to_end_detected += 1
            else:
                analyzer_missed_inside_window_ids.append(ev_id)
        else:
            selector_missed_ids.append(ev_id)

    selector_recall = round(selected_by_scout / total_events, 4) if total_events > 0 else 1.0
    analyzer_conditional_accuracy = (
        round(end_to_end_detected / selected_by_scout, 4) if selected_by_scout > 0 else 0.0
    )
    end_to_end_recall = round(end_to_end_detected / total_events, 4) if total_events > 0 else 1.0
    union_refined_s = round(sum(float(w["end_s"]) - float(w["start_s"]) for w in merged_windows), 4)
    union_fraction_p = round(min(1.0, union_refined_s / max(0.001, source_duration_s)), 4)

    # Go/No-Go requires adaptive to save runtime (< fixed_6fps_runtime_s) without losing more than 2% recall vs fixed 6 FPS
    adopt_adaptive = (adaptive_runtime_s < fixed_6fps_runtime_s) and (end_to_end_recall + 0.02 >= fixed_6fps_recall)

    return {
        "schema_version": "touchline.adaptive_benchmark.v1",
        "total_events": total_events,
        "selector_recall": selector_recall,
        "selector_missed_count": len(selector_missed_ids),
        "selector_missed_event_ids": selector_missed_ids,
        "detailed_analyzer_conditional_recall": analyzer_conditional_accuracy,
        "detailed_analyzer_missed_event_ids": analyzer_missed_inside_window_ids,
        "end_to_end_recall": end_to_end_recall,
        "union_refined_fraction_p": union_fraction_p,
        "adaptive_runtime_s": adaptive_runtime_s,
        "fixed_6fps_runtime_s": fixed_6fps_runtime_s,
        "fixed_6fps_recall": fixed_6fps_recall,
        "decision": "go_adaptive" if adopt_adaptive else "no_go_keep_fixed_rate_fallback",
    }
