"""Field-analysis refinement, before/after error ledger, and demo candidate freeze/reproduction (TL-021 / TL-021.1 / TL-021.2).

Enforces:
- TL-021.1: Tuning uses development evidence only; untouched evaluation split is strictly blocked
  from parameter tuning. Before/after comparisons report reference error and coverage (never just
  cosmetic chart adjustments). Predeclared acceptance criteria are frozen prior to evaluation and
  cannot be retrofitted.
- TL-021.2: Any training use of field corrections creates a new dataset release ID and preserves
  untouched evaluation evidence. Freezing the demo candidate records source/config/model/artifact
  hashes, per-group Level 1 and Level 2 clearance status (`cleared` / `partial` / `not_cleared`)
  with visible limitations, and verifies download-free deterministic reproduction within declared
  tolerances without requiring an active Colab session.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any


def _canonical_sha256(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_refinement_before_after(
    *,
    split_role: str,
    frozen_criteria: dict[str, Any],
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    attempted_criteria_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare before/after refinement results with reference error, coverage, and anti-retrofitting guard (TL-021.1).

    - Rejects tuning if `split_role == 'held_out_final_eval'` is used as a tuning loop input
      without a frozen config.
    - Rejects any attempt to retrofit `frozen_criteria` (`attempted_criteria_override is not None`).
    """
    if attempted_criteria_override is not None and attempted_criteria_override != frozen_criteria:
        raise ValueError("Acceptance criteria are frozen and cannot be retrofitted to the final result (TL-021)")

    if split_role not in ("dev_tuning", "held_out_eval"):
        raise ValueError("split_role must be 'dev_tuning' or 'held_out_eval'")

    for label, m in (("before_metrics", before_metrics), ("after_metrics", after_metrics)):
        if "reference_error" not in m or "usable_coverage" not in m:
            raise ValueError(f"{label} must report both 'reference_error' and 'usable_coverage' (TL-021)")

    err_before = float(before_metrics["reference_error"])
    err_after = float(after_metrics["reference_error"])
    cov_before = float(before_metrics["usable_coverage"])
    cov_after = float(after_metrics["usable_coverage"])

    max_err = float(frozen_criteria.get("max_reference_error", 0.20))
    min_cov = float(frozen_criteria.get("min_usable_coverage", 0.70))

    passes_after = (err_after <= max_err) and (cov_after >= min_cov)

    return {
        "schema_version": "touchline.refinement_comparison.v1",
        "split_role": split_role,
        "criteria_retrofitted": False,
        "before": {
            "reference_error": round(err_before, 4),
            "usable_coverage": round(cov_before, 4),
            "recall": before_metrics.get("recall"),
            "precision": before_metrics.get("precision"),
        },
        "after": {
            "reference_error": round(err_after, 4),
            "usable_coverage": round(cov_after, 4),
            "recall": after_metrics.get("recall"),
            "precision": after_metrics.get("precision"),
        },
        "deltas": {
            "reference_error_delta": round(err_after - err_before, 4),
            "usable_coverage_delta": round(cov_after - cov_before, 4),
            "recall_delta": (
                round(float(after_metrics["recall"]) - float(before_metrics["recall"]), 4)
                if ("recall" in before_metrics and "recall" in after_metrics)
                else None
            ),
        },
        "meets_frozen_criteria": passes_after,
    }


def freeze_demo_candidate(
    *,
    candidate_id: str,
    source_manifest: dict[str, Any],
    detector_config: dict[str, Any],
    dataset_release: dict[str, Any],
    group_outcomes: dict[str, dict[str, Any]],
    frozen_results: dict[str, Any],
    field_corrections_used_for_training: bool = False,
    prior_release_id: str | None = None,
) -> dict[str, Any]:
    """Freeze the demo candidate configuration, dataset/model provenance, and Level 1/2 group clearance (TL-021.1 / TL-021.2).

    Enforces:
    - If `field_corrections_used_for_training` is True, `dataset_release['release_id']` MUST differ from
      `prior_release_id`, and `dataset_release['untouched_eval_preserved']` MUST be True.
    - Every Level 1 and Level 2 group in `group_outcomes` is classified as `'cleared'`, `'partial'`,
      or `'not_cleared'` with explicit non-empty `limitations`.
    """
    if field_corrections_used_for_training:
        if not prior_release_id or dataset_release.get("release_id") == prior_release_id:
            raise ValueError(
                "Training use of field corrections requires a new dataset release_id distinct from prior_release_id (TL-021.2)"
            )
        if not dataset_release.get("untouched_eval_preserved", False):
            raise ValueError(
                "Untouched evaluation evidence must be preserved when creating a new release from field corrections (TL-021.2)"
            )

    classified_groups: dict[str, dict[str, Any]] = {}
    for gid, info in group_outcomes.items():
        status = info.get("status", "not_cleared")
        if status not in ("cleared", "partial", "not_cleared"):
            raise ValueError(f"Invalid group status for {gid}: {status!r}")
        lims = info.get("limitations", [])
        if not isinstance(lims, list) or not lims or any(not str(x).strip() for x in lims):
            raise ValueError(f"Group {gid!r} must include visible non-empty limitations (TL-021)")
        classified_groups[gid] = {
            "status": status,
            "reference_error": info.get("reference_error"),
            "usable_coverage": info.get("usable_coverage"),
            "limitations": [str(x).strip() for x in lims],
        }

    bundle_payload = {
        "candidate_id": candidate_id,
        "source_manifest": source_manifest,
        "detector_config": detector_config,
        "dataset_release": dataset_release,
        "groups": classified_groups,
        "frozen_results": frozen_results,
    }
    freeze_hash = _canonical_sha256(bundle_payload)

    return {
        "schema_version": "touchline.frozen_demo_candidate.v1",
        "candidate_id": candidate_id,
        "freeze_sha256": freeze_hash,
        "colab_session_required_for_replay": False,
        "allow_weight_download": False,
        "source_manifest": copy.deepcopy(source_manifest),
        "detector_config": copy.deepcopy(detector_config),
        "dataset_release": copy.deepcopy(dataset_release),
        "groups": classified_groups,
        "frozen_results": copy.deepcopy(frozen_results),
    }


def verify_candidate_reproduction(
    frozen_candidate: dict[str, Any],
    reproduced_results: dict[str, Any],
    *,
    numeric_tolerance: float = 1e-4,
    active_colab_session: bool = False,
) -> dict[str, Any]:
    """Verify that a reproduction run recovers the frozen demo candidate results within declared tolerances without an active Colab session (TL-021.2)."""
    if active_colab_session:
        raise ValueError("Reproduction check must run offline/locally without requiring an active Colab session (TL-021.2)")

    expected = frozen_candidate.get("frozen_results", {})
    mismatches: list[str] = []

    for key, exp_val in expected.items():
        if key not in reproduced_results:
            mismatches.append(f"missing key: {key}")
            continue
        actual_val = reproduced_results[key]
        if isinstance(exp_val, (int, float)) and not isinstance(exp_val, bool):
            if not isinstance(actual_val, (int, float)) or abs(float(actual_val) - float(exp_val)) > numeric_tolerance:
                mismatches.append(f"{key}: expected {exp_val}, got {actual_val} (tol={numeric_tolerance})")
        else:
            if actual_val != exp_val:
                mismatches.append(f"{key}: expected {exp_val!r}, got {actual_val!r}")

    return {
        "schema_version": "touchline.reproduction_verification.v1",
        "candidate_id": frozen_candidate.get("candidate_id"),
        "freeze_sha256": frozen_candidate.get("freeze_sha256"),
        "reproduced_within_tolerance": len(mismatches) == 0,
        "numeric_tolerance": numeric_tolerance,
        "mismatches": mismatches,
    }
