"""Credit pool reservation ledger (TL-025 / TL-S11) and bounded Meta Evidence Director (TL-026 / TL-S12).

Enforces:
- TL-025: Separate non-interchangeable pools (`meta_model_grant` $150 cap, `modal_compute` $100 cap,
  `baseten_pending` $0 usable). Stage caps sum to each pool; reserve covers outstanding reservations
  and pending charges; no paid job runs on unverified balance/rate/eligibility; local SAM 3 consumes
  runtime compute (`modal_compute` or local/Colab), never the `meta_model_grant` API pool.
- TL-026: Source-backed `EvidencePlan` validation and bounded execution (at most 1 planning request
  and 1 bounded refinement per case; allowlisted actions only: `inspect_frames`, `segment_existing_frames`,
  `compare_existing_observations`, `request_human_review`). Rejects fabricated source IDs, out-of-range
  times, non-existent PTS frames, or budget overruns. Model suggestions never silently alter accepted
  identity, metrics, or reports—only an explicit coach `accept` decision increments the association
  revision. Includes a deterministic fallback (`deterministic_rule_fallback`) when endpoints fail.
"""
from __future__ import annotations

import copy
import math
from typing import Any
from .analysis_policy import PROVIDER_CAPS_USD
from .diagnostics import diagnosed


MODAL_STAGE_CAPS_USD: dict[str, float] = {
    "compatibility_and_sam3_fallback": 10.0,
    "torchvision_finetune_and_eval": 25.0,
    "field_pilot_batch_processing": 20.0,
    "days_6_8_refinement_and_demo": 15.0,
    "sam3_training_smoke_benchmark": 5.0,
    "unallocated_contingency_reserve": 25.0,
}

MODEL_SERVICE_STAGE_CAPS_USD: dict[str, float] = {
    "verify_endpoints_and_metering": 10.0,
    "curated_model_assisted_audit": 30.0,
    "evidence_director_bounded_demo": 25.0,
    "grounded_explanation_experiments": 10.0,
    "unallocated_reserve": 75.0,
}

ALLOWED_EVIDENCE_ACTIONS = frozenset({
    "inspect_frames",
    "segment_existing_frames",
    "compare_existing_observations",
    "request_human_review",
})


class CreditBudgetLedger:
    """Shared reservation protocol for separate credit pools (TL-025 / TL-S11)."""

    def __init__(self):
        if round(sum(MODAL_STAGE_CAPS_USD.values()), 2) != 100.0:
            raise ValueError("Modal stage caps must sum to $100.00")
        if round(sum(MODEL_SERVICE_STAGE_CAPS_USD.values()), 2) != 150.0:
            raise ValueError("Model-service stage caps must sum to $150.00")

        self.pools: dict[str, dict[str, Any]] = {
            "meta_model_grant": {
                "reported_cap_usd": 150.0,
                "verified_balance_usd": None,
                "eligibility_verified": False,
                "stage_caps_usd": dict(MODEL_SERVICE_STAGE_CAPS_USD),
            },
            "modal_compute": {
                "reported_cap_usd": 100.0,
                "verified_balance_usd": None,
                "eligibility_verified": False,
                "stage_caps_usd": dict(MODAL_STAGE_CAPS_USD),
            },
            "baseten_pending": {
                "reported_cap_usd": 0.0,
                "verified_balance_usd": 0.0,
                "eligibility_verified": False,
                "stage_caps_usd": {},
            },
        }
        self.reservations: dict[str, dict[str, Any]] = {}

    @diagnosed('budget.verify')
    def verify_pool_balance(self, pool_name: str, verified_balance_usd: float, *, eligible: bool = True) -> None:
        if pool_name == "baseten_pending":
            raise ValueError("Pending Baseten credits cannot be marked usable until approved and terms confirmed (TL-025)")
        if pool_name not in self.pools:
            raise ValueError(f"Unknown credit pool: {pool_name!r}")
        if isinstance(verified_balance_usd, bool) or not math.isfinite(verified_balance_usd) or verified_balance_usd < 0:
            raise ValueError("verified_balance_usd must be non-negative")
        pool = self.pools[pool_name]
        pool["verified_balance_usd"] = round(min(float(verified_balance_usd), pool["reported_cap_usd"]), 4)
        pool["eligibility_verified"] = bool(eligible)

    def _active_stage_spend_and_reserved(self, pool_name: str, stage: str) -> float:
        total = 0.0
        for res in self.reservations.values():
            if res["pool"] == pool_name and res["stage"] == stage and res["status"] in ("reserved", "pending_reconciliation", "reconciled"):
                total += float(res["effective_charge_usd"])
        return round(total, 4)

    def _active_pool_spend_and_reserved(self, pool_name: str) -> float:
        total = 0.0
        for res in self.reservations.values():
            if res["pool"] == pool_name and res["status"] in ("reserved", "pending_reconciliation", "reconciled"):
                total += float(res["effective_charge_usd"])
        return round(total, 4)

    @diagnosed('budget.reserve')
    def reserve_job(
        self,
        *,
        reservation_id: str,
        pool_name: str,
        stage: str,
        owner: str,
        workload_type: str,
        unit_rate_usd: float | None,
        bounded_quantity: float | None,
        retries: int = 0,
        startup_storage_allowance_usd: float = 0.0,
        dataset_or_input_hash: str,
        cancellation_path: str,
        data_contribution_discount_enabled: bool = False,
    ) -> dict[str, Any]:
        """Reserve worst-case job cost before running any paid request (TL-025)."""
        if pool_name == "baseten_pending":
            raise ValueError("Cannot reserve against pending Baseten pool ($0 usable)")
        if pool_name not in self.pools:
            raise ValueError(f"Unknown pool: {pool_name!r}")
        if not reservation_id or reservation_id in self.reservations:
            raise ValueError("A new unique reservation ID is required; never overwrite a charge")

        # Local SAM 3 consumes runtime compute, never the Meta model API allowance
        if workload_type == "local_sam3" and pool_name == "meta_model_grant":
            raise ValueError("local_sam3 consumes runtime compute (modal_compute), not meta_model_grant API allowance (TL-025)")

        if data_contribution_discount_enabled:
            raise ValueError("Data-contribution discount for identifiable footage is forbidden (TL-S11)")

        pool = self.pools[pool_name]
        if not pool["eligibility_verified"] or pool["verified_balance_usd"] is None:
            raise ValueError(f"Cannot reserve paid job on unverified pool {pool_name!r} (TL-025)")

        if stage not in pool["stage_caps_usd"] or stage in ("unallocated_contingency_reserve", "unallocated_reserve"):
            raise ValueError(f"Stage {stage!r} is not an allocatable operational stage in {pool_name!r}")

        if unit_rate_usd is None or bounded_quantity is None or not all(math.isfinite(v) for v in (unit_rate_usd, bounded_quantity, startup_storage_allowance_usd)) or unit_rate_usd <= 0 or bounded_quantity <= 0 or startup_storage_allowance_usd < 0:
            raise ValueError("Known positive unit_rate_usd and bounded_quantity are required before launch")
        if type(retries) is not int or retries < 0:
            raise ValueError("retries must be non-negative")
        if not dataset_or_input_hash or not cancellation_path:
            raise ValueError("dataset_or_input_hash and cancellation_path are required")

        # Only one active paid GPU reservation initially on modal_compute
        if pool_name == "modal_compute":
            active_gpu = [
                r for r in self.reservations.values()
                if r["pool"] == "modal_compute" and r["status"] == "reserved"
            ]
            if active_gpu:
                raise ValueError("Only one active paid GPU reservation is allowed initially on modal_compute (TL-025)")

        attempts = 1 + int(retries)
        worst_case_usd = round(
            attempts * (float(unit_rate_usd) * float(bounded_quantity) + float(startup_storage_allowance_usd)),
            4,
        )

        stage_cap = pool["stage_caps_usd"][stage]
        stage_used = self._active_stage_spend_and_reserved(pool_name, stage)
        if stage_used + worst_case_usd > stage_cap + 1e-9:
            raise ValueError(
                f"Reservation (${worst_case_usd}) exceeds remaining stage cap for {stage!r} (${round(stage_cap - stage_used, 4)} left)"
            )

        # Reserve must cover outstanding reservations and pending charges without consuming unallocated reserve implicitly
        reserve_key = "unallocated_contingency_reserve" if pool_name == "modal_compute" else "unallocated_reserve"
        reserve_cap = pool["stage_caps_usd"][reserve_key]
        current_ceiling = float(PROVIDER_CAPS_USD["modal" if pool_name == "modal_compute" else "meta"])
        allocatable_ceiling = min(pool["verified_balance_usd"], pool["reported_cap_usd"] - reserve_cap, current_ceiling)
        pool_used = self._active_pool_spend_and_reserved(pool_name)
        if pool_used + worst_case_usd > allocatable_ceiling + 1e-9:
            raise ValueError(
                f"Reservation (${worst_case_usd}) exceeds verified non-reserve pool capacity (${round(allocatable_ceiling - pool_used, 4)} left)"
            )

        record = {
            "reservation_id": reservation_id,
            "pool": pool_name,
            "stage": stage,
            "owner": owner,
            "workload_type": workload_type,
            "attempts": attempts,
            "worst_case_usd": worst_case_usd,
            "measured_usd": None,
            "effective_charge_usd": worst_case_usd,
            "status": "reserved",
            "dataset_or_input_hash": dataset_or_input_hash,
            "cancellation_path": cancellation_path,
        }
        self.reservations[reservation_id] = record
        return copy.deepcopy(record)

    @diagnosed('budget.reconcile')
    def reconcile_job(self, reservation_id: str, *, measured_usd: float | None, billing_finalized: bool) -> dict[str, Any]:
        """Reconcile measured usage or mark pending while retaining reservation until billed (TL-S11)."""
        if reservation_id not in self.reservations:
            raise ValueError(f"Unknown reservation_id: {reservation_id!r}")
        res = self.reservations[reservation_id]
        if measured_usd is not None and (not math.isfinite(measured_usd) or measured_usd < 0):
            raise ValueError("Finite nonnegative measured cost required")
        if billing_finalized and measured_usd is not None:
            if measured_usd < 0 or measured_usd > res["worst_case_usd"] * 1.05:
                raise ValueError("Invalid measured_usd for finalized reconciliation")
            res["measured_usd"] = round(float(measured_usd), 4)
            res["effective_charge_usd"] = res["measured_usd"]
            res["status"] = "reconciled"
        else:
            # Retain worst-case reservation until billing is finalized
            res["measured_usd"] = round(float(measured_usd), 4) if measured_usd is not None else None
            res["effective_charge_usd"] = res["worst_case_usd"]
            res["status"] = "pending_reconciliation"
        return copy.deepcopy(res)


def validate_and_execute_evidence_plan(
    plan: dict[str, Any] | None,
    *,
    run_context: dict[str, Any],
    call_history: list[dict[str, Any]] | None = None,
    endpoint_available: bool = True,
) -> dict[str, Any]:
    """Validate and execute a bounded Meta Evidence Director plan (TL-026 / TL-S12).

    `run_context` requires:
    - `run_id`: str
    - `source_id`: str
    - `valid_time_bounds_s`: tuple[float, float]
    - `available_frames_by_pts`: dict[int, dict]
    - `max_billable_units_budget`: int
    - `local_track_id`: int
    - `current_association_revision`: int
    - `ambiguous_interval_s`: tuple[float, float]
    - `ambiguous_pts_list`: list[int]
    """
    history = list(call_history or [])
    planning_calls = sum(1 for c in history if c.get("phase") == "planning")
    refinement_calls = sum(1 for c in history if c.get("phase") == "refinement")

    # Enforce at most 1 planning request and 1 bounded refinement per case (no recursive loop)
    if len(history) >= 2 or planning_calls > 1 or refinement_calls > 1:
        raise ValueError(
            "Evidence Director policy exceeded: at most 1 planning request and 1 bounded refinement allowed per case (TL-026)"
        )

    # Deterministic fallback when live model endpoint is unavailable or plan is None
    if not endpoint_available or plan is None:
        amb_pts = list(run_context.get("ambiguous_pts_list", []))
        mid_pts = amb_pts[len(amb_pts) // 2] if amb_pts else None
        selected_pts = [p for p in [amb_pts[0] if amb_pts else None, mid_pts, amb_pts[-1] if amb_pts else None] if p is not None]
        # deduplicate preserving order
        dedup_pts = list(dict.fromkeys(selected_pts))
        return {
            "schema_version": "touchline.evidence_card.v1",
            "mode": "deterministic_rule_fallback",
            "replay_labelled": True,
            "run_id": run_context["run_id"],
            "source_id": run_context["source_id"],
            "local_track_id": run_context["local_track_id"],
            "action": "inspect_frames",
            "inspected_pts": dedup_pts,
            "supporting_evidence": ["Uniform motion continuity before crossing"],
            "counter_evidence": ["Bounding box overlap >= 0.50 during crossing interval"],
            "model_suggestion": None,
            "billed_units": 0,
            "coach_decision": "pending",
            "identity_modified": False,
            "association_revision": run_context["current_association_revision"],
        }

    # Validate declarative plan against allowlist, source, time, frame existence, and budget
    action = plan.get("action")
    if action not in ALLOWED_EVIDENCE_ACTIONS:
        raise ValueError(f"Unsupported Evidence Director action: {action!r}")

    if plan.get("source_id") != run_context.get("source_id"):
        raise ValueError(f"Fabricated or mismatched source_id: {plan.get('source_id')!r}")

    if plan.get("run_id") != run_context.get("run_id"):
        raise ValueError(f"Mismatched run_id: {plan.get('run_id')!r}")

    interval = plan.get("interval_s")
    min_t, max_t = run_context["valid_time_bounds_s"]
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or interval[0] < min_t
        or interval[1] > max_t
        or interval[0] >= interval[1]
    ):
        raise ValueError(f"Invalid or out-of-range plan interval_s: {interval!r}")

    requested_pts = plan.get("requested_pts", [])
    avail_frames = run_context.get("available_frames_by_pts", {})
    for pts in requested_pts:
        if pts not in avail_frames and str(pts) not in avail_frames:
            raise ValueError(f"Invented/non-existent frame PTS requested: {pts} (source-rate cap cannot be exceeded)")

    units = int(plan.get("max_billable_units", 0))
    budget = int(run_context.get("max_billable_units_budget", 0))
    if units <= 0 or units > budget:
        raise ValueError(f"Plan max_billable_units ({units}) exceeds case budget ({budget})")

    if not plan.get("supporting_evidence") or not plan.get("counter_evidence"):
        raise ValueError("EvidencePlan must provide both supporting_evidence and counter_evidence (TL-S12)")

    # Execute bounded inspection without modifying accepted identity, metrics, or reports
    return {
        "schema_version": "touchline.evidence_card.v1",
        "mode": "bounded_model_plan",
        "replay_labelled": False,
        "run_id": run_context["run_id"],
        "source_id": run_context["source_id"],
        "local_track_id": run_context["local_track_id"],
        "action": action,
        "interval_s": list(interval),
        "inspected_pts": list(requested_pts),
        "supporting_evidence": list(plan["supporting_evidence"]),
        "counter_evidence": list(plan["counter_evidence"]),
        "model_suggestion": plan.get("model_suggestion"),
        "uncertainty": plan.get("uncertainty", "contested_crossing"),
        "billed_units": units,
        "coach_decision": "pending",
        "identity_modified": False,
        "association_revision": run_context["current_association_revision"],
    }


def apply_coach_evidence_decision(
    evidence_card: dict[str, Any],
    *,
    decision: str,
    reviewer: str,
    confirmed_player_id: str | None = None,
) -> dict[str, Any]:
    """Apply coach accept/reject decision to an EvidenceCard (TL-026 / TL-S12).

    Only an explicit coach `accept` creates a new `association_revision`; `reject` or `pending`
    leaves accepted identity and revisions untouched.
    """
    if decision not in ("accept", "reject"):
        raise ValueError("decision must be 'accept' or 'reject'")
    updated = copy.deepcopy(evidence_card)
    updated["coach_decision"] = decision
    updated["reviewed_by"] = reviewer
    if decision == "accept":
        if not confirmed_player_id:
            raise ValueError("confirmed_player_id required when accepting identity resolution")
        updated["identity_modified"] = True
        updated["confirmed_player_id"] = confirmed_player_id
        updated["association_revision"] = int(updated["association_revision"]) + 1
        updated["dependent_metrics_recompute_required"] = True
    else:
        updated["identity_modified"] = False
        updated["dependent_metrics_recompute_required"] = False
    return updated
