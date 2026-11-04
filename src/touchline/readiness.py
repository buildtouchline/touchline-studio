"""Fail-closed clearance for predeclared Touchline evaluation criteria (TL-010.1).

This module evaluates evidence records; it does not establish that a reference
is independent or that an evaluation was performed. Reviewers must verify both.
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path


REQUIRED_METRICS = {
    "level1_identity_visibility": ("usable_coverage",),
    "level1_spatial_review": ("position_error_p90",),
    "level1_attempt_review": ("evidence_link_rate",),
    "level2_distance": ("distance_error",),
    "level2_speed": ("average_speed_error", "windowed_max_speed_error"),
    "level2_intensity_zones": ("zone_time_error",),
    "level2_activity": ("macro_f1",),
}
METRIC_CONTRACTS = {
    "usable_coverage": ("min", "fraction_of_labelled_visible_time", 0, 1),
    "position_error_p90": ("max", "fraction_of_frame_diagonal", 0, 1),
    "evidence_link_rate": ("min", "fraction_of_reviewed_attempts", 0, 1),
    "distance_error": ("max", "relative_absolute_error", 0, None),
    "average_speed_error": ("max", "relative_absolute_error", 0, None),
    "windowed_max_speed_error": ("max", "relative_absolute_error", 0, None),
    "zone_time_error": ("max", "seconds_per_30_second_drill", 0, 30),
    "macro_f1": ("min", "fraction", 0, 1),
}
RESOURCE_METRICS = ("total_wall_s", "peak_ram_bytes", "peak_vram_bytes", "disk_bytes")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value):
    if not _nonempty(value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _in_domain(value, contract):
    return _number(value) and value >= contract[2] and (contract[3] is None or value <= contract[3])


def _evaluation_set(value):
    if not isinstance(value, dict) or any(
        not _nonempty(value.get(field)) for field in ("id", "source_hash", "labels_hash")
    ):
        return False
    if value.get("split_role") != "held_out" or any(
        not SHA256.fullmatch(value[field]) for field in ("source_hash", "labels_hash")
    ):
        return False
    groups = value.get("capture_group_ids")
    return isinstance(groups, list) and bool(groups) and all(_nonempty(group) for group in groups) and len(groups) == len(set(groups))


def assess_group(group_id, record):
    """Return a derived status and reasons; never trust a supplied status field."""
    if group_id not in REQUIRED_METRICS:
        raise ValueError(f"unknown readiness group: {group_id}")
    if not isinstance(record, dict):
        return {"status": "pending", "reasons": ["group record missing"]}

    criterion = record.get("criterion")
    if not isinstance(criterion, dict) or criterion.get("status") != "frozen":
        return {"status": "pending", "reasons": ["criterion not frozen"]}
    frozen_at = _timestamp(criterion.get("frozen_at"))
    if frozen_at is None or not _nonempty(criterion.get("approved_by")):
        return {"status": "pending", "reasons": ["criterion freeze lacks timestamp or approver"]}
    evaluation_set = criterion.get("evaluation_set")
    if not _evaluation_set(evaluation_set):
        return {"status": "pending", "reasons": ["evaluation set was not frozen with source, labels and capture groups"]}

    metrics = criterion.get("metrics")
    if not isinstance(metrics, dict):
        return {"status": "pending", "reasons": ["metric thresholds missing"]}
    for metric_id in REQUIRED_METRICS[group_id]:
        rule = metrics.get(metric_id)
        contract = METRIC_CONTRACTS[metric_id]
        if (
            not isinstance(rule, dict)
            or rule.get("operator") != contract[0]
            or rule.get("unit") != contract[1]
            or not _in_domain(rule.get("threshold"), contract)
        ):
            return {"status": "pending", "reasons": [f"{metric_id}: threshold incompatible with metric definition"]}

    if (
        record.get("implemented") is not True
        or record.get("tested") is not True
        or not _nonempty(record.get("implementation_ref"))
        or not _nonempty(record.get("test_ref"))
    ):
        return {"status": "pending", "reasons": ["implementation or test evidence missing"]}
    limitations = record.get("limitations")
    if not isinstance(limitations, list) or any(not _nonempty(item) for item in limitations):
        return {"status": "pending", "reasons": ["limitations record missing or invalid"]}
    reference = record.get("reference")
    if (
        not isinstance(reference, dict)
        or not _nonempty(reference.get("method"))
        or not _nonempty(reference.get("evidence_ref"))
    ):
        return {"status": "pending", "reasons": ["independent reference evidence missing"]}

    outcome = record.get("outcome")
    if not isinstance(outcome, dict):
        return {"status": "pending", "reasons": ["evaluation outcome missing"]}
    evaluated_at = _timestamp(outcome.get("evaluated_at"))
    if evaluated_at is None or frozen_at > evaluated_at:
        return {"status": "pending", "reasons": ["criterion was not frozen before evaluation"]}
    if (
        outcome.get("evaluation_set_id") != evaluation_set["id"]
        or outcome.get("source_hash") != evaluation_set["source_hash"]
        or outcome.get("labels_hash") != evaluation_set["labels_hash"]
        or not _nonempty(outcome.get("evidence_ref"))
        or not _nonempty(outcome.get("reviewed_by"))
        or type(outcome.get("sample_count")) is not int
        or outcome["sample_count"] <= 0
    ):
        return {"status": "pending", "reasons": ["evaluation provenance or support missing"]}

    values = outcome.get("metrics")
    if not isinstance(values, dict):
        return {"status": "pending", "reasons": ["metric results missing"]}
    failures = []
    for metric_id in REQUIRED_METRICS[group_id]:
        value = values.get(metric_id)
        if not _in_domain(value, METRIC_CONTRACTS[metric_id]):
            return {"status": "pending", "reasons": [f"{metric_id}: result missing or outside metric domain"]}
        rule = metrics[metric_id]
        passes = value >= rule["threshold"] if rule["operator"] == "min" else value <= rule["threshold"]
        if not passes:
            failures.append(metric_id)
    return {"status": "failed" if failures else "cleared", "reasons": failures}


def assess_resources(record):
    """Apply the same predeclared-before-measurement rule to device budgets."""
    if not isinstance(record, dict) or record.get("status") != "frozen":
        return {"status": "pending", "reasons": ["resource budget not frozen"]}
    frozen_at = _timestamp(record.get("frozen_at"))
    limits = record.get("limits")
    if frozen_at is None or not _nonempty(record.get("approved_by")) or not isinstance(limits, dict):
        return {"status": "pending", "reasons": ["resource budget incomplete"]}
    if any(not _number(limits.get(metric)) or limits[metric] < 0 for metric in RESOURCE_METRICS):
        return {"status": "pending", "reasons": ["resource limits missing or invalid"]}
    outcome = record.get("outcome")
    if not isinstance(outcome, dict):
        return {"status": "pending", "reasons": ["resource measurements missing"]}
    measured_at = _timestamp(outcome.get("measured_at"))
    if measured_at is None or measured_at < frozen_at:
        return {"status": "pending", "reasons": ["resource budget was not frozen before measurement"]}
    if not _nonempty(outcome.get("device")) or not _nonempty(outcome.get("evidence_ref")):
        return {"status": "pending", "reasons": ["resource provenance missing"]}
    values = outcome.get("metrics")
    if not isinstance(values, dict) or any(
        not _number(values.get(metric)) or values[metric] < 0 for metric in RESOURCE_METRICS
    ):
        return {"status": "pending", "reasons": ["resource measurements missing or invalid"]}
    failures = [metric for metric in RESOURCE_METRICS if values[metric] > limits[metric]]
    return {"status": "failed" if failures else "cleared", "reasons": failures}


def assess_scorecard(scorecard):
    """Assess exactly the three Level 1 and four selected Level 2 groups."""
    if not isinstance(scorecard, dict) or scorecard.get("schema_version") != "touchline.readiness.v1":
        raise ValueError("unsupported readiness scorecard")
    groups = scorecard.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(REQUIRED_METRICS):
        raise ValueError("scorecard must contain exactly seven required groups")
    results = {group_id: assess_group(group_id, groups[group_id]) for group_id in REQUIRED_METRICS}
    resources = assess_resources(scorecard.get("resource_budget"))
    return {
        "groups": results,
        "resources": resources,
        "status": "cleared" if resources["status"] == "cleared" and all(
            row["status"] == "cleared" for row in results.values()
        ) else "not_cleared",
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Assess a Touchline readiness scorecard")
    parser.add_argument("scorecard", type=Path)
    args = parser.parse_args(argv)
    result = assess_scorecard(json.loads(args.scorecard.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "cleared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
