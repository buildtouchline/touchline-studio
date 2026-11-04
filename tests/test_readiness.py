import copy
import json
import unittest
from pathlib import Path

from touchline.readiness import METRIC_CONTRACTS, REQUIRED_METRICS, assess_group, assess_resources, assess_scorecard


def complete_group(group_id):
    metrics = {
        metric_id: {"operator": METRIC_CONTRACTS[metric_id][0],
                    "threshold": 0.8 if METRIC_CONTRACTS[metric_id][0] == "min" else 0.1,
                    "unit": METRIC_CONTRACTS[metric_id][1]}
        for metric_id in REQUIRED_METRICS[group_id]
    }
    return {
        "implemented": True,
        "tested": True,
        "implementation_ref": "commit:sha",
        "test_ref": "test-run:sha",
        "limitations": [],
        "reference": {"method": "independent manual labels", "evidence_ref": "labels:sha256"},
        "criterion": {
            "status": "frozen",
            "frozen_at": "2026-10-26T00:00:00Z",
            "approved_by": "reviewer-id",
            "evaluation_set": {"id": "held-out-session", "source_hash": "a" * 64,
                               "labels_hash": "b" * 64, "capture_group_ids": ["capture-1"],
                               "split_role": "held_out"},
            "metrics": metrics,
        },
        "outcome": {
            "evaluated_at": "2026-10-27T00:00:00Z",
            "evaluation_set_id": "held-out-session",
            "source_hash": "a" * 64,
            "labels_hash": "b" * 64,
            "evidence_ref": "evaluation:sha256",
            "reviewed_by": "different-reviewer-id",
            "sample_count": 10,
            "metrics": {metric_id: 0.9 if METRIC_CONTRACTS[metric_id][0] == "min" else 0.05
                        for metric_id in REQUIRED_METRICS[group_id]},
        },
    }


def complete_resources():
    return {
        "status": "frozen",
        "frozen_at": "2026-10-26T00:00:00Z",
        "approved_by": "reviewer-id",
        "limits": {"total_wall_s": 60, "peak_ram_bytes": 1000, "peak_vram_bytes": 0, "disk_bytes": 1000},
        "outcome": {
            "measured_at": "2026-10-27T00:00:00Z",
            "device": "declared-pilot-device",
            "evidence_ref": "resource-log:sha256",
            "metrics": {"total_wall_s": 30, "peak_ram_bytes": 500, "peak_vram_bytes": 0, "disk_bytes": 500},
        },
    }


class ReadinessTests(unittest.TestCase):
    def test_all_seven_groups_need_evidence(self):
        card = {
            "schema_version": "touchline.readiness.v1",
            "groups": {group_id: complete_group(group_id) for group_id in REQUIRED_METRICS},
            "resource_budget": complete_resources(),
        }
        self.assertEqual(assess_scorecard(card)["status"], "cleared")
        card["groups"]["level2_activity"]["reference"]["evidence_ref"] = None
        self.assertEqual(assess_scorecard(card)["status"], "not_cleared")
        self.assertEqual(assess_scorecard(card)["groups"]["level2_activity"]["status"], "pending")

    def test_unset_or_posthoc_criterion_cannot_clear(self):
        row = complete_group("level2_distance")
        row["criterion"]["status"] = "proposed"
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")
        row["criterion"]["status"] = "frozen"
        row["criterion"]["frozen_at"] = "2026-10-28T00:00:00Z"
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")

    def test_failed_threshold_is_not_cleared(self):
        row = complete_group("level1_identity_visibility")
        row["outcome"]["metrics"]["usable_coverage"] = 0.7
        self.assertEqual(assess_group("level1_identity_visibility", row)["status"], "failed")

    def test_speed_requires_average_and_windowed_maximum(self):
        row = complete_group("level2_speed")
        del row["outcome"]["metrics"]["windowed_max_speed_error"]
        self.assertEqual(assess_group("level2_speed", row)["status"], "pending")
        row["outcome"]["metrics"]["windowed_max_speed_error"] = 0.2
        self.assertEqual(assess_group("level2_speed", row)["status"], "failed")

    def test_boolean_zero_support_and_nonfinite_values_do_not_clear(self):
        row = complete_group("level1_attempt_review")
        row["outcome"]["sample_count"] = 0
        self.assertEqual(assess_group("level1_attempt_review", row)["status"], "pending")
        row["outcome"]["sample_count"] = 10
        row["outcome"]["metrics"]["evidence_link_rate"] = float("nan")
        self.assertEqual(assess_group("level1_attempt_review", row)["status"], "pending")
        row["outcome"]["metrics"]["evidence_link_rate"] = True
        self.assertEqual(assess_group("level1_attempt_review", row)["status"], "pending")

    def test_supplied_status_cannot_override_derived_status(self):
        row = copy.deepcopy(complete_group("level2_activity"))
        row["status"] = "cleared"
        row["criterion"]["approved_by"] = None
        self.assertEqual(assess_group("level2_activity", row)["status"], "pending")

    def test_missing_group_rejected(self):
        card = {
            "schema_version": "touchline.readiness.v1",
            "groups": {group_id: complete_group(group_id) for group_id in REQUIRED_METRICS},
            "resource_budget": complete_resources(),
        }
        del card["groups"]["level2_distance"]
        with self.assertRaises(ValueError):
            assess_scorecard(card)

    def test_shipped_template_is_explicitly_pending(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "readiness-criteria.template.json"
        card = json.loads(path.read_text(encoding="utf-8"))
        result = assess_scorecard(card)
        self.assertEqual(result["status"], "not_cleared")
        self.assertTrue(all(row["status"] == "pending" for row in result["groups"].values()))
        self.assertEqual(result["resources"]["status"], "pending")

    def test_unset_or_failed_resource_budget_blocks_clearance(self):
        card = {
            "schema_version": "touchline.readiness.v1",
            "groups": {group_id: complete_group(group_id) for group_id in REQUIRED_METRICS},
            "resource_budget": complete_resources(),
        }
        card["resource_budget"]["status"] = "proposed"
        self.assertEqual(assess_scorecard(card)["status"], "not_cleared")
        card["resource_budget"]["status"] = "frozen"
        card["resource_budget"]["outcome"]["metrics"]["total_wall_s"] = 61
        self.assertEqual(assess_resources(card["resource_budget"])["status"], "failed")
        self.assertEqual(assess_scorecard(card)["status"], "not_cleared")

    def test_wrong_operator_and_negative_error_never_clear(self):
        row = complete_group("level2_distance")
        row["criterion"]["metrics"]["distance_error"]["operator"] = "min"
        row["outcome"]["metrics"]["distance_error"] = 0.9
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")
        row["criterion"]["metrics"]["distance_error"]["operator"] = "max"
        row["outcome"]["metrics"]["distance_error"] = -1.0
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")

    def test_metric_units_and_bounded_fractions_are_checked(self):
        row = complete_group("level1_identity_visibility")
        row["criterion"]["metrics"]["usable_coverage"]["unit"] = "seconds"
        self.assertEqual(assess_group("level1_identity_visibility", row)["status"], "pending")
        row["criterion"]["metrics"]["usable_coverage"]["unit"] = "fraction_of_labelled_visible_time"
        row["outcome"]["metrics"]["usable_coverage"] = 1.1
        self.assertEqual(assess_group("level1_identity_visibility", row)["status"], "pending")

    def test_evaluation_set_is_frozen_and_matches_outcome(self):
        row = complete_group("level2_distance")
        del row["criterion"]["evaluation_set"]
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")
        row = complete_group("level2_distance")
        row["outcome"]["source_hash"] = "different-source"
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")
        row = complete_group("level2_distance")
        row["outcome"]["labels_hash"] = "different-labels"
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")
        row = complete_group("level2_distance")
        row["criterion"]["evaluation_set"]["capture_group_ids"] = []
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")
        row = complete_group("level2_distance")
        row["criterion"]["evaluation_set"]["split_role"] = "train"
        self.assertEqual(assess_group("level2_distance", row)["status"], "pending")


if __name__ == "__main__":
    unittest.main()
