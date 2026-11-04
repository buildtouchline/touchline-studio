import unittest

from touchline.metrics import (
    Calibration,
    Point,
    activity_summary,
    compare_zone_summaries,
    evaluate_activity_intervals,
    intensity_zone_summary,
)


class ActivityAndIntensityZoneTests(unittest.TestCase):
    def setUp(self):
        # 1 pixel = 1 metre identity calibration valid on [0, 100)s
        self.calibration = Calibration(
            matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            independent_check_ref="ref_course_check_01",
            valid_from_s=0.0,
            valid_until_s=100.0,
            image_bounds=(0.0, 0.0, 1920.0, 1080.0),
        )

    def test_activity_summary_uncalibrated_suppressed(self):
        pts = [
            Point(0.0, 10.0, 10.0, "seg1", "rev1"),
            Point(0.5, 11.0, 10.0, "seg1", "rev1"),
        ]
        res = activity_summary(pts, calibration=None)
        self.assertEqual(res["status"], "unavailable")
        self.assertIsNone(res["class_durations_s"])

        invalid_cal = self.calibration.invalidate("camera moved")
        res_inv = activity_summary(pts, calibration=invalid_cal)
        self.assertEqual(res_inv["status"], "unavailable")
        self.assertIn("camera moved", res_inv["reason"])

    def test_activity_summary_missing_gap_is_unknown_never_standing(self):
        # Point 0 -> Point 1: 0.5s stationary (0.0 m/s -> standing)
        # Point 1 -> Point 2: 2.0s gap (> max_gap_s=0.75s) with zero displacement!
        # Must be classified as 'unknown', NEVER 'standing'!
        pts = [
            Point(0.0, 10.0, 10.0, "seg1", "rev1"),
            Point(0.5, 10.0, 10.0, "seg1", "rev1"),
            Point(2.5, 10.0, 10.0, "seg1", "rev1"),
        ]
        res = activity_summary(pts, calibration=self.calibration, max_gap_s=0.75)
        self.assertEqual(res["status"], "estimated")
        self.assertEqual(res["method"], "speed_rule_estimate")
        self.assertAlmostEqual(res["class_durations_s"]["standing"], 0.5)
        self.assertAlmostEqual(res["unknown_duration_s"], 2.0)
        self.assertAlmostEqual(res["eligible_duration_s"], 0.5)
        self.assertEqual(res["intervals"][1]["activity"], "unknown")
        self.assertEqual(res["intervals"][1]["reason"], "gap exceeds eligibility limit")

    def test_activity_summary_variable_sampling_and_transitions(self):
        # Variable cadence:
        # [0.0 -> 0.2] dt=0.2s, dx=0.04m -> 0.2 m/s (standing)
        # [0.2 -> 0.7] dt=0.5s, dx=0.50m -> 1.0 m/s (walking)
        # [0.7 -> 1.1] dt=0.4s, dx=1.20m -> 3.0 m/s (jogging)
        # [1.1 -> 1.6] dt=0.5s, dx=2.50m -> 5.0 m/s (running)
        # [1.6 -> 2.0] dt=0.4s, dx=3.20m -> 8.0 m/s (sprinting)
        pts = [
            Point(0.0, 0.0, 0.0, "seg1", "rev1"),
            Point(0.2, 0.04, 0.0, "seg1", "rev1"),
            Point(0.7, 0.54, 0.0, "seg1", "rev1"),
            Point(1.1, 1.74, 0.0, "seg1", "rev1"),
            Point(1.6, 4.24, 0.0, "seg1", "rev1"),
            Point(2.0, 7.44, 0.0, "seg1", "rev1"),
        ]
        res = activity_summary(pts, calibration=self.calibration, max_gap_s=0.75)
        self.assertEqual(res["status"], "estimated")
        d = res["class_durations_s"]
        self.assertAlmostEqual(d["standing"], 0.2)
        self.assertAlmostEqual(d["walking"], 0.5)
        self.assertAlmostEqual(d["jogging"], 0.4)
        self.assertAlmostEqual(d["running"], 0.5)
        self.assertAlmostEqual(d["sprinting"], 0.4)
        self.assertAlmostEqual(sum(d.values()), res["eligible_duration_s"])
        self.assertIn("jumping", res["unimplemented_classes"])

    def test_evaluate_activity_intervals_confusion_and_coverage(self):
        pred = [
            {"start_s": 0.0, "end_s": 1.0, "activity": "walking"},
            {"start_s": 1.0, "end_s": 2.0, "activity": "unknown"},
            {"start_s": 2.0, "end_s": 3.0, "activity": "running"},
        ]
        ref = [
            {"start_s": 0.0, "end_s": 1.0, "activity": "walking"},
            {"start_s": 1.0, "end_s": 2.0, "activity": "jogging"},
            {"start_s": 2.0, "end_s": 3.0, "activity": "jogging"},
        ]
        ev = evaluate_activity_intervals(pred, ref)
        self.assertAlmostEqual(ev["evaluated_duration_s"], 2.0)
        self.assertAlmostEqual(ev["unknown_predicted_duration_s"], 1.0)
        self.assertAlmostEqual(ev["usable_coverage_ratio"], 2.0 / 3.0, places=3)
        self.assertAlmostEqual(ev["agreement_rate"], 0.5)
        self.assertAlmostEqual(ev["confusion_duration_s"]["walking"]["walking"], 1.0)
        self.assertAlmostEqual(ev["confusion_duration_s"]["jogging"]["running"], 1.0)

    def test_intensity_zones_boundary_and_gap_rules(self):
        # Exact boundary values: 2.0, 4.0, 5.5, 7.0 m/s over 0.5s intervals
        # [0.0 -> 0.5]: dx=1.0m -> 2.0 m/s -> falls in zone_2_moderate [2.0, 4.0)
        # [0.5 -> 1.0]: dx=2.0m -> 4.0 m/s -> falls in zone_3_high_speed_running [4.0, 5.5)
        # [1.0 -> 1.5]: dx=2.75m -> 5.5 m/s -> falls in zone_4_very_high_speed [5.5, 7.0)
        # [1.5 -> 2.0]: dx=3.5m -> 7.0 m/s -> falls in zone_5_sprint_band [7.0, inf)
        # [2.0 -> 4.0]: 2.0s gap (> 0.75s) -> excluded
        pts = [
            Point(0.0, 0.0, 0.0, "seg1", "rev1"),
            Point(0.5, 1.0, 0.0, "seg1", "rev1"),
            Point(1.0, 3.0, 0.0, "seg1", "rev1"),
            Point(1.5, 5.75, 0.0, "seg1", "rev1"),
            Point(2.0, 9.25, 0.0, "seg1", "rev1"),
            Point(4.0, 20.0, 0.0, "seg1", "rev1"),
        ]
        res = intensity_zone_summary(pts, calibration=self.calibration, max_gap_s=0.75)
        self.assertEqual(res["status"], "estimated")
        self.assertAlmostEqual(res["eligible_duration_s"], 2.0)
        self.assertAlmostEqual(res["excluded_duration_s"], 2.0)
        self.assertAlmostEqual(res["total_eligible_distance_m"], 9.25)

        zones = res["zones"]
        self.assertAlmostEqual(zones["zone_1_low"]["time_s"], 0.0)
        self.assertAlmostEqual(zones["zone_2_moderate"]["time_s"], 0.5)
        self.assertAlmostEqual(zones["zone_3_high_speed_running"]["time_s"], 0.5)
        self.assertAlmostEqual(zones["zone_4_very_high_speed"]["time_s"], 0.5)
        self.assertAlmostEqual(zones["zone_5_sprint_band"]["time_s"], 0.5)

        # High-speed running (>= 4.0 m/s) = zone 3 + zone 4 + zone 5
        self.assertAlmostEqual(res["high_speed_running_time_s"], 1.5)
        self.assertAlmostEqual(res["high_speed_running_distance_m"], 2.0 + 2.75 + 3.5)

        # Verify no double counting across zones
        self.assertAlmostEqual(sum(z["time_s"] for z in zones.values()), res["eligible_duration_s"])
        self.assertAlmostEqual(sum(z["distance_m"] for z in zones.values()), res["total_eligible_distance_m"])
        self.assertIn("not validated discrete sprint events", res["sprint_event_counts"])

    def test_compare_zone_summaries_reference_error(self):
        pts = [
            Point(0.0, 0.0, 0.0, "seg1", "rev1"),
            Point(0.5, 2.25, 0.0, "seg1", "rev1"),  # 4.5 m/s -> zone_3
        ]
        pred = intensity_zone_summary(pts, calibration=self.calibration)
        ref = {
            "status": "estimated",
            "zones": {
                "zone_3_high_speed_running": {"time_s": 0.5, "distance_m": 2.0},
            },
        }
        cmp_res = compare_zone_summaries(pred, ref)
        self.assertEqual(cmp_res["status"], "compared")
        z3_err = cmp_res["per_zone_error"]["zone_3_high_speed_running"]
        self.assertAlmostEqual(z3_err["abs_time_error_s"], 0.0)
        self.assertAlmostEqual(z3_err["abs_distance_error_m"], 0.25)


if __name__ == "__main__":
    unittest.main()
