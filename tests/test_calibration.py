import unittest
from touchline.contracts import Box
from touchline.metrics import Calibration, Point, movement_summary


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        # 3x3 identity ground-plane mapping
        self.cal = Calibration(
            matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            independent_check_ref="measured-ground-ref-v1",
            valid_from_s=0.0,
            valid_until_s=20.0,
            image_bounds=(0.0, 0.0, 1920.0, 1080.0),
            source_geometry=(1920, 1080),
            geometry_revision="geom_rev_1",
            calibration_revision="cal_rev_1",
            ground_contact_convention="bottom_center",
        )

    def test_box_ground_contact_bottom_center(self):
        box = Box(10.0, 20.0, 50.0, 100.0)
        # (10 + 50) / 2 = 30.0, y2 = 100.0
        contact = box.ground_contact_point()
        self.assertEqual(contact, (30.0, 100.0))

    def test_box_ground_contact_rejects_box_center(self):
        box = Box(10.0, 20.0, 50.0, 100.0)
        with self.assertRaises(ValueError) as ctx:
            box.ground_contact_point(convention="box_center")
        self.assertIn("box centre cannot be silently treated as a ground contact point", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            box.ground_contact_point(convention="center")
        self.assertIn("box centre cannot be silently treated as a ground contact point", str(ctx.exception))

        with self.assertRaises(ValueError):
            box.ground_contact_point(convention="unsupported_convention")

    def test_point_from_box_success(self):
        box = Box(20.0, 40.0, 60.0, 120.0)
        pt = Point.from_box(1.0, box, "seg1", "review_1", convention="bottom_center", geometry_revision="geom_rev_1")
        self.assertEqual(pt.time_s, 1.0)
        self.assertEqual(pt.x_px, 40.0)
        self.assertEqual(pt.y_px, 120.0)
        self.assertEqual(pt.contact_convention, "bottom_center")
        self.assertEqual(pt.geometry_revision, "geom_rev_1")

    def test_point_from_box_rejects_box_center(self):
        box = Box(20.0, 40.0, 60.0, 120.0)
        with self.assertRaises(ValueError) as ctx:
            Point.from_box(1.0, box, "seg1", convention="box_center")
        self.assertIn("box centre cannot be silently treated as a ground contact point", str(ctx.exception))

    def test_point_direct_init_rejects_box_center(self):
        with self.assertRaises(ValueError) as ctx:
            Point(0.0, 40.0, 80.0, "seg1", contact_convention="box_center")
        self.assertIn("box centre cannot be treated as a ground contact point", str(ctx.exception))

    def test_calibration_projection_success(self):
        box = Box(20.0, 40.0, 60.0, 120.0)
        pt = Point.from_box(1.0, box, "seg1", "review_1", geometry_revision="geom_rev_1")
        x_ground, y_ground = self.cal.project(pt)
        self.assertAlmostEqual(x_ground, 40.0)
        self.assertAlmostEqual(y_ground, 120.0)

    def test_invalid_calibration_suppresses_physical_metrics(self):
        points = [
            Point(0.0, 10.0, 20.0, "s", "rev"),
            Point(0.5, 12.0, 20.0, "s", "rev"),
        ]
        # Valid calibration produces estimated metrics
        res_valid = movement_summary(points, self.cal)
        self.assertEqual(res_valid["status"], "estimated")
        self.assertIsNotNone(res_valid["distance_m"])

        # Invalidated calibration suppresses metrics
        invalid_cal = self.cal.invalidate("camera moved / zoom changed")
        res_invalid = movement_summary(points, invalid_cal)
        self.assertEqual(res_invalid["status"], "unavailable")
        self.assertIsNone(res_invalid["distance_m"])
        self.assertIsNone(res_invalid["average_speed_mps"])
        self.assertIn("calibration invalid: camera moved / zoom changed", res_invalid["reason"])

    def test_project_on_invalid_calibration_raises_error(self):
        invalid_cal = self.cal.invalidate("pitch waterlogged and ground marks obscured")
        pt = Point(1.0, 50.0, 50.0, "s", "rev")
        with self.assertRaises(ValueError) as ctx:
            invalid_cal.project(pt)
        self.assertIn("calibration is invalid: pitch waterlogged", str(ctx.exception))

    def test_source_geometry_change_invalidates_calibration(self):
        # Original geometry was (1920, 1080)
        self.cal.check_geometry_compatibility(1920, 1080)
        with self.assertRaises(ValueError) as ctx:
            self.cal.check_geometry_compatibility(1280, 720)
        self.assertIn("source geometry changed", str(ctx.exception))

    def test_geometry_revision_change_invalidates_calibration(self):
        # Original geometry_revision was 'geom_rev_1'
        self.cal.check_geometry_compatibility(1920, 1080, geometry_revision="geom_rev_1")
        with self.assertRaises(ValueError) as ctx:
            self.cal.check_geometry_compatibility(1920, 1080, geometry_revision="geom_rev_2")
        self.assertIn("geometry revision changed", str(ctx.exception))

    def test_point_geometry_revision_mismatch_fails_project(self):
        mismatched_pt = Point(1.0, 50.0, 50.0, "s", "rev", geometry_revision="mismatched_geom")
        with self.assertRaises(ValueError) as ctx:
            self.cal.project(mismatched_pt)
        self.assertIn("point geometry revision 'mismatched_geom' does not match", str(ctx.exception))

    def test_timestamp_outside_validity_interval_rejected(self):
        before_pt = Point(-1.0, 50.0, 50.0, "s", "rev")
        after_pt = Point(25.0, 50.0, 50.0, "s", "rev")
        with self.assertRaises(ValueError) as ctx:
            self.cal.project(before_pt)
        self.assertIn("outside calibration validity interval", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.cal.project(after_pt)
        self.assertIn("outside calibration validity interval", str(ctx.exception))

    def test_point_outside_image_bounds_rejected(self):
        out_pt = Point(1.0, 2000.0, 500.0, "s", "rev")  # x=2000 > image_bounds max 1920
        with self.assertRaises(ValueError) as ctx:
            self.cal.project(out_pt)
        self.assertIn("outside calibrated image region", str(ctx.exception))

    def test_singular_calibration_matrix_rejected_at_init(self):
        # All zeros matrix is singular
        with self.assertRaises(ValueError) as ctx:
            Calibration((0,) * 9, "ref", 0.0, 10.0, (0, 0, 100, 100))
        self.assertIn("singular calibration", str(ctx.exception))

    def test_calibration_json_lists_coerced_to_tuples(self):
        cal = Calibration(
            matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            independent_check_ref="test_ref",
            valid_from_s=0.0,
            valid_until_s=10.0,
            image_bounds=[0.0, 0.0, 1920.0, 1080.0],
            source_geometry=[1920, 1080],
        )
        self.assertIsInstance(cal.matrix, tuple)
        self.assertIsInstance(cal.image_bounds, tuple)
        self.assertIsInstance(cal.source_geometry, tuple)
        cal.check_geometry_compatibility(1920, 1080)


if __name__ == "__main__":
    unittest.main()
