import json
import math
import unittest
from pathlib import Path
import tempfile

from touchline.analytics import (
    compute_spatial_heatmap,
    compute_track_analytics,
    generate_match_analytics,
)
from touchline.metrics import Calibration
from review_fixtures import review_entry


class AnalyticsTests(unittest.TestCase):
    def test_spatial_heatmap_empty_points(self):
        heatmap = compute_spatial_heatmap([], width=1920, height=1080, grid_cols=16, grid_rows=9)
        self.assertEqual(heatmap["grid_cols"], 16)
        self.assertEqual(heatmap["grid_rows"], 9)
        self.assertEqual(heatmap["total_points"], 0)
        self.assertEqual(heatmap["max_cell_count"], 0)
        self.assertEqual(len(heatmap["density_matrix"]), 9)
        self.assertEqual(len(heatmap["density_matrix"][0]), 16)
        self.assertTrue(all(cell == 0.0 for row in heatmap["density_matrix"] for cell in row))

    def test_spatial_heatmap_density_distribution(self):
        points = [(100, 100), (105, 110), (1900, 1000)]
        heatmap = compute_spatial_heatmap(points, width=1920, height=1080, grid_cols=16, grid_rows=9)
        self.assertEqual(heatmap["total_points"], 3)
        self.assertEqual(heatmap["max_cell_count"], 2)
        # Top-left cell should have normalized density 1.0
        self.assertEqual(heatmap["density_matrix"][0][0], 1.0)
        # Bottom-right cell should have normalized density 0.5 (1/2)
        self.assertEqual(heatmap["density_matrix"][8][15], 0.5)

    def test_track_analytics_uncalibrated_suppresses_physical_metrics(self):
        obs = [
            {
                "local_track_id": 1,
                "player_id": None,
                "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
                "detector_score": 0.9,
                "association_state": "tentative",
                "frame": {"index": 0, "timestamp_s": 0.0, "pts": 0, "width": 320, "height": 180},
            },
            {
                "local_track_id": 1,
                "player_id": None,
                "bbox_xyxy": [15.0, 20.0, 35.0, 60.0],
                "detector_score": 0.85,
                "association_state": "tentative",
                "frame": {"index": 6, "timestamp_s": 0.25, "pts": 6, "width": 320, "height": 180},
            },
        ]
        track = compute_track_analytics(
            track_id=1,
            observations=obs,
            frame_width=320,
            frame_height=180,
            calibration=None,
        )
        self.assertEqual(track["local_track_id"], 1)
        self.assertEqual(track["observation_count"], 2)
        self.assertAlmostEqual(track["visible_duration_s"], 0.25)
        self.assertEqual(track["tracking_quality"], "high")
        self.assertEqual(track["ambiguous_observations"], 0)

        # Physical metrics must be suppressed and marked uncalibrated
        m = track["movement"]
        self.assertEqual(m["status"], "uncalibrated")
        self.assertFalse(m["calibrated"])
        self.assertIsNone(m["distance_m"])
        self.assertIsNone(m["average_speed_mps"])
        self.assertIn("uncalibrated", m["reason"])
        self.assertAlmostEqual(m["image_pixel_distance_px"], 5.0)

    def test_track_analytics_calibrated_produces_physical_metrics(self):
        obs = [
            {
                "local_track_id": 2,
                "player_id": None,
                "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
                "detector_score": 0.9,
                "association_state": "tentative",
                "frame": {"index": 0, "timestamp_s": 0.0, "pts": 0, "width": 320, "height": 180},
            },
            {
                "local_track_id": 2,
                "player_id": None,
                "bbox_xyxy": [15.0, 20.0, 35.0, 60.0],
                "detector_score": 0.88,
                "association_state": "tentative",
                "frame": {"index": 12, "timestamp_s": 0.5, "pts": 12, "width": 320, "height": 180},
            },
        ]
        cal = Calibration(
            matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            independent_check_ref="test_bench_ground_marks",
            valid_from_s=0.0,
            valid_until_s=10.0,
            image_bounds=(0.0, 0.0, 320.0, 180.0),
        )
        track = compute_track_analytics(
            track_id=2,
            observations=obs,
            frame_width=320,
            frame_height=180,
            calibration=cal,
            run_id="fixture-run",
            association=review_entry("fixture-run", 2),
        )
        m = track["movement"]
        self.assertEqual(m["status"], "estimated")
        self.assertTrue(m["calibrated"])
        self.assertIsNotNone(m["distance_m"])
        self.assertAlmostEqual(m["distance_m"], 5.0)
        self.assertAlmostEqual(m["average_speed_mps"], 10.0)

    def test_track_analytics_ambiguity_quality_labels(self):
        obs = [
            {
                "local_track_id": 3,
                "player_id": None,
                "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
                "detector_score": 0.9,
                "association_state": "ambiguous",
                "frame": {"index": 0, "timestamp_s": 0.0, "pts": 0, "width": 320, "height": 180},
            },
            {
                "local_track_id": 3,
                "player_id": None,
                "bbox_xyxy": [12.0, 20.0, 32.0, 60.0],
                "detector_score": 0.85,
                "association_state": "ambiguous",
                "frame": {"index": 6, "timestamp_s": 0.25, "pts": 6, "width": 320, "height": 180},
            },
        ]
        track = compute_track_analytics(
            track_id=3,
            observations=obs,
            frame_width=320,
            frame_height=180,
        )
        self.assertEqual(track["ambiguous_observations"], 2)
        self.assertEqual(track["tracking_quality"], "contested")

    def test_generate_match_analytics_empty_observations(self):
        manifest = {"run_id": "empty-run", "synthetic": True, "source": {"id": "test"}}
        analytics = generate_match_analytics(manifest, [])
        self.assertEqual(analytics["schema_version"], "touchline.analytics.v1")
        self.assertEqual(analytics["run_id"], "empty-run")
        self.assertTrue(analytics["synthetic"])
        self.assertEqual(analytics["quality_indicators"]["total_tracks"], 0)
        self.assertEqual(analytics["player_roster"], [])


if __name__ == "__main__":
    unittest.main()
