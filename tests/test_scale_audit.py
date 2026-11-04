import json
import unittest
from pathlib import Path
import tempfile

from touchline.training.scale_audit import (
    compute_resize_scale,
    compute_effective_box,
    audit_scale,
    audit_scale_file,
)


class ScaleAuditTests(unittest.TestCase):
    def test_compute_resize_scale_landscape(self):
        # 1920x1080 (min_dim=1080, max_dim=1920)
        # min_size/min_dim = 320/1080 = 0.296296
        # max_size/max_dim = 640/1920 = 0.333333
        # scale = min(0.296296, 0.333333) = 320/1080
        scale = compute_resize_scale(1920, 1080, min_size=320, max_size=640)
        self.assertAlmostEqual(scale, 320.0 / 1080.0, places=5)

    def test_compute_resize_scale_portrait(self):
        # 1080x1920 (min_dim=1080, max_dim=1920)
        scale = compute_resize_scale(1080, 1920, min_size=320, max_size=640)
        self.assertAlmostEqual(scale, 320.0 / 1080.0, places=5)

    def test_compute_resize_scale_max_size_cap(self):
        # Ultra-wide image: 3000x500 (min_dim=500, max_dim=3000)
        # min_size/min_dim = 320/500 = 0.64
        # max_size/max_dim = 640/3000 = 0.213333
        # scale should be capped by max_size (640 / 3000)
        scale = compute_resize_scale(3000, 500, min_size=320, max_size=640)
        self.assertAlmostEqual(scale, 640.0 / 3000.0, places=5)

    def test_compute_resize_scale_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_resize_scale(0, 1080)
        with self.assertRaises(ValueError):
            compute_resize_scale(1920, -10)
        with self.assertRaises(ValueError):
            compute_resize_scale(1920, 1080, min_size=800, max_size=600)

    def test_compute_effective_box(self):
        # Box of size 50x100 at scale 0.1 -> effective 5x10, min_dim 5, area 50
        eff_w, eff_h, min_dim, area = compute_effective_box([10, 20, 60, 120], scale=0.1)
        self.assertAlmostEqual(eff_w, 5.0)
        self.assertAlmostEqual(eff_h, 10.0)
        self.assertAlmostEqual(min_dim, 5.0)
        self.assertAlmostEqual(area, 50.0)

    def test_compute_effective_box_invalid(self):
        with self.assertRaises(ValueError):
            compute_effective_box([10, 20, 5, 120], scale=0.1)  # inverted x
        with self.assertRaises(ValueError):
            compute_effective_box([10, 20, 60, 120], scale=0.0)  # zero scale

    def test_audit_scale_threshold_boundaries_and_grouping(self):
        # Frame: 1920x1080 -> scale = 320/1080 ~= 0.296296
        # Box 1: w=20, h=40 -> eff_w = 20 * (320/1080) = 5.925px (< 8px, < 16px)
        # Box 2: w=40, h=80 -> eff_w = 40 * (320/1080) = 11.85px (>= 8px, < 16px)
        # Box 3: w=80, h=160 -> eff_w = 80 * (320/1080) = 23.70px (>= 16px)
        manifest = {
            "release_id": "test-release-001",
            "dataset_version": "0.1.0",
            "samples": [
                {
                    "sample_id": "s1",
                    "split": "train",
                    "capture_group_id": "game-01",
                    "width": 1920,
                    "height": 1080,
                    "objects": [
                        {"class_id": 1, "bbox_xyxy": [100, 100, 120, 140]},  # Box 1 (<8)
                        {"class_id": 1, "bbox_xyxy": [200, 200, 240, 280]},  # Box 2 (<16, >=8)
                    ],
                },
                {
                    "sample_id": "s2",
                    "split": "val",
                    "capture_group_id": "game-02",
                    "width": 1920,
                    "height": 1080,
                    "objects": [
                        {"class_id": 1, "bbox_xyxy": [300, 300, 380, 460]},  # Box 3 (>=16)
                        {"class_id": 0, "bbox_xyxy": [10, 10, 20, 20]},      # Non-person (ignored)
                    ],
                },
            ],
        }

        report = audit_scale(manifest, min_size=320, max_size=640)
        self.assertEqual(report["schema_version"], "touchline.scale_audit.v1")
        self.assertEqual(report["image_count"], 2)

        # Overall summary
        overall = report["overall"]
        self.assertEqual(overall["person_box_count"], 3)
        self.assertEqual(overall["below_8px_count"], 1)
        self.assertEqual(overall["below_16px_count"], 2)
        self.assertAlmostEqual(overall["below_8px_ratio"], 1 / 3, places=3)
        self.assertAlmostEqual(overall["below_16px_ratio"], 2 / 3, places=3)

        # By split
        train_stats = report["by_split"]["train"]
        self.assertEqual(train_stats["person_box_count"], 2)
        self.assertEqual(train_stats["below_8px_count"], 1)
        self.assertEqual(train_stats["below_16px_count"], 2)

        val_stats = report["by_split"]["val"]
        self.assertEqual(val_stats["person_box_count"], 1)
        self.assertEqual(val_stats["below_8px_count"], 0)
        self.assertEqual(val_stats["below_16px_count"], 0)

        # By capture group
        self.assertIn("game-01", report["by_capture_group"])
        self.assertIn("game-02", report["by_capture_group"])
        self.assertEqual(report["by_capture_group"]["game-01"]["below_8px_count"], 1)
        self.assertEqual(report["by_capture_group"]["game-02"]["below_8px_count"], 0)

    def test_audit_scale_file(self):
        manifest = {
            "release_id": "file-release",
            "dataset_version": "1.0",
            "samples": [
                {
                    "sample_id": "f1",
                    "split": "train",
                    "capture_group_id": "g1",
                    "width": 1280,
                    "height": 720,
                    "objects": [{"class_id": 1, "bbox_xyxy": [10, 10, 20, 30]}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(json.dumps(manifest), encoding="utf-8")
            res = audit_scale_file(p, min_size=320, max_size=640)
            self.assertEqual(res["overall"]["person_box_count"], 1)


if __name__ == "__main__":
    unittest.main()
