import json
import unittest
from pathlib import Path
import tempfile

from touchline.data.release_builder import (
    compute_sample_fingerprint,
    ResumableReleaseBuilder,
    build_release_v2,
)
from touchline.data.manifest import validate_manifest

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "ETL image tests require Pillow")
class ResumableReleaseBuilderTests(unittest.TestCase):
    def make_source(self, tmp, count=3):
        root = Path(tmp)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        groups = {}
        for i in range(count):
            name = f"{i}.png"
            Image.new("RGB", (32, 32), (i * 70, 20, 30)).save(root / "images" / name)
            images.append({"id": i, "file_name": name, "width": 32, "height": 32})
            annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": [8, 8, 16, 16]})
            groups[f"input/{name}"] = f"capture-{i}"

        (root / "coco.json").write_text(json.dumps({
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "player"}],
        }))
        item = {
            "id": "input",
            "format": "coco_boxes",
            "images_root": "images",
            "annotations": "coco.json",
        }
        config = {
            "release_id": "synthetic-resumable-etl",
            "dataset_version": "1",
            "reviewer_id": "test_reviewer",
            "annotations_reviewed": True,
            "person_labels_complete": True,
            "training_eligible": False,
            "fixture_only": True,
            "class_aliases": {"player": 1},
            "capture_groups": groups,
            "inputs": [item],
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        return root, config_path

    def test_fingerprint_deterministic_and_sensitive_to_changes(self):
        fp1 = compute_sample_fingerprint("hashA", "input/0.png", [{"box": [0, 0, 10, 10]}], "10.0.0")
        fp2 = compute_sample_fingerprint("hashA", "input/0.png", [{"box": [0, 0, 10, 10]}], "10.0.0")
        self.assertEqual(fp1, fp2)

        # Change image hash -> different fingerprint
        fp3 = compute_sample_fingerprint("hashB", "input/0.png", [{"box": [0, 0, 10, 10]}], "10.0.0")
        self.assertNotEqual(fp1, fp3)

        # Change label -> different fingerprint
        fp4 = compute_sample_fingerprint("hashA", "input/0.png", [{"box": [0, 0, 12, 10]}], "10.0.0")
        self.assertNotEqual(fp1, fp4)

    def test_build_release_fresh_and_resumed_reuses_cached_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, config_path = self.make_source(tmp, count=3)
            out_dir = root / "release"

            # 1. First run: should encode all 3 fresh
            builder1 = ResumableReleaseBuilder(config_path, out_dir, resume=True)
            rel1 = builder1.build()
            self.assertEqual(len(rel1["samples"]), 3)
            self.assertEqual(validate_manifest(rel1), [])

            report1 = json.loads((out_dir / "etl-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report1["status"], "complete")
            self.assertEqual(report1["images_encoded_fresh"], 3)
            self.assertEqual(report1["images_reused_from_cache"], 0)

            # 2. Second run on same output with resume=True: should reuse all 3 from cache
            builder2 = ResumableReleaseBuilder(config_path, out_dir, resume=True)
            rel2 = builder2.build()
            self.assertEqual(len(rel2["samples"]), 3)

            report2 = json.loads((out_dir / "etl-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report2["status"], "complete")
            self.assertEqual(report2["images_encoded_fresh"], 0)
            self.assertEqual(report2["images_reused_from_cache"], 3)
            self.assertEqual(report1["release_sha256"], report2["release_sha256"])

    def test_build_release_detects_corrupt_or_modified_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, config_path = self.make_source(tmp, count=3)
            out_dir = root / "release"

            builder1 = ResumableReleaseBuilder(config_path, out_dir, resume=True)
            builder1.build()

            # Corrupt one cached image file on disk
            sample_file = next((out_dir / "images").glob("*.png"))
            sample_file.write_bytes(b"corrupted_png_bytes")

            # Next run should detect hash mismatch and re-encode fresh
            builder2 = ResumableReleaseBuilder(config_path, out_dir, resume=True)
            rel2 = builder2.build()
            self.assertEqual(len(rel2["samples"]), 3)
            self.assertEqual(validate_manifest(rel2), [])

            report2 = json.loads((out_dir / "etl-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report2["images_encoded_fresh"], 1)
            self.assertEqual(report2["images_reused_from_cache"], 2)


if __name__ == "__main__":
    unittest.main()
