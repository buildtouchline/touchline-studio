import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from touchline.adapters.handoff import EVALUATION_METRICS, assess_handoff, validate_candidate_metadata


class ModelHandoffTests(unittest.TestCase):
    def setUp(self):
        self.bytes = b"synthetic checkpoint fixture"
        self.checkpoint_hash = hashlib.sha256(self.bytes).hexdigest()
        self.manifest_hash = "a" * 64
        self.recipe = {
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "num_classes": 2,
            "class_map": {"0": "background", "1": "person"},
            "backbone_norm": "frozen_batchnorm2d",
            "recipe": "starter-heads-fp32-batch1-v1",
            "promotion_state": "unvalidated_candidate",
            "release_id": "synthetic-release",
            "dataset_version": "1",
            "dataset_manifest_sha256": self.manifest_hash,
            "initial_weights_sha256": "b" * 64,
            "torch": "2.11.0+cu130",
            "torchvision": "0.26.0+cu130",
        }
        self.result = {"checkpoint_sha256": self.checkpoint_hash, "promotion_state": "unvalidated_candidate", "steps": 2, "status": "completed"}
        base = {"manifest_sha256": self.manifest_hash, "split": "val", "sample_ids": ["one", "two"], "images": 2,
                "subset": False, "model": {"adapter": "torchvision-detector-v1", "architecture": self.recipe["architecture"],
                "checkpoint": "b" * 64, "threshold": 0.0},
                "metrics": {metric: 0.1 for metric in EVALUATION_METRICS}}
        self.baseline = base
        self.candidate = copy.deepcopy(base)
        self.candidate["model"]["checkpoint"] = self.checkpoint_hash

    def test_compatible_metadata_and_full_validation_allow_runtime_test(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            checkpoint.write_bytes(self.bytes)
            report = assess_handoff(self.recipe, self.result, self.baseline, self.candidate, checkpoint_path=checkpoint)
        self.assertEqual(report, {"ready_for_runtime_test": True, "issues": []})

    def test_missing_checkpoint_and_subset_fail_closed(self):
        self.candidate["subset"] = True
        report = assess_handoff(self.recipe, self.result, self.baseline, self.candidate)
        self.assertFalse(report["ready_for_runtime_test"])
        self.assertIn("candidate is not full validation", report["issues"])
        self.assertIn("checkpoint bytes have not been hashed by this consumer", report["issues"])

    def test_reviewed_resize_recipe_requires_transform_and_approval_hash(self):
        metadata={**self.recipe,'recipe':'heads-fp32-batch1-reviewed-resize-v2','training_transform':{'min_size':[720],'max_size':1280},'approved_recipe_sha256':'c'*64}
        validate_candidate_metadata(metadata,runtime_torch='2.11.0',runtime_torchvision='0.26.0')
        for changes in ({'approved_recipe_sha256':None},{'training_transform':{'min_size':[1080],'max_size':4096}}):
            with self.assertRaises(ValueError):validate_candidate_metadata({**metadata,**changes},runtime_torch='2.11.0',runtime_torchvision='0.26.0')

    def test_class_head_and_runtime_mismatch_rejected(self):
        for key, value in (("num_classes", True), ("class_map", {"1": "person"}), ("backbone_norm", "batchnorm2d"),
                           ("dataset_manifest_sha256", "bad"), ("recipe", "unknown")):
            with self.subTest(key=key):
                invalid = dict(self.recipe, **{key: value})
                with self.assertRaises(ValueError):
                    validate_candidate_metadata(invalid, runtime_torch="2.11.1", runtime_torchvision="0.26.2")
        with self.assertRaisesRegex(ValueError, "major/minor"):
            validate_candidate_metadata(self.recipe, runtime_torch="2.12.0", runtime_torchvision="0.26.0")

    def test_checkpoint_hash_and_evaluation_lineage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            checkpoint.write_bytes(b"wrong bytes")
            self.candidate["sample_ids"] = ["one", "three"]
            self.candidate["manifest_sha256"] = "c" * 64
            report = assess_handoff(self.recipe, self.result, self.baseline, self.candidate, checkpoint_path=checkpoint)
        self.assertFalse(report["ready_for_runtime_test"])
        self.assertIn("checkpoint bytes differ from training result", report["issues"])
        self.assertIn("baseline and candidate use different validation IDs", report["issues"])
        self.assertIn("candidate uses a different dataset manifest", report["issues"])

    def test_invalid_sample_shape_is_reported_not_crashed(self):
        self.candidate["sample_ids"] = 3
        report = assess_handoff(self.recipe, self.result, self.baseline, self.candidate)
        self.assertFalse(report["ready_for_runtime_test"])
        self.assertIn("candidate sample IDs are missing or duplicated", report["issues"])

    def test_invalid_threshold_and_adapter_fail_closed(self):
        self.candidate["model"]["threshold"] = float("nan")
        self.candidate["model"]["adapter"] = "other"
        report = assess_handoff(self.recipe, self.result, self.baseline, self.candidate)
        self.assertIn("candidate score threshold is invalid", report["issues"])
        self.assertIn("candidate adapter is incompatible", report["issues"])

    def test_missing_or_invalid_evaluation_metrics_fail_closed(self):
        del self.baseline["metrics"]
        self.candidate["metrics"]["ap50"] = float("nan")
        report = assess_handoff(self.recipe, self.result, self.baseline, self.candidate)
        self.assertFalse(report["ready_for_runtime_test"])
        self.assertIn("baseline evaluation metrics are missing", report["issues"])
        self.assertIn("candidate ap50 is invalid", report["issues"])


if __name__ == "__main__":
    unittest.main()
