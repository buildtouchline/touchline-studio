import json
from pathlib import Path
import tempfile
import unittest
from touchline.cli import main
from touchline.contracts import Box, Detection, Frame
from touchline.pipeline import run_pipeline


class CLIAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run_output"

        test_frames = [
            Frame("cli_source", 0, 0, 1, 24, 0.0, 320, 180),
            Frame("cli_source", 1, 1, 1, 24, 1 / 24, 320, 180),
            Frame("cli_source", 2, 2, 1, 24, 2 / 24, 320, 180),
        ]

        class DummyDetector:
            def provenance(self):
                return {"adapter": "dummy"}
            def detect(self, frame):
                if frame.index == 1:
                    return [Detection(Box(10, 10, 50, 50), 0.9)]
                return []

        class DummySampler:
            def select(self, index, timestamp_s):
                return index in (1, 2)

        run_pipeline(test_frames, DummyDetector(), DummySampler(), self.run_dir, {"id": "cli_source"})

        self.gt_file = self.root / "gt.json"
        self.gt_data = [{
            "label_id": "p1",
            "source_id": "cli_source",
            "pts": 1,
            "timestamp_s": 1 / 24,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        self.gt_file.write_text(json.dumps(self.gt_data), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_audit_run_success(self):
        out_file = self.root / "sub" / "ledger.json"
        code = main(["audit-run", str(self.run_dir), str(self.gt_file), "--output", str(out_file)])
        self.assertEqual(code, 0)
        self.assertTrue(out_file.is_file())
        ledger = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(ledger["schema_version"], "touchline.audit_ledger.v1")
        self.assertEqual(ledger["summary"]["detected_and_persisted"], 1)

    def test_audit_run_missing_run_dir(self):
        out_file = self.root / "ledger.json"
        with self.assertRaises(SystemExit) as ctx:
            main(["audit-run", str(self.root / "nonexistent"), str(self.gt_file), "--output", str(out_file)])
        self.assertEqual(ctx.exception.code, 2)

    def test_audit_run_missing_ground_truth(self):
        out_file = self.root / "ledger.json"
        with self.assertRaises(SystemExit) as ctx:
            main(["audit-run", str(self.run_dir), str(self.root / "missing_gt.json"), "--output", str(out_file)])
        self.assertEqual(ctx.exception.code, 2)

    def test_audit_run_same_output_and_input_fails(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["audit-run", str(self.run_dir), str(self.gt_file), "--output", str(self.gt_file)])
        self.assertEqual(ctx.exception.code, 2)

    def test_audit_run_invalid_gt_json(self):
        bad_gt = self.root / "bad.json"
        bad_gt.write_text("{broken", encoding="utf-8")
        out_file = self.root / "ledger.json"
        with self.assertRaises(SystemExit) as ctx:
            main(["audit-run", str(self.run_dir), str(bad_gt), "--output", str(out_file)])
        self.assertEqual(ctx.exception.code, 2)

    def test_audit_run_invalid_threshold(self):
        out_file = self.root / "ledger.json"
        with self.assertRaises(SystemExit) as ctx:
            main(["audit-run", str(self.run_dir), str(self.gt_file), "--output", str(out_file), "--iou-threshold", "1.5"])
        self.assertEqual(ctx.exception.code, 2)

    def test_audit_run_with_failure_cases_template_data(self):
        template_file = Path(__file__).parent.parent / "examples" / "failure-cases.template.json"
        template_data = json.loads(template_file.read_text(encoding="utf-8"))

        # Adapt placeholders to matching run source without frame collisions
        adapted_gt = []
        for i, item in enumerate(template_data):
            entry = dict(item)
            entry["source_id"] = "cli_source"
            if entry.get("bbox_xyxy") is None:
                # Map negative control to frame 2 (clean)
                entry["pts"] = 2
                entry["timestamp_s"] = 2 / 24
            else:
                entry["pts"] = 1
                entry["timestamp_s"] = 1 / 24
            adapted_gt.append(entry)

        gt_path = self.root / "adapted_template_gt.json"
        gt_path.write_text(json.dumps(adapted_gt), encoding="utf-8")
        out_file = self.root / "template_ledger.json"

        code = main(["audit-run", str(self.run_dir), str(gt_path), "--output", str(out_file)])
        self.assertEqual(code, 0)
        self.assertTrue(out_file.is_file())

        ledger = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(ledger["schema_version"], "touchline.audit_ledger.v1")
        self.assertEqual(ledger["summary"]["total_ground_truth"], len(template_data))
        self.assertEqual(ledger["summary"]["empty_controls_clean"], 1)

    def test_audit_run_empty_control_cli(self):
        # Test empty background control producing clean and false positive results via CLI
        ctrl_gt = [
            {"source_id": "cli_source", "pts": 2, "timestamp_s": 2 / 24, "bbox_xyxy": None, "target_category": "empty_background_control"},
            {"source_id": "cli_source", "pts": 1, "timestamp_s": 1 / 24, "bbox_xyxy": None, "target_category": "empty_background_control"},
        ]
        gt_path = self.root / "ctrl_gt.json"
        gt_path.write_text(json.dumps(ctrl_gt), encoding="utf-8")
        out_file = self.root / "ctrl_ledger.json"

        code = main(["audit-run", str(self.run_dir), str(gt_path), "--output", str(out_file)])
        self.assertEqual(code, 0)
        self.assertTrue(out_file.is_file())

        ledger = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(ledger["summary"]["empty_controls_clean"], 1)
        self.assertEqual(ledger["summary"]["false_positive_frames"], 1)
        self.assertEqual(ledger["summary"]["false_positive_observations"], 1)


if __name__ == "__main__":
    unittest.main()
