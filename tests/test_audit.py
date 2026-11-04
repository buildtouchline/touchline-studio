import json
import math
from pathlib import Path
import tempfile
import unittest
from touchline.audit import audit_ground_truth, get_pixel_area_bin
from touchline.contracts import Box, Detection, Frame
from touchline.pipeline import run_pipeline


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run_output"

        # Create a small 5-frame deterministic pipeline run
        test_frames = [
            Frame("test_source_1", i, i * 100, 1, 1000, i * 0.1, 320, 180)
            for i in range(5)
        ]

        class MockDetector:
            def provenance(self):
                return {"adapter": "mock-detector", "threshold": 0.5}

            def detect(self, frame):
                # Frame 1: emits a medium detection Box(10, 10, 50, 50) area=1600
                if frame.index == 1:
                    return [Detection(Box(10.0, 10.0, 50.0, 50.0), 0.9)]
                # Frame 3: sampled, but detector returns empty list (miss)
                return []

        class MockSampler:
            def select(self, index, timestamp_s):
                # Select frames 1 and 3; skip frames 0, 2, 4
                return index in (1, 3)

        self.manifest = run_pipeline(
            test_frames,
            MockDetector(),
            MockSampler(),
            self.run_dir,
            {"id": "test_source_1", "path": "/path/to/test.mp4"},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_pixel_area_size_bins(self):
        self.assertEqual(get_pixel_area_bin(20.0, 20.0), "small")
        self.assertEqual(get_pixel_area_bin(50.0, 50.0), "medium")
        self.assertEqual(get_pixel_area_bin(100.0, 100.0), "large")

    def test_matched_persisted_observation(self):
        # Frame 1 has Box(10, 10, 50, 50)
        gt = [{
            "label_id": "gt_player_1",
            "source_id": "test_source_1",
            "pts": 100,
            "timestamp_s": 0.1,
            "bbox_xyxy": [12.0, 10.0, 52.0, 50.0],
        }]
        report = audit_ground_truth(self.run_dir, gt, iou_threshold=0.5)
        self.assertEqual(report["summary"]["total_ground_truth"], 1)
        self.assertEqual(report["summary"]["detected_and_persisted"], 1)

        entry = report["ledger"][0]
        self.assertEqual(entry["diagnosis"], "detected_and_persisted")
        self.assertEqual(entry["size_bin"], "medium")
        self.assertEqual(entry["evidence"]["decode_status"], "decoded")
        self.assertIsNotNone(entry["evidence"]["matched_observation"])
        self.assertGreaterEqual(entry["evidence"]["matched_observation"]["iou"], 0.5)

        # Tracking status must not claim tracking succeeded across frames
        self.assertEqual(entry["stage_breakdown"]["detection"], "detected")
        self.assertEqual(entry["stage_breakdown"]["detection_persistence"], "persisted_in_observations")
        self.assertEqual(entry["stage_breakdown"]["tracking_status"], "local_track_assigned_tentative")
        self.assertIn("notes", entry)
        self.assertIn("single-frame matching does not establish cross-frame identity continuity", entry["notes"])

    def test_explicitly_unsampled_frame(self):
        # Frame 0 is unsampled_by_sampler
        gt = [{
            "label_id": "gt_player_0",
            "source_id": "test_source_1",
            "pts": 0,
            "timestamp_s": 0.0,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        report = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(report["summary"]["sampling_skip"], 1)
        entry = report["ledger"][0]
        self.assertEqual(entry["diagnosis"], "sampling_skip")
        self.assertEqual(entry["evidence"]["decode_status"], "decoded")
        self.assertEqual(entry["evidence"]["sampling_state"], "unsampled_by_sampler")
        self.assertEqual(entry["stage_breakdown"]["sampling"], "skipped_by_sampler")
        self.assertEqual(entry["stage_breakdown"]["detection"], "not_run_by_design")

    def test_sampled_frame_with_no_matching_observation(self):
        # Frame 3 is sampled, but detector returned no detections
        gt = [{
            "label_id": "gt_player_3",
            "source_id": "test_source_1",
            "pts": 300,
            "timestamp_s": 0.3,
            "bbox_xyxy": [5.0, 5.0, 25.0, 25.0],
        }]
        report = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(report["summary"]["unobserved_in_pipeline"], 1)
        entry = report["ledger"][0]
        self.assertEqual(entry["diagnosis"], "unobserved_in_pipeline")
        self.assertEqual(entry["evidence"]["decode_status"], "decoded")
        self.assertEqual(entry["evidence"]["sampling_state"], "no_detection_not_inactivity")
        # Tracking and detector internal stages must remain explicitly unknown
        self.assertEqual(entry["stage_breakdown"]["detector_proposals"], "unknown_insufficient_evidence")
        self.assertEqual(entry["stage_breakdown"]["detector_filters"], "unknown_insufficient_evidence")
        self.assertEqual(entry["stage_breakdown"]["tracking"], "unknown_insufficient_evidence")
        self.assertIn("limitation", entry)

    def test_frame_absent_from_coverage_ledger_is_unrecorded_not_decode_failure(self):
        # Frame 99 was never recorded in coverage ledger
        gt = [{
            "label_id": "gt_missing_frame",
            "source_id": "test_source_1",
            "pts": 9900,
            "timestamp_s": 9.9,
            "bbox_xyxy": [10.0, 10.0, 110.0, 110.0],
        }]
        report = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(report["summary"]["unrecorded_frame"], 1)

        entry = report["ledger"][0]
        self.assertEqual(entry["diagnosis"], "unrecorded_frame")
        self.assertFalse(entry["evidence"]["coverage_record_found"])
        # Decode status must be unknown, not assumed false or failed
        self.assertEqual(entry["evidence"]["decode_status"], "unknown_absent_from_coverage_ledger")
        self.assertEqual(entry["stage_breakdown"]["decode"], "unknown_absent_from_coverage_ledger")
        self.assertEqual(entry["stage_breakdown"]["sampling"], "unknown")
        self.assertEqual(entry["stage_breakdown"]["detection"], "unknown")
        self.assertEqual(entry["stage_breakdown"]["tracking"], "unknown")

    def test_timestamp_fallback_matching_without_pts(self):
        # Ground truth without 'pts', matching via timestamp_s
        gt = [{
            "label_id": "gt_time_match",
            "source_id": "test_source_1",
            "timestamp_s": 0.1,  # matches frame 1
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        report = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(report["summary"]["detected_and_persisted"], 1)
        self.assertEqual(report["ledger"][0]["frame"]["pts"], 100)

    def test_conflicting_pts_and_timestamp_raises_error(self):
        # pts=100 is at 0.1s, but timestamp_s is set to 0.9s
        gt = [{
            "source_id": "test_source_1",
            "pts": 100,
            "timestamp_s": 0.9,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, gt)
        self.assertIn("conflicting pts", str(ctx.exception))

    def test_ambiguous_timestamp_match_raises_error(self):
        # Create a run with duplicate timestamps in coverage
        cov_path = self.run_dir / "coverage.jsonl"
        lines = cov_path.read_text(encoding="utf-8").splitlines()
        # Add another record with the same timestamp 0.1s but different pts
        extra_record = json.dumps({"frame": {"source_id": "test_source_1", "pts": 101, "timestamp_s": 0.1001, "index": 99}, "state": "unsampled_by_sampler"})
        cov_path.write_text("\n".join(lines + [extra_record]) + "\n", encoding="utf-8")

        # Ground truth without pts will match multiple candidates within 1ms tolerance
        gt = [{
            "source_id": "test_source_1",
            "timestamp_s": 0.1,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, gt)
        self.assertIn("ambiguous timestamp match", str(ctx.exception))

    def test_duplicate_pts_in_coverage_raises_error(self):
        cov_path = self.run_dir / "coverage.jsonl"
        lines = cov_path.read_text(encoding="utf-8").splitlines()
        # Duplicate line for pts 100
        cov_path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
        gt = [{
            "source_id": "test_source_1",
            "pts": 100,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, gt)
        self.assertIn("duplicate coverage record", str(ctx.exception))

    def test_source_mismatch_raises_error(self):
        gt = [{
            "source_id": "different_source_video",
            "pts": 100,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, gt)
        self.assertIn("does not match run source_id", str(ctx.exception))

    def test_invalid_box_and_threshold_validation(self):
        # Inverted box coordinates
        invalid_boxes = [
            [50.0, 50.0, 10.0, 10.0],
            [float("nan"), 10.0, 50.0, 50.0],
            [True, 10.0, 50.0, 50.0],
            [10.0, 10.0, 10.0, 50.0],  # zero width
            [10.0, 10.0, 50.0, 10.0],  # zero height
            [-5.0, 10.0, 50.0, 50.0],  # negative coordinate
        ]
        for box in invalid_boxes:
            with self.assertRaises(ValueError):
                audit_ground_truth(self.run_dir, [{
                    "source_id": "test_source_1",
                    "pts": 100,
                    "bbox_xyxy": box,
                }])

        valid_gt = [{
            "source_id": "test_source_1",
            "pts": 100,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }]
        # Invalid iou_threshold
        for bad_thresh in [0.0, -0.1, 1.1, float("nan"), True, "0.5"]:
            with self.assertRaises(ValueError):
                audit_ground_truth(self.run_dir, valid_gt, iou_threshold=bad_thresh)

    def test_malformed_json_and_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                audit_ground_truth(empty_dir, [])

        # Malformed manifest
        with tempfile.TemporaryDirectory() as bad_dir:
            p = Path(bad_dir)
            (p / "manifest.json").write_text("{broken json", encoding="utf-8")
            (p / "coverage.jsonl").write_text("", encoding="utf-8")
            (p / "observations.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_ground_truth(p, [])

    def test_failure_cases_template_compatibility(self):
        template_path = Path(__file__).parent.parent / "examples" / "failure-cases.template.json"
        self.assertTrue(template_path.is_file(), f"Template file not found at {template_path}")

        raw_content = template_path.read_text(encoding="utf-8")
        template_data = json.loads(raw_content)

        self.assertIsInstance(template_data, list)
        self.assertGreater(len(template_data), 0)

        # Verify placeholder safety, core audit schema, and metadata separation
        for i, item in enumerate(template_data):
            self.assertIsInstance(item, dict)
            self.assertIn("source_id", item)
            self.assertTrue(item["source_id"].startswith("REPLACE-"), f"Item {i} source_id must be a clear placeholder")
            self.assertIn("label_id", item)
            self.assertTrue(item["label_id"].startswith("PLACEHOLDER-"), f"Item {i} label_id must be a clear placeholder")

            # Must have pts or timestamp_s
            self.assertTrue("pts" in item or "timestamp_s" in item)
            if "pts" in item and item["pts"] is not None:
                self.assertIsInstance(item["pts"], int)
                self.assertGreaterEqual(item["pts"], 0)
            if "timestamp_s" in item and item["timestamp_s"] is not None:
                self.assertIsInstance(item["timestamp_s"], (int, float))
                self.assertGreaterEqual(item["timestamp_s"], 0)

            # bbox_xyxy is null for empty_background_control or 4 numbers for positive boxes
            self.assertIn("bbox_xyxy", item)
            box_coords = item["bbox_xyxy"]
            if box_coords is None:
                target_cat = item.get("target_category") or item.get("annotation_metadata", {}).get("target_category")
                self.assertEqual(target_cat, "empty_background_control")
            else:
                self.assertEqual(len(box_coords), 4)
                box = Box(*box_coords)
                self.assertGreater(box.x2, box.x1)
                self.assertGreater(box.y2, box.y1)

            # Metadata separation and coordinate reminder
            self.assertIn("annotation_metadata", item)
            meta = item["annotation_metadata"]
            self.assertIsInstance(meta, dict)
            self.assertIn("coordinate_reminder", meta)
            self.assertIn("not physical distance", meta["coordinate_reminder"])
            self.assertIn("target_category", meta)
            self.assertIn(meta["target_category"], ("miss_case", "successful_control", "empty_background_control"))

        # Verify audit compatibility when populated with actual run source_id and distinct frames
        run_source_id = self.manifest["source"]["id"]
        adapted_cases = []
        for i, item in enumerate(template_data):
            adapted_item = dict(item)
            adapted_item["source_id"] = run_source_id
            # Assign distinct frames: frame 0, 1, 2, 3, 4
            frame_idx = i % 5
            adapted_item["pts"] = frame_idx * 100
            adapted_item["timestamp_s"] = frame_idx * 0.1
            adapted_cases.append(adapted_item)

        ledger = audit_ground_truth(self.run_dir, adapted_cases, iou_threshold=0.5)
        self.assertEqual(ledger["schema_version"], "touchline.audit_ledger.v1")
        self.assertEqual(ledger["summary"]["total_ground_truth"], len(template_data))
        self.assertEqual(len(ledger["ledger"]), len(template_data))

    def test_empty_background_control_clean(self):
        # Frame 3 was sampled and had 0 observations in setUp
        gt = [{
            "label_id": "clean_empty_ctrl",
            "source_id": "test_source_1",
            "pts": 300,
            "timestamp_s": 0.3,
            "bbox_xyxy": None,
            "target_category": "empty_background_control",
        }]
        ledger = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(ledger["summary"]["empty_controls_clean"], 1)
        self.assertEqual(ledger["summary"]["false_positive_frames"], 0)
        self.assertEqual(ledger["summary"]["false_positive_observations"], 0)
        self.assertEqual(ledger["summary"]["empty_controls"]["clean"], 1)
        self.assertEqual(ledger["summary"]["empty_controls"]["evaluated"], 1)

        entry = ledger["ledger"][0]
        self.assertEqual(entry["diagnosis"], "empty_control_verified_clean")
        self.assertIsNone(entry["gt_bbox_xyxy"])
        self.assertEqual(entry["evidence"]["observations_count"], 0)
        self.assertEqual(entry["stage_breakdown"]["detection"], "zero_observations_on_empty_control")

    def test_empty_background_control_false_positive(self):
        # Frame 1 was sampled and had 1 observation in setUp
        gt = [{
            "label_id": "fp_empty_ctrl",
            "source_id": "test_source_1",
            "pts": 100,
            "timestamp_s": 0.1,
            "bbox_xyxy": None,
            "annotation_metadata": {"target_category": "empty_background_control"},
        }]
        ledger = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(ledger["summary"]["empty_controls_clean"], 0)
        self.assertEqual(ledger["summary"]["false_positive_frames"], 1)
        self.assertEqual(ledger["summary"]["false_positive_observations"], 1)
        self.assertEqual(ledger["summary"]["empty_controls"]["false_positive_frames"], 1)
        self.assertEqual(ledger["summary"]["empty_controls"]["false_positive_observations"], 1)

        entry = ledger["ledger"][0]
        self.assertEqual(entry["diagnosis"], "false_positive_observation")
        self.assertEqual(entry["evidence"]["observations_count"], 1)
        self.assertEqual(len(entry["evidence"]["observations"]), 1)
        self.assertAlmostEqual(entry["evidence"]["observations"][0]["detector_score"], 0.9)
        self.assertIn("not that the detection is definitively a non-player", entry["notes"])

    def test_empty_background_control_unsampled(self):
        # Frame 0 was decoded but unsampled
        gt = [{
            "label_id": "unsampled_empty_ctrl",
            "source_id": "test_source_1",
            "pts": 0,
            "timestamp_s": 0.0,
            "bbox_xyxy": None,
            "target_category": "empty_background_control",
        }]
        ledger = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(ledger["summary"]["sampling_skip"], 1)
        self.assertEqual(ledger["summary"]["empty_controls"]["unsampled"], 1)
        self.assertEqual(ledger["summary"]["empty_controls"]["evaluated"], 0)

        entry = ledger["ledger"][0]
        self.assertEqual(entry["diagnosis"], "sampling_skip")
        self.assertEqual(entry["evidence"]["evaluation_status"], "not_evaluated_skipped_by_sampler")

    def test_empty_background_control_unrecorded(self):
        # Frame 999 is absent from coverage
        gt = [{
            "label_id": "unrecorded_empty_ctrl",
            "source_id": "test_source_1",
            "pts": 999,
            "timestamp_s": 99.9,
            "bbox_xyxy": None,
            "target_category": "empty_background_control",
        }]
        ledger = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(ledger["summary"]["unrecorded_frame"], 1)
        self.assertEqual(ledger["summary"]["empty_controls"]["unrecorded"], 1)

        entry = ledger["ledger"][0]
        self.assertEqual(entry["diagnosis"], "unrecorded_frame")
        self.assertEqual(entry["evidence"]["decode_status"], "unknown_absent_from_coverage_ledger")

    def test_empty_background_control_invalid_and_ambiguous_annotations(self):
        # 1. bbox_xyxy null without target_category
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, [{
                "source_id": "test_source_1",
                "pts": 300,
                "bbox_xyxy": None,
            }])
        self.assertIn("target_category is not 'empty_background_control'", str(ctx.exception))

        # 2. bbox_xyxy provided with empty_background_control
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, [{
                "source_id": "test_source_1",
                "pts": 300,
                "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
                "target_category": "empty_background_control",
            }])
        self.assertIn("cannot specify bbox_xyxy for an empty_background_control", str(ctx.exception))

        # 3. missing bbox_xyxy field
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, [{
                "source_id": "test_source_1",
                "pts": 300,
                "target_category": "empty_background_control",
            }])
        self.assertIn("missing required 'bbox_xyxy' field", str(ctx.exception))

        # 4. duplicate empty_background_control on same frame
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, [
                {"source_id": "test_source_1", "pts": 300, "bbox_xyxy": None, "target_category": "empty_background_control"},
                {"source_id": "test_source_1", "pts": 300, "bbox_xyxy": None, "target_category": "empty_background_control"},
            ])
        self.assertIn("duplicate empty_background_control", str(ctx.exception))

        # 5. contradictory annotations on same frame (empty control and positive box)
        with self.assertRaises(ValueError) as ctx:
            audit_ground_truth(self.run_dir, [
                {"source_id": "test_source_1", "pts": 300, "bbox_xyxy": None, "target_category": "empty_background_control"},
                {"source_id": "test_source_1", "pts": 300, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
            ])
        self.assertIn("contradictory annotations", str(ctx.exception))


    def test_invalid_score_threshold(self):
        with self.assertRaises(ValueError):
            audit_ground_truth(self.run_dir, [], score_threshold=-0.1)
        with self.assertRaises(ValueError):
            audit_ground_truth(self.run_dir, [], score_threshold=1.5)
        with self.assertRaises(ValueError):
            audit_ground_truth(self.run_dir, [], score_threshold=True)

    def test_stage_attribution_with_raw_proposals(self):
        # Frame 3 (pts 300) is sampled with no detections in default fixtures.
        # 1. Detector raw miss: proposal exists but far away (IoU < 0.5)
        raw_proposals_file = Path(self.run_dir) / "raw_proposals.jsonl"
        proposals = [
            {
                "frame": {"source_id": "test_source_1", "pts": 300, "timestamp_s": 0.3},
                "bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
                "score": 0.9,
                "label": 1
            }
        ]
        with raw_proposals_file.open("w", encoding="utf-8") as f:
            for p in proposals:
                f.write(json.dumps(p) + "\n")

        gt = [{
            "label_id": "gt_player_miss",
            "source_id": "test_source_1",
            "pts": 300,
            "timestamp_s": 0.3,
            "bbox_xyxy": [5.0, 5.0, 25.0, 25.0]
        }]

        report = audit_ground_truth(self.run_dir, gt)
        self.assertEqual(report["summary"]["detector_raw_miss"], 1)
        entry = report["ledger"][0]
        self.assertEqual(entry["diagnosis"], "detector_raw_miss")
        self.assertEqual(entry["stage_breakdown"]["detector_proposals"], "failed_no_candidate_box")

        # 2. Confidence filtered: proposal overlaps GT (IoU >= 0.5) but score < score_threshold (0.5)
        proposals = [
            {
                "frame": {"source_id": "test_source_1", "pts": 300, "timestamp_s": 0.3},
                "bbox_xyxy": [5.0, 5.0, 25.0, 25.0],
                "score": 0.35,
                "label": 1
            }
        ]
        with raw_proposals_file.open("w", encoding="utf-8") as f:
            for p in proposals:
                f.write(json.dumps(p) + "\n")

        report = audit_ground_truth(self.run_dir, gt, score_threshold=0.5)
        self.assertEqual(report["summary"]["confidence_filtered"], 1)
        entry = report["ledger"][0]
        self.assertEqual(entry["diagnosis"], "confidence_filtered")
        self.assertIn("filtered_sub_threshold_score_0.350", entry["stage_breakdown"]["detector_filters"])

        # 3. Tracker unconfirmed: proposal overlaps GT and score >= score_threshold, but not in observations
        proposals = [
            {
                "frame": {"source_id": "test_source_1", "pts": 300, "timestamp_s": 0.3},
                "bbox_xyxy": [5.0, 5.0, 25.0, 25.0],
                "score": 0.85,
                "label": 1
            }
        ]
        with raw_proposals_file.open("w", encoding="utf-8") as f:
            for p in proposals:
                f.write(json.dumps(p) + "\n")

        report = audit_ground_truth(self.run_dir, gt, score_threshold=0.5)
        self.assertEqual(report["summary"]["tracker_unconfirmed"], 1)
        entry = report["ledger"][0]
        self.assertEqual(entry["diagnosis"], "tracker_unconfirmed")
        self.assertEqual(entry["stage_breakdown"]["tracking"], "dropped_or_unassociated")

        # Clean up
        raw_proposals_file.unlink()

    def test_pipeline_records_raw_proposals_and_audits_stages(self):
        from touchline.contracts import Box, Detection, Frame
        from touchline.pipeline import run_pipeline
        from touchline.video.sampling import FixedRateSampler

        class ProposalDetector:
            def provenance(self):
                return {"adapter": "mock-proposal-detector", "threshold": 0.5}

            def detect_with_proposals(self, frame):
                dets = [Detection(Box(10.0, 10.0, 50.0, 50.0), 0.90)]
                props = [
                    {"bbox_xyxy": [10.0, 10.0, 50.0, 50.0], "score": 0.90, "label": 1},
                    {"bbox_xyxy": [100.0, 100.0, 140.0, 140.0], "score": 0.32, "label": 1},
                ]
                return dets, props

            def detect(self, frame):
                return self.detect_with_proposals(frame)[0]

        frames = [Frame("src_prop", 0, 100, 1, 25, 0.0, 640, 360)]
        out_dir = Path(self.run_dir).parent / "prop_pipeline_run"
        run_pipeline(
            frames,
            ProposalDetector(),
            FixedRateSampler(fps=25.0),
            out_dir,
            source="src_prop",
            synthetic=True,
            record_proposals=True,
        )
        self.assertTrue((out_dir / "raw_proposals.jsonl").is_file())

        gt = [
            {"label_id": "gt_hit", "source_id": "src_prop", "pts": 100, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
            {"label_id": "gt_sub", "source_id": "src_prop", "pts": 100, "bbox_xyxy": [100.0, 100.0, 140.0, 140.0]},
            {"label_id": "gt_miss", "source_id": "src_prop", "pts": 100, "bbox_xyxy": [300.0, 200.0, 340.0, 260.0]},
        ]
        report = audit_ground_truth(out_dir, gt, score_threshold=0.5)
        self.assertEqual(report["summary"]["detected_and_persisted"], 1)
        self.assertEqual(report["summary"]["confidence_filtered"], 1)
        self.assertEqual(report["summary"]["detector_raw_miss"], 1)

if __name__ == "__main__":
    unittest.main()
