"""Offline artifact-contract checks; no video decode, model, or Modal execution."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from touchline.panorama_import import import_completed_panorama


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PanoramaImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.export = root / "export"
        (self.export / "evidence").mkdir(parents=True)
        self.runs = root / "runs"
        self.runs.mkdir()
        self.source = root / "source.mp4"
        self.normalized = root / "normalized.mp4"
        self.frame_map = root / "frame-map.jsonl"
        self.normalization = root / "normalization.json"
        self.source.write_bytes(b"offline-original-fixture")
        self.normalized.write_bytes(b"offline-normalized-fixture")
        self.frame_map.write_text(json.dumps({"output_frame_index": 0, "source_frame_index": 0,
            "source_pts": 300, "source_time_base": [1, 300], "source_relative_time": [0, 1]}) + "\n")
        overlay = b"offline-overlay-fixture"
        (self.export / "overlay24.mp4").write_bytes(overlay)
        summary = {"schema": "touchline.panorama-review.v1", "status": "complete",
                   "run_id": "trial-one", "frames": 1, "source_fps": 24,
                   "source_geometry": [64, 36], "source_sha256": digest(self.normalized.read_bytes()),
                   "frame_map_sha256": digest(self.frame_map.read_bytes()),
                   "identity_confirmed": False, "duration_s": 1 / 24,
                   "counters": {"detector_frames": 1, "raw_proposals": 1,
                                "detector_anchors": 1, "motion_predictions": 1}}
        self.normalization.write_text(json.dumps({
            "schema": "touchline.video-normalization.v1", "status": "complete",
            "source_sha256": digest(self.source.read_bytes()),
            "output_sha256": digest(self.normalized.read_bytes()),
            "frame_map_sha256": digest(self.frame_map.read_bytes()),
            "output_frames": 1, "geometry": [64, 36]}))
        (self.export / "evidence" / "summary.json").write_text(json.dumps(summary))
        (self.export / "receipt.json").write_text(json.dumps({"status": "complete", "run_id": "trial-one",
            "summary": summary, "result_sha256": digest(overlay)}))
        rows = [{"run_id": "trial-one", "frame_index": 0, "timestamp_s": 0,
                 "track_id": 7, "bbox_xyxy": [1, 2, 10, 20],
                 "detector_bbox_xyxy": [1, 2, 10, 20],
                 "evidence": "detector_anchor"},
                {"run_id": "trial-one", "frame_index": 0, "timestamp_s": 0,
                 "track_id": 7, "bbox_xyxy": [1, 2, 10, 20],
                 "detector_bbox_xyxy": None, "detector_score": None,
                 "evidence": "camera_prediction"}]
        (self.export / "evidence" / "observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.export / "evidence" / "raw_proposals.jsonl").write_text(json.dumps({
            "frame_index": 0, "bbox_xyxy": [1, 2, 10, 20], "score": .9}) + "\n")
        (self.export / "evidence" / "coverage.jsonl").write_text(json.dumps({"frame_index": 0}) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def imported(self):
        return import_completed_panorama(export_dir=self.export, source_video=self.source,
            normalized_video=self.normalized, frame_map=self.frame_map,
            normalization_metadata=self.normalization,
            expected_original_sha256=digest(self.source.read_bytes()), runs_root=self.runs)

    def test_complete_verified_bundle_keeps_predictions_separate_and_identity_unconfirmed(self):
        run = self.imported()
        manifest = json.loads((run / "manifest.json").read_text())
        rows = [json.loads(line) for line in (run / "observations.jsonl").read_text().splitlines()]
        predictions = (run / "motion_predictions.jsonl").read_text().splitlines()
        self.assertEqual((manifest["observation_count"], manifest["motion_prediction_count"]), (1, 1))
        self.assertEqual(rows[0]["local_track_id"], 7)
        self.assertEqual(rows[0]["detector_score"], .9)
        self.assertIsNone(rows[0]["player_id"])
        self.assertEqual(rows[0]["frame"]["pts"], 0)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(manifest["source"]["sha256"], digest(self.source.read_bytes()))
        self.assertTrue((run / "overlay24.mp4").is_file())
        self.assertTrue(manifest["normalization_verified"])

    def test_corrupt_export_or_source_does_not_publish_run(self):
        (self.export / "overlay24.mp4").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.imported()
        self.assertEqual(list(self.runs.iterdir()), [])
        (self.export / "overlay24.mp4").write_bytes(b"offline-overlay-fixture")
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            import_completed_panorama(export_dir=self.export, source_video=self.source,
                normalized_video=self.normalized, frame_map=self.frame_map,
                normalization_metadata=self.normalization,
                expected_original_sha256=digest(b"offline-original-fixture"), runs_root=self.runs)
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_failed_receipt_and_duplicate_run_are_rejected(self):
        receipt_path = self.export / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt_path.write_text(json.dumps({**receipt, "status": "failed"}))
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.imported()
        receipt_path.write_text(json.dumps(receipt))
        run = self.imported()
        before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.imported()
        self.assertEqual(before, {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()})

    def test_anchor_without_exact_recorded_proposal_is_rejected(self):
        (self.export / "evidence" / "raw_proposals.jsonl").write_text(json.dumps({
            "frame_index": 0, "bbox_xyxy": [2, 2, 10, 20], "score": .9}) + "\n")
        with self.assertRaisesRegex(ValueError, "exact recorded proposal"):
            self.imported()
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_normalization_must_bind_original_to_analysis_copy(self):
        data = json.loads(self.normalization.read_text())
        data["source_sha256"] = digest(b"unrelated-video")
        self.normalization.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Normalization does not bind"):
            self.imported()
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_summary_observation_count_must_match_export(self):
        summary_path = self.export / "evidence" / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["counters"]["detector_anchors"] = 2
        summary_path.write_text(json.dumps(summary))
        receipt_path = self.export / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["summary"] = summary
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "Observation counts"):
            self.imported()
        self.assertEqual(list(self.runs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
