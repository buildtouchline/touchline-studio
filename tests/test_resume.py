import json
from pathlib import Path
import tempfile
import unittest
from touchline.contracts import Frame
from touchline.fixtures import SyntheticDetector, synthetic_frames
from touchline.pipeline import run_pipeline, validate_run_for_resume
from touchline.video.sampling import Anchor24Sampler, FixedRateSampler


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_successful_resume_from_interrupted_run(self):
        out_dir = self.root / "resumable_run"
        source = {"id": "test_video_source", "fps": 24}

        # Step 1: Run with explicit limit (partial run)
        first_manifest = run_pipeline(
            synthetic_frames(48),
            SyntheticDetector(),
            Anchor24Sampler(),
            out_dir,
            source,
            synthetic=True,
            max_selected=2,
        )
        self.assertEqual(first_manifest["status"], "partial")
        self.assertEqual(first_manifest["selected_frames"], 2)
        # In 24 FPS with Anchor24Sampler (indices 5, 17, 29, 41), selecting 2 means decoding up to index 17
        self.assertEqual(first_manifest["decoded_frames"], 18)
        initial_obs_count = first_manifest["observation_count"]
        self.assertGreater(initial_obs_count, 0)

        # Step 2: Resume the run to completion
        resumed_manifest = run_pipeline(
            synthetic_frames(48),
            SyntheticDetector(),
            Anchor24Sampler(),
            out_dir,
            source,
            synthetic=True,
            resume=True,
        )
        self.assertEqual(resumed_manifest["status"], "complete")
        self.assertEqual(resumed_manifest["decoded_frames"], 48)
        self.assertEqual(resumed_manifest["selected_frames"], 4)
        self.assertEqual(resumed_manifest["resume_count"], 1)
        self.assertEqual(resumed_manifest["run_id"], first_manifest["run_id"])
        self.assertGreater(resumed_manifest["observation_count"], initial_obs_count)

        # Step 3: Verify coverage ledger integrity (all 48 frames present, strictly increasing, no duplicates)
        cov_lines = [json.loads(line) for line in (out_dir / "coverage.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(cov_lines), 48)
        indices = [rec["frame"]["index"] for rec in cov_lines]
        self.assertEqual(indices, list(range(48)))

        # Step 4: Verify observations integrity (no track ID collision)
        obs_lines = [json.loads(line) for line in (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(obs_lines), resumed_manifest["observation_count"])
        # Selected frames are 5, 17, 29, 41
        obs_frame_indices = sorted({obs["frame"]["index"] for obs in obs_lines})
        self.assertEqual(obs_frame_indices, [5, 17, 29, 41])

    def test_resume_rejects_completed_run(self):
        out_dir = self.root / "completed_run"
        source = {"id": "test_video_source"}
        run_pipeline(
            synthetic_frames(24),
            SyntheticDetector(),
            FixedRateSampler(2),
            out_dir,
            source,
            synthetic=True,
        )
        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                synthetic_frames(24),
                SyntheticDetector(),
                FixedRateSampler(2),
                out_dir,
                source,
                synthetic=True,
                resume=True,
            )
        self.assertIn("already marked complete", str(ctx.exception))

    def test_resume_rejects_source_mismatch(self):
        out_dir = self.root / "partial_run_source"
        run_pipeline(
            synthetic_frames(24),
            SyntheticDetector(),
            FixedRateSampler(2),
            out_dir,
            {"id": "source_alpha"},
            synthetic=True,
            max_selected=1,
        )
        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                synthetic_frames(24),
                SyntheticDetector(),
                FixedRateSampler(2),
                out_dir,
                {"id": "source_beta"},  # Different source
                synthetic=True,
                resume=True,
            )
        self.assertIn("source mismatch", str(ctx.exception))

    def test_resume_rejects_detector_mismatch(self):
        out_dir = self.root / "partial_run_detector"
        run_pipeline(
            synthetic_frames(24),
            SyntheticDetector(),
            FixedRateSampler(2),
            out_dir,
            {"id": "src"},
            synthetic=True,
            max_selected=1,
        )

        class DifferentDetector:
            def provenance(self):
                return {"adapter": "different-detector-adapter"}
            def detect(self, frame):
                return []

        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                synthetic_frames(24),
                DifferentDetector(),
                FixedRateSampler(2),
                out_dir,
                {"id": "src"},
                synthetic=True,
                resume=True,
            )
        self.assertIn("detector provenance mismatch", str(ctx.exception))

    def test_resume_rejects_sampler_mismatch(self):
        out_dir = self.root / "partial_run_sampler"
        run_pipeline(
            synthetic_frames(24),
            SyntheticDetector(),
            FixedRateSampler(2),
            out_dir,
            {"id": "src"},
            synthetic=True,
            max_selected=1,
        )
        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                synthetic_frames(24),
                SyntheticDetector(),
                Anchor24Sampler(),  # Different sampler
                out_dir,
                {"id": "src"},
                synthetic=True,
                resume=True,
            )
        self.assertIn("sampler mismatch", str(ctx.exception))

    def test_resume_rejects_missing_required_artifacts(self):
        out_dir = self.root / "partial_run_missing_artifact"
        run_pipeline(
            synthetic_frames(24),
            SyntheticDetector(),
            FixedRateSampler(2),
            out_dir,
            {"id": "src"},
            synthetic=True,
            max_selected=1,
        )
        # Delete observations.jsonl
        (out_dir / "observations.jsonl").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            run_pipeline(
                synthetic_frames(24),
                SyntheticDetector(),
                FixedRateSampler(2),
                out_dir,
                {"id": "src"},
                synthetic=True,
                resume=True,
            )
        self.assertIn("required artifact missing: observations.jsonl", str(ctx.exception))

    def test_resume_rejects_corrupted_artifact(self):
        out_dir = self.root / "partial_run_corrupt"
        run_pipeline(
            synthetic_frames(24),
            SyntheticDetector(),
            FixedRateSampler(2),
            out_dir,
            {"id": "src"},
            synthetic=True,
            max_selected=1,
        )
        # Append corrupt line to coverage.jsonl
        with (out_dir / "coverage.jsonl").open("a", encoding="utf-8") as f:
            f.write("{corrupt line\n")

        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                synthetic_frames(24),
                SyntheticDetector(),
                FixedRateSampler(2),
                out_dir,
                {"id": "src"},
                synthetic=True,
                resume=True,
            )
        self.assertIn("corrupted JSON", str(ctx.exception))

    def test_resume_rejects_conflicting_boundary_frame(self):
        out_dir = self.root / "partial_run_boundary"
        run_pipeline(
            synthetic_frames(24),
            SyntheticDetector(),
            FixedRateSampler(2),
            out_dir,
            {"id": "src"},
            synthetic=True,
            max_selected=1,
        )

        def conflicting_frames():
            for i in range(24):
                # Frame at index 0 has different pts
                pts = 999 if i == 0 else i
                yield Frame("src", i, pts, 1, 24, i / 24, 320, 180)

        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                conflicting_frames(),
                SyntheticDetector(),
                FixedRateSampler(2),
                out_dir,
                {"id": "src"},
                synthetic=True,
                resume=True,
            )
        self.assertIn("boundary frame", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
