import json
from pathlib import Path
import tempfile
import unittest

from touchline.cli import main
from touchline.fixtures import synthetic_frames, SyntheticDetector
from touchline.pipeline import run_pipeline
from touchline.video.sampling import Anchor24Sampler


class ViewerAndInspectTests(unittest.TestCase):
    def test_pipeline_generates_analytics_json_and_interactive_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "demo_run"
            result = run_pipeline(
                synthetic_frames(48),
                SyntheticDetector(),
                Anchor24Sampler(),
                run_dir,
                {"id": "synthetic_viewer_test"},
                synthetic=True,
            )
            self.assertEqual(result["status"], "complete")

            # Check that analytics.json was generated and is valid
            analytics_file = run_dir / "analytics.json"
            self.assertTrue(analytics_file.is_file())
            analytics = json.loads(analytics_file.read_text(encoding="utf-8"))
            self.assertEqual(analytics["schema_version"], "touchline.analytics.v1")
            self.assertTrue(analytics["synthetic"])
            self.assertIn("SYNTHETIC", analytics["banner"])
            self.assertGreater(len(analytics["player_roster"]), 0)

            # Check review.html dashboard
            review_file = run_dir / "review.html"
            self.assertTrue(review_file.is_file())
            html_content = review_file.read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC", html_content)
            self.assertIn("Touchline Football Analytics", html_content)
            self.assertIn("pitchCanvas", html_content)
            self.assertIn("frameSlider", html_content)
            self.assertIn("saveReviewBtn", html_content)

    def test_cli_inspect_subcommand(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "cli_run"
            run_pipeline(
                synthetic_frames(48),
                SyntheticDetector(),
                Anchor24Sampler(),
                run_dir,
                {"id": "synthetic_cli_test"},
                synthetic=True,
            )

            # Remove generated analytics.json to test regeneration via inspect
            (run_dir / "analytics.json").unlink()

            ret = main(["inspect", str(run_dir)])
            self.assertEqual(ret, 0)
            self.assertTrue((run_dir / "analytics.json").is_file())
            self.assertTrue((run_dir / "review.html").is_file())

    def test_cli_inspect_missing_artifacts_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = Path(tmp) / "empty_run"
            empty_dir.mkdir()
            with self.assertRaises(SystemExit) as cm:
                main(["inspect", str(empty_dir)])
            self.assertEqual(cm.exception.code, 2)


    def test_pipeline_initial_analytics_includes_observations_beyond_preview_cap(self):
        class MultiBoxDetector:
            def provenance(self):
                return {"adapter": "multi-box-fixture", "model": "none"}
            def detect(self, frame):
                from touchline.contracts import Box, Detection
                # 10 detections per frame -> 60 frames = 600 observations (> 500 preview limit)
                return [Detection(Box(10 + i * 20, 20, 25 + i * 20, 70), 0.9) for i in range(10)]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "large_obs_run"
            from touchline.video.sampling import FixedRateSampler
            from touchline.fixtures import synthetic_frames
            result = run_pipeline(
                synthetic_frames(60),
                MultiBoxDetector(),
                FixedRateSampler(fps=24),
                run_dir,
                {"id": "synthetic_large_obs"},
                synthetic=True,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["observation_count"], 600)

            analytics_file = run_dir / "analytics.json"
            self.assertTrue(analytics_file.is_file())
            analytics = json.loads(analytics_file.read_text(encoding="utf-8"))

            # The session observation_count must reflect all 600 observations, not just 500
            self.assertEqual(analytics["session"]["observation_count"], 600)
            # The sum of observations across tracks in the roster must be 600
            roster_obs_count = sum(player["observation_count"] for player in analytics["player_roster"])
            self.assertEqual(roster_obs_count, 600)


    def test_tl005_attempt_denominators_and_exclusions_6_of_8_vs_9_of_12(self):
        from touchline.analytics import summarize_attempts

        # Session A: 6 successes, 2 errors (6/8 = 75%), plus 2 inconclusive and 1 pending excluded
        session_a = (
            [{"outcome": "conclusive_success"}] * 6
            + [{"outcome": "conclusive_error"}] * 2
            + [{"outcome": "inconclusive"}] * 2
            + [{"outcome": "pending"}] * 1
        )
        res_a = summarize_attempts(session_a)
        self.assertEqual(res_a["status"], "available")
        self.assertEqual(res_a["conclusive_denominator"], 8)
        self.assertEqual(res_a["success_count"], 6)
        self.assertEqual(res_a["error_count"], 2)
        self.assertAlmostEqual(res_a["success_rate"], 0.75)
        self.assertEqual(res_a["excluded_from_denominator"]["inconclusive_count"], 2)
        self.assertEqual(res_a["excluded_from_denominator"]["pending_count"], 1)
        self.assertEqual(res_a["excluded_from_denominator"]["total_excluded"], 3)

        # Session B: 9 successes, 3 errors (9/12 = 75%), plus 1 inconclusive and 3 pending excluded
        session_b = (
            [{"outcome": "success"}] * 9
            + [{"outcome": "error"}] * 3
            + [{"outcome": "inconclusive"}] * 1
            + [{"outcome": "pending"}] * 3
        )
        res_b = summarize_attempts(session_b)
        self.assertEqual(res_b["status"], "available")
        self.assertEqual(res_b["conclusive_denominator"], 12)
        self.assertEqual(res_b["success_count"], 9)
        self.assertEqual(res_b["error_count"], 3)
        self.assertAlmostEqual(res_b["success_rate"], 0.75)
        self.assertEqual(res_b["excluded_from_denominator"]["total_excluded"], 4)

        # No conclusive attempts -> rate must be unavailable (None), never 0%
        only_obscured = [{"outcome": "inconclusive"}, {"outcome": "pending"}]
        res_empty = summarize_attempts(only_obscured)
        self.assertEqual(res_empty["status"], "unavailable")
        self.assertEqual(res_empty["conclusive_denominator"], 0)
        self.assertIsNone(res_empty["success_rate"])
        self.assertEqual(res_empty["excluded_from_denominator"]["total_excluded"], 2)

if __name__ == "__main__":
    unittest.main()
