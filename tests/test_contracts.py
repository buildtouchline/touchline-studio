import json
from pathlib import Path
import tempfile
import unittest
from touchline.contracts import (
    Box,
    CapabilityEnvelope,
    Detection,
    Frame,
    ObservationRecord,
    SCHEMA_VERSION,
    deserialize_observation,
)
from touchline.fixtures import (
    generate_complete_run_fixture,
    generate_failed_run_fixture,
    generate_partial_run_fixture,
    generate_unsupported_capability_fixture,
)


class ContractAndFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_deserialize_observation_success(self):
        raw_obs = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "test_run_123",
            "frame": {
                "source_id": "match_01",
                "index": 12,
                "pts": 1200,
                "time_base_num": 1,
                "time_base_den": 24,
                "timestamp_s": 0.5,
                "width": 1920,
                "height": 1080,
            },
            "synthetic": True,
            "local_track_id": 4,
            "bbox_xyxy": [100.0, 200.0, 150.0, 320.0],
            "detector_score": 0.88,
            "association_state": "tentative",
            "player_id": None,
        }
        obs = deserialize_observation(raw_obs)
        self.assertEqual(obs.schema_version, SCHEMA_VERSION)
        self.assertEqual(obs.run_id, "test_run_123")
        self.assertEqual(obs.frame.source_id, "match_01")
        self.assertEqual(obs.frame.index, 12)
        self.assertEqual(obs.frame.pts, 1200)
        self.assertAlmostEqual(obs.frame.timestamp_s, 0.5)
        self.assertEqual(obs.local_track_id, 4)
        self.assertEqual(obs.box.values(), [100.0, 200.0, 150.0, 320.0])
        self.assertAlmostEqual(obs.detector_score, 0.88)
        self.assertEqual(obs.association_state, "tentative")
        self.assertIsNone(obs.player_id)
        self.assertTrue(obs.synthetic)

        # Test deserializing from JSON string
        json_str = json.dumps(raw_obs)
        obs_from_json = deserialize_observation(json_str)
        self.assertEqual(obs_from_json.run_id, "test_run_123")

    def test_local_model_id_cannot_be_canonical_player_id(self):
        raw_obs = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "test_run_123",
            "frame": {
                "source_id": "match_01",
                "index": 12,
                "pts": 1200,
                "time_base_num": 1,
                "time_base_den": 24,
                "timestamp_s": 0.5,
                "width": 1920,
                "height": 1080,
            },
            "synthetic": True,
            "local_track_id": 7,
            "bbox_xyxy": [100.0, 200.0, 150.0, 320.0],
            "detector_score": 0.88,
            "association_state": "tentative",
            "player_id": "7",  # Direct adoption of local track ID
        }
        with self.assertRaises(ValueError) as ctx:
            deserialize_observation(raw_obs)
        self.assertIn("cannot be used directly as a confirmed roster player_id", str(ctx.exception))

        raw_obs["player_id"] = "track_7"
        with self.assertRaises(ValueError):
            deserialize_observation(raw_obs)

        raw_obs["player_id"] = "local_7"
        with self.assertRaises(ValueError):
            deserialize_observation(raw_obs)

        # A confirmed roster ID like 'player_mbappe' is permitted
        raw_obs["player_id"] = "player_mbappe"
        obs = deserialize_observation(raw_obs)
        self.assertEqual(obs.player_id, "player_mbappe")

    def test_deserialize_observation_validation_errors(self):
        # Invalid schema version
        with self.assertRaises(ValueError):
            deserialize_observation({"schema_version": "touchline.observation.v999"})

        # Missing frame
        with self.assertRaises(ValueError):
            deserialize_observation({"schema_version": SCHEMA_VERSION})

        # Negative box
        with self.assertRaises(ValueError):
            deserialize_observation({
                "schema_version": SCHEMA_VERSION,
                "run_id": "r1",
                "frame": {"source_id": "s", "index": 0, "pts": 0, "time_base_num": 1, "time_base_den": 24, "timestamp_s": 0.0, "width": 100, "height": 100},
                "local_track_id": 1,
                "bbox_xyxy": [-10, 0, 10, 10],
                "detector_score": 0.5,
            })

    def test_golden_complete_run_fixture(self):
        run_dir = self.root / "complete_fixture"
        manifest = generate_complete_run_fixture(run_dir, count=24)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["decoded_frames"], 24)
        self.assertTrue(manifest["synthetic"])
        self.assertTrue((run_dir / "manifest.json").is_file())
        self.assertTrue((run_dir / "coverage.jsonl").is_file())
        self.assertTrue((run_dir / "observations.jsonl").is_file())
        self.assertTrue((run_dir / "review.html").is_file())

        # Verify provider-independent observation deserialization for every record
        obs_lines = (run_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(obs_lines), 0)
        for line in obs_lines:
            obs = deserialize_observation(line)
            self.assertEqual(obs.schema_version, SCHEMA_VERSION)
            self.assertEqual(obs.run_id, manifest["run_id"])

    def test_golden_partial_run_fixture(self):
        run_dir = self.root / "partial_fixture"
        manifest = generate_partial_run_fixture(run_dir, max_selected=2)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["selected_frames"], 2)
        self.assertIn("explicit selected-frame limit", manifest["stop_reason"])
        self.assertTrue((run_dir / "manifest.json").is_file())
        self.assertTrue((run_dir / "coverage.jsonl").is_file())

    def test_golden_unsupported_capability_fixture(self):
        run_dir = self.root / "unsupported_fixture"
        manifest = generate_unsupported_capability_fixture(run_dir)
        self.assertIn("capabilities", manifest)
        caps = manifest["capabilities"]

        phys = caps["physical_metrics"]
        self.assertFalse(phys["supported"])
        self.assertEqual(phys["status"], "unsupported")
        self.assertIn("calibration", phys["reason"])

        pose = caps["3d_pose"]
        self.assertFalse(pose["supported"])
        self.assertEqual(pose["status"], "unsupported")
        self.assertIn("research scope", pose["reason"])

    def test_golden_failed_run_fixture(self):
        run_dir = self.root / "failed_fixture"
        manifest = generate_failed_run_fixture(run_dir)
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("error", manifest)
        self.assertEqual(manifest["error"]["type"], "RuntimeError")
        self.assertEqual(manifest["error"]["code"], "INTERNAL_ERROR")
        self.assertNotIn("simulated detector runtime failure", manifest["error"]["message"])
        self.assertEqual(manifest["error"]["trace_id"], manifest["diagnostic_trace_id"])
        # Partial coverage was retained despite failure
        self.assertTrue((run_dir / "coverage.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
