import json
from pathlib import Path
import tempfile
import unittest

from touchline.cli import main
from touchline.contracts import Box, Detection
from touchline.fixtures import synthetic_frames, SyntheticDetector
from touchline.pipeline import run_pipeline
from touchline.tracking import (
    IoUTracker,
    PlayerAssociation,
    PlayerAssociationLedger,
    TrackingEvent,
)
from touchline.video.sampling import Anchor24Sampler


class TrackingTests(unittest.TestCase):
    def test_tracking_lifecycle_events_ledger(self):
        tracker = IoUTracker(threshold=0.3, max_gap_s=0.5, run_id="session_01")

        # Frame 0: Track 1 starts
        res1 = tracker.update(0.0, [Detection(Box(10, 10, 50, 50), 0.9)])
        self.assertEqual(len(res1), 1)
        self.assertEqual(res1[0]["local_track_id"], 1)

        # Frame 1 (0.2s): Track 1 matches
        res2 = tracker.update(0.2, [Detection(Box(12, 10, 52, 50), 0.9)])
        self.assertEqual(res2[0]["local_track_id"], 1)

        # Frame 2 (0.4s): Competing detections cause ambiguous crossing
        res3 = tracker.update(
            0.4,
            [
                Detection(Box(13, 10, 53, 50), 0.9),
                Detection(Box(14, 10, 54, 50), 0.85),
            ],
        )
        self.assertTrue(any(r["association_state"] == "ambiguous" for r in res3))

        # Frame 3 (1.2s): Time elapsed > max_gap_s (0.8s gap > 0.5s) -> prior tracks terminated
        res4 = tracker.update(1.2, [Detection(Box(100, 100, 150, 150), 0.9)])
        self.assertNotIn(1, [r["local_track_id"] for r in res4])

        events = tracker.get_event_ledger()
        event_types = [e["event_type"] for e in events]
        self.assertIn("started", event_types)
        self.assertIn("matched", event_types)
        self.assertIn("crossing_ambiguous", event_types)
        self.assertIn("terminated", event_types)

    def test_player_association_and_target_selection(self):
        ledger = PlayerAssociationLedger(run_id="run_match_001")
        assoc = ledger.associate(
            local_track_id=1,
            player_id="P10",
            player_name="Lionel Messi",
            is_target=True,
            reviewed_by="coach_analyst",
            notes="Starting right winger",
        )
        self.assertEqual(assoc.local_track_id, 1)
        self.assertEqual(assoc.player_id, "P10")
        self.assertEqual(assoc.player_name, "Lionel Messi")
        self.assertTrue(assoc.is_target)
        self.assertEqual(assoc.revision, 1)

        dict_form = ledger.to_dict()
        self.assertEqual(dict_form["run_id"], "run_match_001")
        self.assertIn(1, dict_form["target_tracks"])
        self.assertEqual(dict_form["associations"]["1"]["player_id"], "P10")

        # Test reload from dict
        loaded = PlayerAssociationLedger.from_dict(dict_form)
        self.assertEqual(loaded.run_id, "run_match_001")
        loaded_assoc = loaded.get_association(1)
        self.assertIsNotNone(loaded_assoc)
        self.assertEqual(loaded_assoc.player_id, "P10")
        self.assertTrue(loaded_assoc.is_target)

    def test_track_reassignment_creates_explicit_correction(self):
        ledger = PlayerAssociationLedger(run_id="run_match_002")
        # First association
        ledger.associate(local_track_id=2, player_id="Player_A", is_target=False)

        # Correction / Re-association
        ledger.associate(
            local_track_id=2,
            player_id="Player_B",
            is_target=True,
            reviewed_by="senior_coach",
            notes="Corrected identity swap after crossing",
        )

        assoc = ledger.get_association(2)
        self.assertEqual(assoc.player_id, "Player_B")
        self.assertEqual(assoc.revision, 2)
        self.assertTrue(assoc.is_target)

        # Corrections history must record previous and new revisions
        self.assertEqual(len(ledger.corrections), 1)
        correction = ledger.corrections[0]
        self.assertEqual(correction["local_track_id"], 2)
        self.assertEqual(correction["previous_player_id"], "Player_A")
        self.assertEqual(correction["new_player_id"], "Player_B")
        self.assertEqual(correction["previous_revision"], 1)
        self.assertEqual(correction["new_revision"], 2)
        self.assertIn("analytics.json", correction["stale_derivatives"])

    def test_reused_local_track_ids_isolated_across_sessions(self):
        ledger_a = PlayerAssociationLedger(run_id="session_match_A")
        ledger_b = PlayerAssociationLedger(run_id="session_match_B")

        # Track 7 in Session A is Player_Alpha
        ledger_a.associate(local_track_id=7, player_id="Player_Alpha")
        # Track 7 in Session B is Player_Beta
        ledger_b.associate(local_track_id=7, player_id="Player_Beta")

        self.assertNotEqual(
            ledger_a.get_association(7).player_id,
            ledger_b.get_association(7).player_id,
        )
        self.assertEqual(ledger_a.run_id, "session_match_A")
        self.assertEqual(ledger_b.run_id, "session_match_B")

    def test_rejection_of_raw_track_id_as_player_id(self):
        ledger = PlayerAssociationLedger(run_id="session_03")
        # Direct raw track number "7"
        with self.assertRaises(ValueError) as cm:
            ledger.associate(local_track_id=7, player_id="7")
        self.assertIn("cannot be used directly as confirmed player_id", str(cm.exception))

        # "track_07"
        with self.assertRaises(ValueError) as cm:
            ledger.associate(local_track_id=7, player_id="track_07")
        self.assertIn("cannot be used directly as confirmed player_id", str(cm.exception))

    def test_cli_associate_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_cli_test"
            run_pipeline(
                synthetic_frames(24),
                SyntheticDetector(),
                Anchor24Sampler(),
                run_dir,
                {"id": "cli_source"},
                synthetic=True,
            )

            # Associate track 1 with target player P09
            ret = main([
                "associate",
                str(run_dir),
                "--track", "1",
                "--player", "P09",
                "--name", "Marcus Rashford",
                "--target",
                "--reviewer", "head_coach",
                "--notes", "Confirmed striker target",
            ])
            self.assertEqual(ret, 0)

            # Check associations.json exists and is valid
            assoc_file = run_dir / "associations.json"
            self.assertTrue(assoc_file.is_file())
            assoc_data = json.loads(assoc_file.read_text(encoding="utf-8"))
            self.assertEqual(assoc_data["associations"]["1"]["player_id"], "P09")
            self.assertTrue(assoc_data["associations"]["1"]["is_target"])

            # Check analytics.json updated with confirmed player and target roster
            analytics_file = run_dir / "analytics.json"
            analytics = json.loads(analytics_file.read_text(encoding="utf-8"))
            self.assertEqual(analytics["quality_indicators"]["confirmed_players_count"], 1)
            self.assertEqual(analytics["quality_indicators"]["target_tracks_count"], 1)
            self.assertEqual(len(analytics["target_players"]), 1)
            self.assertEqual(analytics["target_players"][0]["player_id"], "P09")

            # Check review.html contains target badge
            review_html = (run_dir / "review.html").read_text(encoding="utf-8")
            self.assertIn("TARGET", review_html)


if __name__ == "__main__":
    unittest.main()
