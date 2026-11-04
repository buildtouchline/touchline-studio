from pathlib import Path
import unittest

from touchline.analytics import generate_match_analytics
from touchline.metrics import Calibration
from review_fixtures import review_entry, review_ledger
from touchline.reports import (
    approve_report_version,
    check_report_access,
    compare_two_sessions,
    compare_two_video_runs,
    export_approved_player_package,
    generate_player_report_draft,
    update_report_draft,
)


def _sample_observations(track_id=7, count=6, dt=0.25):
    obs = []
    for i in range(count):
        t = round(i * dt, 4)
        obs.append({
            "local_track_id": track_id,
            "bbox_xyxy": [100.0 + i * 8.0, 200.0, 140.0 + i * 8.0, 300.0],
            "detector_score": 0.92,
            "association_state": "confirmed",
            "frame": {
                "index": i,
                "pts": i * 1000,
                "timestamp_s": t,
                "width": 1920,
                "height": 1080,
            },
        })
    return obs


class ReportsAndTwoSessionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "run_id": "run-session-001",
            "source": "match_clip_01.mp4",
            "synthetic": False,
            "decoded_frames": 6,
            "selected_frames": 6,
            "sampler": "FixedRateSampler",
            "detector": {"adapter": "torchvision-detector-v1"},
        }
        self.obs = _sample_observations(track_id=7, count=6, dt=0.25)
        self.calibration = Calibration(
            matrix=(52.0 / 1920.0, 0.0, 0.0, 0.0, 34.0 / 1080.0, 0.0, 0.0, 0.0, 1.0),
            independent_check_ref="pitch_check_v1",
            valid_from_s=0.0,
            valid_until_s=60.0,
            image_bounds=(0.0, 0.0, 1920.0, 1080.0),
            source_geometry=(1920, 1080),
            geometry_revision="g1",
            calibration_revision="c1",
        )

    def test_tl015_report_draft_suppresses_uncalibrated_metrics_without_plausible_text(self):
        analytics_uncal = generate_match_analytics(
            self.manifest,
            self.obs,
            calibration=None,
            associations=review_ledger(self.manifest["run_id"],7),
        )
        draft = generate_player_report_draft(analytics_uncal, local_track_id=7)
        self.assertEqual(draft["publication_state"], "draft")
        self.assertEqual(draft["delivery_state"], "not_sent")
        self.assertFalse(draft["public_published"])

        claims_by_id = {c["claim_id"]: c for c in draft["claims"]}
        self.assertEqual(claims_by_id["exposure_visibility"]["status"], "supported")
        self.assertEqual(claims_by_id["physical_workload"]["status"], "suppressed")
        self.assertIsNone(claims_by_id["physical_workload"]["value"])
        self.assertIn("uncalibrated", claims_by_id["physical_workload"]["reason"])
        self.assertEqual(claims_by_id["activity_and_intensity_zones"]["status"], "suppressed")
        self.assertIsNone(claims_by_id["activity_and_intensity_zones"]["value"])
        self.assertEqual(claims_by_id["objective_attempts"]["status"], "unavailable")
        self.assertIsNone(claims_by_id["objective_attempts"]["value"])
        self.assertEqual(claims_by_id["deferred_constructs"]["status"], "unavailable")
        self.assertIsNone(claims_by_id["deferred_constructs"]["value"])

    def test_tl015_exact_version_approval_and_edit_invalidation(self):
        analytics_cal = generate_match_analytics(
            self.manifest,
            self.obs,
            calibration=self.calibration,
            associations=review_ledger(self.manifest["run_id"],7,revision=2),
        )
        draft = generate_player_report_draft(
            analytics_cal,
            local_track_id=7,
            coach_notes={"what_went_well": "Good scanning before receiving."},
            recipient_scope={
                "team_id": "U14_A",
                "authorized_player_id": "P07",
                "allowed_recipients": ["parent_p07@example.com"],
                "parent_sharing_authorized": True,
            },
        )
        v1_hash = draft["version_hash"]

        # Wrong hash must fail
        with self.assertRaises(ValueError):
            approve_report_version(draft, reviewer="coach_1", expected_version_hash="wrong_hash")

        approved = approve_report_version(draft, reviewer="coach_1", expected_version_hash=v1_hash)
        self.assertEqual(approved["publication_state"], "approved_for_sharing")
        self.assertEqual(approved["review_state"], "approved")
        self.assertEqual(approved["delivery_state"], "not_sent")
        self.assertFalse(approved["public_published"])

        # Editing text or identity revision invalidates approval and creates revision 2 draft
        edited = update_report_draft(
            approved,
            coach_notes={"what_went_well": "Edited text after review"},
            new_association_revision=3,
        )
        self.assertEqual(edited["revision"], 2)
        self.assertNotEqual(edited["version_hash"], v1_hash)
        self.assertEqual(edited["publication_state"], "draft")
        self.assertEqual(edited["review_state"], "pending")
        self.assertIsNone(edited["approval"])
        self.assertEqual(len(edited["history"]), 1)
        self.assertTrue(edited["history"][0]["marked_stale"])

    def test_tl015_recipient_scope_and_unauthorized_parent_access(self):
        analytics_cal = generate_match_analytics(
            self.manifest,
            self.obs,
            calibration=self.calibration,
            associations=review_ledger(self.manifest["run_id"],7),
        )
        draft = generate_player_report_draft(
            analytics_cal,
            local_track_id=7,
            recipient_scope={
                "team_id": "U14_A",
                "authorized_player_id": "P07",
                "allowed_recipients": ["parent_p07@example.com"],
                "parent_sharing_authorized": True,
            },
        )

        # Parent cannot access unapproved draft
        res_unapproved = check_report_access(
            draft,
            requester_id="parent_p07@example.com",
            requester_role="parent",
            target_player_id="P07",
        )
        self.assertFalse(res_unapproved["authorized"])

        approved = approve_report_version(
            draft,
            reviewer="coach_1",
            expected_version_hash=draft["version_hash"],
        )

        # Authorized parent for P07 succeeds
        res_ok = check_report_access(
            approved,
            requester_id="parent_p07@example.com",
            requester_role="parent",
            target_player_id="P07",
        )
        self.assertTrue(res_ok["authorized"])

        # Another child's parent or wrong target_player_id is denied
        res_wrong_parent = check_report_access(
            approved,
            requester_id="parent_p09@example.com",
            requester_role="parent",
            target_player_id="P07",
        )
        self.assertFalse(res_wrong_parent["authorized"])

        res_cross_child = check_report_access(
            approved,
            requester_id="parent_p07@example.com",
            requester_role="parent",
            target_player_id="P09",
        )
        self.assertFalse(res_cross_child["authorized"])

    def test_tl016_single_session_cannot_satisfy_two_session_gate(self):
        res = compare_two_sessions(
            {"session_id": "S1", "run_id": "R1", "confirmed_player_id": "P07"},
            None,
        )
        self.assertFalse(res["two_session_gate_satisfied"])
        self.assertFalse(res["comparable"])
        self.assertEqual(res["status"], "incomplete_single_session")

    def test_tl016_track7_without_confirmed_player_or_incompatible_drill_prevents_scoring(self):
        # Track 7 in two sessions without confirmed_player_id does NOT establish the same player
        s1 = {
            "session_id": "S1",
            "run_id": "R1",
            "local_track_id": 7,
            "confirmed_player_id": None,
            "objective_id": "OBJ-PASS-01",
            "criteria_revision": "v1",
            "drill_id": "rondo_4v2",
            "visible_duration_s": 10.0,
            "attempts": [{"outcome": "conclusive_success"}] * 6 + [{"outcome": "conclusive_error"}] * 2,
        }
        s2 = {
            "session_id": "S2",
            "run_id": "R2",
            "local_track_id": 7,
            "confirmed_player_id": None,
            "objective_id": "OBJ-PASS-01",
            "criteria_revision": "v1",
            "drill_id": "rondo_4v2",
            "visible_duration_s": 12.0,
            "attempts": [{"outcome": "conclusive_success"}] * 9 + [{"outcome": "conclusive_error"}] * 3,
        }
        res = compare_two_sessions(s1, s2)
        self.assertFalse(res["comparable"])
        self.assertIsNone(res["automatic_improvement_score"])
        self.assertIsNone(res["attempt_rate_comparison"]["descriptive_diff_percentage_points"])
        self.assertTrue(any("unconfirmed_player_identity" in r for r in res["incompatibility_reasons"]))

    def test_tl016_six_of_eight_vs_nine_of_twelve_equal_75pct_with_exclusions(self):
        s1_attempts = (
            [{"outcome": "conclusive_success"}] * 6
            + [{"outcome": "conclusive_error"}] * 2
            + [{"outcome": "inconclusive"}] * 2
        )
        s2_attempts = (
            [{"outcome": "conclusive_success"}] * 9
            + [{"outcome": "conclusive_error"}] * 3
            + [{"outcome": "inconclusive"}] * 1
            + [{"outcome": "pending"}] * 1
        )
        s1 = {
            "session_id": "2026-10-21-session-a",
            "run_id": "run-a",
            "local_track_id": 1,
            "confirmed_player_id": "P07",
            "association_revision": 1,
            "objective_id": "OBJ-1V1-EXIT",
            "criteria_revision": "v1",
            "drill_id": "wing_1v1_channel",
            "visible_duration_s": 15.0,
            "attempts": s1_attempts,
        }
        s2 = {
            "session_id": "2026-10-28-session-b",
            "run_id": "run-b",
            "local_track_id": 4,
            "confirmed_player_id": "P07",
            "association_revision": 1,
            "objective_id": "OBJ-1V1-EXIT",
            "criteria_revision": "v1",
            "drill_id": "wing_1v1_channel",
            "visible_duration_s": 18.5,
            "attempts": s2_attempts,
        }
        for session in (s1,s2):
            session["identity_review"] = review_entry(session["run_id"],session["local_track_id"])
        comp = compare_two_sessions(s1, s2)
        self.assertTrue(comp["comparable"])
        self.assertTrue(comp["two_session_gate_satisfied"])
        self.assertIsNone(comp["automatic_improvement_score"])

        arc = comp["attempt_rate_comparison"]
        self.assertEqual(arc["session_a_conclusive_str"], "6/8")
        self.assertEqual(arc["session_b_conclusive_str"], "9/12")
        self.assertEqual(arc["session_a_rate_pct"], 75.0)
        self.assertEqual(arc["session_b_rate_pct"], 75.0)
        self.assertEqual(arc["descriptive_diff_percentage_points"], 0.0)
        self.assertEqual(arc["session_a_excluded"]["inconclusive_count"], 2)
        self.assertEqual(arc["session_b_excluded"]["inconclusive_count"], 1)
        self.assertEqual(arc["session_b_excluded"]["pending_count"], 1)

    def test_tl016_compare_two_video_runs_from_mp4_run_dirs(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "video_session_a"
            run_b = root / "video_session_b"
            run_a.mkdir()
            run_b.mkdir()

            ana_a = generate_match_analytics(
                {**self.manifest, "run_id": "run-mp4-a", "source": "session_a.mp4"},
                _sample_observations(track_id=1, count=6, dt=0.25),
                calibration=self.calibration,
                associations=review_ledger("run-mp4-a",1),
            )
            ana_b = generate_match_analytics(
                {**self.manifest, "run_id": "run-mp4-b", "source": "session_b.mp4"},
                _sample_observations(track_id=4, count=8, dt=0.25),
                calibration=self.calibration,
                associations=review_ledger("run-mp4-b",4),
            )
            (run_a / "analytics.json").write_text(json.dumps(ana_a), encoding="utf-8")
            (run_b / "analytics.json").write_text(json.dumps(ana_b), encoding="utf-8")
            (run_a / "associations.json").write_text(json.dumps(review_ledger("run-mp4-a",1)))
            (run_b / "associations.json").write_text(json.dumps(review_ledger("run-mp4-b",4)))

            comp = compare_two_video_runs(
                run_a,
                run_b,
                confirmed_player_id="P07",
                objective_id="OBJ-SPEED-01",
                drill_id="transition_sprint",
            )
            self.assertTrue(comp["comparable"])
            self.assertEqual(comp["player_id"], "P07")
            self.assertEqual(comp["physical_movement_comparison"]["status"], "estimated_internal")
            self.assertIsNotNone(comp["physical_movement_comparison"]["delta_distance_m"])
            self.assertEqual(comp["zone_comparison"]["status"], "compared")

    def test_tl045_report_with_aligned_voiceover_cues(self):
        analytics = generate_match_analytics(
            self.manifest,
            self.obs,
            calibration=self.calibration,
            associations=review_ledger(self.manifest["run_id"], 7, revision=1),
        )
        transcript = {
            "schema_version": "touchline.transcript.v1",
            "segments": [
                {
                    "segment_id": "seg_001",
                    "text": "Press the full back now!",
                    "video_start_s": 0.5,
                    "video_end_s": 1.0,
                    "nearest_frame_pts": 500,
                },
                {
                    "segment_id": "seg_002",
                    "text": "Unrelated shout after player left pitch",
                    "video_start_s": 50.0,
                    "video_end_s": 52.0,
                }
            ],
            "manual_cues": [
                {
                    "cue_id": "cue_001",
                    "note": "Sharp turn into box",
                    "target_track_id": 7,
                    "video_start_s": 0.75,
                    "nearest_frame_pts": 750,
                }
            ]
        }
        draft = generate_player_report_draft(
            analytics,
            local_track_id=7,
            transcript=transcript,
            coach_notes={"what_went_well": "Good positioning"}
        )
        cues = draft["coach_observations"]["aligned_voice_cues"]
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["segment_id"], "seg_001")
        self.assertEqual(cues[0]["text"], "Press the full back now!")
        self.assertEqual(cues[1]["cue_id"], "cue_001")
        # Ensure approved version hash binds the aligned voice cues
        approved = approve_report_version(
            draft,
            reviewer="coach_test",
            expected_version_hash=draft["version_hash"],
            notes="Approved with collar mic voice evidence"
        )
        self.assertEqual(approved["publication_state"], "approved_for_sharing")

    def test_tl043_export_approved_player_package(self):
        import tempfile
        analytics = generate_match_analytics(
            self.manifest,
            self.obs,
            calibration=self.calibration,
            associations=review_ledger(self.manifest["run_id"], 7, revision=1),
        )
        draft = generate_player_report_draft(
            analytics,
            local_track_id=7,
            recipient_scope={
                "parent_sharing_authorized": True,
                "allowed_recipients": ["parent_01"],
            },
            coach_notes={"what_went_well": "Explosive acceleration"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "export"
            # Unapproved draft must fail export
            with self.assertRaises(PermissionError):
                export_approved_player_package(draft, out_dir, requester_id="parent_01")

            approved = approve_report_version(
                draft,
                reviewer="coach_test",
                expected_version_hash=draft["version_hash"],
            )

            # Unauthorized parent must fail export
            with self.assertRaises(PermissionError):
                export_approved_player_package(approved, out_dir, requester_id="parent_unauthorized")

            # Authorized parent succeeds
            res = export_approved_player_package(approved, out_dir, requester_id="parent_01")
            self.assertEqual(res["status"], "exported")
            self.assertTrue(Path(res["report_file"]).is_file())
            self.assertTrue(Path(res["summary_file"]).is_file())

            # Export content must be clean and not leak private tokens
            content = Path(res["report_file"]).read_text(encoding="utf-8")
            self.assertIn("P07", content)
            self.assertNotIn("sk_", content)


if __name__ == "__main__":
    unittest.main()
