import unittest

from touchline.contracts import Frame
from touchline.exemplars import (
    ContourMotionDetector,
    VersionedExemplarIndex,
    switch_or_rollback_provider,
)
from touchline.fixtures import SyntheticDetector


class ProviderAndExemplarTests(unittest.TestCase):
    def setUp(self):
        self.frames = [
            Frame("src_1", 0, 100, 1, 25, 0.0, 1920, 1080),
            Frame("src_1", 1, 200, 1, 25, 0.04, 1920, 1080),
            Frame("src_1", 2, 300, 1, 25, 0.08, 1920, 1080),
        ]
        self.initial_state = {
            "provider_revision": 1,
            "provider_provenance": {"adapter": "synthetic-detector-v1"},
            "detections_by_pts": {
                "100": [{"bbox_xyxy": [10.0, 20.0, 50.0, 100.0], "score": 0.9}],
            },
            "player_associations": {"1": {"player_id": "P07", "player_name": "Alex", "revision": 2}},
            "objectives": {"OBJ-01": {"title": "Scanning before receiving"}},
            "approved_annotations": [{"annotation_id": "ann-1", "pts": 100, "outcome": "conclusive_success"}],
            "source_evidence": {"source_id": "src_1", "sha256": "abc123"},
            "provider_history": [],
        }

    def test_tl013_provider_replacement_rollback_and_model_absent_fallback(self):
        alt_provider = ContourMotionDetector(threshold=0.5)
        replaced = switch_or_rollback_provider(
            self.initial_state,
            alt_provider,
            self.frames,
            action="replace",
        )
        self.assertEqual(replaced["provider_revision"], 2)
        self.assertEqual(replaced["provider_provenance"]["adapter"], "contour-motion-detector-v1")
        self.assertTrue(replaced["outputs_changed"])
        self.assertTrue(replaced["dependent_results_stale"])
        # Canonical IDs and annotations remain intact
        self.assertEqual(replaced["player_associations"]["1"]["player_id"], "P07")
        self.assertEqual(replaced["objectives"]["OBJ-01"]["title"], "Scanning before receiving")
        self.assertEqual(len(replaced["approved_annotations"]), 1)
        self.assertEqual(len(replaced["provider_history"]), 1)

        # Rollback to SyntheticDetector
        rolled_back = switch_or_rollback_provider(
            replaced,
            SyntheticDetector(),
            self.frames,
            action="rollback",
        )
        self.assertEqual(rolled_back["provider_revision"], 3)
        self.assertEqual(rolled_back["action"], "rollback")
        self.assertEqual(len(rolled_back["provider_history"]), 2)
        self.assertEqual(rolled_back["player_associations"]["1"]["player_id"], "P07")

        # Model absent fallback keeps manual review and evidence available
        no_model = switch_or_rollback_provider(
            rolled_back,
            None,
            self.frames,
            action="disable_model",
        )
        self.assertEqual(no_model["mode"], "manual_review_only")
        self.assertTrue(no_model["manual_review_available"])
        self.assertEqual(no_model["player_associations"]["1"]["player_id"], "P07")
        self.assertEqual(len(no_model["approved_annotations"]), 1)

    def test_tl014_isolated_namespaces_permission_filtering_and_overlap_exclusion(self):
        idx = VersionedExemplarIndex("dinov2_vitb14", "crop_224_norm", "v1", dim=4)
        idx.add_clip(
            clip_id="clip_query",
            source_id="match_1",
            start_s=10.0,
            end_s=14.0,
            start_pts=250,
            end_pts=350,
            team_id="TEAM_A",
            player_id="P07",
            feature_vector=[1.0, 0.0, 0.0, 0.0],
            attempt_outcome="conclusive_success",
            review_state="approved",
        )
        # Overlapping clip from same source_id (12.0s - 16.0s overlaps 10.0s - 14.0s)
        idx.add_clip(
            clip_id="clip_overlap",
            source_id="match_1",
            start_s=12.0,
            end_s=16.0,
            start_pts=300,
            end_pts=400,
            team_id="TEAM_A",
            player_id="P07",
            feature_vector=[0.99, 0.01, 0.0, 0.0],
            attempt_outcome="conclusive_success",
            review_state="approved",
        )
        # Valid non-overlapping clip on TEAM_A
        idx.add_clip(
            clip_id="clip_valid_a",
            source_id="match_1",
            start_s=30.0,
            end_s=34.0,
            start_pts=750,
            end_pts=850,
            team_id="TEAM_A",
            player_id="P07",
            feature_vector=[0.9, 0.2, 0.0, 0.0],
            attempt_outcome="conclusive_error",
            review_state="approved",
        )
        # Unauthorized clip on TEAM_B (must never leak in results or pool count)
        idx.add_clip(
            clip_id="clip_secret_b",
            source_id="match_2",
            start_s=50.0,
            end_s=54.0,
            start_pts=1250,
            end_pts=1350,
            team_id="TEAM_B",
            player_id="P99",
            feature_vector=[1.0, 0.0, 0.0, 0.0],
            attempt_outcome="conclusive_success",
            review_state="approved",
        )

        # Incompatible namespace must raise ValueError
        with self.assertRaises(ValueError):
            idx.search(
                query_clip_id="clip_query",
                query_source_id="match_1",
                query_start_s=10.0,
                query_end_s=14.0,
                query_vector=[1.0, 0.0, 0.0, 0.0],
                query_namespace="other_encoder::crop::v1",
                authorized_teams=["TEAM_A"],
            )

        res = idx.search(
            query_clip_id="clip_query",
            query_source_id="match_1",
            query_start_s=10.0,
            query_end_s=14.0,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            authorized_teams=["TEAM_A"],
        )
        # TEAM_B clip is excluded BEFORE counting pool size (only 3 TEAM_A clips exist)
        self.assertEqual(res["authorized_candidate_pool_size"], 3)
        # Both clip_query and clip_overlap are excluded
        self.assertEqual(res["excluded_self_or_overlapping"], 2)
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(res["results"][0]["clip_id"], "clip_valid_a")
        self.assertFalse(res["results"][0]["is_correctness_probability"])

        # Relevance feedback is recorded separately from attempt outcome
        fb = idx.record_relevance_feedback(
            query_clip_id="clip_query",
            candidate_clip_id="clip_valid_a",
            relevance="relevant",
            reviewer="coach_1",
        )
        self.assertEqual(fb["relevance"], "relevant")
        self.assertEqual(fb["candidate_attempt_outcome"], "conclusive_error")

        # Encoder change rebuilds isolated index and preserves approved annotations
        rebuilt = idx.rebuild_with_new_encoder(
            new_encoder_id="vjepa_vitl16",
            new_preprocessor_id="tubelet_16f",
            new_recipe_version="v2",
            new_dim=3,
            reembed_fn=lambda item: item["feature_vector"][:3],
        )
        self.assertEqual(rebuilt.namespace, "vjepa_vitl16::tubelet_16f::v2")
        self.assertEqual(len(rebuilt._approved_annotations), 4)

        # Revocation removes clip from search
        self.assertTrue(rebuilt.revoke_or_delete_clip("clip_valid_a"))
        res_after_revoke = rebuilt.search(
            query_clip_id="clip_query",
            query_source_id="match_1",
            query_start_s=10.0,
            query_end_s=14.0,
            query_vector=[1.0, 0.0, 0.0],
            authorized_teams=["TEAM_A"],
        )
        self.assertEqual(len(res_after_revoke["results"]), 0)


if __name__ == "__main__":
    unittest.main()
