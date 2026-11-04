import unittest

from touchline.freeze import (
    evaluate_refinement_before_after,
    freeze_demo_candidate,
    verify_candidate_reproduction,
)
from touchline.video.adaptive import (
    AdaptiveWindowController,
    evaluate_adaptive_benchmark,
    execute_window_with_oom_retry,
    reconcile_single_ownership_observations,
)


class AdaptiveAndFreezeTests(unittest.TestCase):
    def test_tl012_adaptive_controller_triggers_and_discontinuity_boundary(self):
        ctrl = AdaptiveWindowController(
            enter_motion_threshold=0.25,
            exit_motion_threshold=0.12,
            min_detail_duration_s=0.5,
            cooldown_duration_s=0.5,
            padding_s=0.5,
            max_contiguous_gap_s=2.0,
        )
        # Segment 1: t=0.0..3.0s (has midpoint motion spike between similar endpoints at t=1.0s and crossing at t=1.5s)
        # Segment 2: t=10.0..12.0s (discontinuous timeline cut; camera_change trigger at t=10.5s)
        samples = [
            {"timestamp_s": 0.0, "pts": 0, "motion_score": 0.05, "reasons": []},
            {"timestamp_s": 0.5, "pts": 12, "motion_score": 0.05, "reasons": []},
            {"timestamp_s": 1.0, "pts": 24, "motion_score": 0.04, "midpoint_motion_spike": True, "reasons": []},
            {"timestamp_s": 1.5, "pts": 36, "motion_score": 0.10, "reasons": ["track_uncertain"]},
            {"timestamp_s": 2.0, "pts": 48, "motion_score": 0.05, "reasons": []},
            {"timestamp_s": 2.5, "pts": 60, "motion_score": 0.05, "reasons": []},
            {"timestamp_s": 10.0, "pts": 240, "motion_score": 0.05, "timeline_discontinuity": True, "reasons": []},
            {"timestamp_s": 10.5, "pts": 252, "motion_score": 0.08, "reasons": ["camera_change"]},
            {"timestamp_s": 11.5, "pts": 276, "motion_score": 0.04, "reasons": []},
        ]
        sched = ctrl.schedule_pass_a(samples, source_duration_s=12.0)
        self.assertEqual(sched["contiguous_segments_count"], 2)
        # Ensure windows from segment 0 and segment 1 were NEVER merged across the discontinuity (2.5s -> 10.0s)
        self.assertEqual(len(sched["merged_windows"]), 2)
        w0, w1 = sched["merged_windows"][0], sched["merged_windows"][1]
        self.assertLessEqual(w0["end_s"], 3.0)
        self.assertGreaterEqual(w1["start_s"], 10.0)
        self.assertIn("track_uncertain", w0["reasons"])
        self.assertIn("player_motion", w0["reasons"])
        self.assertTrue(w1["camera_uncalibrated_warning"])

    def test_tl012_single_ownership_reconciliation_prevents_duplicates(self):
        base_obs = [
            {"local_track_id": 1, "bbox_xyxy": [10, 10, 50, 90], "frame": {"pts": 100, "timestamp_s": 0.0}},
            {"local_track_id": 1, "bbox_xyxy": [12, 10, 52, 90], "frame": {"pts": 200, "timestamp_s": 0.5}},
        ]
        # Overlapping detail windows both emit PTS 200, plus a refined PTS 150
        detail_obs = [
            {"local_track_id": 1, "bbox_xyxy": [11, 10, 51, 90], "frame": {"pts": 150, "timestamp_s": 0.25}},
            {"local_track_id": 1, "bbox_xyxy": [13, 10, 53, 90], "frame": {"pts": 200, "timestamp_s": 0.5}},
            {"local_track_id": 1, "bbox_xyxy": [13, 10, 53, 90], "frame": {"pts": 200, "timestamp_s": 0.5}},
        ]
        rec = reconcile_single_ownership_observations(base_obs, detail_obs, run_revision=2)
        self.assertEqual(rec["final_count"], 3)  # PTS 100, 150, 200
        self.assertEqual(rec["audit_coexistence_count"], 1)  # baseline PTS 200 kept for audit only
        self.assertEqual(rec["duplicates_prevented"], 1)  # duplicate detail PTS 200 prevented
        pts_200 = [o for o in rec["final_observations"] if o["frame"]["pts"] == 200][0]
        self.assertEqual(pts_200["owner"], "detail_pass_b")
        self.assertEqual(pts_200["bbox_xyxy"], [13, 10, 53, 90])

    def test_tl012_oom_cleanup_and_bounded_subwindow_retry(self):
        frames = list(range(8))

        def flaky_worker(sub_frames, quality):
            if len(sub_frames) > 4:
                raise MemoryError("CUDA out of memory on 8-frame batch")
            return [{"frame": f, "min_size": quality["min_size"]} for f in sub_frames]

        res = execute_window_with_oom_retry(
            frames,
            flaky_worker,
            max_retries=2,
            quality_settings={"min_size": 640, "max_size": 1280},
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["oom_retries"], 1)
        self.assertEqual(res["final_chunk_size"], 4)
        self.assertTrue(res["quality_settings_preserved"])
        self.assertEqual(len(res["results"]), 8)

    def test_tl012_selector_vs_analyzer_benchmark_and_go_no_go(self):
        events = [
            {"event_id": "e1", "timestamp_s": 1.2, "category": "crossing"},
            {"event_id": "e2", "timestamp_s": 1.8, "category": "sprint_onset"},
            {"event_id": "e3", "timestamp_s": 8.0, "category": "distant_runner"},
        ]
        windows = [{"start_s": 0.5, "end_s": 2.5}]
        # e1 and e2 are inside window (e1 detected, e2 missed by analyzer); e3 missed by selector
        bench = evaluate_adaptive_benchmark(
            ground_truth_events=events,
            merged_windows=windows,
            detailed_detected_event_ids=["e1"],
            adaptive_runtime_s=12.0,
            fixed_6fps_runtime_s=20.0,
            fixed_6fps_recall=0.90,
            source_duration_s=10.0,
        )
        self.assertAlmostEqual(bench["selector_recall"], 0.6667, places=3)
        self.assertEqual(bench["selector_missed_event_ids"], ["e3"])
        self.assertEqual(bench["detailed_analyzer_missed_event_ids"], ["e2"])
        self.assertAlmostEqual(bench["detailed_analyzer_conditional_recall"], 0.5, places=3)
        self.assertEqual(bench["union_refined_fraction_p"], 0.2)
        self.assertEqual(bench["decision"], "no_go_keep_fixed_rate_fallback")

    def test_tl021_before_after_refinement_freeze_and_reproduction(self):
        frozen_crit = {"max_reference_error": 0.15, "min_usable_coverage": 0.80}
        # Retrofitting criteria must fail
        with self.assertRaises(ValueError):
            evaluate_refinement_before_after(
                split_role="dev_tuning",
                frozen_criteria=frozen_crit,
                before_metrics={"reference_error": 0.42, "usable_coverage": 0.292},
                after_metrics={"reference_error": 0.11, "usable_coverage": 0.840},
                attempted_criteria_override={"max_reference_error": 0.50, "min_usable_coverage": 0.20},
            )

        comp = evaluate_refinement_before_after(
            split_role="dev_tuning",
            frozen_criteria=frozen_crit,
            before_metrics={"reference_error": 0.42, "usable_coverage": 0.2925, "recall": 0.2925, "precision": 0.747},
            after_metrics={"reference_error": 0.11, "usable_coverage": 0.8396, "recall": 0.8396, "precision": 0.9175},
        )
        self.assertTrue(comp["meets_frozen_criteria"])
        self.assertAlmostEqual(comp["deltas"]["recall_delta"], 0.5471, places=4)

        # Field corrections used for training without a new release_id must fail
        with self.assertRaises(ValueError):
            freeze_demo_candidate(
                candidate_id="cand-01",
                source_manifest={"source": "SNMOT-097"},
                detector_config={"min_size": 640, "max_size": 1280},
                dataset_release={"release_id": "rel-v1", "untouched_eval_preserved": True},
                group_outcomes={
                    "level1_identity_visibility": {
                        "status": "cleared",
                        "reference_error": 0.08,
                        "usable_coverage": 0.84,
                        "limitations": ["Single broadcast camera angle"],
                    }
                },
                frozen_results={"detected": 178, "recall": 0.8396},
                field_corrections_used_for_training=True,
                prior_release_id="rel-v1",
            )

        frozen = freeze_demo_candidate(
            candidate_id="cand-snmot097-640p",
            source_manifest={"source": "SNMOT-097"},
            detector_config={"min_size": 640, "max_size": 1280, "threshold": 0.5},
            dataset_release={"release_id": "rel-v2", "untouched_eval_preserved": True},
            group_outcomes={
                "level1_identity_visibility": {
                    "status": "cleared",
                    "reference_error": 0.082,
                    "usable_coverage": 0.8396,
                    "limitations": ["Evaluated on 1080p fixed/broadcast camera; severe 3+ player pileups remain contested."],
                },
                "level2_speed": {
                    "status": "partial",
                    "reference_error": 0.12,
                    "usable_coverage": 0.8396,
                    "limitations": ["Internal homography trajectory estimate; independent timing-gate validation pending."],
                },
            },
            frozen_results={"detected": 178, "recall": 0.8396, "precision": 0.9175},
            field_corrections_used_for_training=True,
            prior_release_id="rel-v1",
        )
        self.assertFalse(frozen["colab_session_required_for_replay"])

        repro = verify_candidate_reproduction(
            frozen,
            {"detected": 178, "recall": 0.83961, "precision": 0.91752},
            numeric_tolerance=1e-4,
            active_colab_session=False,
        )
        self.assertTrue(repro["reproduced_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
