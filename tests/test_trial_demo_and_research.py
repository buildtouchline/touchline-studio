import unittest

from touchline.trial import (
    align_android_coach_audio,
    build_labelled_3d_replay_metadata,
    compile_coaching_objective,
    evaluate_field_trial_summary,
    evaluate_learned_temporal_selector,
    evaluate_sam3_smoke_benchmark,
    evaluate_specialist_head_experiment,
    verify_hackathon_demo_package,
)


class TrialDemoAndResearchTests(unittest.TestCase):
    def test_tl011_field_trial_summary_honest_level2_counts_and_permissions(self):
        # Missing internal permission must fail
        with self.assertRaises(ValueError):
            evaluate_field_trial_summary(
                trial_id="trial-01",
                level1_usable=True,
                coach_audio_aligned=True,
                preflight_checks={"access": True, "recording": True, "calibration": True, "target_identity": True, "disk": True, "hardware": True},
                level2_group_statuses={
                    "level2_distance": "cleared",
                    "level2_speed": "cleared",
                    "level2_intensity_zones": "partial",
                    "level2_activity": "not_cleared",
                },
                gaps_recorded=[{"start_s": 4.0, "end_s": 5.5}],
                permissions={"internal_analysis_permitted": False, "public_demo_permitted": False},
            )

        res = evaluate_field_trial_summary(
            trial_id="trial-01",
            level1_usable=True,
            coach_audio_aligned=True,
            preflight_checks={"access": True, "recording": True, "calibration": True, "target_identity": True, "disk": True, "hardware": True},
            level2_group_statuses={
                "level2_distance": "cleared",
                "level2_speed": "cleared",
                "level2_intensity_zones": "partial",
                "level2_activity": "not_cleared",
            },
            gaps_recorded=[{"start_s": 4.0, "end_s": 5.5}],
            permissions={"internal_analysis_permitted": True, "public_demo_permitted": False},
        )
        self.assertTrue(res["preflight_all_logged_and_passed"])
        self.assertEqual(res["level2_cleared_count"], 2)
        self.assertFalse(res["above_half_cleared"])  # fewer than 4 passed L2 groups is not called above-half cleared
        self.assertEqual(res["trial_status"], "completed_with_gaps")
        self.assertFalse(res["permissions"]["public_demo_permitted"])

    def test_tl028_android_coach_audio_clap_alignment_and_tolerance(self):
        meta = {
            "session_uuid": "sess-android-001",
            "monotonic_start_ns": 1000000000,
            "monotonic_stop_ns": 61000000000,
            "audio_sha256": "a" * 64,
            "actual_sample_rate_hz": 48000,
            "codec": "aac_lc",
            "container": "m4a",
            "permission_granted": True,
            "discontinuities": [{"at_ms": 30000, "reason": "notification_interruption"}],
        }
        # Wall-clock only sync is rejected
        with self.assertRaises(ValueError):
            align_android_coach_audio(
                {**meta, "sync_method": "wall_clock_only"},
                [],
                [1.0, 50.0],
            )

        markers = [
            {"marker_id": "m1", "marker_type": "clap_sync", "media_offset_ms": 2000, "uncertainty_ms": 20},
            {"marker_id": "m2", "marker_type": "clap_sync", "media_offset_ms": 52000, "uncertainty_ms": 25},
        ]
        aligned = align_android_coach_audio(
            meta,
            markers,
            video_clap_timestamps_s=[3.5, 53.55],
            tolerance_s=0.05,
        )
        self.assertEqual(aligned["alignment_status"], "aligned")
        self.assertAlmostEqual(aligned["slope_a"], 1.001, places=4)
        self.assertEqual(aligned["discontinuities_preserved"], 1)

        # Exceeding tolerance returns unknown visibly
        bad_markers = [
            {"marker_id": "m1", "marker_type": "clap_sync", "media_offset_ms": 2000, "uncertainty_ms": 180},
            {"marker_id": "m2", "marker_type": "clap_sync", "media_offset_ms": 52000, "uncertainty_ms": 180},
        ]
        unk = align_android_coach_audio(meta, bad_markers, [3.5, 53.55], tolerance_s=0.05)
        self.assertEqual(unk["alignment_status"], "unknown")
        self.assertIsNone(unk["slope_a"])

    def test_tl027_sam3_smoke_benchmark_budget_and_no_go_isolation(self):
        # Exceeding $5 cap or claiming LoRA fails
        with self.assertRaises(ValueError):
            evaluate_sam3_smoke_benchmark(
                projected_or_measured_cost_usd=6.50,
                elapsed_wall_seconds=120.0,
                step_time_s=1.2,
                peak_vram_gb=14.2,
            )
        with self.assertRaises(ValueError):
            evaluate_sam3_smoke_benchmark(
                projected_or_measured_cost_usd=2.10,
                elapsed_wall_seconds=120.0,
                step_time_s=1.2,
                peak_vram_gb=14.2,
                claimed_lora_or_quantized=True,
            )

        oom_res = evaluate_sam3_smoke_benchmark(
            projected_or_measured_cost_usd=1.85,
            elapsed_wall_seconds=95.0,
            step_time_s=None,
            peak_vram_gb=15.9,
            oom_or_failed=True,
            failure_reason="CUDA OOM on high-res mask decoder backward pass",
        )
        self.assertEqual(oom_res["status"], "no_go_documented")
        self.assertFalse(oom_res["blocks_torchvision_field_pipeline"])

    def test_tl022_hackathon_demo_package_verification(self):
        frozen_res = {"recall": 0.8396, "precision": 0.9175, "distance_m": 13.27}
        claims = [
            {
                "claim_id": "c1",
                "source_type": "frozen_result",
                "metric_key": "recall",
                "displayed_value": 0.8396,
                "is_sped_up": True,
                "sped_up_labelled": True,
            },
            {
                "claim_id": "c2",
                "source_type": "frozen_result",
                "metric_key": "distance_m",
                "displayed_value": 13.27,
                "is_experimental": True,
                "experimental_labelled": True,
            },
            {
                "claim_id": "c3",
                "source_type": "approved_annotation",
                "annotation_id": "ann-p07-1",
            },
        ]
        # Single-session claiming durable development must fail
        with self.assertRaises(ValueError):
            verify_hackathon_demo_package(
                storyboard_claims=[{**claims[0], "claims_durable_development": True}],
                frozen_candidate_results=frozen_res,
                approved_annotation_ids=["ann-p07-1"],
                video_metadata={"sessions_count": 1, "overlays_legible": True, "captions_included": True},
                permissions={"public_demo_permitted": True},
            )

        pkg = verify_hackathon_demo_package(
            storyboard_claims=claims,
            frozen_candidate_results=frozen_res,
            approved_annotation_ids=["ann-p07-1"],
            video_metadata={
                "sessions_count": 1,
                "overlays_legible": True,
                "captions_included": True,
                "local_backup_path": "C:/backups/demo_day8.mp4",
            },
            permissions={"public_demo_permitted": True},
        )
        self.assertEqual(pkg["status"], "verified_ready_for_local_delivery")
        self.assertFalse(pkg["public_upload_triggered"])

    def test_tl017_to_tl020_research_guardrails(self):
        # TL-017: Specialist head no-go outcome when not beating baseline
        spec_res = evaluate_specialist_head_experiment(
            task_id="pass_completion_head",
            encoder_namespace="dinov2::crop::v1",
            head_version="mlp_v1",
            team_permissions_verified=True,
            evaluated_against_independent_held_out_labels=True,
            specialist_macro_f1=0.71,
            baseline_macro_f1=0.74,
            resource_overhead_ratio=1.3,
        )
        self.assertEqual(spec_res["outcome"], "no_go_research_outcome")
        self.assertFalse(spec_res["feature_marked_implemented"])
        self.assertFalse(spec_res["online_update_permitted"])

        # TL-018: Declarative objective compiler flags unsupported scanning/technique prerequisites
        compiled = compile_coaching_objective(
            "Check if P07 scans over shoulder twice before receiving under pressure",
            {
                "objective_id": "OBJ-SCAN-01",
                "requested_capabilities": ["person_detection", "head_scanning_pose", "manual_attempt_review"],
            },
        )
        self.assertEqual(compiled["missing_prerequisites"], ["head_scanning_pose"])
        self.assertFalse(compiled["can_auto_recognize_skill"])
        self.assertTrue(compiled["human_review_required"])

        # TL-019: 3D body replay labels estimated geometry and rejects measured joint angle claims
        with self.assertRaises(ValueError):
            build_labelled_3d_replay_metadata(
                source_id="src_1",
                pts_interval=(100, 200),
                camera_distance_m=12.0,
                occlusion_ratio=0.1,
                claimed_measured_joint_angles=True,
            )
        replay = build_labelled_3d_replay_metadata(
            source_id="src_1",
            pts_interval=(100, 200),
            camera_distance_m=32.0,
            occlusion_ratio=0.45,
        )
        self.assertEqual(replay["geometry_status"], "estimated_reconstruction")
        self.assertEqual(replay["decision"], "no_go_unreliable_input")

        # TL-020: Learned temporal selector blocked when extra compute doesn't beat cheap scout
        ablation = evaluate_learned_temporal_selector(
            namespace="vjepa::tubelet::v1",
            threshold_frozen_before_eval=True,
            learned_recall=0.85,
            learned_false_trigger_rate=0.18,
            learned_total_cost_s=28.0,
            cheap_scout_recall=0.84,
            cheap_scout_false_trigger_rate=0.15,
            cheap_scout_total_cost_s=14.0,
        )
        self.assertFalse(ablation["promote_learned_selector"])
        self.assertEqual(ablation["decision"], "keep_cheap_motion_scout")


if __name__ == "__main__":
    unittest.main()
