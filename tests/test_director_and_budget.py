import unittest

from touchline.director import (
    CreditBudgetLedger,
    apply_coach_evidence_decision,
    validate_and_execute_evidence_plan,
)


class CreditBudgetAndDirectorTests(unittest.TestCase):
    def test_tl025_separate_credit_pools_and_stage_caps(self):
        ledger = CreditBudgetLedger()

        # Pending Baseten cannot be verified or reserved
        with self.assertRaises(ValueError):
            ledger.verify_pool_balance("baseten_pending", 100.0)

        # Unverified pool cannot reserve a job
        with self.assertRaises(ValueError):
            ledger.reserve_job(
                reservation_id="res-1",
                pool_name="modal_compute",
                stage="torchvision_finetune_and_eval",
                owner="stream_A",
                workload_type="gpu_finetune",
                unit_rate_usd=1.02,
                bounded_quantity=2.0,
                retries=1,
                dataset_or_input_hash="sha256:abc",
                cancellation_path="modal app stop",
            )

        ledger.verify_pool_balance("modal_compute", 100.0, eligible=True)
        ledger.verify_pool_balance("meta_model_grant", 150.0, eligible=True)

        # Local SAM 3 cannot be charged to meta_model_grant API pool
        with self.assertRaises(ValueError):
            ledger.reserve_job(
                reservation_id="res-sam",
                pool_name="meta_model_grant",
                stage="curated_model_assisted_audit",
                owner="stream_A",
                workload_type="local_sam3",
                unit_rate_usd=1.0,
                bounded_quantity=2.0,
                dataset_or_input_hash="sha256:def",
                cancellation_path="local kill",
            )

        # Reserve valid Modal GPU job (1 + 1 retry = 2 attempts * $5 = $10 <= $25 cap)
        res = ledger.reserve_job(
            reservation_id="res-modal-1",
            pool_name="modal_compute",
            stage="torchvision_finetune_and_eval",
            owner="stream_A",
            workload_type="gpu_finetune",
            unit_rate_usd=2.5,
            bounded_quantity=2.0,
            retries=1,
            dataset_or_input_hash="sha256:abc",
            cancellation_path="modal app stop",
        )
        self.assertEqual(res["worst_case_usd"], 10.0)
        self.assertEqual(res["status"], "reserved")

        # Concurrent second GPU reservation on modal_compute is blocked while first is active
        with self.assertRaises(ValueError):
            ledger.reserve_job(
                reservation_id="res-modal-2",
                pool_name="modal_compute",
                stage="field_pilot_batch_processing",
                owner="stream_B",
                workload_type="batch_infer",
                unit_rate_usd=1.0,
                bounded_quantity=1.0,
                dataset_or_input_hash="sha256:123",
                cancellation_path="modal app stop",
            )

        # Reconcile first job as finalized ($4.20 actual)
        rec = ledger.reconcile_job("res-modal-1", measured_usd=4.20, billing_finalized=True)
        self.assertEqual(rec["status"], "reconciled")
        self.assertEqual(rec["effective_charge_usd"], 4.20)

    def test_tl026_evidence_director_validation_bounded_calls_and_coach_authority(self):
        ctx = {
            "run_id": "run-snmot-097",
            "source_id": "SNMOT-097",
            "valid_time_bounds_s": (0.0, 2.0),
            "available_frames_by_pts": {6: {}, 10: {}, 14: {}, 18: {}},
            "max_billable_units_budget": 10,
            "local_track_id": 1,
            "current_association_revision": 1,
            "ambiguous_interval_s": (0.24, 0.72),
            "ambiguous_pts_list": [6, 10, 14, 18],
        }

        # Reject unsupported action or invented PTS frame
        with self.assertRaises(ValueError):
            validate_and_execute_evidence_plan(
                {
                    "run_id": "run-snmot-097",
                    "source_id": "SNMOT-097",
                    "action": "auto_confirm_identity",
                    "interval_s": [0.24, 0.72],
                    "requested_pts": [10, 14],
                    "max_billable_units": 4,
                    "supporting_evidence": ["a"],
                    "counter_evidence": ["b"],
                },
                run_context=ctx,
            )

        with self.assertRaises(ValueError):
            validate_and_execute_evidence_plan(
                {
                    "run_id": "run-snmot-097",
                    "source_id": "SNMOT-097",
                    "action": "inspect_frames",
                    "interval_s": [0.24, 0.72],
                    "requested_pts": [10, 999],  # PTS 999 does not exist
                    "max_billable_units": 4,
                    "supporting_evidence": ["a"],
                    "counter_evidence": ["b"],
                },
                run_context=ctx,
            )

        # Valid bounded plan executes without modifying identity
        valid_plan = {
            "run_id": "run-snmot-097",
            "source_id": "SNMOT-097",
            "action": "inspect_frames",
            "interval_s": [0.24, 0.72],
            "requested_pts": [10, 14],
            "max_billable_units": 4,
            "supporting_evidence": ["Jersey colour and velocity vector match Track #1 prior to PTS 6"],
            "counter_evidence": ["High IoU overlap (0.62) with Track #8 at PTS 10"],
            "model_suggestion": "Maintain Track #1 -> P07 after crossing",
        }
        card = validate_and_execute_evidence_plan(valid_plan, run_context=ctx, call_history=[])
        self.assertEqual(card["mode"], "bounded_model_plan")
        self.assertFalse(card["identity_modified"])
        self.assertEqual(card["association_revision"], 1)

        # Exceeding 1 planning + 1 refinement call raises ValueError
        with self.assertRaises(ValueError):
            validate_and_execute_evidence_plan(
                valid_plan,
                run_context=ctx,
                call_history=[{"phase": "planning"}, {"phase": "refinement"}],
            )

        # Endpoint failure falls back deterministically
        fallback_card = validate_and_execute_evidence_plan(
            None,
            run_context=ctx,
            endpoint_available=False,
        )
        self.assertEqual(fallback_card["mode"], "deterministic_rule_fallback")
        self.assertTrue(fallback_card["replay_labelled"])
        self.assertFalse(fallback_card["identity_modified"])

        # Coach reject leaves revision unchanged; coach accept increments revision
        rejected = apply_coach_evidence_decision(card, decision="reject", reviewer="coach_1")
        self.assertFalse(rejected["identity_modified"])
        self.assertEqual(rejected["association_revision"], 1)

        accepted = apply_coach_evidence_decision(
            card,
            decision="accept",
            reviewer="coach_1",
            confirmed_player_id="P07",
        )
        self.assertTrue(accepted["identity_modified"])
        self.assertEqual(accepted["association_revision"], 2)


if __name__ == "__main__":
    unittest.main()
