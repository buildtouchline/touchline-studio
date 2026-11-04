from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from touchline.analysis_policy import preflight_analysis_video, validate_analysis_frames
from touchline.contracts import Frame
from touchline.spending import SpendingLedger, units


def frames(width=1920, height=1080, rate=24, count=6, origin=120):
    return [Frame('policy-fixture',i,origin+i,1,rate,i/rate,width,height) for i in range(count)]


class AnalysisPolicyTests(unittest.TestCase):
    def test_accept_exact_1080p24_nonzero_pts(self):
        self.assertEqual(validate_analysis_frames(frames())['frames_verified'],6)

    def test_reject_other_cadences_accept_landscape_resolutions(self):
        for rate in (25,30,60):
            with self.assertRaises(ValueError):validate_analysis_frames(frames(rate=rate,count=60))
        for width,height in ((1280,720),(4096,1080)):
            self.assertEqual(validate_analysis_frames(frames(width=width,height=height))['width'],width)
        with self.assertRaises(ValueError):validate_analysis_frames(frames(width=720,height=1280))

    def test_reject_geometry_changes(self):
        with self.assertRaises(ValueError):
            validate_analysis_frames([*frames(count=3),frames(width=4096)[3]])

    def test_reject_gap_vfr_and_empty(self):
        f=frames()
        with self.assertRaises(ValueError):validate_analysis_frames([f[0],*f[2:]])
        with self.assertRaises(ValueError):validate_analysis_frames([])
        with self.assertRaises(ValueError):validate_analysis_frames(f[:1])
        f=[Frame('vfr',i,p,1,2400,p/2400,1920,1080) for i,p in enumerate((0,100,205,300))]
        with self.assertRaises(ValueError):validate_analysis_frames(f)

    def test_rounding_quantization_not_drift(self):
        f=[Frame('quantized',i,round(i*1000/24),1,1000,i/24,1920,1080) for i in range(100)]
        validate_analysis_frames(f)

    def test_preflight_rejects_23976_before_decode(self):
        info={'width':1920,'height':1080,'is_rotated':False,'average_fps':float(Fraction(24000,1001))}
        with patch('touchline.analysis_policy.probe_video',return_value=info),patch('touchline.analysis_policy.decode_video') as decode:
            with self.assertRaises(ValueError):preflight_analysis_video('fixture.mp4')
            decode.assert_not_called()

    def test_full_source_validated_despite_claimed_metadata(self):
        info={'width':1920,'height':1080,'is_rotated':False,'average_fps':24.0}
        f=frames(count=120);f[-1]=Frame('policy-fixture',119,999,1,24,5,1920,1080)
        with patch('touchline.analysis_policy.probe_video',return_value=info),patch('touchline.analysis_policy.decode_video',return_value=iter(f)):
            with self.assertRaises(ValueError):preflight_analysis_video('fixture.mp4')

    def test_actual_encoded_media_preflight(self):
        from test_decode import create_test_mp4
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.mp4'
            create_test_mp4(p,frame_count=4,width=1920,height=1080,fps=24)
            self.assertTrue(preflight_analysis_video(p)['timing_verified'])
            create_test_mp4(p,frame_count=4,width=1920,height=1080,fps=30)
            with self.assertRaises(ValueError):preflight_analysis_video(p)

    def test_cli_rejects_input_before_constructing_detector(self):
        from touchline.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.mp4';p.write_bytes(b'invalid fixture')
            with patch('touchline.analysis_policy.preflight_analysis_video',side_effect=ValueError('wrong cadence')),patch('touchline.adapters.torchvision.TorchvisionDetector') as detector:
                with self.assertRaises(SystemExit):main(['analyze',str(p),'--output',str(Path(tmp)/'run')])
                detector.assert_not_called()


class SpendingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'usage.sqlite3'
        self.ledger=SpendingLedger(self.path)
    def tearDown(self):self.tmp.cleanup()
    def verify(self,provider='meta',prior='0',credit='150'):
        self.ledger.verify_pool(provider,prior_spend_usd=prior,available_credit_usd=credit)
    def reserve(self,id='one',cost='1',provider='meta'):
        return self.ledger.reserve(provider,id,fingerprint='source-config-model-hash',worst_case_usd=cost)

    def test_no_unverified_launch_and_half_cap(self):
        with self.assertRaises(ValueError):self.reserve()
        self.verify();self.reserve(cost='75')
        with self.assertRaises(ValueError):self.reserve('two',cost='0.000001')

    def test_four_independent_pools_and_existing_spend(self):
        for provider,cap in [('modal','50'),('meta','75'),('baseten','50'),('fireworks','50')]:
            self.verify(provider,prior=cap,credit='100')
            with self.assertRaises(ValueError):self.reserve(provider,provider=provider)

    def test_credit_smaller_than_budget(self):
        self.verify(credit='0.75')
        with self.assertRaises(ValueError):self.reserve(cost='1')

    def test_restart_and_idempotency_not_retry_permission(self):
        self.verify();self.assertTrue(self.reserve()['created'])
        restarted=SpendingLedger(self.path)
        duplicate=restarted.reserve('meta','one',fingerprint='source-config-model-hash',worst_case_usd='1')
        self.assertFalse(duplicate['created'])
        with self.assertRaises(ValueError):self.reserve(cost='2')
        with self.assertRaises(ValueError):self.verify()

    def test_pending_retains_reservation_until_final_charge(self):
        self.verify();self.reserve(cost='75')
        self.ledger.reconcile('one',actual_usd='1',finalized=False)
        with self.assertRaises(ValueError):self.reserve('two')
        self.ledger.reconcile('one',actual_usd='1',finalized=True)
        self.reserve('two',cost='74')

    def test_unknown_charge_cannot_release_or_rewrite_final(self):
        self.verify();self.reserve()
        with self.assertRaises(ValueError):self.ledger.reconcile('one',finalized=True)
        self.ledger.reconcile('one',actual_usd='0.5',finalized=True)
        with self.assertRaises(ValueError):self.ledger.reconcile('one',actual_usd='0',finalized=True)

    def test_overrun_persisted_and_blocks_even_after_restart(self):
        self.verify();self.reserve()
        self.ledger.reconcile('one',actual_usd='2',finalized=True)
        with self.assertRaises(ValueError):self.reserve('two')
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT state,charge FROM reservations WHERE id='one'").fetchone(),('overrun',2000000))
        restarted=SpendingLedger(self.path)
        with self.assertRaises(ValueError):restarted.reserve('meta','three',fingerprint='x',worst_case_usd='1')

    def test_atomic_parallel_reservations_cannot_double_spend(self):
        self.verify(credit='1')
        def attempt(n):
            try:
                SpendingLedger(self.path).reserve('meta',str(n),fingerprint='x',worst_case_usd='0.6')
                return True
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(attempt,range(2))),1)

    def test_invalid_money_and_submicro_rounding(self):
        for value in (float('nan'),float('inf'),'-1',True,None,'garbage'):
            with self.assertRaises(ValueError):units(value)
        self.assertEqual(units('0.00000001'),1)

    def test_legacy_modal_budget_cannot_exceed_half_pool(self):
        from touchline.director import CreditBudgetLedger
        old=CreditBudgetLedger();old.verify_pool_balance('modal_compute',100)
        def reserve(id,stage,cost):
            return old.reserve_job(reservation_id=id,pool_name='modal_compute',stage=stage,owner='test',workload_type='gpu',unit_rate_usd=cost,bounded_quantity=1,dataset_or_input_hash='hash',cancellation_path='stop')
        reserve('first','torchvision_finetune_and_eval',25)
        old.reconcile_job('first',measured_usd=25,billing_finalized=True)
        reserve('second','field_pilot_batch_processing',20)
        old.reconcile_job('second',measured_usd=20,billing_finalized=True)
        with self.assertRaises(ValueError):reserve('third','compatibility_and_sam3_fallback',6)
        with self.assertRaises(ValueError):reserve('first','compatibility_and_sam3_fallback',1)
        with self.assertRaises(ValueError):reserve('nan','compatibility_and_sam3_fallback',float('nan'))
