import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from touchline.cpu_sandbox_jobs import CpuSandboxAllowance,sandbox_options,extract_results

TOOLS=Path(__file__).resolve().parents[1]/'tools'
with patch.object(sys,'path',[str(TOOLS),*sys.path]):
    spec=importlib.util.spec_from_file_location('cpu_sandbox_controller_test',TOOLS/'run_modal_cpu_analysis.py')
    cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)


class CpuSandboxTests(unittest.TestCase):
    def test_cloud_cpu_resources_have_no_gpu_credentials_mounts_or_network(self):
        options=sandbox_options()
        self.assertIsNone(options['gpu']);self.assertEqual(options['cpu'],(4,4))
        self.assertEqual(options['memory'],(8192,8192));self.assertEqual(options['timeout'],7200)
        self.assertTrue(options['block_network']);self.assertEqual(options['secrets'],[])
        self.assertEqual(options['volumes'],{})

    def test_ten_dollar_hold_is_single_use_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'allowance.sqlite3'
            result=CpuSandboxAllowance(p).reserve('first','a'*64)
            self.assertEqual(result['reserved_usd'],'10.00');self.assertIsNone(result['gpu'])
            with self.assertRaises(sqlite3.IntegrityError):CpuSandboxAllowance(p).reserve('second','b'*64)

    def test_worker_event_rejects_unexpected_payload_and_gpu_stages(self):
        item={'schema':'touchline.cpu-worker-event.v1','stage':'cpu_analysis','status':'progress','elapsed_s':20,'coverage_rows':48}
        self.assertEqual(cpu.clean_event(item),item)
        for extra in ({'raw_error':'private-fixture-token'},{'stage':'cuda'},{'coverage_rows':float('nan')}):
            with self.assertRaises(ValueError):cpu.clean_event({**item,**extra})

    def test_remote_checks_share_cpu_hold_and_cannot_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.sqlite3';budget=CpuSandboxAllowance(path)
            with self.assertRaises(ValueError):budget.reserve_remote_checks('checks','b'*64)
            with self.assertRaises(ValueError):budget.reserve_remote_checks_retry('retry','c'*64)
            budget.reserve('analysis','a'*64)
            record=budget.reserve_remote_checks('checks','b'*64)
            self.assertTrue(record['shares_original_hold'])
            with self.assertRaises(sqlite3.IntegrityError):CpuSandboxAllowance(path).reserve_remote_checks('again','c'*64)
            self.assertTrue(budget.reserve_remote_checks_retry('retry','c'*64)['shares_original_hold'])
            with self.assertRaises(sqlite3.IntegrityError):CpuSandboxAllowance(path).reserve_remote_checks_retry('third','d'*64)

    def test_results_cannot_escape_overwrite_or_include_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'result.zip'
            for name in ('../escape.json','run/../../escape.json','run/candidate.pt'):
                with zipfile.ZipFile(p,'w') as z:z.writestr(name,'{}')
                with self.assertRaises(ValueError):extract_results(p,root/'output')
            with zipfile.ZipFile(p,'w') as z:z.writestr('run/manifest.json','{}')
            extract_results(p,root/'output')
            with self.assertRaises(ValueError):extract_results(p,root/'output')

    def test_tracking_trial_shares_hold_and_is_single_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'allowance.sqlite3';budget=CpuSandboxAllowance(p)
            with self.assertRaises(ValueError):budget.reserve_tracking_trial('trial','b'*64)
            budget.reserve('parent','a'*64)
            self.assertTrue(budget.reserve_tracking_trial('trial','b'*64)['shares_original_hold'])
            with self.assertRaises(sqlite3.IntegrityError):CpuSandboxAllowance(p).reserve_tracking_trial('repeat','c'*64)

    def test_event_checks_share_hold_and_are_single_use(self):
        for method in ('reserve_event_checks','reserve_final_checks','reserve_handover_checks'):
            with self.subTest(method=method),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'budget.sqlite3';budget=CpuSandboxAllowance(p)
                with self.assertRaises(ValueError):getattr(budget,method)('checks','b'*64)
                budget.reserve('first','a'*64)
                self.assertTrue(getattr(budget,method)('checks','b'*64)['shares_original_hold'])
                with self.assertRaises(sqlite3.IntegrityError):getattr(CpuSandboxAllowance(p),method)('repeat','c'*64)
