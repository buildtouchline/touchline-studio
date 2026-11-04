import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

TOOLS=Path(__file__).resolve().parents[1]/'tools'
with patch.object(sys,'path',[str(TOOLS),*sys.path]):
    spec=importlib.util.spec_from_file_location('touchline_function_probe_test',TOOLS/'probe_modal_function.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class FunctionProbeTests(unittest.TestCase):
    def fixture(self):
        app=Mock();app.function.return_value=lambda f:f
        modal=SimpleNamespace(App=Mock(return_value=app),Image=SimpleNamespace(from_id=Mock(return_value='image')))
        result,worker=probe.build_probe(modal,'im-fixture','test-probe')
        return modal,app,worker

    def test_probe_has_no_source_secrets_mounts_network_or_retry(self):
        modal,app,_=self.fixture()
        modal.App.assert_called_once_with('test-probe',include_source=False)
        config=app.function.call_args.kwargs
        self.assertEqual(config['gpu'],'T4');self.assertEqual(config['retries'],0)
        self.assertEqual(config['max_containers'],1);self.assertEqual(config['min_containers'],0)
        self.assertEqual(config['timeout'],30);self.assertEqual(config['startup_timeout'],30)
        self.assertTrue(config['block_network']);self.assertTrue(config['restrict_modal_access'])
        self.assertTrue(config['single_use_containers']);self.assertEqual(config['secrets'],[])
        self.assertEqual(config['volumes'],{});self.assertFalse(config['include_source'])

    def test_worker_returns_only_validated_gpu_metadata(self):
        _,_,worker=self.fixture()
        with patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='Tesla T4, 15360 MiB, 570.124.06')) as command:
            self.assertEqual(worker()['status'],'passed')
            self.assertEqual(command.call_args.kwargs['timeout'],10)
        with patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='fixture-private-token')):
            self.assertEqual(worker(),{'status':'invalid_output'})

    def test_duplicate_and_unapproved_probes_never_load_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'runs/probe';work.mkdir(parents=True)
            (work/'gpu-function-main-t4.json').write_text('{}',encoding='utf-8')
            with patch.object(probe,'ROOT',root),patch.object(probe,'modal_client') as client:
                with self.assertRaises(ValueError):probe.run(work,approved=False)
                with self.assertRaises(FileExistsError):probe.run(work,approved=True)
                client.assert_not_called()

    def test_post_payment_retry_needs_reconciled_prior_failure_and_is_single_use(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'runs/probe';work.mkdir(parents=True)
            original=work/'gpu-function-main-t4.json'
            original.write_text(json.dumps({'status':'failed','app_stop':'unknown','flags':['billing_mentioned']}),encoding='utf-8')
            with patch.object(probe,'ROOT',root),patch.object(probe,'modal_client') as client:
                with self.assertRaises(ValueError):probe.run(work,approved=True,after_payment=True)
                original.write_text(json.dumps({'status':'failed','app_stop':'confirmed','flags':['billing_mentioned'],'run_id':'previous'}),encoding='utf-8')
                (work/'gpu-function-main-t4-after-payment.json').write_text('{}',encoding='utf-8')
                with self.assertRaises(FileExistsError):probe.run(work,approved=True,after_payment=True)
                client.assert_not_called()
