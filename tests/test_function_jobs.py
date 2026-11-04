from pathlib import Path
import hashlib
import importlib.util
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from touchline.function_jobs import AnalysisAllowance,FILES,validate_payload,load_frozen_inputs

TOOLS=Path(__file__).resolve().parents[1]/'tools'
with patch.object(sys,'path',[str(TOOLS),*sys.path]):
    spec=importlib.util.spec_from_file_location('analysis_function_test',TOOLS/'modal_analysis_function.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    log_spec=importlib.util.spec_from_file_location('analysis_log_test',TOOLS/'fetch_modal_analysis_logs.py')
    log_module=importlib.util.module_from_spec(log_spec);log_spec.loader.exec_module(log_module)


class FunctionJobTests(unittest.TestCase):
    def fixture(self):
        files={n:b'fixed-reviewed-input' for n in FILES}
        hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
        return files,hashes

    def test_all_inputs_are_bound_and_extra_files_rejected(self):
        files,hashes=self.fixture();digest=validate_payload(files,hashes)
        self.assertEqual(len(digest),64)
        self.assertEqual(digest,validate_payload(dict(reversed(list(files.items()))),hashes))
        with self.assertRaises(ValueError):validate_payload({**files,'.env':b'fixture'},hashes)
        with self.assertRaises(ValueError):validate_payload({**files,'candidate.pt':b'changed'},hashes)
        with self.assertRaises(ValueError):validate_payload({**files,'candidate.pt':'not bytes'},hashes)

    def test_size_bound_precedes_paid_dispatch(self):
        files,hashes=self.fixture()
        with patch('touchline.function_jobs.MAX_INPUT_BYTES',1):
            with self.assertRaises(ValueError):validate_payload(files,hashes)

    def test_job_cannot_redirect_input_reads_to_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'runs/current';work.mkdir(parents=True)
            files,hashes=self.fixture()
            paths={'code.zip':work/'code.zip','candidate.pt':root/'artifacts/models/football-person/candidate.pt',
                'video24.mp4':root/'runs/video-trial-20261101/normalized/video24.mp4',
                'worker.py':root/'tools/modal_video_worker.py','check_cuda_tracking.py':root/'tools/check_cuda_tracking.py'}
            for name,path in paths.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(files[name])
            specification={'input_paths':{n:str(p) for n,p in paths.items()},'hashes':hashes}
            loaded,_=load_frozen_inputs(specification,root,work);self.assertEqual(loaded,files)
            specification['input_paths']['candidate.pt']=str(root/'not-approved.bin')
            with self.assertRaises(ValueError):load_frozen_inputs(specification,root,work)

    def test_one_dollar_claim_survives_restarts_and_refuses_new_run_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.sqlite3'
            result=AnalysisAllowance(path).reserve('first','a'*64)
            self.assertEqual(result['reserved_usd'],'1.00')
            with self.assertRaises(sqlite3.IntegrityError):AnalysisAllowance(path).reserve('second','b'*64)

    def test_explicit_retry_reuses_hold_once_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.sqlite3';allowance=AnalysisAllowance(path)
            with self.assertRaises(ValueError):allowance.reserve_retry('retry','b'*64,'c'*32)
            allowance.reserve('first','a'*64)
            retry=allowance.reserve_retry('retry','b'*64,'c'*32)
            self.assertEqual(retry['reserved_usd'],'1.00')
            with self.assertRaises(sqlite3.IntegrityError):AnalysisAllowance(path).reserve_retry('third','b'*64,'c'*32)

    def test_log_fetch_keeps_stages_and_categories_without_raw_secret_text(self):
        text=('private-fixture-token InternalFailure\n'
              'timestamp TOUCHLINE_EVENT {"stage":"cuda","elapsed_s":2.5}\n'
              'TOUCHLINE_EVENT {"stage":"export","elapsed_s":10,"secret":"private-fixture-token"}')
        report=log_module.summarize(text)
        self.assertEqual(report['flags'],{'internal_failure':1})
        self.assertEqual(report['events'],[{'stage':'cuda','elapsed_s':2.5}])
        self.assertNotIn('private-fixture-token',str(report))
        self.assertFalse(report['raw_retained'])

    def test_function_is_frozen_single_use_no_credentials_or_network(self):
        app=Mock();app.function.return_value=lambda f:f
        modal=SimpleNamespace(App=Mock(return_value=app),Image=SimpleNamespace(from_id=Mock(return_value='image')))
        _,worker=module.build_app(modal,'im-fixture','fixed-analysis')
        config=app.function.call_args.kwargs
        self.assertEqual(config['gpu'],'T4');self.assertEqual(config['timeout'],540)
        self.assertEqual(config['retries'],0);self.assertEqual(config['max_containers'],1)
        self.assertTrue(config['single_use_containers']);self.assertTrue(config['block_network'])
        self.assertTrue(config['restrict_modal_access']);self.assertFalse(config['include_source'])
        self.assertEqual(config['secrets'],[]);self.assertEqual(config['volumes'],{})
        self.assertEqual(worker({'unexpected':b''},{}),{'status':'failed','failure_code':'INPUT_INVALID'})
        with self.assertRaises(ValueError):module.build_app(modal,'mutable-tag','fixed-analysis')
