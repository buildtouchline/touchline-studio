from contextlib import closing, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from touchline.diagnostics import (DiagnosticJournal, DiagnosticUnavailable, current_trace_id,
    diagnosed, diagnostic_session, diagnostic_span, diagnostic_progress, diagnostic_summary,
    failure_details, local_doctor, record_failure)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def rows(self,path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]

    def test_nested_stages_share_trace_and_restore_context(self):
        with diagnostic_session(self.root/'diag') as journal:
            with diagnostic_span('pipeline'):
                with diagnostic_span('media.preflight'):
                    self.assertEqual(current_trace_id(),journal.trace_id)
                    diagnostic_progress('pipeline',decoded_frames=24,selected_frames=2)
        self.assertIsNone(current_trace_id())
        rows=self.rows(journal.path)
        self.assertEqual({r['trace_id'] for r in rows},{journal.trace_id})
        self.assertEqual(rows[1]['parent_span_id'],rows[0]['span_id'])
        self.assertEqual(diagnostic_summary(journal.path)['state'],'recorded')

    def test_exception_payload_and_arguments_never_persist(self):
        secret='fixture-only-private-token-123'
        @diagnosed('provider.baseten')
        def fail(**kwargs):raise RuntimeError('Authorization Bearer '+secret+' /private/person@example.invalid/file')
        with diagnostic_session(self.root/'diag') as journal:
            with self.assertRaises(RuntimeError):fail(api_key=secret,headers={'Authorization':secret})
        raw=journal.path.read_text(encoding='utf-8')
        for forbidden in (secret,'Authorization','person@example.invalid','/private/'):
            self.assertNotIn(forbidden,raw)
        self.assertIn('INTERNAL_ERROR',raw)
        self.assertEqual(diagnostic_summary(journal.path)['state'],'failed')

    def test_counter_and_schema_injection_denied(self):
        journal=DiagnosticJournal(self.root/'diag')
        for counters in ({'api_key':'fixture'},{'step':float('nan')},{'step':-1},{'step':True}):
            with self.assertRaises(ValueError):journal.event('pipeline','progress',counters=counters)
        with self.assertRaises(ValueError):journal.event('secret-stage','started')
        with self.assertRaises(ValueError):journal.event('pipeline','failed',failure={'code':'INPUT_INVALID','raw':'fixture'})

    def test_http_and_resource_failure_categories(self):
        class RemoteError(Exception):pass
        for status,expected in [(401,'AUTH_REJECTED'),(403,'AUTH_REJECTED'),(429,'RATE_LIMITED'),(503,'PROVIDER_UNAVAILABLE')]:
            exc=RemoteError('fixture body');exc.status_code=status
            self.assertEqual(failure_details(exc,'provider.meta')['code'],expected)
        self.assertEqual(failure_details(TimeoutError(),'provider.meta')['code'],'TIMEOUT')
        self.assertEqual(failure_details(RuntimeError('CUDA out of memory'),'detector.infer')['code'],'RESOURCE_EXHAUSTED')
        self.assertEqual(failure_details(ValueError('non-finite loss'),'training')['code'],'TRAINING_NONFINITE')
        self.assertEqual(failure_details(ValueError(),'budget.reserve')['code'],'BUDGET_REJECTED')

    def test_journal_limit_marks_summary_incomplete(self):
        journal=DiagnosticJournal(self.root/'diag',max_events=4)
        for i in range(20):journal.event('pipeline','progress',counters={'step':i})
        self.assertLessEqual(len(self.rows(journal.path)),4)
        summary=diagnostic_summary(journal.path)
        self.assertTrue(summary['truncated']);self.assertEqual(summary['state'],'incomplete')

    def test_unfinished_span_is_not_reported_as_complete(self):
        journal=DiagnosticJournal(self.root/'diag')
        journal.event('pipeline','started',span_id='a'*32)
        self.assertEqual(diagnostic_summary(journal.path)['state'],'incomplete')

    def test_report_rejects_credential_names_before_opening(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('Must not open')):
            for name in ('.env','.env.example','.modal.toml','some-other-log.jsonl'):
                with self.assertRaises(ValueError):diagnostic_summary(self.root/name)

    def test_summary_does_not_forward_unknown_text(self):
        journal=DiagnosticJournal(self.root/'diag');journal.event('doctor','completed')
        row=self.rows(journal.path)[0];row['untrusted_note']='fixture-private'
        journal.path.write_text(json.dumps(row)+'\n',encoding='utf-8')
        self.assertNotIn('fixture-private',json.dumps(diagnostic_summary(journal.path)))

    def test_session_context_is_restored_after_failure(self):
        with self.assertRaises(ValueError):
            with diagnostic_session(self.root/'diag'):
                with diagnostic_span('data.etl'):raise ValueError('fixture')
        self.assertIsNone(current_trace_id())

    def test_cancellation_recorded(self):
        with diagnostic_session(self.root/'diag') as journal:
            with self.assertRaises(KeyboardInterrupt):
                with diagnostic_span('pipeline'):raise KeyboardInterrupt()
        self.assertIn('cancelled',[r['event'] for r in self.rows(journal.path)])
        self.assertEqual(diagnostic_summary(journal.path)['state'],'failed')

    def test_failure_only_inference_avoids_per_frame_success_logs(self):
        @diagnosed('detector.infer',failures_only=True)
        def infer():return 1
        with diagnostic_session(self.root/'diag') as journal:
            with diagnostic_span('pipeline'):
                for i in range(100):infer()
        self.assertEqual(len(self.rows(journal.path)),2)

    def test_doctor_does_not_load_secrets_or_provider_sdks(self):
        with patch('touchline.diagnostics.importlib.util.find_spec',return_value=None),patch('pathlib.Path.read_text',side_effect=AssertionError('No file reads')):
            report=local_doctor()
        self.assertEqual(report['credentials'],'not inspected')
        self.assertEqual(report['paid_calls'],0)
        self.assertEqual(report['env_loader'],'explicit operator-only; not invoked by doctor')

    def test_cli_doctor_and_argument_errors_are_safe(self):
        from touchline.cli import main
        with diagnostic_session(self.root/'diag') as journal,redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['doctor']),0)
        self.assertEqual(json.loads(output.getvalue())['trace_id'],journal.trace_id)
        with diagnostic_session(self.root/'diag'),redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit):main(['--fixture-private-token'])
        self.assertNotIn('fixture-private-token',error.getvalue())
        self.assertIn('Trace:',error.getvalue())

    def test_pipeline_failure_artifact_has_safe_diagnostic_reference(self):
        from touchline.pipeline import run_pipeline
        from touchline.fixtures import synthetic_frames
        from touchline.video.sampling import Anchor24Sampler
        class BrokenDetector:
            def provenance(self):return {'adapter':'fixture'}
            def detect(self,frame):raise RuntimeError('fixture-private-key /private/path')
        with diagnostic_session(self.root/'diag') as journal:
            with self.assertRaises(RuntimeError):run_pipeline(synthetic_frames(),BrokenDetector(),Anchor24Sampler(),self.root/'run',{'id':'fixture'},synthetic=True)
        manifest=json.loads((self.root/'run'/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['diagnostic_trace_id'],journal.trace_id)
        self.assertEqual(manifest['error']['trace_id'],journal.trace_id)
        self.assertNotIn('fixture-private-key',json.dumps(manifest['error']))
        self.assertEqual(manifest['status'],'failed')

    def test_budget_overrun_survives_diagnostic_write_failure(self):
        from touchline.spending import SpendingLedger
        with diagnostic_session(self.root/'diag'):
            path=self.root/'budget.sqlite3';ledger=SpendingLedger(path)
            ledger.verify_pool('meta',prior_spend_usd=0,available_credit_usd=150)
            ledger.reserve('meta','job',fingerprint='fixture',worst_case_usd=1)
            with patch('touchline.spending.diagnostic_issue',side_effect=DiagnosticUnavailable('write failed')):
                with self.assertRaises(DiagnosticUnavailable):ledger.reconcile('job',actual_usd=2,finalized=True)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT blocked FROM pools WHERE provider='meta'").fetchone()[0],1)
                self.assertEqual(db.execute("SELECT state FROM reservations WHERE id='job'").fetchone()[0],'overrun')

    def test_duplicate_trace_sequence_rejected(self):
        journal=DiagnosticJournal(self.root/'diag');journal.event('doctor','completed')
        text=journal.path.read_text(encoding='utf-8');journal.path.write_text(text+text,encoding='utf-8')
        with self.assertRaises(ValueError):diagnostic_summary(journal.path)
