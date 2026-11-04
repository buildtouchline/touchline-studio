from contextlib import redirect_stdout
import io,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from touchline.run_records import new_record,write_record,export_run_logs
from touchline.training.runner import train_detector
from touchline.diagnostics import diagnostic_session,diagnostic_span


class RunRecordTests(unittest.TestCase):
    def test_training_run_identity_and_parent_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'training'
            with patch('touchline.training.runner._train_detector',return_value={'status':'completed','checkpoint_sha256':'a'*64}) as implementation:
                result=train_detector('fixture.json',output,allow_download=True,run_id='run-1',parent_run_id='experiment-1')
            record=json.loads((output/'run-record.json').read_text(encoding='utf-8'))
            self.assertEqual(record['run_id'],'run-1');self.assertEqual(record['parent_run_id'],'experiment-1')
            self.assertEqual(record['backend'],'python');self.assertEqual(result['run_id'],'run-1')
            self.assertEqual(record['status'],'completed')
            self.assertEqual(implementation.call_args.kwargs['run_record']['run_id'],'run-1')

    def test_failed_training_has_identity_without_raw_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'training'
            with patch('touchline.training.runner._train_detector',side_effect=RuntimeError('fixture-private-token')):
                with self.assertRaises(RuntimeError):train_detector('fixture.json',output,allow_download=True)
            raw=(output/'run-record.json').read_text(encoding='utf-8')
            self.assertNotIn('fixture-private-token',raw);self.assertEqual(json.loads(raw)['status'],'failed')

    def test_existing_training_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'sentinel').write_text('original')
            with self.assertRaises(FileExistsError):train_detector('fixture.json',root,allow_download=True)
            self.assertEqual((root/'sentinel').read_text(),'original');self.assertFalse((root/'run-record.json').exists())

    def test_export_keeps_only_safe_summary_and_numeric_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            with diagnostic_session(root/'diag') as journal:
                with diagnostic_span('training'):pass
            record=new_record('training',run_id='run-1');record.update(status='failed',diagnostic_trace_id=journal.trace_id,untrusted='fixture-private-value')
            write_record(run,record)
            (run/'losses.jsonl').write_text(json.dumps({'step':1,'epoch':0,'loss':2,'api_key':'fixture-private-value'})+'\n')
            result=export_run_logs(run,root/'export',diagnostics_dir=root/'diag')
            self.assertTrue(result['trace_available']);self.assertEqual(result['loss_record_count'],1)
            self.assertNotIn('fixture-private-value',(root/'export'/'support.json').read_text()+(root/'export'/'losses.jsonl').read_text())

    def test_legacy_run_without_id_is_not_given_fabricated_training_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'result.json').write_text('{"status":"completed"}')
            with self.assertRaises(ValueError):export_run_logs(root,root/'support')
