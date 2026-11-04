import json
from contextlib import closing
from pathlib import Path
import runpy
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ACCESS=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/check_provider_access.py'))


class ProviderAccessTests(unittest.TestCase):
    def client(self,status=200,body=None):
        response=Mock(status_code=status)
        response.iter_bytes.return_value=[json.dumps(body if body is not None else {'data':[{'id':'sam-3.1'}]}).encode()]
        context=Mock();context.__enter__=Mock(return_value=response);context.__exit__=Mock(return_value=False)
        client=Mock();client.stream.return_value=context
        return client

    def test_only_pinned_model_presence_is_emitted(self):
        client=self.client(body={'data':[{'id':'sam-3.1'},{'id':'fixture-secret-echo'}]})
        result=ACCESS['probe_catalog'](client,'https://api.meta.ai/v1/models','fixture-private-key',('sam-3.1',))
        self.assertEqual(result['requested_models'],{'sam-3.1':True})
        self.assertNotIn('fixture',json.dumps(result))
        self.assertEqual(client.stream.call_args.args[0],'GET')

    def test_missing_key_never_sends_request(self):
        client=Mock()
        self.assertEqual(ACCESS['probe_catalog'](client,'https://api.meta.ai/v1/models','',()),{'state':'key_missing'})
        client.stream.assert_not_called()

    def test_failure_response_and_exception_text_are_never_forwarded(self):
        client=self.client(status=401,body={'error':'fixture-private-key'})
        self.assertEqual(ACCESS['probe_catalog'](client,'https://api.meta.ai/v1/models','fixture',()),
                         {'state':'http_rejected','http_status':401})
        client.stream.side_effect=RuntimeError('fixture-private-key')
        result=ACCESS['probe_catalog'](client,'https://api.meta.ai/v1/models','fixture',())
        self.assertNotIn('fixture',json.dumps(result))

    def test_oversized_catalog_is_rejected(self):
        client=self.client();client.stream.return_value.__enter__.return_value.iter_bytes.return_value=[b'x'*(ACCESS['MAX_BODY']+1)]
        result=ACCESS['probe_catalog'](client,'https://api.meta.ai/v1/models','fixture',())
        self.assertEqual(result['state'],'response_too_large')

    def test_presence_is_not_claimed_as_authenticated_or_paid(self):
        settings=SimpleNamespace(credentials=SimpleNamespace(get=lambda name:'fixture-private-key'))
        result=ACCESS['check_access'](settings,self.client())
        self.assertEqual(result['paid_requests'],0)
        self.assertEqual(result['results']['modal']['state'],'tokens_configured_not_authenticated')
        self.assertEqual(result['results']['sarvam']['state'],'key_configured_not_authenticated')
        self.assertNotIn('fixture',json.dumps(result))

    def test_modal_authentication_launches_no_job_and_suppresses_sdk_output(self):
        settings=SimpleNamespace(credentials=SimpleNamespace(get=lambda name:'fixture-private-key'))
        client=Mock();factory=Mock(return_value=client)
        result=ACCESS['probe_modal'](settings,factory)
        self.assertEqual(result,{'state':'authenticated_hello','jobs_launched':0})
        client.hello.assert_called_once_with()
        client.Sandbox.create.assert_not_called()
        factory.side_effect=RuntimeError('fixture-private-key')
        self.assertEqual(ACCESS['probe_modal'](settings,factory)['state'],'authentication_or_connection_failed')

    def test_hello_uses_exact_sdk_endpoint_model_and_output_bound(self):
        settings=SimpleNamespace(credentials=SimpleNamespace(get=lambda name:'fixture-private-key'))
        client=Mock();client.chat.completions.create.return_value=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=8,completion_tokens=2,total_tokens=10))
        context=Mock();context.__enter__=Mock(return_value=client);context.__exit__=Mock(return_value=False)
        factory=Mock(return_value=context)
        result=ACCESS['probe_baseten_hello'](settings,factory)
        args=client.chat.completions.create.call_args.kwargs
        self.assertEqual(args['model'],'meta-models/muse-spark-open-1026')
        self.assertEqual(args['max_tokens'],32)
        self.assertEqual(factory.call_args.kwargs['max_retries'],0)
        self.assertNotIn('fixture',json.dumps(result))

    def test_hello_does_not_forward_provider_error_text(self):
        settings=SimpleNamespace(credentials=SimpleNamespace(get=lambda name:'fixture-private-key'))
        class Rejected(Exception):status_code=401
        factory=Mock(side_effect=Rejected('fixture-private-key'))
        self.assertEqual(ACCESS['probe_baseten_hello'](settings,factory),{'state':'http_rejected','http_status':401})

    def test_one_dollar_reservation_survives_restart_and_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'allowance.sqlite3'
            ACCESS['reserve_single_smoke'](path)
            with self.assertRaises(sqlite3.IntegrityError):ACCESS['reserve_single_smoke'](path)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT reserved_usd FROM smoke').fetchall(),[('1.00',)])

    def test_only_one_explicit_retry_after_401_reuses_retained_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'allowance.sqlite3';receipt=Path(directory)/'receipt.json'
            ACCESS['reserve_single_smoke'](path)
            receipt.write_text(json.dumps({'results':{'baseten':{'state':'connection_or_response_failed'}}}))
            with self.assertRaises(ValueError):ACCESS['reserve_single_smoke'](path,retry_receipt=receipt)
            receipt.write_text(json.dumps({'results':{'baseten':{'state':'http_rejected','http_status':401}}}))
            ACCESS['reserve_single_smoke'](path,retry_receipt=receipt)
            with self.assertRaises(sqlite3.IntegrityError):ACCESS['reserve_single_smoke'](path,retry_receipt=receipt)
            with closing(sqlite3.connect(path)) as db:self.assertEqual(db.execute('SELECT reserved_usd FROM smoke').fetchall(),[('1.00',)])
