import copy
from datetime import datetime,timezone
import io,json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch
from touchline.repair import TARGET,CHECK_ID,prepare_case,validate_proposal,stage_proposal,RepairAttempts,failure_action,run_repair_cycle
from touchline.code_adviser import BasetenCodeAdviser,ModelQuote,create_baseten_client,MODEL,BASE_URL
from touchline.repair_checks import ModalRepairChecks,evaluate_results,named_modal_factory,read_worker_result,MAX_RESULT_BYTES
from touchline.spending import SpendingLedger


def quote():return ModelQuote('1','2',8192,datetime.now(timezone.utc),True)
def expected_output():
    return json.dumps({'anchor':[5,17,29,41],'fixed':list(range(0,48,4)),
        'gap_rejected':True,'wrong_fps_rejected':True,'merged':[[0,2.0]]})


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.source=self.root/'source'
        target=self.source/TARGET;target.parent.mkdir(parents=True)
        current=Path(__file__).resolve().parents[1]/TARGET
        target.write_bytes(current.read_bytes())
        self.packet=prepare_case(self.source)
        self.proposal={'schema':'touchline.patch-proposal.v1','baseline_id':self.packet['baseline_id'],'check_id':CHECK_ID,
            'replacement':{'path':TARGET,'before_sha256':self.packet['source']['sha256'],'content':self.packet['source']['content']+'\n# fixture repair candidate\n'}}
        self.ledger=SpendingLedger(self.root/'usage.sqlite3')
        self.ledger.verify_pool('baseten',prior_spend_usd=0,available_credit_usd=100)
        self.ledger.verify_pool('modal',prior_spend_usd=0,available_credit_usd=100)
        self.attempts=RepairAttempts(self.root/'attempts.sqlite3')
    def tearDown(self):self.tmp.cleanup()
    def client(self,text=None,finish='stop',error=None):
        message=SimpleNamespace(content=text or json.dumps(self.proposal),refusal=None,tool_calls=None)
        create=Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message,finish_reason=finish)]),side_effect=error)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_staging_preserves_original_and_backup_without_executing_code(self):
        original=(self.source/TARGET).read_bytes()
        result=stage_proposal(self.packet,self.proposal,self.root/'staged',source_root=self.source)
        self.assertEqual(result['promotion_state'],'not_promoted')
        self.assertEqual((self.source/TARGET).read_bytes(),original)
        self.assertEqual((self.root/'staged'/'baseline.py').read_bytes(),original)
        self.assertTrue((self.root/'staged'/'candidate.py').is_file())

    def test_protected_path_stale_hash_and_extra_command_rejected(self):
        for changed in (
            {**self.proposal,'command':'arbitrary shell'},
            {**self.proposal,'baseline_id':'a'*64},
            {**self.proposal,'replacement':{**self.proposal['replacement'],'path':'src/touchline/spending.py'}},
            {**self.proposal,'replacement':{**self.proposal['replacement'],'path':'../../.env'}}):
            with self.assertRaises(ValueError):validate_proposal(self.packet,changed)

    def test_unsafe_import_dynamic_execution_and_noop_rejected(self):
        for code in ('import os\n','eval("1")\n','x=object.__class__\n',self.packet['source']['content']):
            proposal=copy.deepcopy(self.proposal);proposal['replacement']['content']=code
            with self.assertRaises(ValueError):validate_proposal(self.packet,proposal)

    def test_source_change_invalidates_staged_candidate(self):
        (self.source/TARGET).write_text('changed',encoding='utf-8')
        with self.assertRaises(ValueError):stage_proposal(self.packet,self.proposal,self.root/'staged',source_root=self.source)

    def test_repair_cannot_stage_over_source_directory(self):
        with self.assertRaises(ValueError):stage_proposal(self.packet,self.proposal,self.source/'src'/'new',source_root=self.source)

    def test_data_oom_and_auth_failures_do_not_automatically_edit_code(self):
        self.assertEqual(failure_action('RESOURCE_EXHAUSTED'),'parameter_review')
        for code in ('DATA_INVALID','AUTH_REJECTED','BUDGET_REJECTED'):
            with self.assertRaises(ValueError):prepare_case(self.source,failure_code=code)

    def test_attempt_caps_and_restart_idempotency(self):
        cid=self.packet['case_id']
        self.assertTrue(self.attempts.claim(cid,'first'))
        self.assertFalse(RepairAttempts(self.root/'attempts.sqlite3').claim(cid,'first'))
        with self.assertRaises(ValueError):self.attempts.claim(cid,'parallel')
        self.attempts.finish('first','rejected')
        self.assertTrue(self.attempts.claim(cid,'second'));self.attempts.finish('second','rejected')
        with self.assertRaises(ValueError):self.attempts.claim(cid,'third')

    def test_sdk_configuration_has_explicit_key_timeout_no_hidden_retries(self):
        factory=Mock(return_value='fixture-client')
        self.assertEqual(create_baseten_client('fixture-key',sdk_factory=factory),'fixture-client')
        self.assertEqual(factory.call_args.kwargs,{'api_key':'fixture-key','base_url':BASE_URL,'max_retries':0,'timeout':30.0})

    def test_adviser_disabled_without_source_review_or_unverified_quote(self):
        client=self.client();adviser=BasetenCodeAdviser(client,self.ledger,self.attempts)
        with self.assertRaises(ValueError):adviser.propose(self.packet,attempt_id='one',quote=quote())
        bad=ModelQuote('1','2',8192,datetime.now(timezone.utc),False)
        with self.assertRaises(ValueError):adviser.propose(self.packet,attempt_id='one',quote=bad,enabled=True,source_reviewed_for_model=True)
        client.chat.completions.create.assert_not_called()

    def test_adviser_uses_pinned_model_validates_result_and_retains_reservation(self):
        client=self.client();adviser=BasetenCodeAdviser(client,self.ledger,self.attempts)
        result=adviser.propose(self.packet,attempt_id='one',quote=quote(),enabled=True,source_reviewed_for_model=True)
        self.assertEqual(result,self.proposal);self.assertEqual(client.chat.completions.create.call_args.kwargs['model'],MODEL)
        with self.assertRaises(ValueError):adviser.propose(self.packet,attempt_id='one',quote=quote(),enabled=True,source_reviewed_for_model=True)
        self.assertEqual(client.chat.completions.create.call_count,1)
        with self.ledger._transaction() as db:
            self.assertEqual(db.execute('SELECT state FROM reservations').fetchone()[0],'pending')

    def test_truncation_or_invalid_model_json_never_becomes_a_patch(self):
        for n,(text,finish) in enumerate([('not json','stop'),(json.dumps(self.proposal),'length')]):
            client=self.client(text,finish);adviser=BasetenCodeAdviser(client,self.ledger,self.attempts)
            with self.assertRaises(ValueError):adviser.propose(self.packet,attempt_id='bad'+str(n),quote=quote(),enabled=True,source_reviewed_for_model=True)

    def test_provider_failure_blocks_blind_repair_replay(self):
        client=self.client(error=TimeoutError('fixture-private-token'));adviser=BasetenCodeAdviser(client,self.ledger,self.attempts)
        with self.assertRaises(RuntimeError):adviser.propose(self.packet,attempt_id='one',quote=quote(),enabled=True,source_reviewed_for_model=True)
        with self.assertRaises(ValueError):adviser.propose(self.packet,attempt_id='two',quote=quote(),enabled=True,source_reviewed_for_model=True)
        self.assertEqual(client.chat.completions.create.call_count,1)

    def test_worker_success_message_does_not_decide_pass(self):
        self.assertTrue(evaluate_results(expected_output()))
        wrong=json.loads(expected_output());wrong['anchor']=[6,18]
        self.assertFalse(evaluate_results(json.dumps(wrong)))
        with self.assertRaises(ValueError):evaluate_results('{"passed":true}')

    def sandbox(self,*,termination_error=False):
        filesystem=SimpleNamespace(write_text=Mock())
        process=SimpleNamespace(stdout=io.BytesIO(expected_output().encode()),wait=Mock(),returncode=0)
        terminate=Mock(side_effect=RuntimeError('fixture') if termination_error else None)
        return SimpleNamespace(object_id='sb-fixture',filesystem=filesystem,exec=Mock(return_value=process),terminate=terminate)

    def test_optional_modal_check_disabled_by_default_and_terminates_when_enabled(self):
        sandbox=self.sandbox();factory=Mock(return_value=sandbox,image_id='im-fixture');checker=ModalRepairChecks(factory,self.ledger,image_id='im-fixture')
        with self.assertRaises(ValueError):checker.check(self.packet,self.proposal,attempt_id='one',worst_case_usd='1')
        factory.assert_not_called()
        receipt=checker.check(self.packet,self.proposal,attempt_id='one',worst_case_usd='1',enabled=True,quote_verified=True,run_dir=self.root/'check')
        self.assertTrue(receipt['passed']);self.assertEqual(receipt['run_id'],'repair-one')
        self.assertEqual(receipt['promotion_state'],'human_review_required')
        sandbox.terminate.assert_called_once_with(wait=True)
        self.assertEqual(sandbox.exec.call_args.args[:2],('python','-I'))
        record=json.loads((self.root/'check'/'run-record.json').read_text())
        self.assertEqual(record['worker_id'],'sb-fixture')
        self.assertEqual(record['image_id'],'im-fixture')
        self.assertEqual(receipt['image_id'],'im-fixture')
        self.assertEqual(sandbox.exec.call_args.kwargs['text'],False)
        self.assertEqual(sandbox.exec.call_args.kwargs['bufsize'],-1)

    def test_unknown_worker_termination_blocks_more_modal_dispatch(self):
        sandbox=self.sandbox(termination_error=True);checker=ModalRepairChecks(Mock(return_value=sandbox,image_id='im-fixture'),self.ledger,image_id='im-fixture')
        with self.assertRaises(RuntimeError):checker.check(self.packet,self.proposal,attempt_id='one',worst_case_usd='1',enabled=True,quote_verified=True,run_dir=self.root/'check')
        with self.assertRaises(ValueError):self.ledger.reserve('modal','another',fingerprint='fixture',worst_case_usd='1')

    def test_named_modal_factory_is_isolated_and_has_no_secret_or_data_mounts(self):
        sandbox_api=SimpleNamespace(create=Mock(return_value='worker'))
        modal=SimpleNamespace(App=SimpleNamespace(lookup=Mock(return_value='app')),Image=SimpleNamespace(from_id=Mock(return_value='image')),Sandbox=sandbox_api)
        with patch.dict('sys.modules',{'modal':modal}):
            self.assertEqual(named_modal_factory('touchline-repair','im-fixture')(job_name='repair-fixture'),'worker')
        modal.Image.from_id.assert_called_once_with('im-fixture')
        kw=sandbox_api.create.call_args.kwargs
        self.assertTrue(kw['block_network']);self.assertEqual(kw['secrets'],[]);self.assertEqual(kw['volumes'],{})
        self.assertEqual(kw['runtime'],'gvisor');self.assertNotIn('gpu',kw)

    def test_complete_optional_cycle_leaves_live_source_unchanged_and_records_check(self):
        before=(self.source/TARGET).read_bytes()
        adviser=BasetenCodeAdviser(self.client(),self.ledger,self.attempts)
        checker=ModalRepairChecks(Mock(return_value=self.sandbox(),image_id='im-fixture'),self.ledger,image_id='im-fixture')
        result=run_repair_cycle(self.source,self.root/'cycle',adviser=adviser,checker=checker,
            attempt_id='cycle',quote=quote(),check_worst_case_usd='1',enabled=True,
            source_reviewed_for_model=True,check_quote_verified=True)
        self.assertTrue(result['passed']);self.assertFalse(result['live_source_changed'])
        self.assertEqual(result['promotion_state'],'human_review_required')
        self.assertEqual((self.source/TARGET).read_bytes(),before)
        self.assertTrue((self.root/'cycle'/'check-receipt.json').is_file())
        with self.attempts.transaction() as db:
            self.assertEqual(db.execute('SELECT state FROM repair_attempts').fetchone()[0],'checked')

    def test_stale_check_receipt_stops_promotion_and_blocks_replay(self):
        adviser=BasetenCodeAdviser(self.client(),self.ledger,self.attempts)
        checker=SimpleNamespace(check=Mock(return_value={'case_id':'a'*64,'passed':True,'worker_terminated':True}))
        with self.assertRaises(ValueError):
            run_repair_cycle(self.source,self.root/'cycle',adviser=adviser,checker=checker,
                attempt_id='cycle',quote=quote(),check_worst_case_usd='1',enabled=True,
                source_reviewed_for_model=True,check_quote_verified=True)
        with self.assertRaises(ValueError):self.attempts.claim(self.packet['case_id'],'replay')
        self.assertEqual((self.source/TARGET).read_bytes(),self.packet['source']['content'].encode())

    def test_stream_limit_stops_before_reading_remaining_chunks(self):
        consumed=[]
        def chunks():
            consumed.append(1);yield b'a'*MAX_RESULT_BYTES
            consumed.append(2);yield b'b'
            consumed.append(3);raise AssertionError('Do not drain untrusted output after overflow')
        with self.assertRaises(ValueError):read_worker_result(chunks())
        self.assertEqual(consumed,[1,2])
        self.assertEqual(read_worker_result([b'a'*MAX_RESULT_BYTES]),'a'*MAX_RESULT_BYTES)
        # Byte chunks may split a UTF-8 codepoint; only decode the bounded result.
        self.assertEqual(read_worker_result([b'\xc3',b'\xa9']),'é')
        with self.assertRaises(ValueError):read_worker_result([b'\xc3'])
        with self.assertRaises(ValueError):read_worker_result(['not binary'])

    def test_output_overflow_terminates_worker_and_retains_reservation(self):
        sandbox=self.sandbox();sandbox.exec.return_value.stdout=iter([b'x'*(MAX_RESULT_BYTES+1)])
        checker=ModalRepairChecks(Mock(return_value=sandbox,image_id='im-fixture'),self.ledger,image_id='im-fixture')
        with self.assertRaises(RuntimeError):
            checker.check(self.packet,self.proposal,attempt_id='overflow',worst_case_usd='1',enabled=True,quote_verified=True,run_dir=self.root/'overflow')
        sandbox.terminate.assert_called_once_with(wait=True)
        sandbox.exec.return_value.wait.assert_not_called()
        record=json.loads((self.root/'overflow'/'run-record.json').read_text())
        self.assertEqual(record['status'],'failed')
        with self.ledger._transaction() as db:
            self.assertEqual(db.execute("SELECT state FROM reservations WHERE id='repair-modal-overflow'").fetchone()[0],'pending')

    def test_mutable_image_names_and_factory_mismatch_rejected_before_dispatch(self):
        for image in ('reviewed-python-image','reviewed-python-image:latest','',None):
            with self.assertRaises(ValueError):named_modal_factory('touchline-repair',image)
        with self.assertRaises(ValueError):
            ModalRepairChecks(named_modal_factory('touchline-repair','im-first'),self.ledger,image_id='im-second')

    def test_reused_attempt_cannot_change_image_under_same_reservation(self):
        first=ModalRepairChecks(Mock(return_value=self.sandbox(),image_id='im-first'),self.ledger,image_id='im-first')
        first.check(self.packet,self.proposal,attempt_id='pinned',worst_case_usd='1',enabled=True,quote_verified=True,run_dir=self.root/'pinned')
        factory=Mock(return_value=self.sandbox(),image_id='im-second')
        second=ModalRepairChecks(factory,self.ledger,image_id='im-second')
        with self.assertRaises(ValueError):
            second.check(self.packet,self.proposal,attempt_id='pinned',worst_case_usd='1',enabled=True,quote_verified=True,run_dir=self.root/'changed')
        factory.assert_not_called()

    def test_unknown_reconciliation_is_audited_idempotent_and_preserves_cap(self):
        cid=self.packet['case_id'];self.attempts.claim(cid,'uncertain');self.attempts.finish('uncertain','unknown')
        evidence=dict(reviewer_id='operator-a',evidence_sha256='a'*64,worker_terminated=True,billing_finalized=True)
        self.assertTrue(self.attempts.reconcile_unknown('uncertain',**evidence)['created'])
        self.assertFalse(RepairAttempts(self.root/'attempts.sqlite3').reconcile_unknown('uncertain',**evidence)['created'])
        with self.assertRaises(ValueError):self.attempts.reconcile_unknown('uncertain',**{**evidence,'evidence_sha256':'b'*64})
        with self.attempts.transaction() as db:
            self.assertEqual(db.execute('SELECT state FROM repair_attempts').fetchone()[0],'rejected')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM repair_reconciliations').fetchone()[0],1)
        self.assertTrue(self.attempts.claim(cid,'last'));self.attempts.finish('last','rejected')
        with self.assertRaises(ValueError):self.attempts.claim(cid,'extra')

    def test_reconciliation_needs_complete_evidence_and_does_not_unblock_budget(self):
        cid=self.packet['case_id'];self.attempts.claim(cid,'uncertain');self.attempts.finish('uncertain','unknown')
        evidence=dict(reviewer_id='operator-a',evidence_sha256='a'*64,worker_terminated=True,billing_finalized=True)
        for change in ({'worker_terminated':False},{'billing_finalized':False},{'evidence_sha256':'not-a-hash'},{'reviewer_id':''}):
            with self.assertRaises(ValueError):self.attempts.reconcile_unknown('uncertain',**{**evidence,**change})
        with self.assertRaises(ValueError):self.attempts.claim(cid,'still-blocked')
        self.ledger.block_pool('modal')
        self.attempts.reconcile_unknown('uncertain',**evidence)
        with self.assertRaises(ValueError):self.ledger.reserve('modal','job',fingerprint='fixture',worst_case_usd='1')
        with self.assertRaises(ValueError):self.attempts.reconcile_unknown('missing',**evidence)
        self.attempts.claim(cid,'active')
        with self.assertRaises(ValueError):self.attempts.reconcile_unknown('active',**evidence)
