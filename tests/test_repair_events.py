import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from touchline.repair import TARGET,CHECK_ID
from touchline.repair_events import RepairEventQueue,failure_event,validate_event,dispatch_event,publish_job_failure,registered_code_handler
import json
from touchline.data.manifest import sha256_file


class RepairEventTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        p=self.root/TARGET;p.parent.mkdir(parents=True);p.write_text('value=1\n')
        self.source={'path':TARGET,'sha256':sha256_file(p),'check_id':CHECK_ID}
        self.queue=RepairEventQueue(self.root/'queue.sqlite3')
    def tearDown(self):self.tmp.cleanup()
    def event(self,code='INTERNAL_ERROR',source=None):return failure_event('run-1','attempt-1','training',code,source=source)
    def dispatch(self,event,**kwargs):
        self.queue.enqueue(event)
        return dispatch_event(self.queue,event['event_id'],source_root=self.root,enabled=True,worker_terminated=True,**kwargs)

    def test_idempotent_delivery_and_restart_prevent_duplicate_call(self):
        e=self.event(source=self.source);call=Mock(return_value={'live_source_changed':False})
        self.assertTrue(self.queue.enqueue(e)['created']);self.assertFalse(self.queue.enqueue(e)['created'])
        self.assertEqual(self.dispatch(e,handlers={'diagnosis':call})['state'],'major_approval_required')
        reopened=RepairEventQueue(self.root/'queue.sqlite3')
        self.assertEqual(dispatch_event(reopened,e['event_id'],handlers={'diagnosis':call},enabled=True,worker_terminated=True)['reason'],'duplicate_or_unknown_event')
        call.assert_called_once()

    def test_no_dispatch_before_worker_termination_or_enablement(self):
        e=self.event();self.queue.enqueue(e)
        with self.assertRaises(ValueError):dispatch_event(self.queue,e['event_id'],handlers={})
        self.assertEqual(dispatch_event(self.queue,e['event_id'],handlers={},enabled=True)['reason'],'worker_reconciliation_required')
        self.assertIsNotNone(self.queue.claim(e['event_id']))

    def test_provider_data_and_budget_errors_never_trigger_source_repair(self):
        for code in ('AUTH_REJECTED','PROVIDER_UNAVAILABLE','BUDGET_REJECTED','DATA_INVALID','INPUT_INVALID'):
            with self.subTest(code=code):
                call=Mock();result=self.dispatch(self.event(code),handlers={'code_repair':call,'diagnosis':call})
                call.assert_not_called();self.assertIn(result['state'],('provider_reconciliation','review_required'))

    def test_only_registered_reproduced_failure_can_stage_checks(self):
        call=Mock(return_value={'live_source_changed':False,'worker_terminated':True,'passed':True})
        result=self.dispatch(self.event(source=self.source),handlers={'code_repair':call},failure_reproduced=True)
        self.assertEqual(result['state'],'checked');self.assertEqual(result['promotion'],'not performed');call.assert_called_once()

    def test_stale_hash_blocks_any_adviser_call(self):
        e=self.event(source=self.source);(self.root/TARGET).write_text('changed')
        call=Mock();self.assertEqual(self.dispatch(e,handlers={'diagnosis':call})['state'],'unknown');call.assert_not_called()

    def test_payload_injection_and_unsafe_paths_are_rejected(self):
        e=self.event()
        for changed in ({**e,'raw_error':'private data'},{**e,'stage':'arbitrary'},{**e,'event_id':'a'*64}):
            with self.assertRaises(ValueError):validate_event(changed)
        for path in ('../.env','src/touchline/../../.env','C:/private.py'):
            with self.assertRaises(ValueError):self.event(source={**self.source,'path':path})

    def test_unknown_claim_is_not_automatically_retried(self):
        e=self.event();self.queue.enqueue(e);self.assertIsNotNone(self.queue.claim(e['event_id']))
        self.assertIsNone(RepairEventQueue(self.root/'queue.sqlite3').claim(e['event_id']))

    def test_handler_failure_keeps_unknown_state_without_retry(self):
        result=self.dispatch(self.event(source=self.source),handlers={'diagnosis':Mock(side_effect=RuntimeError('secret fixture'))})
        self.assertEqual(result['state'],'unknown');self.assertNotIn('secret',str(result))

    def test_unsafe_live_mutation_receipt_fails_closed(self):
        result=self.dispatch(self.event(source=self.source),handlers={'diagnosis':Mock(return_value={'live_source_changed':True})})
        self.assertEqual(result['state'],'unknown')

    def test_actual_controller_failure_emits_fixed_event_not_raw_error(self):
        work=self.root/'training-attempt';work.mkdir()
        (work/'worker.events.jsonl').write_text(json.dumps({'stage':'failed','failure_code':'SOURCE_IMAGE_HASH_MISMATCH'})+'\n')
        result=publish_job_failure(work,{'status':'failed','worker_terminated':True,'raw_error':'private fixture'},queue_path=self.root/'job-events.sqlite3',stage='training')
        event=json.loads((work/'repair-event.json').read_text())
        self.assertTrue(result['created']);self.assertEqual(event['failure_code'],'DATA_INVALID');self.assertNotIn('private',str(event))
        self.assertFalse(publish_job_failure(work,{'status':'failed','worker_terminated':True},queue_path=self.root/'job-events.sqlite3',stage='training')['created'])

    def test_successful_job_does_not_invent_a_failure(self):
        self.assertIsNone(publish_job_failure(self.root,{'status':'complete'},queue_path=self.root/'events.sqlite3',stage='analysis'))

    def test_missing_dependency_is_not_misclassified_as_source_defect(self):
        work=self.root/'dependency-attempt';work.mkdir()
        (work/'worker.events.jsonl').write_text(json.dumps({'stage':'failed','failure_code':'DEPENDENCY_MISSING'})+'\n')
        publish_job_failure(work,{'status':'failed','worker_terminated':True},queue_path=self.root/'dep-events.sqlite3',stage='training')
        self.assertEqual(json.loads((work/'repair-event.json').read_text())['failure_code'],'DEPENDENCY_MISSING')

    def test_registered_handler_rechecks_lineage_before_any_paid_call(self):
        e=self.event(source=self.source);adviser=Mock()
        handler=registered_code_handler(self.root,self.root/'out',adviser=adviser,checker=Mock(),quote=Mock(),check_worst_case_usd='0.10')
        (self.root/TARGET).write_text('changed=2\n')
        with self.assertRaises(ValueError):handler(e)
        adviser.propose.assert_not_called()
