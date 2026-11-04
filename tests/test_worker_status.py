import json,unittest
from touchline.worker_status import parse_worker_status

class WorkerStatusTests(unittest.TestCase):
    def record(self,**extra):return {'stage':'analysis','status':'progress','frames':480,'selected_frames':120,'ids':12,'event_seq':3,'worker_epoch':1793700000.,**extra}
    def test_reconnected_controller_can_accept_a_later_snapshot_without_stdout_history(self):
        item=parse_worker_status(json.dumps(self.record(event_seq=81)));self.assertEqual(item['event_seq'],81)
    def test_payloads_unknown_fields_and_nonfinite_counters_are_rejected(self):
        for extra in ({'headers':'forbidden'},{'frames':float('nan')},{'ids':25},{'event_seq':False},{'worker_epoch':float('inf')}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):parse_worker_status(json.dumps(self.record(**extra)))
    def test_completion_and_failure_are_explicit_states(self):
        self.assertEqual(parse_worker_status(json.dumps(self.record(stage='complete',status='complete')))['status'],'complete')
        self.assertEqual(parse_worker_status(json.dumps(self.record(status='failed',error_category='ValueError')))['error_category'],'ValueError')
