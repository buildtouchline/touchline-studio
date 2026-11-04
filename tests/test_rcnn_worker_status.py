import importlib.util,io,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
TOOLS=Path(__file__).resolve().parents[1]/'tools'

class RcnnWorkerStatusTests(unittest.TestCase):
    def worker(self):
        spec=importlib.util.spec_from_file_location('rcnn_status_test',TOOLS/'modal_rcnn_gpu_worker.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    def test_atomic_state_remains_parseable_after_concurrent_events(self):
        module=self.worker()
        with tempfile.TemporaryDirectory() as tmp,patch.object(module.sys,'__stdout__',io.StringIO()):
            module.STATUS_PATH=Path(tmp)/'status.json';threads=[threading.Thread(target=module.emit,args=('analysis',),kwargs={'status':'progress','frames':i}) for i in range(8)]
            for thread in threads:thread.start()
            for thread in threads:thread.join()
            state=json.loads(module.STATUS_PATH.read_text());self.assertEqual(state['event_seq'],8);self.assertEqual(state['stage'],'analysis');self.assertFalse(module.STATUS_PATH.with_suffix('.tmp').exists())
    def test_late_heartbeat_does_not_overwrite_terminal_success(self):
        module=self.worker()
        with tempfile.TemporaryDirectory() as tmp,patch.object(module.sys,'__stdout__',io.StringIO()):
            module.STATUS_PATH=Path(tmp)/'status.json';module.emit('complete',status='complete');module.emit('complete',status='heartbeat')
            state=json.loads(module.STATUS_PATH.read_text());self.assertEqual(state['status'],'complete');self.assertEqual(state['event_seq'],1)
