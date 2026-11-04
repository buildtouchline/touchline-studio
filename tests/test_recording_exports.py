import importlib.util,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
TOOLS=Path(__file__).resolve().parents[1]/'tools'
with patch.object(sys,'path',[str(TOOLS),*sys.path]):
    spec=importlib.util.spec_from_file_location('recording_controller',TOOLS/'run_modal_recording.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RecordingExportTests(unittest.TestCase):
    def test_export_rejects_escape_symlink_duplicate_and_unexpected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'out';output.mkdir();archive=root/'output.zip'
            for name in ('../escape.json','preview/../../escape.json','candidate.pt','.env'):
                with zipfile.ZipFile(archive,'w') as z:z.writestr(name,'{}')
                with self.subTest(name=name),self.assertRaises(ValueError):m.extract(archive,output)
    def test_export_streams_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'out';output.mkdir();archive=root/'output.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('normalized/video24.mp4',b'fixture')
            m.extract(archive,output);self.assertEqual((output/'normalized/video24.mp4').read_bytes(),b'fixture')
            with self.assertRaises(ValueError):m.extract(archive,output)
