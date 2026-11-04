import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
TOOLS=Path(__file__).resolve().parents[1]/'tools'
with patch.object(sys,'path',[str(TOOLS),*sys.path]):
    spec=importlib.util.spec_from_file_location('handover_launcher',TOOLS/'run_handover_job.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class HandoverLauncherTests(unittest.TestCase):
    def config(self):return json.loads((TOOLS.parent/'examples/handover-session.template.json').read_text())
    def test_rejects_dated_runs_secret_fields_and_unbounded_caps(self):
        for changes in ({'run_id':'modal-ap50-20261103'},{'API_KEY':'fixture'},{'job_cap_usd':'100'},{'owner_budget_confirmed':'yes'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):m.parse_config({**self.config(),**changes})
    def test_asset_path_cannot_escape_private_handover_folder(self):
        for value in ('../outside.pt','.env','src/touchline/__init__.py'):
            with self.assertRaises(ValueError):m.asset(value)
