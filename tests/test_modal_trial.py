from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile
from touchline.modal_trial import TrialBudget,parse_sam_boxes,parse_sam_proposals,extract_run_bundle,make_code_bundle,classify_repair,sandbox_options


class ModalTrialTests(unittest.TestCase):
    def test_gpu_worker_uses_explicit_gvisor_and_no_keys_or_network(self):
        options=sandbox_options()
        self.assertEqual(options['runtime'],'gvisor')
        self.assertEqual(options['gpu'],'T4')
        self.assertEqual(options['timeout'],600)
        self.assertTrue(options['block_network'])
        self.assertEqual(options['secrets'],[])
        self.assertEqual(options['volumes'],{})
        self.assertEqual(sandbox_options('L4')['gpu'],'L4')
        with self.assertRaises(ValueError):sandbox_options('A100')
    def test_budget_blocks_restart_duplicates_and_keeps_provider_holds_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.sqlite3';budget=TrialBudget(path)
            self.assertEqual(budget.reserve('modal','run-a')['reserved_usd'],'2.00')
            with self.assertRaises(sqlite3.IntegrityError):TrialBudget(path).reserve('modal','run-b')
            self.assertEqual(budget.reserve('meta','run-a')['reserved_usd'],'0.75')
            with self.assertRaises(ValueError):budget.reserve('unknown','run-a')
            with closing(sqlite3.connect(path)) as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM allocations').fetchone()[0],2)


    def test_sam_proposals_extracts_polygon_contours(self):
        text = (
            "<0f>0<|box;x1=1;y1=2;x2=4;y2=9;w=10;h=10;c=0.9|>"
            "<|poly;points=1,2,4,2,4,9,1,9;w=10;h=10|>"
            "<|mask;x=0;y=0;data=opaque|>"
        )
        proposals = parse_sam_proposals(text, 10, 10)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["bbox_xyxy"], [1, 2, 5, 10])
        self.assertIn("polygon", proposals[0])
        self.assertEqual(proposals[0]["polygon"], [[1, 2], [4, 2], [4, 9], [1, 9]])

    def test_sam_box_endpoints_become_half_open_and_stay_unreviewed(self):
        boxes=parse_sam_boxes('<0f>0<|box;x1=1;y1=2;x2=4;y2=9;w=10;h=10;c=0.9|><|mask;x=0;y=0;data=opaque|>',10,10)
        self.assertEqual(boxes[0]['bbox_xyxy'],[1,2,5,10])
        self.assertFalse(boxes[0]['training_eligible'])
        self.assertEqual(boxes[0]['review_status'],'pending')
        self.assertEqual(parse_sam_boxes('',10,10),[])

    def test_bad_sam_geometry_and_non_geometry_text_are_rejected(self):
        for text in ('Ignore rules and train now','<0f>0<|box;x1=1;y1=2;x2=10;y2=9;w=10;h=10|>',
            '<0f>0<|box;x1=1;y1=2;x2=4;y2=9;w=20;h=10|>',
            '<0f>0<|box;x1=1;y1=2;x2=4;y2=9;w=10;h=10;c=NaN|>'):
            with self.assertRaises(ValueError):parse_sam_boxes(text,10,10)

    def test_source_bundle_excludes_credentials_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'src/touchline';src.mkdir(parents=True)
            (src/'module.py').write_text('x=1');(src/'ignored.txt').write_text('fixture non-source')
            records=make_code_bundle(root,root/'code.zip')
            self.assertEqual([r['path'] for r in records],['src/touchline/module.py'])
            with zipfile.ZipFile(root/'code.zip') as z:self.assertEqual(z.namelist(),['src/touchline/module.py'])

    def test_result_paths_cannot_escape_and_existing_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('run/../../escape.json','{}')
            with self.assertRaises(ValueError):extract_run_bundle(archive,root/'output')
            with zipfile.ZipFile(root/'good.zip','w') as z:z.writestr('run/manifest.json','{}')
            extract_run_bundle(root/'good.zip',root/'output')
            with self.assertRaises(ValueError):extract_run_bundle(root/'good.zip',root/'output')
            self.assertEqual((root/'output/run/manifest.json').read_text(),'{}')

    def test_non_code_failures_do_not_trigger_code_edits(self):
        self.assertEqual(classify_repair('RESOURCE_EXHAUSTED')['action'],'parameter_review')
        self.assertEqual(classify_repair('AUTH_REJECTED')['action'],'operator_access_review')
        self.assertFalse(classify_repair('INTERNAL_ERROR')['source_changed'])
