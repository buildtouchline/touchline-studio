"""Verify the notebook backup helper with local synthetic files only."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


class CompleteBackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save=staticmethod(runpy.run_path(str(Path(__file__).resolve().parents[1]/'notebooks/analysis_backup.py'))['save_complete_analysis'])

    def fixture(self,root,label):
        work=root/label;run=work/'run';run.mkdir(parents=True)
        video=root/'video24.mp4'
        if not video.exists():video.write_bytes(b'synthetic-video-bytes-for-copy-test')
        (run/'manifest.json').write_text(json.dumps({'run_id':label,'status':'partial',
            'source':{'id':hashlib.sha256(video.read_bytes()).hexdigest()}}),encoding='utf-8')
        for name in ('observations.jsonl','coverage.jsonl','raw_proposals.jsonl'):(run/name).write_text('{}\n',encoding='utf-8')
        preview=work/'preview-fixture';preview.mkdir()
        for name in ('overlay24.mp4','focus24.mp4','players.jpg','track-index.json'):(preview/name).write_bytes(b'fixture-'+name.encode())
        evaluation=work/'evaluation-fixture';evaluation.mkdir()
        (evaluation/'tracking-evaluation.json').write_text('{}',encoding='utf-8')
        (work/'private.txt').write_text('synthetic-unrelated-file',encoding='utf-8')
        return work,run,video

    def test_all_outputs_parent_video_and_snapshots_are_verified_without_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work,run,video=self.fixture(root,'current-fixture')
            parent_work,parent_run,_=self.fixture(root,'parent-fixture')
            drive=root/'drive';drive.mkdir()
            saved=self.save(work,run,drive,video,parent_work=parent_work,parent_run_dir=parent_run,
                snapshots={'preview_result':{'schema':'fixture'}})
            receipt=json.loads((saved/'TRANSFER_COMPLETE.json').read_text(encoding='utf-8'))
            self.assertTrue(receipt['parent_included'])
            self.assertEqual(receipt['restore_run_subfolder'],'current/run')
            self.assertIn('current/preview-fixture/players.jpg',receipt['sha256'])
            self.assertIn('current/preview-fixture/focus24.mp4',receipt['sha256'])
            self.assertIn('current/evaluation-fixture/tracking-evaluation.json',receipt['sha256'])
            self.assertIn('parent/run/raw_proposals.jsonl',receipt['sha256'])
            self.assertIn('video/video24.mp4',receipt['sha256'])
            self.assertFalse(list(saved.rglob('private.txt')))
            for relative,digest in receipt['sha256'].items():
                self.assertEqual(hashlib.sha256((saved/relative).read_bytes()).hexdigest(),digest)
            self.assertEqual(json.loads((saved/'notebook-results.json').read_text())['results']['preview_result'],{'schema':'fixture'})
            self.assertNotEqual(saved,self.save(work,run,drive,video))

    def test_wrong_video_leaves_no_success_marker_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work,run,video=self.fixture(root,'fixture');drive=root/'drive';drive.mkdir()
            before=(run/'manifest.json').read_bytes()
            wrong=root/'wrong.mp4';wrong.write_bytes(b'wrong fixture media')
            with self.assertRaisesRegex(ValueError,'differs'):
                self.save(work,run,drive,wrong)
            self.assertFalse(list(drive.rglob('TRANSFER_COMPLETE.json')))
            self.assertEqual((run/'manifest.json').read_bytes(),before)
            self.assertTrue(video.is_file())

    def test_missing_required_artifacts_fail_before_creating_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work,run,video=self.fixture(root,'fixture');drive=root/'drive';drive.mkdir()
            (run/'coverage.jsonl').unlink()
            with self.assertRaisesRegex(ValueError,'Missing required'):
                self.save(work,run,drive,video)
            self.assertFalse((drive/'complete-results').exists())
