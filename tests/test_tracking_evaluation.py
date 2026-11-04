import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from touchline.tracking_evaluation import evaluate_tracking


@unittest.skipUnless(importlib.util.find_spec('scipy'),'optional tracking evaluation extra not installed')
class TrackingEvaluationTests(unittest.TestCase):
    def fixture(self,root,ids=(7,7,7,7)):
        run=root/'run';run.mkdir()
        source_hash='a'*64
        (run/'manifest.json').write_text(json.dumps({'run_id':'fixture','synthetic':True,'status':'partial',
            'source':{'id':source_hash},'detector':{'checkpoint':'b'*64},'tracker':{'type':'fixture'}}),encoding='utf-8')
        coverage=[];observations=[];reference=[]
        for n,tid in enumerate(ids):
            index=n*4;box=[10+n,20,30+n,80];f={'index':index,'width':320,'height':180}
            coverage.append({'frame':f,'state':'detections_present'})
            if tid is not None:observations.append({'frame':f,'local_track_id':tid,'bbox_xyxy':box})
            reference.append({'frame_index':index,'people':[{'reference_id':'reviewed-A','bbox_xyxy':box}]})
        for name,rows in (('coverage.jsonl',coverage),('observations.jsonl',observations)):
            (run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        ref=root/'reference.json'
        ref.write_text(json.dumps({'schema':'touchline.tracking-reference.v1','reviewed':True,
            'reviewer_id':'synthetic-review-fixture','source_sha256':source_hash,'frames':reference}),encoding='utf-8')
        return run,ref

    def test_perfect_identity_score_does_not_accept_synthetic_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ref=self.fixture(Path(tmp));result=evaluate_tracking(run,ref)
            self.assertEqual(result['idf1'],1)
            self.assertEqual(result['detection_recall'],1)
            self.assertFalse(result['minimum_target_met_on_reviewed_scope'])
            self.assertFalse(result['full_video_accuracy_verified'])

    def test_duplicate_ids_lower_identity_score_even_when_detection_is_perfect(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ref=self.fixture(Path(tmp),(7,7,23,23));result=evaluate_tracking(run,ref)
            self.assertEqual(result['idf1'],.5)
            self.assertEqual(result['detection_recall'],1)
            self.assertEqual(result['detection_precision'],1)
            self.assertEqual(result['identity_changes_among_reviewed_matches'],1)
            self.assertEqual(result['predicted_ids_per_reference']['reviewed-A'],[7,23])

    def test_missing_detections_count_as_identity_false_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ref=self.fixture(Path(tmp),(7,None,None,7));result=evaluate_tracking(run,ref)
            self.assertEqual(result['idfn'],2)
            self.assertAlmostEqual(result['idf1'],2/3)

    def test_reviewed_empty_frames_count_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ref=self.fixture(Path(tmp));data=json.loads(ref.read_text())
            data['frames'][-1]['people']=[];ref.write_text(json.dumps(data),encoding='utf-8')
            result=evaluate_tracking(run,ref)
            self.assertEqual(result['idfp'],1)
            self.assertEqual(result['detection_precision'],.75)

    def test_wrong_source_and_unreviewed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ref=self.fixture(Path(tmp));original=json.loads(ref.read_text())
            for change in ({'source_sha256':'c'*64},{'reviewed':False},{'reviewer_id':None}):
                ref.write_text(json.dumps({**original,**change}),encoding='utf-8')
                with self.assertRaises(ValueError):evaluate_tracking(run,ref)

    def test_unsampled_reference_frames_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,ref=self.fixture(Path(tmp));p=run/'coverage.jsonl'
            rows=[json.loads(s) for s in p.read_text().splitlines()]
            rows[0]['state']='unsampled_by_sampler';p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            with self.assertRaises(ValueError):evaluate_tracking(run,ref)
