import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from touchline.contracts import Frame,SCHEMA_VERSION
from touchline.metrics import Calibration
from touchline.player_analysis import generate_profiles,score_profiles,analyze_run
from review_fixtures import review_ledger


def inputs(*,synthetic=True):
    source='f'*64
    manifest={'run_id':'fixture-run','synthetic':synthetic,'source':{'id':source},'status':'complete',
        'selected_frames':3,'decoded_frames':25,'detector':{'checkpoint':'a'*64},'sampler':'fixture','sampler_config':{}}
    rows=[]
    for track in (1,2):
        for n,index in enumerate((0,12,24)):
            frame=Frame(source,index,index,1,24,index/24,1920,1080)
            rows.append({'schema_version':SCHEMA_VERSION,'run_id':'fixture-run','local_track_id':track,
                'frame':frame.metadata(),'bbox_xyxy':[100+n*10,100,140+n*10,200],
                'detector_score':0.9,'association_state':'matched','synthetic':synthetic})
    return manifest,rows


def calibration():
    return Calibration((0.01,0.,0.,0.,0.01,0.,0.,0.,1.),'fixture-reference',0,5,(0,0,1920,1080),
        source_geometry=(1920,1080),geometry_revision='fixture-geometry',calibration_revision='fixture-calibration')


def rubric():
    return {'schema':'touchline.rubric.v1','rubric_id':'fixture','version':'1','drill_id':'movement-drill','context':'fixture-context',
        'reviewed':True,'reviewed_by':'fixture-coach','dimensions':[{'metric':'distance_m','units':'metres',
            'lower':0,'upper':1,'weight':1,'minimum_support_ratio':0.5,'minimum_observations':2,'direction':'higher','focus_below':60,
            'plan_template':{'drill':'coach-selected movement drill','success_criterion':'coach-defined target','review_window':'next reviewed session'}}]}


class ProfileTests(unittest.TestCase):
    def test_all_tracks_retained_and_unknown_identity_not_invented(self):
        m,rows=inputs();result=generate_profiles(m,rows)
        self.assertEqual(len(result['profiles']),2);self.assertEqual(result['unresolved_tracks'],[1,2])
        first=result['profiles'][0]
        self.assertIsNone(first['player_id']);self.assertIsNone(first['metrics']['distance_m']['value'])
        self.assertEqual(first['evidence']['pts'],[0,12,24])
        self.assertEqual(first['sample_support']['ratio'],1)
        self.assertAlmostEqual(first['sample_support']['selected_fraction'],3/25)

    def test_reviewed_identity_does_not_replace_calibration(self):
        m,rows=inputs();out=generate_profiles(m,rows,associations=review_ledger('fixture-run',1))
        first=out['profiles'][0]
        self.assertEqual(first['player_id'],'P07')
        self.assertIsNone(first['metrics']['average_speed_mps']['value'])
        self.assertEqual(out['unresolved_tracks'],[2])

    def test_contract_fixture_calibrated_metric_and_draft_score(self):
        # False here tests a contract branch only; this is not real performance evidence.
        m,rows=inputs(synthetic=False)
        context={'drill_id':'movement-drill','context':'fixture-context','reviewed':True,'reviewed_by':'fixture-coach'}
        out=generate_profiles(m,rows,associations=review_ledger('fixture-run',1),calibration=calibration(),session_context=context)
        self.assertAlmostEqual(out['profiles'][0]['metrics']['distance_m']['value'],0.2)
        scores=score_profiles(out,rubric())
        self.assertAlmostEqual(scores['results'][0]['coach_rubric_total'],20)
        self.assertIsNone(scores['results'][1]['coach_rubric_total'])
        self.assertFalse(scores['approved'])
        self.assertEqual(scores['results'][0]['improvement_plan_draft'][0]['status'],'draft_requires_coach_review')

    def test_fixture_or_unreviewed_context_and_rubric_stay_unscored(self):
        for synthetic in (True,False):
            m,rows=inputs(synthetic=synthetic)
            out=generate_profiles(m,rows,associations=review_ledger('fixture-run',1),calibration=calibration())
            self.assertIsNone(score_profiles(out,rubric())['results'][0]['coach_rubric_total'])
        unreviewed=rubric();unreviewed['reviewed']=False
        self.assertIsNone(score_profiles(out,unreviewed)['results'][0]['coach_rubric_total'])

    def test_cross_run_duplicate_pts_and_source_mismatch_rejected(self):
        m,rows=inputs()
        changed=copy.deepcopy(rows);changed[0]['run_id']='other'
        with self.assertRaises(ValueError):generate_profiles(m,changed)
        with self.assertRaises(ValueError):generate_profiles(m,rows+[rows[0]])
        changed=copy.deepcopy(rows);changed[1]['frame']['pts']=13
        with self.assertRaises(ValueError):generate_profiles(m,changed)
        m,rows=inputs(synthetic=False);m['source']['id']='other'
        with self.assertRaises(ValueError):generate_profiles(m,rows)

    def test_score_rejects_modified_profile_and_invalid_weights(self):
        m,rows=inputs();out=generate_profiles(m,rows)
        bad=rubric();bad['dimensions'][0]['weight']=0.5
        with self.assertRaises(ValueError):score_profiles(out,bad)
        out['profiles'][0]['metrics']['distance_m']['value']=99
        with self.assertRaises(ValueError):score_profiles(out,rubric())

    def test_partial_coverage_and_empty_observations_are_explicit(self):
        m,rows=inputs();m['status']='partial'
        out=generate_profiles(m,rows)
        self.assertEqual(out['profiles'][0]['sample_support']['remaining_source_coverage'],'unknown')
        m['selected_frames']=0;m['decoded_frames']=0
        self.assertEqual(generate_profiles(m,[])['profiles'],[])

    def test_cli_is_additive_local_and_does_not_load_credentials(self):
        from touchline.cli import main
        m,rows=inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            (run/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
            (run/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            with patch('touchline.settings.load_operator_settings',side_effect=AssertionError('No credential loading')),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['player-analysis',str(run),'--output',str(root/'profiles')]),0)
            self.assertTrue((root/'profiles'/'review.md').is_file())
            self.assertEqual(analyze_run(run)['coverage']['observed_tracks'],2)
