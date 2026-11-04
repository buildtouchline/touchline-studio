"""Independent regression cases: synthetic data, no credentials or external APIs."""
import json
import copy
from pathlib import Path
import tempfile
import unittest

from touchline.analytics import compute_track_analytics, generate_match_analytics
from touchline.metrics import Calibration
from touchline.reports import (approve_report_version, check_report_access,
                              compare_two_sessions, compare_two_video_runs, generate_player_report_draft)
from review_fixtures import review_entry, review_ledger


def observations():
    return [{'local_track_id':7,'bbox_xyxy':[100+i*8,200,140+i*8,300],
             'detector_score':.9,'association_state':'tentative',
             'frame':{'index':i,'pts':i*1000,'timestamp_s':i*.25,'width':1920,'height':1080}}
            for i in range(6)]


class ReviewRegressions(unittest.TestCase):
    def reviewed_draft(self):
        analytics=generate_match_analytics({'run_id':'fixture-run','source':'fixture','synthetic':True},
            observations(),associations=review_ledger('fixture-run',7,'fixture-player'))
        return generate_player_report_draft(analytics,7,recipient_scope={
            'allowed_recipients':['fixture-parent'],'parent_sharing_authorized':True})

    def test_calibration_without_identity_review_cannot_produce_metres(self):
        cal=Calibration((52/1920,0,0,0,34/1080,0,0,0,1),'synthetic-survey',0,60,
                        (0,0,1920,1080),source_geometry=(1920,1080))
        track=compute_track_analytics(7,observations(),1920,1080,calibration=cal)
        self.assertFalse(track['movement'].get('identity_reviewed',False))
        self.assertIsNone(track['movement']['distance_m'])

    def test_explicit_track_cannot_acquire_the_requested_player_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('a','b'):
                folder=root/name;folder.mkdir()
                (folder/'analytics.json').write_text(json.dumps({'run_id':'fixture-'+name,
                    'player_roster':[{'local_track_id':7,'confirmed_player_id':None,'visible_duration_s':10}]}))
            result=compare_two_video_runs(root/'a',root/'b',confirmed_player_id='fixture-player',
                                          local_track_id_a=7,local_track_id_b=7)
            self.assertFalse(result['comparable'])

    def test_same_run_cannot_be_presented_as_two_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'analytics.json').write_text(json.dumps({'run_id':'fixture-same',
                'player_roster':[{'local_track_id':7,'confirmed_player_id':'fixture-player','visible_duration_s':10}]}))
            result=compare_two_video_runs(folder,folder,confirmed_player_id='fixture-player')
            self.assertFalse(result['comparable'])

    def test_parent_cannot_read_approved_content_after_mutation(self):
        manifest={'run_id':'fixture-run','source':'fixture','synthetic':True,'decoded_frames':6,'selected_frames':6}
        analytics=generate_match_analytics(manifest,observations(),associations=review_ledger('fixture-run',7,'fixture-player'))
        draft=generate_player_report_draft(analytics,7,recipient_scope={
            'team_id':'fixture-team','allowed_recipients':['fixture-parent'],'parent_sharing_authorized':True})
        approved=approve_report_version(draft,reviewer='fixture-coach',expected_version_hash=draft['version_hash'])
        approved['coach_observations']['general_notes']='Changed after approval'
        access=check_report_access(approved,requester_id='fixture-parent',requester_role='parent',target_player_id='fixture-player')
        self.assertFalse(access['authorized'])

    def test_missing_foreign_revoked_and_malformed_reviews_fail_closed(self):
        cal=Calibration((1,0,0,0,1,0,0,0,1),'fixture-survey',0,60,(0,0,1920,1080))
        variants=[None,{'player_id':'fixture-player'},
            {**review_entry('fixture-run',7),'run_id':'other-run'},
            {**review_entry('fixture-run',7),'local_track_id':8},
            {**review_entry('fixture-run',7),'confirmed':False},
            {**review_entry('fixture-run',7),'decision':'revoked'},
            {**review_entry('fixture-run',7),'reviewed_by':None},
            {**review_entry('fixture-run',7),'revision':True},
            {**review_entry('fixture-run',7),'revision':0},
            {**review_entry('fixture-run',7),'player_id':' '},
            {**review_entry('fixture-run',7),'review_ref':'invented-track-review'}]
        for association in variants:
            with self.subTest(association=association):
                row=compute_track_analytics(7,observations(),1920,1080,calibration=cal,
                                             run_id='fixture-run',association=association)
                self.assertIsNone(row['confirmed_player_id'])
                self.assertIsNone(row['movement']['distance_m'])

    def test_observation_player_label_is_not_review_evidence(self):
        obs=observations()
        for row in obs:row['player_id']='fixture-player'
        result=compute_track_analytics(7,obs,1920,1080,run_id='fixture-run')
        self.assertIsNone(result['confirmed_player_id'])

    def test_reviewed_identity_does_not_make_ambiguous_segments_eligible(self):
        obs=observations()
        for row in obs:row['association_state']='ambiguous'
        cal=Calibration((1,0,0,0,1,0,0,0,1),'fixture-survey',0,60,(0,0,1920,1080))
        result=compute_track_analytics(7,obs,1920,1080,calibration=cal,run_id='fixture-run',
                                       association=review_entry('fixture-run',7))
        self.assertIsNone(result['movement']['distance_m'])

    def test_positive_review_does_not_allow_duplicate_run_session_or_source(self):
        a={'session_id':'session-a','run_id':'run-a','local_track_id':7,'confirmed_player_id':'P07',
           'association_revision':1,'identity_review':review_entry('run-a',7),'objective_id':'objective',
           'criteria_revision':'v1','drill_id':'drill','visible_duration_s':2}
        b={**a,'session_id':'session-b','run_id':'run-b','identity_review':review_entry('run-b',7)}
        self.assertTrue(compare_two_sessions(a,b)['comparable'])
        for key,value in [('run_id','run-a'),('session_id','session-a'),
                          ('source_video','same-recording.mp4'),('source_sha256','a'*64)]:
            left=copy.deepcopy(a);right=copy.deepcopy(b)
            if key.startswith('source_'):left[key]=value
            right[key]=value
            with self.subTest(key=key):self.assertFalse(compare_two_sessions(left,right)['comparable'])

    def test_stale_or_differently_scoped_ledger_cannot_confirm_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('a','b'):
                path=root/name;path.mkdir()
                analytics=generate_match_analytics({'run_id':name,'source':name+'.mp4'},observations(),
                                                  associations=review_ledger(name,7))
                (path/'analytics.json').write_text(json.dumps(analytics))
                (path/'associations.json').write_text(json.dumps(review_ledger(name,7)))
            self.assertTrue(compare_two_video_runs(root/'a',root/'b',confirmed_player_id='P07')['comparable'])
            for ledger in [review_ledger('foreign-run',7),review_ledger('b',7,revision=2),
                           review_ledger('b',7,'other-player')]:
                with self.subTest(ledger=ledger):
                    (root/'b'/'associations.json').write_text(json.dumps(ledger))
                    self.assertFalse(compare_two_video_runs(root/'a',root/'b',confirmed_player_id='P07')['comparable'])

    def test_approval_checks_current_content_before_recording_decision(self):
        draft=self.reviewed_draft()
        expected=draft['version_hash']
        draft['coach_observations']['general_notes']='Changed before approval'
        with self.assertRaises(ValueError):approve_report_version(draft,reviewer='fixture-coach',expected_version_hash=expected)

    def test_every_shared_projection_field_is_bound_to_exact_approval(self):
        draft=self.reviewed_draft()
        approved=approve_report_version(draft,reviewer='fixture-coach',expected_version_hash=draft['version_hash'])
        roundtrip=json.loads(json.dumps(approved))
        self.assertTrue(check_report_access(roundtrip,requester_id='fixture-parent',requester_role='parent')['authorized'])
        changes={'player':{'confirmed_player_id':'other-player','identity_confirmed':True},
                 'revision':2,'recipient_scope':{'allowed_recipients':['new-parent'],'parent_sharing_authorized':True},
                 'claims':[],'source_lineage':{},'objective':{'objective_id':'other-objective'},'report_id':'other-report'}
        for key,value in changes.items():
            changed=copy.deepcopy(roundtrip);changed[key]=value
            recipient='new-parent' if key=='recipient_scope' else 'fixture-parent'
            with self.subTest(key=key):
                self.assertFalse(check_report_access(changed,requester_id=recipient,requester_role='parent')['authorized'])

    def test_approval_revision_mismatch_or_malformed_approval_denies_parent(self):
        draft=self.reviewed_draft()
        approved=approve_report_version(draft,reviewer='fixture-coach',expected_version_hash=draft['version_hash'])
        for approval in [{**approved['approval'],'approved_revision':99},'not-an-approval']:
            changed=copy.deepcopy(approved);changed['approval']=approval
            with self.subTest(approval=approval):
                self.assertFalse(check_report_access(changed,requester_id='fixture-parent',requester_role='parent')['authorized'])
