import unittest
import numpy as np
from touchline.motion_tracking import MotionTracker,estimate_motion
def detection(box,feature=(1.,0.),score=.9):return {'bbox_xyxy':box,'score':score,'feature':np.asarray(feature)}
class MotionTrackingTests(unittest.TestCase):
    def test_camera_translation_is_not_player_displacement(self):
        t=MotionTracker(200,100);t.update(0,[detection([20,20,30,50])])
        camera=np.array([[1.,0.,7.],[0.,1.,0.]])
        rows=t.update(.1,[detection([27,20,37,50])],camera=camera,camera_valid=True)
        self.assertEqual(rows[0]['track_id'],1);np.testing.assert_allclose(rows[0]['camera_compensated_delta_px'],[0,0],atol=1e-6)
    def test_flow_and_appearance_keep_ids_through_crossing(self):
        t=MotionTracker(200,100);a=detection([20,20,30,50]);b=detection([60,20,70,50],(0.,1.))
        t.update(0,[a,b]);t.update(.1,[a,b])
        rows=t.update(.2,[detection([20,20,30,50],(0.,1.)),detection([60,20,70,50])],foreground={1:[40,0],2:[-40,0]})
        self.assertEqual({r['track_id'] for r in rows},{1,2});self.assertGreater(rows[0]['footpoint_xy'][0],rows[1]['footpoint_xy'][0])
    def test_low_confidence_does_not_create_tracks_and_lost_box_is_hidden(self):
        t=MotionTracker(200,100);self.assertEqual(t.update(0,[detection([20,20,30,50],score=.2)]),[]);self.assertEqual(t.next_id,1)
        t.update(.1,[detection([20,20,30,50])]);t.update(.2,[detection([20,20,30,50])]);self.assertEqual(t.update(1.1,[]),[])
    def test_invalid_or_reversed_time_and_boxes_are_rejected(self):
        t=MotionTracker(200,100);t.update(0,[])
        with self.assertRaises(ValueError):t.update(0,[])
        with self.assertRaises(ValueError):t.update(.1,[detection([30,20,20,50])])
    def test_background_camera_motion_on_translated_texture(self):
        import cv2
        rng=np.random.default_rng(3);a=rng.integers(0,256,(120,200),dtype=np.uint8);b=cv2.warpAffine(a,np.float32([[1,0,3],[0,1,2]]),(200,120))
        matrix,valid,quality,_=estimate_motion(a,b,{},200)
        self.assertTrue(valid);np.testing.assert_allclose(matrix[:,2],[3,2],atol=.5)
    def test_unique_appearance_recovers_long_gap_without_forcing_duplicate_identity(self):
        t=MotionTracker(500,200);a=detection([20,20,40,80]);t.update(0,[a]);t.update(.1,[a]);t.update(1.,[])
        rows=t.update(3.,[detection([60,20,80,80])]);self.assertEqual(rows[0]['track_id'],1);self.assertEqual(t.next_id,2)
    def test_nested_box_does_not_start_a_second_track_for_the_same_body(self):
        t=MotionTracker(200,150);a=detection([20,20,60,100]);t.update(0,[a]);t.update(.1,[a,detection([30,30,50,80])])
        self.assertEqual(t.next_id,2)
    def test_session_cap_keeps_extra_detection_unassigned_instead_of_recycling_ids(self):
        t=MotionTracker(500,200,max_session_ids=1);a=detection([20,20,40,80]);t.update(0,[a]);t.update(.1,[a,detection([200,20,220,80],(0.,1.))])
        self.assertEqual(t.next_id,2);self.assertEqual(len(t.unassigned),1);self.assertEqual(t.unassigned[0]['reason'],'session_id_capacity_reached')
    def test_dormant_identity_cannot_steal_an_active_players_detection(self):
        t=MotionTracker(500,200);a=detection([20,20,40,80]);b=detection([80,20,100,80],(0.,1.));t.update(0,[a,b]);t.update(.1,[a,b]);t.update(.5,[b])
        t.tracks[1].box=np.array([80.,20.,100.,80.]);t.tracks[1].feature=np.array([0.,1.])
        rows=t.update(1.,[detection([82,20,102,80],(0.,1.))]);self.assertEqual([r['track_id'] for r in rows],[2])
    def test_invalid_flow_is_explicitly_rejected_without_nan_or_false_verification(self):
        for delta in ([float('nan'),0],[float('inf'),1],[1,2,3],'bad'):
            with self.subTest(delta=delta):
                t=MotionTracker(200,100);a=detection([20,20,40,80]);t.update(0,[a]);t.update(.1,[a])
                rows=t.update(.2,[],foreground={1:delta})
                self.assertEqual(len(rows),1);self.assertFalse(rows[0]['foreground_flow_valid'])
                self.assertEqual(rows[0]['foreground_flow_rejection_reason'],'invalid_displacement')
                self.assertTrue(np.isfinite(rows[0]['bbox_xyxy']).all())
    def test_implausible_flow_is_not_silently_clamped_into_a_verified_edge_box(self):
        t=MotionTracker(200,100);a=detection([20,20,40,80]);t.update(0,[a]);t.update(.1,[a])
        rows=t.update(.2,[],foreground={1:[100000,0]})
        self.assertEqual(rows[0]['foreground_flow_rejection_reason'],'implausible_displacement')
        self.assertFalse(rows[0]['foreground_flow_valid']);self.assertLess(rows[0]['bbox_xyxy'][2],100)
    def test_valid_offscreen_camera_exit_is_hidden_instead_of_pinned_to_border(self):
        t=MotionTracker(200,100);a=detection([170,20,190,80]);t.update(0,[a]);t.update(.1,[a])
        rows=t.update(.2,[],camera=np.array([[1.,0.,40.],[0.,1.,0.]]),camera_valid=True)
        self.assertEqual(rows,[]);self.assertGreater(t.tracks[1].box[0],200)
    def test_unconfirmed_birth_stays_unassigned_and_does_not_consume_a_session_id(self):
        t=MotionTracker(200,100);a=detection([20,20,40,80]);a['new_track_allowed']=False
        t.update(0,[a]);self.assertEqual(t.next_id,1)
        self.assertEqual(t.unassigned[0]['reason'],'semantic_confirmation_required')
        a['new_track_allowed']=True;t.update(.1,[a]);self.assertEqual(t.next_id,2)
        a['new_track_allowed']=False;rows=t.update(.2,[a]);self.assertEqual(rows[0]['track_id'],1)
    def test_ambiguous_dormant_match_does_not_create_a_duplicate_identity(self):
        t=MotionTracker(300,150);t.update(0,[detection([20,20,40,80]),detection([80,20,100,80])])
        for track in t.tracks.values():track.box=np.array([50.,20.,70.,80.])
        rows=t.update(1.5,[detection([50,20,70,80])])
        self.assertEqual(rows,[]);self.assertEqual(t.next_id,3)
        self.assertEqual(t.unassigned[0]['reason'],'ambiguous_reidentification')
