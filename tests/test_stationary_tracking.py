import unittest
import copy
import numpy as np
from touchline.stationary_tracking import StationaryTracker


def detection(box,feature=(1.,0.),teacher=False):
    return {'bbox_xyxy':box,'score':.95,'feature':np.array(feature),'teacher_supported':teacher}


class StationaryTrackingTests(unittest.TestCase):
    def test_false_short_tracklets_do_not_exhaust_session_slots(self):
        tracker=StationaryTracker(1000,500,max_session_ids=1)
        tracker.update(0,[detection([5,5,15,30])]);tracker.update(1,[])
        self.assertEqual(tracker.next_id,1)
        for i in range(6):rows=tracker.update(2+i/6,[detection([100+i,100,130+i,180])])
        self.assertEqual(rows[0]['track_id'],1)

    def test_lost_session_record_is_not_deleted_or_recycled(self):
        tracker=StationaryTracker(1000,500,max_session_ids=1)
        for i in range(6):tracker.update(i/6,[detection([100,100,130,180])])
        tracker.update(2,[])
        rows=tracker.update(3,[detection([100,100,130,180])])
        self.assertEqual(rows[0]['track_id'],1)
        tracker.update(30,[])
        self.assertTrue(any(t.session_id==1 for t in tracker.tracks.values()))
        self.assertEqual(tracker.next_id,2)

    def test_exact_detector_anchor_is_not_shifted_by_box_smoothing(self):
        tracker=StationaryTracker(1000,500)
        for i in range(6):rows=tracker.update(i/6,[detection([100+i*3,100,130+i*3,180])])
        self.assertEqual(rows[0]['bbox_xyxy'],[115.,100.,145.,180.])
        self.assertEqual(rows[0]['evidence'],'detector_anchor')
        self.assertFalse(rows[0]['identity_confirmed'])

    def test_predicted_boxes_are_labelled_and_stop_after_bound(self):
        tracker=StationaryTracker(1000,500)
        for i in range(6):tracker.update(i/6,[detection([100,100,130,180])])
        rows=tracker.update(1,[])
        self.assertEqual(rows[0]['evidence'],'kalman_prediction')
        self.assertIsNone(rows[0]['detector_bbox_xyxy'])
        self.assertEqual(tracker.update(2,[]),[])

    def test_teacher_supported_births_still_need_persistent_detections(self):
        tracker=StationaryTracker(1000,500,teacher_births_only=True)
        for i in range(8):self.assertEqual(tracker.update(i/6,[detection([100,100,130,180])]),[])
        rows=tracker.update(8/6,[detection([100,100,130,180],teacher=True)])
        self.assertEqual(rows[0]['track_id'],1)

    def test_invalid_geometry_and_time_never_reach_assignment(self):
        tracker=StationaryTracker(1000,500)
        with self.assertRaises(ValueError):tracker.update(0,[detection([10,10,9,20])])
        with self.assertRaises(ValueError):tracker.update(0,[])

    def test_pending_duplicate_cannot_steal_confirmed_tracks_detection(self):
        tracker=StationaryTracker(1000,500,max_session_ids=1)
        for i in range(6):tracker.update(i/6,[detection([100,100,130,180])])
        known=next(t for t in tracker.tracks.values() if t.session_id==1)
        pending=copy.deepcopy(known);pending.token=999;pending.session_id=None;pending.hits=1
        pending.box=np.array([103.,100.,133.,180.]);pending.state[:4]=tracker._measurement(pending.box)
        tracker.tracks[999]=pending
        rows=tracker.update(1,[detection([103,100,133,180])])
        anchor=next(r for r in rows if r['track_id']==1)
        self.assertEqual(anchor['evidence'],'detector_anchor')
        self.assertEqual(anchor['detector_bbox_xyxy'],[103.,100.,133.,180.])
        self.assertEqual(tracker.tracks[999].hits,1)

    def test_foreground_measurement_moves_prediction_without_faking_anchor(self):
        tracker=StationaryTracker(1000,500)
        for i in range(6):tracker.update(i/6,[detection([100,100,130,180])])
        rows=tracker.update(1,[],foreground={1:[3.,0.]})
        self.assertEqual(rows[0]['evidence'],'optical_flow_prediction')
        self.assertAlmostEqual(rows[0]['footpoint_xy'][0],118.)
        self.assertIsNone(rows[0]['detector_bbox_xyxy'])
