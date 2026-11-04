"""Run the real optional CPU association backend; never load a detector/GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from touchline.byte_tracking import ByteTrackTracker
from touchline.contracts import Box,Detection
from touchline.fixtures import synthetic_frames
from touchline.pipeline import run_pipeline
from touchline.tracking import IoUTracker
from touchline.tracking_replay import retrack_run
from touchline.video.sampling import FixedRateSampler


@unittest.skipUnless(importlib.util.find_spec('supervision'),'optional tracking extra not installed')
class ByteTrackingTests(unittest.TestCase):
    def detection(self,x,score=.9):return Detection(Box(x,50,x+30,120),score)

    def test_motion_prediction_keeps_id_after_short_occlusion(self):
        tracker=ByteTrackTracker(6)
        rows=[];baseline=[];iou=IoUTracker()
        for i,x in enumerate((10,16,22,28,None,None,46,52)):
            output=tracker.update(i/6,[] if x is None else [self.detection(x)])
            baseline.extend(iou.update(i/6,[] if x is None else [self.detection(x)]))
            rows.extend(output)
            if x is None:self.assertEqual(output,[])
        self.assertGreaterEqual(len(rows),4)
        self.assertEqual(len({r['local_track_id'] for r in rows}),1)
        self.assertGreater(len({r['local_track_id'] for r in baseline}),1)
        self.assertTrue(all(r['player_id'] is None for r in rows))

    def test_low_score_recovers_existing_track_but_does_not_spawn_a_new_one(self):
        tracker=ByteTrackTracker(6)
        for i in range(4):rows=tracker.update(i/6,[self.detection(10+i*2)])
        tid=rows[0]['local_track_id']
        recovered=tracker.update(4/6,[self.detection(18,.2),self.detection(180,.2)])
        self.assertEqual(len(recovered),1)
        self.assertEqual(recovered[0]['local_track_id'],tid)

    def test_expired_tracks_are_not_automatically_reidentified(self):
        tracker=ByteTrackTracker(6,max_gap_s=.5)
        for i in range(3):first=tracker.update(i/6,[self.detection(20)])
        for i in range(3,10):self.assertEqual(tracker.update(i/6,[]),[])
        later=[]
        for i in range(10,14):later=tracker.update(i/6,[self.detection(20)])
        self.assertNotEqual(first[0]['local_track_id'],later[0]['local_track_id'])

    def test_irregular_update_cadence_is_not_silently_mistimed(self):
        tracker=ByteTrackTracker(6);tracker.update(0,[])
        with self.assertRaises(ValueError):tracker.update(.5,[])

    def test_provenance_uses_update_rate_not_source_rate(self):
        tracker=ByteTrackTracker(6)
        self.assertEqual(tracker.backend.max_time_lost,9)
        self.assertEqual(tracker.provenance()['update_fps'],6)
        self.assertFalse(tracker.provenance()['appearance_reid'])

    def parent(self,root):
        class Recorded:
            def provenance(self):return {'checkpoint':'a'*64,'device':'fixture-cpu'}
            def detect(self,frame):return [Detection(Box(10+frame.index,50,40+frame.index,120),.9)]
        output=root/'parent'
        run_pipeline(synthetic_frames(40),Recorded(),FixedRateSampler(6),output,
            {'id':'synthetic-source'},synthetic=True,max_selected=8,record_proposals=True)
        return output

    def test_replay_preserves_parent_partial_status_and_runs_no_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=self.parent(root)
            before={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
            result=retrack_run(parent,root/'replayed')
            self.assertEqual(result['status'],'partial')
            self.assertEqual(result['selected_frames'],8)
            self.assertEqual(result['tracker']['type'],'bytetrack-supervision')
            self.assertEqual(result['detector_runtime']['inference_calls'],0)
            self.assertEqual(result['detector']['checkpoint'],'a'*64)
            self.assertFalse(result['resume_supported'])
            self.assertNotEqual(result['run_id'],json.loads(before['manifest.json'])['run_id'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()})

    def test_replay_requires_recorded_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=self.parent(root)
            (parent/'raw_proposals.jsonl').unlink()
            with self.assertRaises(ValueError):retrack_run(parent,root/'replayed')
            self.assertFalse((root/'replayed').exists())

    def test_custom_tracker_resume_is_rejected_before_output_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=self.parent(root)
            before=(parent/'manifest.json').read_bytes()
            with self.assertRaises(ValueError):
                run_pipeline([],None,FixedRateSampler(6),parent,{},resume=True,tracker=ByteTrackTracker(6))
            self.assertEqual((parent/'manifest.json').read_bytes(),before)


class ByteTrackingConfigurationTests(unittest.TestCase):
    def test_invalid_rates_rejected_without_importing_backend(self):
        for fps in (0,5,30,float('nan'),True):
            with self.assertRaises(ValueError):ByteTrackTracker(fps)
