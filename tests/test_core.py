import json
from pathlib import Path
import tempfile
import unittest
from touchline.contracts import Box,Detection,Frame
from touchline.fixtures import SyntheticDetector,synthetic_frames
from touchline.pipeline import run_pipeline
from touchline.tracking import IoUTracker
from touchline.video.sampling import Anchor24Sampler,FixedRateSampler,merge_intervals


class CoreTests(unittest.TestCase):
    def test_anchor_schedule_crosses_seconds(self):
        sampler=Anchor24Sampler()
        self.assertEqual([i for i in range(48) if sampler.select(i,i/24)],[5,17,29,41])

    def test_anchor_rejects_vfr_and_skipped_decode(self):
        with self.assertRaises(ValueError):Anchor24Sampler().select(1,1/24)
        sampler=Anchor24Sampler();sampler.select(0,0)
        with self.assertRaises(ValueError):sampler.select(1,0.1)

    def test_fixed_rate_uses_actual_timestamps(self):
        sampler=FixedRateSampler(2)
        self.assertEqual([i for i,t in enumerate([0,.1,.51,.7,1.1]) if sampler.select(i,t)],[0,2,4])

    def test_sampling_rejects_duplicate_time(self):
        sampler=FixedRateSampler();sampler.select(0,0)
        with self.assertRaises(ValueError):sampler.select(1,0)

    def test_merge_padding_and_source_bounds(self):
        self.assertEqual(merge_intervals([(1,2),(2.1,3)],.2,3),[(.8,3)])
        with self.assertRaises(ValueError):merge_intervals([(3,2)])

    def test_box_and_score_validation(self):
        self.assertEqual(Box(0,0,10,10).iou(Box(0,0,10,10)),1)
        with self.assertRaises(ValueError):Box(10,0,0,10)
        with self.assertRaises(ValueError):Detection(Box(0,0,10,10),float('nan'))

    def test_frame_metadata_does_not_copy_image(self):
        class Uncopyable:
            def __deepcopy__(self,memo):raise AssertionError('image must not be copied')
        frame=Frame('s',0,0,1,24,0,10,10,Uncopyable())
        self.assertNotIn('image',frame.metadata())

    def test_tracker_continuity_is_not_roster_identity(self):
        tracker=IoUTracker()
        first=tracker.update(0,[Detection(Box(1,1,11,11),.9)])
        second=tracker.update(.5,[Detection(Box(2,1,12,11),.9)])
        self.assertEqual(first[0]['local_track_id'],second[0]['local_track_id'])
        self.assertIsNone(second[0]['player_id'])
        third=tracker.update(2,[Detection(Box(2,1,12,11),.9)])
        self.assertNotEqual(second[0]['local_track_id'],third[0]['local_track_id'])

    def test_tracker_flags_competing_matches(self):
        tracker=IoUTracker()
        tracker.update(0,[Detection(Box(0,0,20,20),.9)])
        output=tracker.update(.5,[Detection(Box(1,0,21,20),.9),Detection(Box(2,0,22,20),.8)])
        self.assertTrue(all(x['association_state']=='ambiguous' for x in output))

    def test_synthetic_pipeline_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            result=run_pipeline(synthetic_frames(),SyntheticDetector(),Anchor24Sampler(),out,{'id':'fixture'},synthetic=True)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['selected_frames'],8)
            self.assertTrue(result['synthetic']);self.assertIn('SYNTHETIC', (out/'review.html').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):run_pipeline(synthetic_frames(),SyntheticDetector(),Anchor24Sampler(),out,{'id':'fixture'})

    def test_limit_is_partial_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=run_pipeline(synthetic_frames(),SyntheticDetector(),Anchor24Sampler(),Path(tmp)/'r',{},max_selected=2)
            self.assertEqual(result['status'],'partial')

    def test_failed_run_keeps_manifest(self):
        def broken():
            yield next(synthetic_frames())
            raise ValueError('fixture decode failure')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'r'
            with self.assertRaises(ValueError):run_pipeline(broken(),SyntheticDetector(),FixedRateSampler(),out,{})
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'failed')


    def test_coverage_ledger_records_all_frames_and_distinguishes_skipped(self):
        test_frames = [Frame("source_fixture", i, i, 1, 24, i / 24, 320, 180) for i in range(10)]

        class SpyDetector:
            def __init__(self):
                self.calls = []
            def provenance(self):
                return {"adapter": "spy-detector"}
            def detect(self, frame):
                self.calls.append(frame.index)
                if frame.index == 1:
                    return [Detection(Box(10, 10, 40, 60), 0.85)]
                return []

        class CustomSampler:
            def select(self, index, timestamp_s):
                return index in (1, 3)

        detector = SpyDetector()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run_coverage'
            result = run_pipeline(test_frames, detector, CustomSampler(), out, {'id': 'cov_test'})
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['decoded_frames'], 10)
            self.assertEqual(result['selected_frames'], 2)

            self.assertEqual(detector.calls, [1, 3])

            coverage_records = [json.loads(line) for line in (out / 'coverage.jsonl').read_text(encoding='utf-8').splitlines() if line]
            self.assertEqual(len(coverage_records), 10)
            self.assertEqual([r['frame']['index'] for r in coverage_records], list(range(10)))

            self.assertEqual(coverage_records[1]['state'], 'detections_present')
            self.assertEqual(coverage_records[1]['detections'], 1)

            self.assertEqual(coverage_records[3]['state'], 'no_detection_not_inactivity')
            self.assertEqual(coverage_records[3]['detections'], 0)

            self.assertEqual(coverage_records[0]['state'], 'unsampled_by_sampler')
            self.assertIsNone(coverage_records[0]['detections'])

            skipped = [r for r in coverage_records if r['frame']['index'] not in (1, 3)]
            self.assertEqual(len(skipped), 8)
            self.assertTrue(all(r['state'] == 'unsampled_by_sampler' and r['detections'] is None for r in skipped))

            # Explicitly assert timestamp_s and pts preservation on skipped frames
            self.assertAlmostEqual(coverage_records[0]['frame']['timestamp_s'], 0.0)
            self.assertEqual(coverage_records[0]['frame']['pts'], 0)
            self.assertAlmostEqual(coverage_records[4]['frame']['timestamp_s'], 4 / 24)
            self.assertEqual(coverage_records[4]['frame']['pts'], 4)
            for r in skipped:
                idx = r['frame']['index']
                self.assertEqual(r['frame']['pts'], idx)
                self.assertAlmostEqual(r['frame']['timestamp_s'], idx / 24)


if __name__=='__main__':unittest.main()
