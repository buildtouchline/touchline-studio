"""Exercise saved-run visual exports using CPU media fixtures only."""
import json
from pathlib import Path
import tempfile
import unittest

import av
from PIL import Image
from test_decode import create_test_mp4
from touchline.data.manifest import sha256_file
from touchline.video.preview import render_run_preview


class VisualPreviewTests(unittest.TestCase):
    def fixture(self, root):
        source = create_test_mp4(root/'source.mp4',24,24,320,180)
        run = root/'run';run.mkdir()
        manifest={'run_id':'visual-fixture','synthetic':True,'status':'partial','decoded_frames':16,
            'source':{'id':sha256_file(source),'path':str(source)},
            'detector':{'checkpoint':'a'*64,'device':'cpu'}}
        (run/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        coverage=[];observations=[];proposals=[]
        for i in range(16):
            frame={'index':i,'timestamp_s':i/24}
            coverage.append({'frame':frame,'state':'detections_present' if i%4==0 else 'unsampled_by_sampler'})
            if i%4==0:
                observations.append({'frame':frame,'local_track_id':7,'bbox_xyxy':[80,60,120,140],'detector_score':.9})
                proposals.append({'frame':frame,'bbox_xyxy':[200,60,220,140],'score':.2})
        for name,rows in (('coverage.jsonl',coverage),('observations.jsonl',observations),('raw_proposals.jsonl',proposals)):
            (run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        return source,run

    def test_real_overlay_has_boxes_preserves_cadence_and_caps_to_processed_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,run=self.fixture(root)
            original=sha256_file(source);observations=sha256_file(run/'observations.jsonl')
            output=root/'preview';result=render_run_preview(run,output)
            with av.open(str(output/'overlay24.mp4')) as video:
                self.assertEqual(video.streams.video[0].average_rate,24)
                frames=list(video.decode(video=0))
            self.assertEqual(len(frames),16)
            image=frames[0].to_image()
            r,g,b=image.getpixel((80,100));self.assertGreater(g,r+60)
            self.assertEqual(Image.open(output/'contact-sheet.jpg').size,(960,882))
            self.assertEqual(Image.open(output/'players.jpg').size,(960,300))
            index=json.loads((output/'track-index.json').read_text(encoding='utf-8'))
            self.assertEqual(index['gallery_ids'],[7])
            self.assertEqual(index['tracks'][0]['observations'],4)
            self.assertFalse(index['tracks_are_roster_identities'])
            self.assertFalse(result['held_overlays_are_new_evidence'])
            self.assertEqual(result['run_status'],'partial')
            self.assertEqual(sha256_file(source),original)
            self.assertEqual(sha256_file(run/'observations.jsonl'),observations)
            with self.assertRaises(FileExistsError):render_run_preview(run,output)

    def test_zero_hold_does_not_present_unsampled_frames_as_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,run=self.fixture(root)
            output=root/'preview';render_run_preview(run,output,hold_seconds=0,max_seconds=.25)
            with av.open(str(output/'overlay24.mp4')) as video:frames=list(video.decode(video=0))
            self.assertEqual(len(frames),6)
            r,g,b=frames[1].to_image().getpixel((80,100))
            self.assertLess(abs(g-r),30)

    def test_wrong_source_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,run=self.fixture(root)
            wrong=create_test_mp4(root/'other.mp4',4,24,320,180)
            with self.assertRaises(ValueError):render_run_preview(run,root/'preview',source=wrong)
            self.assertFalse((root/'preview').exists())

    def test_invalid_preview_limits_do_not_write(self):
        for duration in (0,121,float('nan')):
            with self.assertRaises(ValueError):render_run_preview('missing','unused',max_seconds=duration)

    def test_explicit_full_run_renders_all_analyzed_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,run=self.fixture(root)
            result=render_run_preview(run,root/'full',max_seconds=None,full_run=True)
            self.assertEqual(result['render_scope'],'full_analyzed_run')
            with av.open(str(root/'full/overlay24.mp4')) as video:
                self.assertEqual(len(list(video.decode(video=0))),16)
            with self.assertRaises(ValueError):render_run_preview(run,root/'bad',full_run=True)

    def test_focus_video_stays_at_24_fps_and_does_not_bridge_missing_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,run=self.fixture(root)
            path=run/'observations.jsonl'
            rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['frame']['index']!=4),encoding='utf-8')
            output=root/'focus';result=render_run_preview(run,output,focus_track_id=7)
            with av.open(str(output/'focus24.mp4')) as video:
                self.assertEqual(video.streams.video[0].average_rate,24)
                self.assertEqual((video.streams.video[0].width,video.streams.video[0].height),(960,540))
                frames=list(video.decode(video=0))
            self.assertEqual(len(frames),16)
            self.assertEqual(result['focus_track_id'],7)
            self.assertEqual(result['focus_sha256'],sha256_file(output/'focus24.mp4'))
            self.assertLess(max(frames[4].to_image().getpixel((400,300))),5)
            self.assertFalse(result['tracks_are_roster_identities'])

    def test_focus_requires_an_observed_track_in_the_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,run=self.fixture(root)
            for target in (99,-1,'7'):
                with self.assertRaises(ValueError):render_run_preview(run,root/'focus',focus_track_id=target)
                self.assertFalse((root/'focus').exists())
