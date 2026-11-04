"""Real CPU encode/decode checks; no GPU, Drive, credentials or model calls."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from test_decode import create_test_mp4
from touchline.analysis_policy import preflight_analysis_video
from touchline.data.manifest import sha256_file
from touchline.video.normalize import normalize_video


class VideoNormalizationTests(unittest.TestCase):
    def check_conversion(self, rate, frames, width=256, height=64):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = create_test_mp4(root/'source.mp4', frames, rate, width, height)
            original = sha256_file(source)
            output = root/'converted'
            destination = normalize_video(source, output)
            check = preflight_analysis_video(destination)
            self.assertEqual((check['width'],check['height']),(width,height))
            result = json.loads((output/'normalization.json').read_text(encoding='utf-8'))
            mapping = [json.loads(s) for s in (output/'frame-map.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(result['output_frames'],len(mapping))
            self.assertEqual(check['frames_verified'],len(mapping))
            self.assertAlmostEqual(len(mapping)/24, frames/rate, delta=1/24)
            self.assertEqual(sha256_file(source),original)
            self.assertEqual(result['source_sha256'],original)
            self.assertEqual(result['output_sha256'],sha256_file(destination))
            self.assertEqual(result['frame_map_sha256'],sha256_file(output/'frame-map.jsonl'))
            for i, row in enumerate(mapping):
                self.assertEqual(row['output_frame_index'],i)
                self.assertEqual(row['source_frame_index'],min(frames-1,(i*rate)//24))
                self.assertLessEqual(Fraction(*row['source_relative_time']),Fraction(i,24))
            with self.assertRaises(FileExistsError):normalize_video(source,output)
            return mapping

    def test_25_to_24_preserves_duration_geometry_source_and_maps_drops(self):
        rows = self.check_conversion(25,50)
        self.assertEqual(len(rows),48)
        self.assertLess(len(rows),50)

    def test_lower_rate_duplicates_with_exact_source_mapping(self):
        rows = self.check_conversion(12,12)
        self.assertEqual(len(rows),24)
        self.assertEqual(rows[0]['source_frame_index'],rows[1]['source_frame_index'])

    def test_24_fps_copy_has_identity_mapping(self):
        self.check_conversion(24,12)

    def test_actual_4096_panorama_is_accepted(self):
        self.check_conversion(25,5,4096,1080)

    def test_portrait_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=create_test_mp4(root/'portrait.mp4',4,25,64,128)
            with self.assertRaises(ValueError):normalize_video(source,root/'copy')
            self.assertFalse((root/'copy').exists())

    def test_bounded_excerpt_keeps_original_frame_map_and_full_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=create_test_mp4(root/'source.mp4',50,25,256,64)
            digest=sha256_file(source);output=root/'excerpt'
            video=normalize_video(source,output,max_seconds=.5)
            result=json.loads((output/'normalization.json').read_text())
            rows=[json.loads(s) for s in (output/'frame-map.jsonl').read_text().splitlines()]
            self.assertEqual(result['source_processing_scope'],'bounded_excerpt')
            self.assertEqual(result['output_frames'],12)
            self.assertEqual(result['source_duration_used_s'],.5)
            self.assertEqual(result['source_sha256'],digest)
            self.assertEqual(sha256_file(source),digest)
            self.assertEqual(preflight_analysis_video(video)['frames_verified'],12)
            self.assertEqual([r['source_frame_index'] for r in rows],[(i*25)//24 for i in range(12)])

    def test_invalid_excerpt_limits_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for duration in (0,-1,True,float('inf'),121):
                with self.assertRaises(ValueError):normalize_video(root/'absent.mp4',root/'copy',max_seconds=duration)
            self.assertFalse((root/'copy').exists())
