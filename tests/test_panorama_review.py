import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('panorama_worker', Path(__file__).parents[1] / 'tools/modal_panorama_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PanoramaReviewTests(unittest.TestCase):
    def test_scope_rejects_other_footage_or_sampling(self):
        job = {k: 'a' * 64 for k in ('video_sha256', 'weights_sha256', 'code_sha256', 'frame_map_sha256')}
        job.update(expected_frames=5760, geometry=[4096, 1080], detector_stride=4, session_id_cap=24, stationary_camera=True)
        self.assertIs(worker.validate_job(job), job)
        for key, value in [('expected_frames', 4320), ('detector_stride', 8), ('stationary_camera', False), ('geometry', [1920, 1080])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_job({**job, key: value})

    def test_single_frame_candidate_cannot_consume_an_id(self):
        self.assertFalse(worker.confirmed_birth([0, 0, 10, 20], .9, [], lambda a, b: 1))
        self.assertFalse(worker.confirmed_birth([0, 0, 10, 20], .9, [([0, 0, 10, 20], .4)], lambda a, b: 1))
        self.assertFalse(worker.confirmed_birth([0, 0, 10, 20], .9, [([50, 0, 60, 20], .9)], lambda a, b: 0))
        self.assertTrue(worker.confirmed_birth([0, 0, 10, 20], .9, [([1, 0, 11, 20], .9)], lambda a, b: .8))
