"""Synthetic telemetry checks; no Torch/CUDA runtime or credentials are accessed."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock
from touchline.adapters.torchvision import TorchvisionDetector


class DetectorRuntimeStatsTests(unittest.TestCase):
    def detector(self, device):
        detector=TorchvisionDetector.__new__(TorchvisionDetector)
        detector.device=device;detector.actual_device=device
        detector._input_device=device;detector._inference_calls=12
        detector._detector_wall_seconds=2.5;detector.torch=Mock()
        return detector

    def test_cuda_stats_report_actual_model_and_input_placement(self):
        detector=self.detector('cuda:0')
        detector.torch.cuda.get_device_name.return_value='FAKE CUDA FIXTURE'
        detector.torch.cuda.max_memory_allocated.return_value=123456
        stats=detector.runtime_stats()
        self.assertEqual(stats['model_parameter_device'],'cuda:0')
        self.assertEqual(stats['last_input_device'],'cuda:0')
        self.assertEqual(stats['inference_calls'],12)
        self.assertEqual(stats['gpu_peak_allocated_bytes'],123456)
        detector.torch.cuda.max_memory_allocated.assert_called_once_with('cuda:0')

    def test_cpu_stats_do_not_query_cuda(self):
        detector=self.detector('cpu');stats=detector.runtime_stats()
        self.assertNotIn('gpu_peak_allocated_bytes',stats)
        detector.torch.cuda.get_device_name.assert_not_called()

    def test_pipeline_persists_metrics_from_inference_process(self):
        from touchline.fixtures import SyntheticDetector, synthetic_frames
        from touchline.pipeline import run_pipeline
        from touchline.video.sampling import FixedRateSampler
        class Instrumented(SyntheticDetector):
            def runtime_stats(self):return {'model_parameter_device':'cpu','inference_calls':2}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            run_pipeline(synthetic_frames(8),Instrumented(),FixedRateSampler(6),output,{'id':'synthetic-source'},synthetic=True)
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['detector_runtime']['inference_calls'],2)
            self.assertEqual(manifest['detector_runtime']['model_parameter_device'],'cpu')
