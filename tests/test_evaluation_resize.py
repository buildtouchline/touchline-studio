import unittest
from unittest.mock import patch
from touchline.training.evaluate import evaluate_detector
from touchline.cli import main


class EvaluationResizeTests(unittest.TestCase):
    def test_invalid_resize_fails_before_reading_data_or_loading_model(self):
        for changes in ({'min_size':1080},{'min_size':True,'max_size':4096},{'min_size':4096,'max_size':1080},{'min_size':0,'max_size':4096}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):evaluate_detector('missing','unused',**changes)
    def test_cli_passes_explicit_model_input_config_without_metric_changes(self):
        with patch('touchline.training.evaluate.evaluate_detector',return_value={}) as evaluate:
            self.assertEqual(main(['evaluate-detector','release.json','--output','unused','--min-size','1080','--max-size','4096']),0)
            self.assertEqual(evaluate.call_args.kwargs,{'min_size':1080,'max_size':4096})
