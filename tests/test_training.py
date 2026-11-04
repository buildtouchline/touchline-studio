import unittest
from touchline.adapters.torchvision import TorchvisionDetector
from touchline.training.runner import train_detector


class TrainingGuardTests(unittest.TestCase):
    def test_no_implicit_download_or_training(self):
        with self.assertRaises(ValueError):TorchvisionDetector()
        with self.assertRaises(ValueError):train_detector('not-a-dataset.json','unused-output')

    def test_training_caps_validate_before_runtime(self):
        with self.assertRaises(ValueError):train_detector('missing','unused',epochs=0,allow_download=True)

    def test_warm_start_requires_existing_candidate_before_output_creation(self):
        with self.assertRaises(ValueError):train_detector('missing','unused',initial_checkpoint='missing.pt')

    def test_minor_tuning_boundaries_reject_before_runtime_or_output(self):
        for changes in ({'learning_rate':float('nan')},{'learning_rate':True},{'learning_rate':.1},{'weight_decay':0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                train_detector('missing','unused',allow_download=True,**changes)


if __name__=='__main__':unittest.main()
