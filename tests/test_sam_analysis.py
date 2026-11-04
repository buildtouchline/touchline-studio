import unittest
from touchline.sam_analysis import compare_boxes,FRAME_INDICES


class SamAnalysisTests(unittest.TestCase):
    def test_exact_sampled_frames_and_one_to_one_agreement(self):
        self.assertEqual(FRAME_INDICES,(0,60,120,180,240,300,360,420))
        box={'bbox_xyxy':[0,0,10,10]}
        r=compare_boxes([box],[box,box],20,20)
        self.assertEqual(len(r['matched']),1);self.assertEqual(len(r['sam_only']),1)
    def test_disagreements_do_not_become_truth_or_training_labels(self):
        r=compare_boxes([], [{'bbox_xyxy':[1,1,9,9]}],20,20)
        self.assertEqual(r['sam_only'],[0]);self.assertFalse(r['training_eligible']);self.assertIsNone(r['identity_accuracy'])
    def test_invalid_geometry_rejected(self):
        for b in ([0,0,30,30],[0,0,float('nan'),10]):
            with self.assertRaises(ValueError):compare_boxes([{'bbox_xyxy':b}],[],20,20)
    def test_no_detections_is_not_perfect_accuracy(self):
        r=compare_boxes([],[],20,20)
        self.assertIsNone(r['identity_accuracy']);self.assertFalse(r['ground_truth_available'])
