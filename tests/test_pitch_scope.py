import unittest
import numpy as np
from touchline.pitch_scope import estimate_pitch_mask,footpoint_scope

class PitchScopeTests(unittest.TestCase):
    def test_pitch_footpoint_is_distinguished_from_background_without_changing_boxes(self):
        image=np.full((100,160,3),120,dtype=np.uint8);image[45:]=[70,130,50]
        mask=estimate_pitch_mask(image);self.assertIsNotNone(mask)
        self.assertEqual(footpoint_scope([20,20,40,80],160,100,mask),'inside_estimated_pitch')
        self.assertEqual(footpoint_scope([20,5,40,30],160,100,mask),'outside_estimated_pitch')
    def test_no_reliable_pitch_is_unknown_instead_of_declaring_all_people_outside(self):
        image=np.full((100,160,3),120,dtype=np.uint8)
        self.assertIsNone(estimate_pitch_mask(image));self.assertEqual(footpoint_scope([20,20,40,80],160,100,None),'unknown')
    def test_small_isolated_green_background_patch_is_not_a_pitch(self):
        image=np.full((100,160,3),120,dtype=np.uint8);image[5:15,5:15]=[70,130,50]
        self.assertIsNone(estimate_pitch_mask(image))
    def test_teal_netting_does_not_extend_the_grass_scope_into_the_background(self):
        image=np.full((100,160,3),120,dtype=np.uint8);image[45:]=[70,130,50];image[:,130:]=[20,120,110]
        mask=estimate_pitch_mask(image);self.assertIsNotNone(mask)
        self.assertEqual(mask[10,140],0);self.assertNotEqual(mask[80,60],0)
