import copy,unittest
from touchline.repair_policy import CHECKS
from touchline.training_trials import admit_minor_trial


class MinorTrialTests(unittest.TestCase):
    def setUp(self):
        self.b={'recipe_sha256':'a'*64,'checkpoint_sha256':'b'*64,'learning_rate':.001,'weight_decay':.0005}
        self.p={'kind':'tuning','baseline_sha256':'a'*64,'rollback_sha256':'b'*64,'changes':{'learning_rate':.0008},'expected_outcome':'Hypothesis only','validation_checks':sorted(CHECKS)}
        self.r={'status':'complete','restored_images':1710,'release_sha256':'c'*64,'input_hashes':{'candidate.pt':'b'*64},'training':{'checkpoint_sha256':'d'*64}}
        self.e={'split':'val','subset':False,'images':570,'manifest_sha256':'c'*64,'model':{'checkpoint':'d'*64},'metrics':{'ap50':.3}}
    def test_only_isolated_minor_trial_admitted(self):
        result=admit_minor_trial(self.p,self.b,self.r,self.e)
        self.assertEqual(result['learning_rate'],.0008);self.assertEqual(result['promotion'],'not performed')
    def test_incomplete_images_stale_weights_and_subset_validation_block(self):
        for r,e in (({**self.r,'restored_images':100},self.e),({**self.r,'input_hashes':{}},self.e),(self.r,{**self.e,'subset':True}),(self.r,{**self.e,'images':30})):
            with self.assertRaises(ValueError):admit_minor_trial(self.p,self.b,r,e)
    def test_large_recipe_change_cannot_auto_execute(self):
        with self.assertRaises(ValueError):admit_minor_trial({**self.p,'changes':{'learning_rate':.1}},self.b,self.r,self.e)
