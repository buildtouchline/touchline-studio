import copy,unittest
from touchline.repair import canonical_hash
from touchline.repair_policy import CHECKS
from touchline.training.recipes import validate_resize_recipe
class ReviewedResizeTests(unittest.TestCase):
    def doc(self):
        baseline={'recipe_sha256':'a'*64,'checkpoint_sha256':'b'*64,'learning_rate':.0008,'weight_decay':.0005}
        proposal={'kind':'tuning','baseline_sha256':'a'*64,'rollback_sha256':'b'*64,'changes':{'training_min_size':720,'training_max_size':1280,'learning_rate':.0005,'weight_decay':.0005,'epochs':4,'max_steps':2000,'max_seconds':2700,'freeze_backbone':True},'expected_outcome':'Hypothesis only','validation_checks':sorted(CHECKS)}
        digest=canonical_hash(proposal)
        return {'baseline':baseline,'proposal':proposal,'proposal_sha256':digest,'approval':{'approved':True,'proposal_sha256':digest,'baseline_sha256':'a'*64,'reviewer_id':'fixture-reviewer'}}
    def test_approval_cannot_be_reused_for_changed_recipe_or_checkpoint(self):
        for what in ('missing','changed','checkpoint'):
            doc=self.doc();checkpoint='b'*64
            if what=='missing':doc['approval']=None
            if what=='changed':doc['proposal']['changes']['training_min_size']=1080
            if what=='checkpoint':checkpoint='c'*64
            with self.subTest(what=what),self.assertRaises(ValueError):validate_resize_recipe(doc,checkpoint)
    def test_only_registered_exact_recipe_is_allowed(self):
        self.assertEqual(validate_resize_recipe(self.doc(),'b'*64)['max_steps'],2000)
        doc=self.doc();doc['proposal']['changes']['max_steps']=9999;doc['proposal_sha256']=canonical_hash(doc['proposal']);doc['approval']['proposal_sha256']=doc['proposal_sha256']
        with self.assertRaises(ValueError):validate_resize_recipe(doc,'b'*64)
