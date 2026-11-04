import copy
import unittest
from touchline.repair import canonical_hash
from touchline.repair_policy import assess_change,CHECKS


class RepairPolicyTests(unittest.TestCase):
    def setUp(self):
        self.baseline={'recipe_sha256':'a'*64,'checkpoint_sha256':'b'*64,'learning_rate':.001,'weight_decay':.0005}
        self.proposal={'kind':'tuning','baseline_sha256':'a'*64,'rollback_sha256':'b'*64,
            'changes':{'learning_rate':.0005},'expected_outcome':'Reduce unstable loss; held-out improvement is unverified',
            'validation_checks':sorted(CHECKS)}
        self.gates=dict(reviewed_data=True,frozen_evaluation=True,isolated_candidate=True,budget_approved=True)

    def test_minor_trial_requires_reviewed_data_isolation_budget_and_independent_checks(self):
        result=assess_change(self.proposal,self.baseline,**self.gates)
        self.assertEqual(result['state'],'minor_staged_trial_allowed')
        self.assertEqual(result['promotion'],'human_review_required')
        for key in self.gates:
            self.assertEqual(assess_change(self.proposal,self.baseline,**{**self.gates,key:False})['state'],'blocked')
        with self.assertRaises(ValueError):assess_change({**self.proposal,'validation_checks':[]},self.baseline,**self.gates)

    def test_large_recipe_change_needs_exact_human_approval_and_never_bypasses_data_review(self):
        proposal={**self.proposal,'changes':{'learning_rate':.01}}
        self.assertEqual(assess_change(proposal,self.baseline,**self.gates)['state'],'major_approval_required')
        approval={'approved':True,'proposal_sha256':canonical_hash(proposal),'baseline_sha256':'a'*64,'reviewer_id':'owner-a'}
        self.assertEqual(assess_change(proposal,self.baseline,approval=approval,**self.gates)['state'],'major_staged_checks_allowed')
        self.assertEqual(assess_change(proposal,self.baseline,approval=approval,**{**self.gates,'reviewed_data':False})['state'],'blocked')
        changed=copy.deepcopy(proposal);changed['changes']['learning_rate']=.02
        with self.assertRaises(ValueError):assess_change(changed,self.baseline,approval=approval,**self.gates)

    def test_major_code_cannot_modify_protected_evaluation_budget_or_credentials(self):
        for path in ('src/touchline/spending.py','src/touchline/settings.py','src/touchline/data/etl.py','../../.env',
            'src/touchline/repair_events.py','src/touchline/training_trials.py','tools/run_modal_training_recovery.py','tools/run_handover_job.py'):
            proposal={**self.proposal,'kind':'code','changes':{path:'replacement'}}
            with self.assertRaises(ValueError):assess_change(proposal,self.baseline,**self.gates)
        proposal={**self.proposal,'kind':'code','changes':{'src/touchline/video/sampling.py':'replacement'}}
        self.assertEqual(assess_change(proposal,self.baseline,**self.gates)['state'],'major_approval_required')

    def test_stale_baseline_rollback_and_invalid_values_fail_before_trials(self):
        for change in ({'baseline_sha256':'c'*64},{'rollback_sha256':'c'*64},{'changes':{'learning_rate':float('nan')}},{'changes':{'learning_rate':-.01}}):
            with self.assertRaises(ValueError):assess_change({**self.proposal,**change},self.baseline,**self.gates)
