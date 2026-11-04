"""Exact, human-reviewed resize trials; never an automatic minor repair."""
from ..repair_policy import assess_change
RESIZE_RECIPE='heads-fp32-batch1-reviewed-resize-v2'
def validate_resize_recipe(document,checkpoint_sha256):
    if not isinstance(document,dict) or not {'proposal','proposal_sha256','baseline','approval'}.issubset(document):raise ValueError('Reviewed resize recipe required')
    proposal=document['proposal'];baseline=document['baseline']
    decision=assess_change(proposal,baseline,reviewed_data=True,frozen_evaluation=True,isolated_candidate=True,budget_approved=True,approval=document['approval'])
    if decision['state']!='major_staged_checks_allowed' or decision['proposal_sha256']!=document['proposal_sha256'] or baseline['checkpoint_sha256']!=checkpoint_sha256:raise ValueError('Exact reviewed recipe/rollback checkpoint required')
    changes=proposal['changes']
    expected={'training_min_size':720,'training_max_size':1280,'learning_rate':.0005,'weight_decay':.0005,'epochs':4,'max_steps':2000,'max_seconds':2700,'freeze_backbone':True}
    if changes!=expected or any(type(changes[k]) is not type(v) for k,v in expected.items()):raise ValueError('Only the exact approved resize recipe is registered')
    return changes
