"""Deterministic change classification. Decisions never execute or promote code."""
import math
from pathlib import PurePosixPath
import re
from .repair import canonical_hash

CHECKS={'finite-loss','checkpoint-reload','held-out-regression','budget-and-time-cap','rollback'}
PROTECTED=('spending','settings','diagnostic','evaluation','rubric','repair','contracts','manifest',
    'training_trials','cpu_sandbox_jobs','function_jobs','modal_trial','code_adviser','tools/run_modal',
    'tools/modal_dataset_recovery_worker','tools/modal_recovered_gpu_worker','tools/request_training_tuning','tools/run_handover_job',
    'training/recipes','adapters/handoff','tools/adopt_modal_recording','tools/fetch_soccertrack_intake',
    'tools/request_soccertrack_sam_review','tools/modal_recording_worker','tools/modal_resize_training_worker','tools/modal_soccertrack_audit_worker')


def assess_change(proposal,baseline,*,reviewed_data=False,frozen_evaluation=False,
                  isolated_candidate=False,budget_approved=False,approval=None):
    """Minor recipes may run staged tests; major diffs need exact human approval.

    Approval fields are trusted-controller attestations, not authentication.
    Promotion remains a separate human-reviewed artifact/metrics operation.
    """
    required={'kind','baseline_sha256','changes','expected_outcome','validation_checks','rollback_sha256'}
    if not isinstance(proposal,dict) or set(proposal)!=required:raise ValueError('Invalid change proposal')
    for value in (proposal['baseline_sha256'],proposal['rollback_sha256']):
        if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('Exact lineage required')
    if proposal['baseline_sha256']!=baseline['recipe_sha256'] or proposal['rollback_sha256']!=baseline['checkpoint_sha256']:
        raise ValueError('Stale proposal or rollback checkpoint')
    if not isinstance(proposal['expected_outcome'],str) or not 1<=len(proposal['expected_outcome'])<=2048:
        raise ValueError('Expected outcome is required and is not a guarantee')
    checks=proposal['validation_checks']
    if not isinstance(checks,list) or any(not isinstance(c,str) for c in checks) or not CHECKS.issubset(checks):
        raise ValueError('Independent validation and rollback checks are required')
    changes=proposal['changes']
    if not isinstance(changes,dict) or not changes:raise ValueError('Nonempty typed changes required')
    if not (isolated_candidate is True and budget_approved is True):return {'state':'blocked','reason':'isolation_or_budget_missing'}
    identifier=canonical_hash(proposal)
    minor=proposal['kind']=='tuning' and set(changes).issubset({'learning_rate','weight_decay'})
    if minor:
        for key,value in changes.items():
            previous=baseline.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0 or type(previous) not in (int,float) or not math.isfinite(previous) or previous<=0:
                raise ValueError('Finite positive baseline and recipe values required')
            if not previous*.5<=value<=previous*2:minor=False
    if proposal['kind']=='tuning' and not (reviewed_data is True and frozen_evaluation is True):
        return {'state':'blocked','reason':'reviewed_data_or_frozen_evaluation_missing'}
    if proposal['kind']=='code':
        for path,diff in changes.items():
            parsed=PurePosixPath(path)
            if parsed.is_absolute() or '..' in parsed.parts or '\\' in path or not path.startswith(('src/touchline/','tools/')):
                raise ValueError('Unsafe change path')
            if not path.endswith('.py') or '/data/' in path or any(word in path for word in PROTECTED):
                raise ValueError('Protected control cannot be edited by repair automation')
            if not isinstance(diff,str) or not 1<=len(diff.encode())<=32768:raise ValueError('Bounded exact replacement required')
    elif proposal['kind']!='tuning':raise ValueError('Unknown proposal kind')
    if minor:
        return {'state':'minor_staged_trial_allowed','proposal_sha256':identifier,'promotion':'human_review_required',
            'caps':'unchanged','checks':sorted(CHECKS)}
    if approval is None:return {'state':'major_approval_required','proposal_sha256':identifier,
        'expected_outcome':proposal['expected_outcome'],'promotion':'human_review_required'}
    if (not isinstance(approval,dict) or approval.get('approved') is not True or
        approval.get('proposal_sha256')!=identifier or approval.get('baseline_sha256')!=proposal['baseline_sha256'] or
        not isinstance(approval.get('reviewer_id'),str) or not re.fullmatch('[a-zA-Z0-9_-]{1,80}',approval['reviewer_id'])):
        raise ValueError('Human approval must bind the exact unchanged proposal and baseline')
    return {'state':'major_staged_checks_allowed','proposal_sha256':identifier,'promotion':'human_review_required'}
