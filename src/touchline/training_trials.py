"""Admit one minor adviser trial only after a verified independent baseline."""
from .repair_policy import assess_change


def admit_minor_trial(proposal,baseline,receipt,evaluation):
    if receipt.get('status')!='complete' or receipt.get('restored_images')!=1710:
        raise ValueError('Complete hash-verified restored release required')
    if receipt.get('input_hashes',{}).get('candidate.pt')!=baseline['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint changed')
    if (evaluation.get('split')!='val' or evaluation.get('subset') is not False or evaluation.get('images')!=570 or
        evaluation.get('manifest_sha256')!=receipt.get('release_sha256') or
        evaluation.get('model',{}).get('checkpoint')!=receipt.get('training',{}).get('checkpoint_sha256')):
        raise ValueError('Full independent validation baseline required')
    decision=assess_change(proposal,baseline,reviewed_data=True,frozen_evaluation=True,isolated_candidate=True,budget_approved=True)
    if decision['state']!='minor_staged_trial_allowed':raise ValueError('Only registered minor tuning can execute automatically')
    return {'decision':decision,'learning_rate':proposal['changes'].get('learning_rate',baseline['learning_rate']),
        'weight_decay':proposal['changes'].get('weight_decay',baseline['weight_decay']),
        'baseline_ap50':evaluation['metrics']['ap50'],'promotion':'not performed'}
