"""Fresh local-terminal → Modal CPU analysis/validation; no default paid execution."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,re,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.data.manifest import sha256_file
from touchline.spending import SpendingLedger
from run_modal_remote_checks import make_bundle
from run_modal_training_recovery import phase
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error

KEYS={'run_id','operation','native','snapshot_image_id','checkpoint','checkpoint_sha256','video24','baseline_observations',
      'reference_evaluation','owner_budget_confirmed','prior_upper_bound_usd','available_headroom_usd','job_cap_usd'}

def parse_config(value):
    if not isinstance(value,dict) or set(value)!=KEYS:raise ValueError('Use the exact handover config schema; no secrets')
    if not isinstance(value['run_id'],str) or not re.fullmatch(r'takeover-[a-zA-Z0-9_-]{1,60}',value['run_id']):raise ValueError('Fresh takeover- run ID required')
    if value['operation'] not in {'analysis','evaluate'}:raise ValueError('Supported operations: analysis, evaluate')
    if type(value['native']) is not bool:raise ValueError('Explicit inference resolution selection required')
    if not isinstance(value['snapshot_image_id'],str) or not re.fullmatch('im-[a-zA-Z0-9_-]{1,150}',value['snapshot_image_id']):raise ValueError('Shared immutable snapshot ID required')
    if not isinstance(value['checkpoint_sha256'],str) or not re.fullmatch('[a-f0-9]{64}',value['checkpoint_sha256']):raise ValueError('Selected checkpoint hash required')
    from touchline.spending import units
    if not 500000<=units(value['job_cap_usd'])<=2000000:raise ValueError('CPU job cap must be $0.50–$2.00')
    units(value['prior_upper_bound_usd']);units(value['available_headroom_usd'])
    if type(value['owner_budget_confirmed']) is not bool:raise ValueError('Explicit owner budget attestation required')
    return value

def asset(value):
    if not isinstance(value,str) or not value:raise ValueError('Required private artifact path missing')
    p=(ROOT/value).resolve()
    if not p.is_relative_to((ROOT/'artifacts/handover').resolve()) or any(x.is_symlink() for x in (ROOT/value,*(ROOT/value).parents)) or not p.is_file():raise ValueError('Use real files under artifacts/handover')
    return p

def run(config_path,*,execute=False):
    config=parse_config(json.loads(config_path.read_text(encoding='utf-8')))
    checkpoint=asset(config['checkpoint'])
    if sha256_file(checkpoint)!=config['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
    inputs={'candidate.pt':checkpoint}
    if config['operation']=='evaluate':
        inputs.update({'reference.json':asset(config['reference_evaluation']),'worker.py':ROOT/'tools/modal_ap50_worker.py'})
    else:
        inputs.update({'video24.mp4':asset(config['video24']),'baseline-observations.jsonl':asset(config['baseline_observations']),
            'worker.py':ROOT/'tools/modal_candidate_analysis_worker.py'})
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    work=ROOT/'runs'/config['run_id']
    if work.exists():
        prepared=json.loads((work/'prepared.json').read_text())
        if prepared['config_sha256']!=identity:raise ValueError('Run ID already belongs to another configuration')
    else:
        work.mkdir(parents=True);make_bundle(work/'code.zip')
        save(work/'prepared.json',{'config_sha256':identity,'code_sha256':sha256_file(work/'code.zip'),'inputs':{n:sha256_file(p) for n,p in inputs.items()}})
    prepared=json.loads((work/'prepared.json').read_text())
    if prepared['inputs']!={n:sha256_file(p) for n,p in inputs.items()}:raise ValueError('Prepared inputs changed')
    if prepared['code_sha256']!=sha256_file(work/'code.zip'):raise ValueError('Prepared source changed')
    if not execute:return {'status':'prepared','run_id':config['run_id'],'provider_called':False,'credentials_loaded':False}
    if config['owner_budget_confirmed'] is not True:raise ValueError('Owner must confirm prior charges/holds and available headroom')
    ledger_path=ROOT/'runs/handover-spending.sqlite3';new_ledger=not ledger_path.exists();ledger=SpendingLedger(ledger_path)
    if new_ledger:ledger.verify_pool('modal',prior_spend_usd=config['prior_upper_bound_usd'],available_credit_usd=config['available_headroom_usd'])
    reservation='handover-'+config['run_id']
    held=ledger.reserve('modal',reservation,fingerprint=identity,worst_case_usd=config['job_cap_usd'])
    if not held['created']:raise ValueError('Job already dispatched; inspect/reconcile it, do not repeat')
    result=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-training-recovery',client=client,environment_name='main',create_if_missing=False)
            image=modal.Image.from_id(config['snapshot_image_id'],client=client)
        inputs['code.zip']=work/'code.zip'
        result=phase(work/(config['run_id']+'-worker'),modal=modal,client=client,app=app,image=image,inputs=inputs,
            job={'mode':config['operation'],'native':config['native'],'fresh_source':True,'sam_review':False,
                 'selection':{'promotion':'not performed','handover_run_id':config['run_id'],'native_resolution':config['native']}},is_analysis=True)
        save(work/'result.json',result)
        return {'status':result['status'],'worker_terminated':result['worker_terminated'],'run_id':config['run_id']}
    finally:
        ledger.reconcile(reservation,finalized=False)
        # Keep the pool blocked after uncertain create/termination; owner reconciles.
        if result is None or result.get('worker_terminated') is not True:ledger.block_pool('modal')

if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--execute-approved',action='store_true');args=p.parse_args()
    try:print(json.dumps(run(args.config,execute=args.execute_approved)))
    except Exception as exc:print(json.dumps({'status':'blocked_or_failed',**safe_error(exc)}));raise SystemExit(1) from None
