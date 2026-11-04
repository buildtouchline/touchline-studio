"""Explicit one-dollar Modal analysis trial; input preparation never reads keys."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from touchline.function_jobs import AnalysisAllowance,load_frozen_inputs,MAX_OUTPUT_BYTES
from touchline.modal_trial import make_code_bundle,extract_run_bundle,classify_repair
from touchline.data.manifest import sha256_file
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error
from modal_analysis_function import build_app
from fetch_modal_analysis_logs import fetch


def prepare(work):
    previous=ROOT/'runs/video-trial-20261101'
    success=json.loads((previous/'gpu-function-main-t4-after-payment.json').read_text(encoding='utf-8'))
    if success.get('status')!='passed' or success.get('app_stop')!='confirmed':raise ValueError('Reconciled GPU capability required')
    prepared=json.loads((previous/'prepared.json').read_text(encoding='utf-8'))
    work.mkdir(parents=True,exist_ok=False)
    make_code_bundle(ROOT,work/'code.zip')
    inputs={'code.zip':work/'code.zip','candidate.pt':ROOT/'artifacts/models/football-person/candidate.pt',
        'video24.mp4':previous/'normalized/video24.mp4','worker.py':ROOT/'tools/modal_video_worker.py',
        'check_cuda_tracking.py':ROOT/'tools/check_cuda_tracking.py'}
    hashes={n:sha256_file(p) for n,p in inputs.items()}
    if hashes['candidate.pt']!=prepared['checkpoint_sha256'] or hashes['video24.mp4']!=prepared['video_sha256']:
        raise ValueError('Reviewed input selection changed')
    from touchline.analysis_policy import preflight_analysis_video
    from touchline.video.decode import probe_video
    preflight_analysis_video(inputs['video24.mp4'])
    info=probe_video(inputs['video24.mp4'])
    if info['duration_s'] is None or not 0<info['duration_s']<=20.01:raise ValueError('20-second excerpt required')
    image=json.loads((previous/'modal-job.json').read_text(encoding='utf-8'))['image_id']
    specification={'schema':'touchline.modal-analysis-job.v1','environment':'main','gpu':'T4','image_id':image,
        'input_paths':{n:str(p.resolve()) for n,p in inputs.items()},'hashes':hashes,
        'source_parent':'video-trial-20261101','excerpt_seconds':20,'detector_samples_per_second':6,
        'min_size':1080,'max_size':4096,'training_eligible':False,'training':'not run'}
    save(work/'job-specification.json',specification)
    return specification


def execute(work,*,approved=False,retry_after_logs=False):
    if approved is not True:raise ValueError('One additional dollar analysis approval required')
    specification=json.loads((work/'job-specification.json').read_text(encoding='utf-8'))
    files,fingerprint=load_frozen_inputs(specification,ROOT,work)
    claim=work/'modal-function.json'
    record={'schema':'touchline.modal-analysis-attempt.v1','attempt_id':uuid.uuid4().hex,'status':'claimed',
        'environment':'main','gpu':'T4','fingerprint':fingerprint,'training':'not run','app_stop':'not_created'}
    with claim.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    allowance=AnalysisAllowance(ROOT/'runs/modal-analysis-20261102-allowance.sqlite3')
    if retry_after_logs is True:
        parent_work=ROOT/'runs/modal-analysis-20261102'
        parent=json.loads((parent_work/'modal-function.json').read_text(encoding='utf-8'))
        logs=json.loads((parent_work/'provider-log-summary.json').read_text(encoding='utf-8'))
        inspection=json.loads((parent_work/'failure-inspection.json').read_text(encoding='utf-8'))
        if parent.get('app_stop')!='confirmed' or logs.get('fetch_succeeded') is not True or inspection.get('exception_type')!='InternalFailure':
            raise ValueError('Reconciled provider failure with fetched logs required before retry')
        record['parent_attempt_id']=parent['attempt_id']
        record['allocation']=allowance.reserve_retry(work.name,fingerprint,parent['attempt_id'])
    else:record['allocation']=allowance.reserve(work.name,fingerprint)
    save(claim,record);started=time.monotonic()
    def event(stage,status,**data):
        item={'attempt_id':record['attempt_id'],'utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_s':round(time.monotonic()-started,3),'stage':stage,'status':status,**data}
        with (work/'modal-function.events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    app=call=client=None;result=None
    try:
        event('configuration','loading_privately')
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={})
            modal,client=modal_client(settings,work)
            app,analyze=build_app(modal,specification['image_id'],'touchline-analysis-'+record['attempt_id'][:12])
        event('app','starting',image_id=specification['image_id'],gpu='T4',environment='main')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            with app.run(client=client,environment_name='main',detach=False):
                record.update(app_id=app.app_id,status='app_running',app_stop='pending');save(claim,record)
                call=analyze.spawn(files,specification['hashes'])
                record.update(function_call_id=call.object_id,status='submitted');save(claim,record)
                result=call.get(timeout=600)
        if not isinstance(result,dict) or result.get('status')!='complete':
            code=result.get('failure_code','INTERNAL_ERROR') if isinstance(result,dict) else 'INTERNAL_ERROR'
            if code not in {'INPUT_INVALID','INTERNAL_ERROR','RESOURCE_EXHAUSTED','DEPENDENCY_MISSING','OUTPUT_LIMIT','WORKER_PROTOCOL'}:code='INTERNAL_ERROR'
            record['worker_failure_code']=code
            record['repair']=classify_repair({'OUTPUT_LIMIT':'RESOURCE_EXHAUSTED','WORKER_PROTOCOL':'INTERNAL_ERROR'}.get(code,code))
            if isinstance(result,dict) and result.get('stage') in {'input','cuda','parity','analysis','export'}:record['worker_failure_stage']=result['stage']
            raise RuntimeError('Frozen analysis worker failed')
        bundle=result.get('result_zip')
        if type(bundle) is not bytes or len(bundle)>MAX_OUTPUT_BYTES:raise ValueError('Bounded output required')
        with (work/'result.zip').open('xb') as stream:stream.write(bundle)
        record['status']='downloaded';record['output_sha256']=hashlib.sha256(bundle).hexdigest()
        event('output','downloaded',bytes=len(bundle))
    except BaseException as exc:
        if type(exc).__name__=='InternalFailure':record['repair']=classify_repair('PROVIDER_UNAVAILABLE')
        record.update(status='failed',**safe_error(exc));event('analysis','failed',**safe_error(exc))
    finally:
        if call is not None and record['status']!='downloaded':
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):call.cancel(terminate_containers=True)
                record['call_cancel']='confirmed'
            except BaseException:record['call_cancel']='unknown'
        if app is not None and app.app_id is not None:
            record['app_id']=app.app_id
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):app.stop(client=client)
                record['app_stop']='confirmed';event('app.stop','confirmed',app_id=app.app_id)
            except BaseException as exc:record['app_stop']='unknown';record['stop_error']=safe_error(exc)
        record['duration_s']=round(time.monotonic()-started,3);save(claim,record)
        if record.get('app_id') and client is not None:
            try:
                logs=fetch(work,settings)
                record['provider_logs_fetched']=logs['fetch_succeeded'];save(claim,record)
                event('provider.logs','fetched' if logs['fetch_succeeded'] else 'failed',bytes=logs['raw_bytes'])
            except Exception as exc:
                record['provider_logs_fetched']=False;record['log_fetch_error']=safe_error(exc);save(claim,record)
    if record['status']!='downloaded' or record['app_stop']!='confirmed':return record
    try:
        extract_run_bundle(work/'result.zip',work)
        manifest=json.loads((work/'run/manifest.json').read_text(encoding='utf-8'))
        gpu=json.loads((work/'run/gpu-receipt.json').read_text(encoding='utf-8'))
        runtime=manifest['detector_runtime']
        if (manifest['detector']['checkpoint']!=specification['hashes']['candidate.pt'] or
            manifest['status']!='complete' or manifest['decoded_frames']!=480 or manifest['selected_frames']!=120 or
            gpu.get('synthetic_cuda_parity')!='passed' or gpu.get('cuda_nms')!='passed' or
            runtime.get('inference_calls')!=120 or not runtime.get('model_parameter_device','').startswith('cuda') or
            not runtime.get('last_input_device','').startswith('cuda') or gpu.get('input_hashes')!=specification['hashes']):
            raise ValueError('Output lineage, excerpt or CUDA device evidence mismatch')
    except Exception as exc:
        record.update(status='output_validation_failed',**safe_error(exc));save(claim,record)
        event('output.validation','failed',**safe_error(exc));return record
    record.update(status='complete',run_id=manifest['run_id'],decoded_frames=manifest['decoded_frames'],
        selected_frames=manifest['selected_frames'],observation_count=manifest['observation_count'])
    save(claim,record);event('analysis','complete',run_id=record['run_id'])
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','execute'])
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--approved-one-dollar',action='store_true')
    parser.add_argument('--approved-retry-after-logs',action='store_true')
    args=parser.parse_args()
    try:
        work=args.work.resolve()
        if not work.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Project runs directory required')
        result=prepare(work) if args.phase=='prepare' else execute(work,approved=args.approved_one_dollar,retry_after_logs=args.approved_retry_after_logs)
        print(json.dumps({'status':result.get('status','prepared'),'work':str(work),
            **{k:result[k] for k in ('run_id','app_stop','decoded_frames','selected_frames','observation_count','worker_failure_code','worker_failure_stage') if k in result}},indent=2))
        if args.phase=='execute' and result['status']!='complete':raise SystemExit(1)
    except Exception as exc:
        print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
