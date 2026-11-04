"""Single approved ephemeral T4 Function probe, separate from Sandbox creation."""
import argparse
from contextlib import redirect_stderr,redirect_stdout
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from probe_modal_gpu import safe_error,parse_gpu_output
from run_modal_video_trial import Discard,modal_client,save


def build_probe(modal,image_id,name):
    # Only a serialized stdlib-only function is sent; no checkout source,
    # credentials, video, model or dataset is mounted/uploaded.
    app=modal.App(name,include_source=False)
    image=modal.Image.from_id(image_id)
    @app.function(image=image,gpu='T4',serialized=True,include_source=False,
                  cpu=(.25,.25),memory=(512,512),timeout=30,startup_timeout=30,
                  min_containers=0,max_containers=1,buffer_containers=0,
                  scaledown_window=5,retries=0,single_use_containers=True,
                  secrets=[],volumes={},env={},block_network=True,
                  restrict_modal_access=True)
    def check_gpu():
        import re
        import subprocess
        result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=False)
        if result.returncode!=0:return {'status':'command_failed'}
        raw=result.stdout
        if len(raw.encode())>1024:return {'status':'invalid_output'}
        fields=[v.strip() for v in raw.strip().split(',')]
        if (len(fields)!=3 or not re.fullmatch(r'(?:NVIDIA )?(?:Tesla )?T4',fields[0]) or
            not re.fullmatch(r'\d+ MiB',fields[1]) or not re.fullmatch(r'\d+(?:\.\d+){1,3}',fields[2])):
            return {'status':'invalid_output'}
        return {'status':'passed','gpu_output':raw.strip()}
    return app,check_gpu


def run(work,*,approved=False,after_payment=False):
    if approved is not True:raise ValueError('Explicit single Function probe approval required')
    work=Path(work).resolve()
    if not work.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Project run directory required')
    suffix='-after-payment' if after_payment is True else ''
    if after_payment is True:
        previous_function=json.loads((work/'gpu-function-main-t4.json').read_text(encoding='utf-8'))
        if (previous_function.get('status')!='failed' or previous_function.get('app_stop')!='confirmed' or
            'billing_mentioned' not in previous_function.get('flags',[])):
            raise ValueError('A reconciled billing-rejected Function attempt is required')
    claim=work/f'gpu-function-main-t4{suffix}.json'
    record={'schema':'touchline.modal-gpu-function-probe.v1','run_id':uuid.uuid4().hex,
        'environment':'main','gpu':'T4','status':'claimed','app_stop':'not_created',
        'execution_timeout_s':30,'startup_timeout_s':30,'result_wait_s':60,
        'training_performed':False,'endpoint_deployed':False,
        'authorization':'user explicitly requested one separate GPU Function probe; existing Modal hold retained'}
    if after_payment is True:
        record['authorization']='user reports payment method added and explicitly requests one retry; original Modal hold retained'
        record['previous_run_id']=previous_function['run_id']
    with claim.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    started=time.monotonic()
    def event(stage,status,**data):
        item={'run_id':record['run_id'],'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_s':round(time.monotonic()-started,3),'stage':stage,'status':status,**data}
        with (work/f'gpu-function-main-t4{suffix}.events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    app=call=client=None
    try:
        previous=json.loads((work/'gpu-probe-main-t4.json').read_text(encoding='utf-8'))
        hold=json.loads((work/'modal-job.json').read_text(encoding='utf-8'))
        if previous.get('worker_terminated')!='confirmed_no_worker':raise ValueError('Sandbox reconciliation required')
        if hold.get('allocation',{}).get('reserved_usd')!='2.00':raise ValueError('Preserved Modal hold required')
        event('credentials','loading_privately')
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={})
            modal,client=modal_client(settings,work)
            app,check=build_probe(modal,previous['image_id'],'touchline-gpu-function-'+record['run_id'][:12])
        record['image_id']=previous['image_id']
        event('function.app','starting',environment='main',gpu='T4',retries=0)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            with app.run(client=client,environment_name='main',detach=False):
                record.update(status='app_running',app_id=app.app_id)
                save(claim,record)
                event('function.app','ready',app_id=app.app_id)
                call=check.spawn()
                record['function_call_id']=call.object_id;save(claim,record)
                event('function.call','submitted',function_call_id=call.object_id)
                result=call.get(timeout=60)
                if not isinstance(result,dict) or set(result)!={'status','gpu_output'} or result['status']!='passed':
                    raise RuntimeError('GPU command did not pass')
                record.update(status='passed',gpu_metadata=parse_gpu_output(result['gpu_output']))
        event('nvidia-smi','passed',**record['gpu_metadata'])
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc))
        event('function.probe','failed',**safe_error(exc))
    finally:
        if call is not None and record['status']!='passed':
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):call.cancel(terminate_containers=True)
                event('function.cancel','confirmed')
            except BaseException as exc:
                record['cancel_error']=safe_error(exc);event('function.cancel','unknown',**safe_error(exc))
        if app is not None and app.app_id is not None:
            record['app_id']=app.app_id
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):app.stop(client=client)
                record['app_stop']='confirmed'
                event('function.app.stop','confirmed',app_id=app.app_id)
            except BaseException as exc:
                record['app_stop']='unknown';record['stop_error']=safe_error(exc)
                event('function.app.stop','unknown',**safe_error(exc))
        record['duration_s']=round(time.monotonic()-started,3)
        record['billing']='unreconciled; original Modal hold retained; no further retry'
        save(claim,record)
        event('function.probe','finished',result=record['status'],app_stop=record['app_stop'])
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--approved-single-function',action='store_true')
    parser.add_argument('--after-payment',action='store_true')
    args=parser.parse_args()
    try:
        result=run(args.work,approved=args.approved_single_function,after_payment=args.after_payment)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['status']=='passed' else 1)
    except Exception as exc:
        print(json.dumps({'status':'not_dispatched',**safe_error(exc)}))
        raise SystemExit(1) from None
