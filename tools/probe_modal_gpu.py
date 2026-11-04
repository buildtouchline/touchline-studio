"""One explicitly approved, logged T4 capability probe; no video or training."""
import argparse
from contextlib import redirect_stderr,redirect_stdout
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import re
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,modal_client,save

ENVIRONMENT='main'
APP='touchline-video-trial'
GPU='T4'
LIFETIME=60


def safe_error(exc):
    """Classify private provider wording; never return raw text or identifiers."""
    text=str(exc).lower()
    patterns={
        'gpu_mentioned':r'\bgpu(?:s)?\b',
        'sandbox_mentioned':r'\bsandbox(?:es)?\b',
        'concurrency_mentioned':r'concurren',
        'quota_mentioned':r'\bquota\b',
        'limit_mentioned':r'\blimit(?:s|ed)?\b',
        'billing_mentioned':r'\bbilling\b|payment method',
        'credit_mentioned':r'\bcredits?\b',
        'workspace_mentioned':r'\bworkspace\b',
        'environment_mentioned':r'\benvironment\b',
        'gpu_unavailable_wording':r'gpu.{0,100}(?:not available|unavailable|disabled|not enabled|not supported)',
        'access_wording':r'permission|access denied|unauthorized|not authorized|not permitted',
        'beta_wording':r'\bbeta\b|\bpreview\b',
        'capacity_wording':r'capacity|no available',
        'runtime_mentioned':r'\bruntime\b|\bgvisor\b',
        'network_mentioned':r'network|outbound',
        'timeout_mentioned':r'timeout|timed out',
        'startup_mentioned':r'startup|initialization',
        'serialization_mentioned':r'serializ|pickle',
        'missing_module_wording':r'no module named',
    }
    category=type(exc).__name__
    if category not in {'InvalidError','AuthError','PermissionDeniedError','NotFoundError',
                        'TimeoutError','ValueError','RuntimeError','ConnectionError','KeyboardInterrupt',
                        'FunctionTimeoutError','ExecutionError','RemoteError','ModuleNotFoundError',
                        'ResourceExhaustedError','SerializationError','DeserializationError','RequestSizeError',
                        'ClientClosed','InternalError','ServiceError','ImageBuildError','ImportError',
                        'AttributeError','TypeError','KeyError','AssertionError','CancelledError',
                        'FunctionCancelledError','SystemExit','FileNotFoundError','OSError','GRPCError','InternalFailure'}:
        category='ProviderOrRuntimeError'
    return {'error_category':category,'flags':[k for k,v in patterns.items() if re.search(v,text)],
            'raw_error_retained':False}


def parse_gpu_output(raw):
    lines=raw.strip().splitlines()
    if len(lines)!=1:raise ValueError('One GPU result required')
    fields=[v.strip() for v in lines[0].split(',')]
    if len(fields)!=3 or not re.fullmatch(r'(?:NVIDIA )?(?:Tesla )?T4',fields[0]):
        raise ValueError('Unexpected GPU result')
    if not re.fullmatch(r'\d+ MiB',fields[1]) or not re.fullmatch(r'\d+(?:\.\d+){1,3}',fields[2]):
        raise ValueError('Invalid bounded GPU metadata')
    return {'gpu_name':'T4','memory_mib':int(fields[1].split()[0]),'driver_version':fields[2]}


def run(work,*,approved=False):
    if approved is not True:raise ValueError('Explicit one-probe approval required')
    work=Path(work).resolve()
    if not work.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Project run directory required')
    # Refuse duplicates before credential loading or any network call.
    claim=work/'gpu-probe-main-t4.json'
    record={'schema':'touchline.modal-gpu-probe.v1','run_id':uuid.uuid4().hex,
        'environment':ENVIRONMENT,'app':APP,'gpu':GPU,'lifetime_s':LIFETIME,
        'status':'claimed','worker_terminated':False,'training_performed':False,
        'authorization':'user-requested single main-environment diagnostic; prior Modal $2 hold retained'}
    with claim.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    started=time.monotonic()
    def event(stage,status,**values):
        entry={'run_id':record['run_id'],'timestamp_utc':datetime.now(timezone.utc).isoformat(),
               'elapsed_s':round(time.monotonic()-started,3),'stage':stage,'status':status,**values}
        with (work/'gpu-probe-main-t4.events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(entry)+'\n')
        print(json.dumps(entry),flush=True)
    sandbox=None;modal=None;client=None;attempted=False
    name='gpu-probe-'+record['run_id']
    try:
        prior=json.loads((work/'modal-job.json').read_text(encoding='utf-8'))
        if prior.get('worker_terminated')!='confirmed_no_worker':raise ValueError('Prior worker reconciliation required')
        if prior.get('allocation',{}).get('reserved_usd')!='2.00':raise ValueError('Preserved Modal hold required')
        event('credentials','loading_privately')
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={})
            modal,client=modal_client(settings,work)
            app=modal.App.lookup(APP,environment_name=ENVIRONMENT,client=client,create_if_missing=False)
        event('environment','selected',environment=ENVIRONMENT)
        event('image','preparing',image_kind='plain_debian_python312')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            image=modal.Image.debian_slim(python_version='3.12')
            image.build(app)
        record['image_id']=image.object_id
        event('image','ready',image_id=image.object_id)
        record['status']='creating';save(claim,record)
        event('sandbox.create','started',gpu=GPU,runtime='gvisor',lifetime_s=LIFETIME)
        attempted=True
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox=modal.Sandbox.create(app=app,client=client,image=image,name=name,
                gpu=GPU,runtime='gvisor',timeout=LIFETIME,
                cpu=(.25,.25),memory=(512,512),block_network=True,secrets=[],volumes={},env={})
        record.update(status='running',worker_id=sandbox.object_id)
        save(claim,record);event('sandbox.create','created',worker_id=sandbox.object_id)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            process=sandbox.exec('nvidia-smi','--query-gpu=name,memory.total,driver_version',
                '--format=csv,noheader',timeout=15,text=False,bufsize=-1)
            from touchline.repair_checks import read_worker_result
            raw=read_worker_result(process.stdout);process.wait()
            if process.returncode!=0:raise RuntimeError('GPU diagnostic command failed')
        record['gpu_metadata']=parse_gpu_output(raw)
        record['status']='passed'
        event('nvidia-smi','passed',**record['gpu_metadata'])
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc))
        event('probe','failed',**safe_error(exc))
    finally:
        if sandbox is None and attempted and modal is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):
                    sandbox=modal.Sandbox.from_name(APP,name,environment_name=ENVIRONMENT,client=client)
            except modal.exception.NotFoundError:
                record['worker_terminated']='confirmed_no_worker'
                event('reconcile','no_running_worker')
            except BaseException as exc:
                record['reconciliation']=safe_error(exc)
                event('reconcile','unknown',**safe_error(exc))
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
                event('sandbox.terminate','confirmed')
            except BaseException as exc:
                record['status']='termination_unknown';record['termination_error']=safe_error(exc)
                event('sandbox.terminate','unknown',**safe_error(exc))
        record['duration_s']=round(time.monotonic()-started,3)
        record['billing']='unreconciled; original hold retained; no further retry'
        save(claim,record)
        event('probe','finished',result=record['status'],worker_terminated=record['worker_terminated'])
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--approved-single-probe',action='store_true')
    args=parser.parse_args()
    try:
        result=run(args.work,approved=args.approved_single_probe)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['status']=='passed' else 1)
    except Exception as exc:
        print(json.dumps({'status':'not_dispatched',**safe_error(exc)}))
        raise SystemExit(1) from None
