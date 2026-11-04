"""One approved full-video Modal CPU Sandbox job with streamed safe logs."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from touchline.cpu_sandbox_jobs import CpuSandboxAllowance,sandbox_options,extract_results
from touchline.modal_trial import make_code_bundle
from touchline.data.manifest import sha256_file
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error


def clean_event(item):
    if not isinstance(item,dict) or set(item)-{'schema','stage','status','elapsed_s','coverage_rows','decoded_frames','selected_frames','observations','failure_code'}:
        raise ValueError('Unexpected worker event')
    if item.get('schema')!='touchline.cpu-worker-event.v1' or item.get('stage') not in {'input','cpu_runtime','cpu_analysis','profiles','full_overlay','export','complete'}:
        raise ValueError('Unknown worker stage')
    if item.get('status') not in {'started','progress','complete','failed'}:raise ValueError('Unknown worker state')
    for k in ('elapsed_s','coverage_rows','decoded_frames','selected_frames','observations'):
        if k in item and (type(item[k]) not in (int,float) or not 0<=item[k]<=10000000):raise ValueError('Invalid worker counter')
    if 'failure_code' in item and item['failure_code'] not in {'RESOURCE_EXHAUSTED','INPUT_INVALID','INTERNAL_ERROR'}:raise ValueError('Invalid worker failure')
    return item


def run(work,approved=False):
    if approved is not True:raise ValueError('Explicit CPU Sandbox ten-dollar approval required')
    work=Path(work).resolve()
    if not work.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Project run folder required')
    source=ROOT/'runs/cpu-full-video-20261102/normalized'
    normalization=json.loads((source/'normalization.json').read_text(encoding='utf-8'))
    if normalization.get('status')!='complete' or normalization.get('output_frames')!=5760:raise ValueError('Complete full-video normalization required')
    work.mkdir(parents=True,exist_ok=False)
    make_code_bundle(ROOT,work/'code.zip')
    inputs={'code.zip':work/'code.zip','video24.mp4':source/'video24.mp4',
        'candidate.pt':ROOT/'artifacts/models/football-person/candidate.pt','cpu_worker.py':ROOT/'tools/modal_cpu_worker.py'}
    hashes={n:sha256_file(p) for n,p in inputs.items()}
    selected=json.loads(inputs['candidate.pt'].with_name('result.json').read_text(encoding='utf-8'))['checkpoint_sha256']
    if hashes['candidate.pt']!=selected or hashes['video24.mp4']!=normalization['output_sha256']:raise ValueError('Input selection changed')
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    allocation=CpuSandboxAllowance(ROOT/'runs/modal-cpu-analysis-20261102-allowance.sqlite3').reserve(work.name,fingerprint)
    record={'schema':'touchline.modal-cpu-attempt.v1','attempt_id':uuid.uuid4().hex,'status':'prepared',
        'environment':'main','gpu':None,'allocation':allocation,'fingerprint':fingerprint,'hashes':hashes,
        'worker_terminated':'not_created','training':'not run','cpu':4,'memory_mib':8192,'timeout_s':7200}
    save(work/'modal-cpu.json',record);started=time.monotonic();sandbox=None;complete=False
    modal=client=None;creation_attempted=False
    def event(stage,status,**details):
        item={'attempt_id':record['attempt_id'],'utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_s':round(time.monotonic()-started,2),'stage':stage,'status':status,**details}
        with (work/'controller.events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:
        event('configuration','loading_privately')
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={})
            modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-cpu-analysis',client=client,environment_name='main',create_if_missing=True)
            image_id='im-QcRMDynaNwmRjN6PN4KbJr'
            creation_attempted=True
            sandbox=modal.Sandbox.create(app=app,client=client,image=modal.Image.from_id(image_id,client=client),
                name=work.name,**sandbox_options())
        record.update(status='created',worker_id=sandbox.object_id,image_id=image_id,worker_terminated='pending')
        save(work/'modal-cpu.json',record);event('sandbox','created',worker_id=sandbox.object_id)
        for name,path in inputs.items():
            with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(path,'/work/'+name)
            event('input','uploaded',name=name,bytes=path.stat().st_size)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.write_text(json.dumps({'hashes':hashes}),'/work/job.json')
            process=sandbox.exec('python','-I','/work/cpu_worker.py',timeout=6500,text=False,bufsize=-1)
        record['status']='running';save(work/'modal-cpu.json',record)
        buffer=bytearray();total=0
        for chunk in process.stdout:
            if not isinstance(chunk,bytes):raise ValueError('Binary worker log expected')
            total+=len(chunk)
            if total>1024*1024 or len(chunk)>65536-len(buffer):raise ValueError('Worker logs exceeded bounds')
            buffer.extend(chunk)
            while b'\n' in buffer:
                line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
                if not line.strip():continue
                item=clean_event(json.loads(line))
                with (work/'worker.events.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'attempt_id':record['attempt_id'],**item})+'\n')
                if item['status']=='complete':complete=True
                event('worker.'+item['stage'],item['status'],**{k:v for k,v in item.items() if k in {'coverage_rows','decoded_frames','selected_frames','observations','failure_code'}})
        process.wait()
        if process.returncode!=0 or not complete or buffer.strip():raise RuntimeError('CPU worker did not complete')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            size=sandbox.filesystem.stat('/work/cpu-results.zip').size
            if size>1024**3:raise ValueError('CPU output ZIP exceeded budget')
            sandbox.filesystem.copy_to_local('/work/cpu-results.zip',work/'cpu-results.zip')
        record.update(status='downloaded',result_sha256=sha256_file(work/'cpu-results.zip'))
        event('results','downloaded',bytes=size)
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc));event('cpu.analysis','failed',**safe_error(exc))
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):
                    if sandbox.filesystem.stat('/work/cpu-failure.zip').size>64*1024*1024:raise ValueError('Failure evidence too large')
                    sandbox.filesystem.copy_to_local('/work/cpu-failure.zip',work/'cpu-failure.zip')
                extract_results(work/'cpu-failure.zip',work)
                record['failure_artifacts_fetched']=True
            except Exception:record['failure_artifacts_fetched']=False
    finally:
        if sandbox is None and creation_attempted:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):
                    sandbox=modal.Sandbox.from_name('touchline-cpu-analysis',work.name,client=client,environment_name='main')
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception as exc:record['worker_terminated']='unknown';record['reconciliation_error']=safe_error(exc)
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True;event('sandbox.terminate','confirmed')
            except BaseException as exc:record['worker_terminated']='unknown';record['termination_error']=safe_error(exc)
        record['duration_s']=round(time.monotonic()-started,2);save(work/'modal-cpu.json',record)
    if record['status']=='downloaded' and record['worker_terminated'] is True:
        extract_results(work/'cpu-results.zip',work)
        receipt=json.loads((work/'cpu-receipt.json').read_text(encoding='utf-8'))
        if receipt['input_hashes']!=hashes or receipt['decoded_frames']!=5760 or receipt['selected_frames']!=1440:
            raise ValueError('CPU result lineage mismatch')
        target=work/'source';target.mkdir()
        for name in ('video24.mp4','frame-map.jsonl','normalization.json'):shutil.copyfile(source/name,target/name)
        if sha256_file(target/'video24.mp4')!=hashes['video24.mp4']:raise ValueError('Source restore copy failed')
        record.update(status='complete',run_id=receipt['run_id'],decoded_frames=5760,selected_frames=1440,
            observation_count=receipt['observation_count'],source_restored=True)
        save(work/'modal-cpu.json',record);event('cpu.analysis','complete',run_id=record['run_id'])
    if record['status']=='failed':
        from touchline.repair_events import publish_job_failure
        record['repair_event']=publish_job_failure(work,record,queue_path=ROOT/'runs/repair-events.sqlite3',stage='analysis')
        save(work/'modal-cpu.json',record)
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--approved-ten-dollar',action='store_true')
    args=parser.parse_args()
    try:
        result=run(args.work,approved=args.approved_ten_dollar)
        print(json.dumps({k:result[k] for k in ('status','worker_terminated','run_id','decoded_frames','selected_frames','observation_count') if k in result},indent=2))
        if result['status']!='complete':raise SystemExit(1)
    except Exception as exc:
        print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
