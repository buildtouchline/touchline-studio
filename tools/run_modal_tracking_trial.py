"""One bounded Modal CPU recipe comparison within the retained ten-dollar hold."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,logging,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from touchline.cpu_sandbox_jobs import CpuSandboxAllowance,extract_results
from touchline.data.manifest import sha256_file
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error


def run(work):
    parent=ROOT/'runs/modal-cpu-full-video-20261102'
    previous=json.loads((parent/'modal-cpu.json').read_text())
    if previous['status']!='complete' or previous['worker_terminated'] is not True:raise ValueError('Terminated complete parent required')
    work.mkdir(parents=True,exist_ok=False);make_bundle(work/'checks-code.zip')
    inputs={name:parent/'run'/name for name in ('manifest.json','coverage.jsonl','raw_proposals.jsonl')}
    inputs.update({'checks-code.zip':work/'checks-code.zip','video24.mp4':parent/'source/video24.mp4',
        'worker.py':ROOT/'tools/modal_tracking_worker.py'})
    hashes={name:sha256_file(p) for name,p in inputs.items()}
    if hashes['video24.mp4']!=previous['hashes']['video24.mp4']:raise ValueError('Source changed')
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    allocation=CpuSandboxAllowance(ROOT/'runs/modal-cpu-analysis-20261102-allowance.sqlite3').reserve_tracking_trial(work.name,fingerprint)
    record={'status':'prepared','allocation':allocation,'hashes':hashes,'worker_terminated':'not_created','gpu':None}
    save(work/'tracking-trial.json',record);sandbox=None;creation=False;modal=client=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-tracking-review',client=client,environment_name='main',create_if_missing=True)
            creation=True
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id(previous['image_id'],client=client),
                runtime='gvisor',gpu=None,cpu=(2,2),memory=(4096,4096),timeout=1800,block_network=True,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'})
        record.update(worker_id=sandbox.object_id,worker_terminated='pending',status='running');save(work/'tracking-trial.json',record)
        print(json.dumps({'stage':'sandbox','status':'created','worker_id':sandbox.object_id}),flush=True)
        for name,p in inputs.items():
            with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(p,'/work/'+name)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.write_text(json.dumps({'hashes':hashes}),'/work/job.json')
            process=sandbox.exec('python','-I','/work/worker.py',timeout=1500,text=False,bufsize=-1)
        total=0
        for chunk in process.stdout:
            total+=len(chunk)
            if total>65536:raise ValueError('Worker log budget exceeded')
            with (work/'worker.events.jsonl').open('ab') as stream:stream.write(chunk)
            print(chunk.decode('utf-8'),end='',flush=True)
        process.wait()
        if process.returncode!=0:raise RuntimeError('Tracking worker failed')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/tracking-results.zip').size>512*1024**2:raise ValueError('Results too large')
            sandbox.filesystem.copy_to_local('/work/tracking-results.zip',work/'tracking-results.zip')
        extract_results(work/'tracking-results.zip',work)
        receipt=json.loads((work/'cpu-receipt.json').read_text())
        if receipt['input_hashes']!=hashes or receipt['status']!='complete':raise ValueError('Receipt lineage mismatch')
        record.update(status='complete',receipt=receipt)
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc))
        if sandbox is not None:
            for name in ('tests.log','artifacts.log','trial.log'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/'+name).size<1024**2:sandbox.filesystem.copy_to_local('/work/'+name,work/name)
                except Exception:pass
    finally:
        if sandbox is None and creation:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox=modal.Sandbox.from_name('touchline-tracking-review',work.name,client=client,environment_name='main')
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception:record['worker_terminated']='unknown'
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'tracking-trial.json',record)
    if record['status']=='failed':
        from touchline.repair_events import publish_job_failure
        record['repair_event']=publish_job_failure(work,record,queue_path=ROOT/'runs/repair-events.sqlite3',stage='tracking')
        save(work/'tracking-trial.json',record)
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-existing-cap',action='store_true');args=parser.parse_args()
    if not args.approved_existing_cap:raise SystemExit('Explicit existing-cap scope required')
    result=run(ROOT/'runs/modal-tracking-trial-20261102')
    print(json.dumps({'status':result['status'],'worker_terminated':result['worker_terminated']}))
    if result['status']!='complete':raise SystemExit(1)
