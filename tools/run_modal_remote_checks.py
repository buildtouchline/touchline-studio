"""Run code validation only in a bounded Modal CPU Sandbox, without user data."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import logging
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from touchline.cpu_sandbox_jobs import CpuSandboxAllowance
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error

RUNNER=r'''import hashlib,json,os,re,subprocess,sys,zipfile
from pathlib import Path
work=Path('/work');project=work/'project';project.mkdir()
with zipfile.ZipFile(work/'checks-code.zip') as archive:
 for entry in archive.infolist():
  target=(project/entry.filename).resolve()
  if not target.is_relative_to(project) or '..' in Path(entry.filename).parts or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe code archive')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(entry))
os.chdir(project)
with (work/'checks.log').open('wb') as log:
 result=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
data=(work/'checks.log').read_bytes()
if len(data)>1024*1024:raise ValueError('Test log exceeds budget')
text=data.decode('utf-8',errors='replace');counts=re.findall(r'Ran (\d+) tests?',text)
with (work/'artifact-checks.log').open('wb') as log:
 artifacts=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py','artifacts'],stdout=log,stderr=subprocess.STDOUT,timeout=120)
summary={'status':'passed' if result.returncode==0 and artifacts.returncode==0 else 'failed','tests':int(counts[-1]) if counts else 0,'test_exit_code':result.returncode,'artifact_exit_code':artifacts.returncode,'log_sha256':hashlib.sha256(data).hexdigest(),'gpu_requested':False,'execution':'Modal CPU Sandbox','source_sha256':hashlib.sha256((work/'checks-code.zip').read_bytes()).hexdigest()}
try:summary['artifacts']=json.loads((work/'artifact-checks.log').read_text())
except Exception:summary['artifacts']={'status':'unparsed'}
summary['failed_test_names']=re.findall(r'^(?:FAIL|ERROR): ([a-zA-Z0-9_ .()]+)',text,re.MULTILINE)
(work/'checks-result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
'''


def make_bundle(path):
    candidates=[]
    for folder in ('src','tests','tools','docs','notebooks','examples','.github'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or any(v in {'.venv','__pycache__','runs','artifacts'} for v in p.parts):continue
            if p.name=='.env' or p.name.startswith('.env.'):continue
            if p.suffix.lower() in {'.py','.md','.json','.ipynb','.toml','.csv','.ini','.txt','.template'} or p.name=='CODEOWNERS':candidates.append(p)
    for name in ('pyproject.toml','README.md','AGENTS.md','Skills.md','SKILL.md','.gitignore'):
        if (ROOT/name).is_file():candidates.append(ROOT/name)
    if sum(p.stat().st_size for p in candidates)>20*1024*1024:raise ValueError('Expected small source-only archive')
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(set(candidates)):
            if any(v.is_symlink() for v in (p,*p.parents)):raise ValueError('Linked source refused')
            archive.write(p,p.relative_to(ROOT).as_posix())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(work,*,retry_package_fix=False,event_checks=False,final_checks=False,handover_checks=False):
    if sum((retry_package_fix,event_checks,final_checks,handover_checks))>1:raise ValueError('Choose one bounded check phase')
    parent=json.loads((ROOT/'runs/modal-cpu-full-video-20261102/modal-cpu.json').read_text(encoding='utf-8'))
    if parent.get('status')!='complete' or parent.get('worker_terminated') is not True:raise ValueError('Reconciled CPU analysis required')
    work.mkdir(parents=True,exist_ok=False);digest=make_bundle(work/'checks-code.zip')
    allowance=CpuSandboxAllowance(ROOT/'runs/modal-cpu-analysis-20261102-allowance.sqlite3')
    if handover_checks:allocation=allowance.reserve_handover_checks(work.name,digest)
    elif final_checks:allocation=allowance.reserve_final_checks(work.name,digest)
    elif event_checks:
        if retry_package_fix:raise ValueError('Choose one bounded check phase')
        allocation=allowance.reserve_event_checks(work.name,digest)
    elif retry_package_fix:
        previous=json.loads((ROOT/'runs/modal-cpu-checks-20261102/remote-checks.json').read_text(encoding='utf-8'))
        if (previous.get('worker_terminated') is not True or previous.get('result',{}).get('test_exit_code')!=0 or
            previous.get('result',{}).get('artifact_exit_code')!=1):raise ValueError('Reconciled packaging-only failure required')
        allocation=allowance.reserve_remote_checks_retry(work.name,digest)
    else:allocation=allowance.reserve_remote_checks(work.name,digest)
    image_id=parent['image_id']
    if final_checks or handover_checks:
        recovery=json.loads((ROOT/'runs/modal-training-recovery-20261102/recovery.json').read_text())
        if recovery.get('status')!='complete':raise ValueError('Completed corrected training image required')
        image_id=recovery['evaluation_image_id']
    record={'status':'prepared','source_sha256':digest,'allocation':allocation,'worker_terminated':'not_created','image_id':image_id}
    save(work/'remote-checks.json',record);sandbox=None;modal=client=None;creation=False
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-cpu-checks',client=client,environment_name='main',create_if_missing=True)
            creation=True
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id(image_id,client=client),
                gpu=None,runtime='gvisor',cpu=(2,2),memory=(4096,4096),timeout=900,block_network=True,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'})
            record.update(worker_id=sandbox.object_id,status='running',worker_terminated='pending');save(work/'remote-checks.json',record)
            sandbox.filesystem.copy_from_local(work/'checks-code.zip','/work/checks-code.zip')
            sandbox.filesystem.write_text(RUNNER,'/work/checks.py')
            process=sandbox.exec('python','-I','/work/checks.py',timeout=780,text=False,bufsize=-1)
        from touchline.repair_checks import read_worker_result
        result=json.loads(read_worker_result(process.stdout));process.wait()
        if process.returncode!=0:raise RuntimeError('Remote validation worker failed')
        for name in ('checks.log','artifact-checks.log','checks-result.json'):
            with redirect_stdout(Discard()),redirect_stderr(Discard()):
                if sandbox.filesystem.stat('/work/'+name).size>1024*1024:raise ValueError('Output log too large')
                sandbox.filesystem.copy_to_local('/work/'+name,work/name)
        record.update(status=result['status'],result=result)
    except BaseException as exc:record.update(status='failed',**safe_error(exc))
    finally:
        if sandbox is None and creation:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox=modal.Sandbox.from_name('touchline-cpu-checks',work.name,environment_name='main',client=client)
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception:record['worker_terminated']='unknown'
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'remote-checks.json',record)
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-cpu-budget',action='store_true')
    parser.add_argument('--retry-package-fix',action='store_true');parser.add_argument('--event-checks',action='store_true');parser.add_argument('--final-checks',action='store_true');parser.add_argument('--handover-checks',action='store_true');args=parser.parse_args()
    try:
        if not args.approved_cpu_budget:raise ValueError('Explicit CPU budget scope required')
        result=run(ROOT/'runs'/('modal-handover-checks-20261103' if args.handover_checks else ('modal-final-checks-20261102' if args.final_checks else ('modal-event-checks-20261102' if args.event_checks else ('modal-cpu-checks-20261102-retry' if args.retry_package_fix else 'modal-cpu-checks-20261102')))),retry_package_fix=args.retry_package_fix,event_checks=args.event_checks,final_checks=args.final_checks,handover_checks=args.handover_checks)
        print(json.dumps(result,indent=2))
        if result['status']!='passed':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
