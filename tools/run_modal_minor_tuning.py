"""Execute the accepted Baseten minor proposal after the independent baseline."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,sqlite3,sys,zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.data.manifest import sha256_file
from touchline.training_trials import admit_minor_trial
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error


def run():
    parent=ROOT/'runs/modal-reviewed-training-20261102';prior=json.loads((parent/'training-job.json').read_text())
    if prior['status']!='complete' or prior['worker_terminated'] is not True:raise ValueError('Completed/reconciled independent baseline required')
    advice_path=ROOT/'runs/training-tuning-advice-20261102/proposal.json';advice=json.loads(advice_path.read_text())
    if advice['status']!='proposal_classified':raise ValueError('Classified adviser proposal required')
    baseline_eval=json.loads((parent/'candidate-val/evaluation.json').read_text())
    plan=admit_minor_trial(advice['proposal'],advice['baseline'],prior['receipt'],baseline_eval)
    # The adviser cannot change the evaluator or the frozen data between trials.
    with zipfile.ZipFile(parent/'checks-code.zip') as archive:
        evaluator_sha=hashlib.sha256(archive.read('src/touchline/training/evaluate.py')).hexdigest()
    if sha256_file(ROOT/'src/touchline/training/evaluate.py')!=evaluator_sha:raise ValueError('Independent evaluator changed')
    work=ROOT/'runs/modal-minor-tuning-20261102';work.mkdir(exist_ok=False);make_bundle(work/'checks-code.zip')
    inputs={'checks-code.zip':work/'checks-code.zip','worker.py':ROOT/'tools/modal_minor_tuning_worker.py',
        'dataset.zip':parent/'training-results.zip','advice.json':advice_path,
        'baseline-receipt.json':parent/'training-receipt.json','baseline-evaluation.json':parent/'candidate-val/evaluation.json',
        'candidate.pt':ROOT.parent/'Models-Trained/person-20261031T115722222540Z/candidate-train/candidate.pt'}
    hashes={name:sha256_file(p) for name,p in inputs.items()}
    if hashes['candidate.pt']!=advice['baseline']['checkpoint_sha256']:raise ValueError('Warm-start/rollback checkpoint changed')
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    with sqlite3.connect(ROOT/'runs/modal-minor-tuning-20261102-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY, fingerprint TEXT, reserved_micro_usd INTEGER)')
        db.execute('INSERT INTO claims VALUES (?,?,?)',('rulebook-minor-trial-half-budget-20261102',fingerprint,3000000))
    record={'status':'prepared','hashes':hashes,'worker_terminated':'not_created','reserved_usd':'3.00','cumulative_modal_ceiling_usd':'50.00',
        'billing':'unreconciled','plan':plan,'evaluator_sha256':evaluator_sha,'promotion':'not performed'}
    save(work/'minor-job.json',record);sandbox=None;creation=False;modal=client=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-rulebook-tuning',client=client,environment_name='main',create_if_missing=True);creation=True
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id('im-QcRMDynaNwmRjN6PN4KbJr',client=client),
                runtime='gvisor',gpu='T4',cpu=(4,4),memory=(8192,8192),timeout=3600,block_network=True,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'})
        record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'minor-job.json',record)
        print(json.dumps({'stage':'sandbox','worker_id':sandbox.object_id,'status':'created'}),flush=True)
        for name,p in inputs.items():
            with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(p,'/work/'+name)
            print(json.dumps({'stage':'input','name':name,'bytes':p.stat().st_size}),flush=True)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.write_text(json.dumps({'hashes':hashes}),'/work/job.json')
            process=sandbox.exec('python','-I','/work/worker.py',timeout=3000,text=False,bufsize=-1)
        total=0
        for chunk in process.stdout:
            total+=len(chunk)
            if total>1024**2:raise ValueError('Worker log cap exceeded')
            with (work/'worker.events.jsonl').open('ab') as stream:stream.write(chunk)
            print(chunk.decode('utf-8'),end='',flush=True)
        process.wait()
        if process.returncode:raise RuntimeError('Minor candidate worker failed')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/minor-results.zip').size>256*1024**2:raise ValueError('Minor candidate export exceeds cap')
            sandbox.filesystem.copy_to_local('/work/minor-results.zip',work/'minor-results.zip')
        with zipfile.ZipFile(work/'minor-results.zip') as archive:
            if sum(e.file_size for e in archive.infolist())>512*1024**2:raise ValueError('Expanded export exceeds cap')
            for e in archive.infolist():
                p=(work/e.filename).resolve()
                if (not p.is_relative_to(work) or p.exists() or (e.external_attr>>16)&0o170000==0o120000 or
                    Path(e.filename).parts[0] not in {'candidate','candidate-val','diagnostics','minor-trial-receipt.json','tests.log','artifacts.log','training.log'}):raise ValueError('Unsafe minor trial export')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
        receipt=json.loads((work/'minor-trial-receipt.json').read_text())
        if receipt['input_hashes']!=hashes or receipt.get('rollback_checkpoint_unchanged') is not True:raise ValueError('Minor trial lineage/rollback mismatch')
        record.update(status='complete',receipt=receipt)
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc))
        if sandbox is not None:
            for name in ('tests.log','artifacts.log','training.log'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/'+name).size<1024**2:sandbox.filesystem.copy_to_local('/work/'+name,work/name)
                except Exception:pass
    finally:
        if sandbox is None and creation:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox=modal.Sandbox.from_name('touchline-rulebook-tuning',work.name,client=client,environment_name='main')
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception:record['worker_terminated']='unknown'
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'minor-job.json',record)
    if record['status']=='failed':
        from touchline.repair_events import publish_job_failure
        record['repair_event']=publish_job_failure(work,record,queue_path=ROOT/'runs/repair-events.sqlite3',stage='training');save(work/'minor-job.json',record)
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-rulebook-trial',action='store_true');args=parser.parse_args()
    if not args.approved_rulebook_trial:raise SystemExit('Explicit bounded rulebook trial required')
    try:
        result=run();print(json.dumps({'status':result['status'],'worker_terminated':result['worker_terminated']}))
        if result['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'blocked',**safe_error(exc)}));raise SystemExit(1) from None
