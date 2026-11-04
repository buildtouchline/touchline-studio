"""One five-dollar T4 recovery/training trial under the cumulative Modal $50 ceiling."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,logging,sqlite3,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.data.manifest import sha256_file
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error


def run():
    source=ROOT.parent/'Models-Trained/person-20261031T115722222540Z'
    reviewed=json.loads((source/'soccernet-staging/etl-config.json').read_text())
    if not all(reviewed.get(k) is True for k in ('annotations_reviewed','person_labels_complete','training_eligible')):raise ValueError('Reviewed data required')
    work=ROOT/'runs/modal-reviewed-training-20261102';work.mkdir(exist_ok=False)
    make_bundle(work/'checks-code.zip')
    inputs={'checks-code.zip':work/'checks-code.zip','worker.py':ROOT/'tools/modal_training_worker.py',
        'release.json':source/'release.json','candidate.pt':source/'candidate-train/candidate.pt'}
    hashes={name:sha256_file(p) for name,p in inputs.items()}
    if hashes['release.json']!='03b75589845449c4c068df012024c02a14119037310cefbee97a63617e5ff303' or hashes['candidate.pt']!='22ede06aaf1fdb695570cb0ba103f997e71edb7759b2228b47d6a85d2b4c31a1':raise ValueError('Selected training inputs changed')
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    # Preserve all earlier ledgers. This is a separate, single bounded phase;
    # the latest owner ceiling is cumulative $50, not a per-job allowance.
    with sqlite3.connect(ROOT/'runs/modal-reviewed-training-20261102-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY, fingerprint TEXT, reserved_micro_usd INTEGER)')
        db.execute('INSERT INTO claims VALUES (?,?,?)',('reviewed-training-20261102-half-budget',fingerprint,5000000))
    record={'status':'prepared','worker_terminated':'not_created','hashes':hashes,'reserved_usd':'5.00',
        'cumulative_modal_ceiling_usd':'50.00','billing':'unreconciled','gpu':'T4','timeout_s':3600,
        'training':'not started','reviewed_config_sha256':sha256_file(source/'soccernet-staging/etl-config.json')}
    save(work/'training-job.json',record);sandbox=None;creation=False;modal=client=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-reviewed-training',client=client,environment_name='main',create_if_missing=True)
            creation=True
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id('im-QcRMDynaNwmRjN6PN4KbJr',client=client),
                runtime='gvisor',gpu='T4',cpu=(4,4),memory=(8192,8192),timeout=3600,block_network=False,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'})
        record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'training-job.json',record)
        print(json.dumps({'stage':'sandbox','worker_id':sandbox.object_id,'status':'created'}),flush=True)
        for name,p in inputs.items():
            with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(p,'/work/'+name)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.write_text(json.dumps({'hashes':hashes,'parent_run_id':'79ba69be-028e-4419-a4ef-57cd60b4f3c0'}),'/work/job.json')
            process=sandbox.exec('python','-I','/work/worker.py',timeout=3300,text=False,bufsize=-1)
        total=0
        for chunk in process.stdout:
            total+=len(chunk)
            if total>1024**2:raise ValueError('Worker logs exceed bound')
            with (work/'worker.events.jsonl').open('ab') as stream:stream.write(chunk)
            print(chunk.decode('utf-8'),end='',flush=True)
        process.wait()
        if process.returncode:raise RuntimeError('Training worker did not complete')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/training-results.zip').size>5*1024**3:raise ValueError('Results too large')
            sandbox.filesystem.copy_to_local('/work/training-results.zip',work/'training-results.zip')
        # Keep the immutable dataset archive; extract only small candidate/evaluation
        # artifacts here. Images can be extracted later by the operator.
        with zipfile.ZipFile(work/'training-results.zip') as archive:
            for e in archive.infolist():
                p=(work/e.filename).resolve()
                if not p.is_relative_to(work) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe training results')
                if e.filename.startswith('dataset/images/'):continue
                if p.exists() or e.file_size>256*1024**2:raise ValueError('Invalid output file')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
        receipt=json.loads((work/'training-receipt.json').read_text())
        if receipt['input_hashes']!=hashes:raise ValueError('Result lineage mismatch')
        record.update(status='complete',training='completed bounded trial',receipt=receipt)
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc))
        if sandbox is not None:
            for name in ('tests.log','artifacts.log','training.log','evaluation.log'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/'+name).size<1024**2:sandbox.filesystem.copy_to_local('/work/'+name,work/name)
                except Exception:pass
    finally:
        if sandbox is None and creation:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox=modal.Sandbox.from_name('touchline-reviewed-training',work.name,client=client,environment_name='main')
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception:record['worker_terminated']='unknown'
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'training-job.json',record)
    if record['status']=='failed':
        from touchline.repair_events import publish_job_failure
        record['repair_event']=publish_job_failure(work,record,queue_path=ROOT/'runs/repair-events.sqlite3',stage='training')
        save(work/'training-job.json',record)
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-half-budget',action='store_true');args=parser.parse_args()
    if not args.approved_half_budget:raise SystemExit('Explicit half-budget scope required')
    try:
        result=run();print(json.dumps({'status':result['status'],'worker_terminated':result['worker_terminated']}))
        if result['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'blocked',**safe_error(exc)}));raise SystemExit(1) from None
