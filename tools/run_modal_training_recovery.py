"""One bounded CPU recovery plus baseline/minor T4 trials; persistent stage exports."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,sqlite3,sys,uuid,zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.data.manifest import sha256_file
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error


def extract_small(archive_path,output):
    with zipfile.ZipFile(archive_path) as archive:
        if sum(e.file_size for e in archive.infolist())>512*1024**2:raise ValueError('Recovery results too large')
        for e in archive.infolist():
            p=(output/e.filename).resolve()
            if (not p.is_relative_to(output) or p.exists() or (e.external_attr>>16)&0o170000==0o120000 or
                Path(e.filename).parts[0] not in {'candidate','candidate-val','baseline-val','diagnostics','training-receipt.json','minor-trial-receipt.json','training.log','run','profiles','preview','review','analysis-receipt.json','evaluation-comparison.json','checks.json','RESULTS.html'}):raise ValueError('Unsafe recovery artifact')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))


def phase(work,*,modal,client,app,image,inputs,job,gpu=None,is_analysis=False):
    work.mkdir();record={'status':'prepared','attempt_id':uuid.uuid4().hex,'worker_terminated':'not_created','gpu':gpu,'image_id':image.object_id}
    hashes={n:sha256_file(p) for n,p in inputs.items()};record['hashes']=hashes;save(work/'job-receipt.json',record)
    sandbox=None;creation=False
    try:
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            creation=True
            sandbox=modal.Sandbox.create(app=app,client=client,name='touchline-recovery-'+work.name+'-20261102',image=image,
                runtime='gvisor',gpu=gpu,cpu=(4,4) if gpu or is_analysis else (8,8),memory=(8192,8192),timeout=900 if is_analysis else (3600 if gpu else 2400),
                block_network=gpu is not None or is_analysis,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'})
        record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'job-receipt.json',record)
        print(json.dumps({'phase':work.name,'stage':'sandbox','status':'created','worker_id':sandbox.object_id}),flush=True)
        for name,p in inputs.items():
            with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(p,'/work/'+name)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.write_text(json.dumps({**job,'hashes':hashes}),'/work/job.json')
            process=sandbox.exec('python','-I','/work/worker.py',timeout=780 if is_analysis else (3000 if gpu else 2100),text=False,bufsize=-1)
        buffer=bytearray();total=0;complete=False
        for chunk in process.stdout:
            total+=len(chunk)
            if total>1024**2:raise ValueError('Worker log bound exceeded')
            buffer.extend(chunk)
            while b'\n' in buffer:
                line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
                if not line.strip():continue
                item=json.loads(line)
                if not isinstance(item,dict) or set(item)-{'stage','status','elapsed_s','expected_bytes','bytes','mode','exit_code','tests','pycocotools','verified','images','learning_rate','gpu','steps','checkpoint_sha256','model','ap50','error_category','failure_code'}:raise ValueError('Unexpected worker event')
                with (work/'worker.events.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
                print(json.dumps({'phase':work.name,**item}),flush=True)
                if item['stage']=='checkpoint':
                    # Save completed weights before evaluation can fail.
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_to_local('/work/candidate/candidate.pt',work/'checkpoint-recovery.pt')
                    if sha256_file(work/'checkpoint-recovery.pt')!=item['checkpoint_sha256']:raise ValueError('Checkpoint recovery hash mismatch')
                    record['checkpoint_exported_before_validation']=True;save(work/'job-receipt.json',record)
                if item['stage']=='complete':complete=True
        process.wait()
        if process.returncode or not complete:raise RuntimeError('Recovery phase failed')
        if gpu is None and not is_analysis:
            with redirect_stdout(Discard()),redirect_stderr(Discard()):
                sandbox.filesystem.copy_to_local('/work/data-receipt.json',work/'data-receipt.json')
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):snapshot=sandbox.snapshot_filesystem(timeout=120,ttl=604800)
                record['snapshot_image_id']=snapshot.object_id;record['snapshot_ttl_seconds']=604800
            except Exception as exc:
                record['snapshot_error']=safe_error(exc)
                # Preserve verified data before shutdown even if snapshots fail.
                command="from pathlib import Path; import zipfile; w=Path('/work'); z=zipfile.ZipFile(w/'dataset-export.zip','w',zipfile.ZIP_STORED); [z.write(p,p.relative_to(w).as_posix()) for p in (w/'dataset').rglob('*') if p.is_file()]; z.close()"
                with redirect_stdout(Discard()),redirect_stderr(Discard()):
                    export_process=sandbox.exec('python','-I','-c',command,timeout=180);export_process.wait()
                    if export_process.returncode or sandbox.filesystem.stat('/work/dataset-export.zip').size>5*1024**3:raise RuntimeError('Dataset fallback export failed')
                    sandbox.filesystem.copy_to_local('/work/dataset-export.zip',work/'dataset-export.zip')
                record['dataset_archive_sha256']=sha256_file(work/'dataset-export.zip')
        else:
            with redirect_stdout(Discard()),redirect_stderr(Discard()):
                if sandbox.filesystem.stat('/work/gpu-results.zip').size>256*1024**2:raise ValueError('GPU output bound exceeded')
                sandbox.filesystem.copy_to_local('/work/gpu-results.zip',work/'gpu-results.zip')
            extract_small(work/'gpu-results.zip',work)
        record['status']='complete'
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc))
        if sandbox is not None:
            for name in ('tests.log','artifacts.log','training.log','data-receipt.json','candidate/result.json'):
                try:
                    target=work/('failure-'+name.replace('/','-'))
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/'+name).size<1024**2:sandbox.filesystem.copy_to_local('/work/'+name,target)
                except Exception:pass
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):
                    if sandbox.filesystem.stat('/work/gpu-failure.zip').size<256*1024**2:sandbox.filesystem.copy_to_local('/work/gpu-failure.zip',work/'gpu-failure.zip')
            except Exception:pass
    finally:
        if sandbox is None and creation:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox=modal.Sandbox.from_name('touchline-training-recovery','touchline-recovery-'+work.name+'-20261102',client=client,environment_name='main')
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception:record['worker_terminated']='unknown'
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'job-receipt.json',record)
    if record['status']=='failed':
        from touchline.repair_events import publish_job_failure
        record['repair_event']=publish_job_failure(work,record,queue_path=ROOT/'runs/repair-events.sqlite3',stage='analysis' if is_analysis else ('training' if gpu else 'restore'))
        save(work/'job-receipt.json',record)
    if record['status']!='complete' or record['worker_terminated'] is not True:raise RuntimeError('Phase failed or remains unreconciled')
    return record


def run():
    previous=json.loads((ROOT/'runs/modal-reviewed-training-20261102/training-job.json').read_text())
    if previous['status']!='failed' or previous['worker_terminated'] is not True:raise ValueError('Reconciled failed baseline required')
    work=ROOT/'runs/modal-training-recovery-20261102';work.mkdir(exist_ok=False)
    with sqlite3.connect(ROOT/'runs/modal-training-recovery-20261102-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY, reserved_micro_usd INTEGER)')
        db.execute('INSERT INTO claims VALUES (?,?)',('separated-training-recovery-half-budget',9000000))
    record={'status':'prepared','reserved_usd':'9.00','cumulative_modal_ceiling_usd':'50.00','billing':'unreconciled','prior_holds':'retained'}
    save(work/'recovery.json',record);make_bundle(work/'checks-code.zip')
    source=ROOT.parent/'Models-Trained/person-20261031T115722222540Z'
    inputs={'checks-code.zip':work/'checks-code.zip','release.json':source/'release.json',
        'candidate.pt':source/'candidate-train/candidate.pt','advice.json':ROOT/'runs/training-tuning-advice-20261102/proposal.json',
        'worker.py':ROOT/'tools/modal_dataset_recovery_worker.py'}
    if sha256_file(inputs['release.json'])!='03b75589845449c4c068df012024c02a14119037310cefbee97a63617e5ff303' or sha256_file(inputs['candidate.pt'])!='22ede06aaf1fdb695570cb0ba103f997e71edb7759b2228b47d6a85d2b4c31a1':raise ValueError('Reviewed inputs changed')
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-training-recovery',client=client,environment_name='main',create_if_missing=True)
            image=modal.Image.from_id('im-QcRMDynaNwmRjN6PN4KbJr',client=client).pip_install('pycocotools==2.0.11')
            image.build(app)
        record['evaluation_image_id']=image.object_id;save(work/'recovery.json',record)
        data=phase(work/'data',modal=modal,client=client,app=app,image=image,inputs=inputs,job={})
        record['data']=data;save(work/'recovery.json',record)
        snapshot=modal.Image.from_id(data['snapshot_image_id'],client=client) if data.get('snapshot_image_id') else image
        gpu_inputs={'worker.py':ROOT/'tools/modal_recovered_gpu_worker.py','candidate.pt':inputs['candidate.pt'],
            'checks-code.zip':inputs['checks-code.zip'],'advice.json':inputs['advice.json'],'data-receipt.json':work/'data/data-receipt.json'}
        if not data.get('snapshot_image_id'):gpu_inputs['dataset.zip']=work/'data/dataset-export.zip'
        baseline=phase(work/'baseline',modal=modal,client=client,app=app,image=snapshot,inputs=gpu_inputs,
            job={'mode':'baseline','parent_run_id':'79ba69be-028e-4419-a4ef-57cd60b4f3c0'},gpu='T4')
        record['baseline']=baseline;save(work/'recovery.json',record)
        gpu_inputs={**gpu_inputs,'baseline-receipt.json':work/'baseline/training-receipt.json',
            'baseline-evaluation.json':work/'baseline/candidate-val/evaluation.json'}
        minor=phase(work/'minor',modal=modal,client=client,app=app,image=snapshot,inputs=gpu_inputs,
            job={'mode':'minor','parent_run_id':json.loads((work/'baseline/training-receipt.json').read_text())['training']['run_id']},gpu='T4')
        record.update(minor=minor,status='complete');save(work/'recovery.json',record)
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc));save(work/'recovery.json',record);raise
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--approved-half-budget-recovery',action='store_true');args=p.parse_args()
    if not args.approved_half_budget_recovery:raise SystemExit('Explicit bounded recovery scope required')
    try:r=run();print(json.dumps({'status':r['status']}))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
