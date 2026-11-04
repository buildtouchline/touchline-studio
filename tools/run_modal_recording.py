"""One owner-requested PXL full recording in a bounded Modal CPU Sandbox."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,shutil,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,modal_client,save
from run_modal_remote_checks import make_bundle
from probe_modal_gpu import safe_error
from touchline.data.manifest import sha256_file
from touchline.spending import SpendingLedger

ALLOWED={'run','profiles','preview','diagnostics','normalized','recording-receipt.json','RESULTS.html','checks.json','processing.log','tests.log','artifacts.log'}
def extract(path,output,*,max_bytes=8*1024**3):
    if type(max_bytes) is not int or not 0<max_bytes<=24*1024**3:raise ValueError('Bounded result size required')
    output=output.resolve()
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();names=set()
        if len(entries)>20000 or sum(e.file_size for e in entries)>max_bytes:raise ValueError('Result archive exceeds bound')
        for e in entries:
            p=(output/e.filename).resolve()
            if e.filename in names or not Path(e.filename).parts or Path(e.filename).parts[0] not in ALLOWED or not p.is_relative_to(output) or p.exists() or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe result entry')
            names.add(e.filename)
        for e in entries:
            p=output/e.filename;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(e) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
def run(*,verified_retry=False):
    work=ROOT/('runs/modal-pxl-analysis-20261103-retry' if verified_retry else 'runs/modal-pxl-analysis-20261103');work.mkdir(exist_ok=False)
    if verified_retry:
        previous=json.loads((ROOT/'runs/modal-pxl-analysis-20261103/job-receipt.json').read_text())
        probe=json.loads((ROOT/'runs/modal-soccertrack-audit-20261103/review/pxl-probe.json').read_text())
        if previous['status']!='failed' or previous['worker_terminated'] is not True or probe['is_rotated'] or not 2<=probe['duration_s']<=900:raise ValueError('Reconciled failure and verified source metadata required')
    video=ROOT/'artifacts/videos/PXL_20261101_045045814~2.mp4'
    if video.is_symlink() or not video.is_file() or not 0<video.stat().st_size<=5*1024**3:raise ValueError('Expected selected recording under 5 GiB')
    model=ROOT/('runs/modal-resize-training-20261103/candidate/candidate.pt' if verified_retry else 'runs/modal-training-recovery-20261102/minor/candidate/candidate.pt')
    expected='6340f81877408e693dcd4734d84214c29095011a184e4c8187a98139ef3b7d80' if verified_retry else 'b9200825a2bb3e39589fb705ced502531136b2cc70496c0a0d9a8519e2e99063'
    if sha256_file(model)!=expected:raise ValueError('Selected checkpoint changed')
    make_bundle(work/'code.zip')
    inputs={'recording.mp4':video,'candidate.pt':model,'worker.py':ROOT/'tools/modal_recording_worker.py','code.zip':work/'code.zip'}
    hashes={n:sha256_file(p) for n,p in inputs.items()};fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    ledger_path=ROOT/'runs/soccertrack-pxl-20261103-budget.sqlite3';fresh=not ledger_path.exists();ledger=SpendingLedger(ledger_path)
    # Conservative prior upper bound includes all old holds, never releases them.
    if fresh:ledger.verify_pool('modal',prior_spend_usd='35.00',available_credit_usd='15.00')
    claim_id='pxl-full-cpu-20261103-retry' if verified_retry else 'pxl-full-cpu-20261103';cap='4.00' if verified_retry else '5.00'
    claim=ledger.reserve('modal',claim_id,fingerprint=fingerprint,worst_case_usd=cap)
    if not claim['created']:raise ValueError('Consumed recording claim; no duplicate dispatch')
    record={'status':'prepared','hashes':hashes,'gpu':None,'cpu':4,'memory_mib':8192,'timeout_s':10800,
        'reserved_usd':cap,'prior_upper_bound_usd':'35.00','billing':'unreconciled','worker_terminated':'not_created','training':'not run'}
    save(work/'job-receipt.json',record);sandbox=None;creation=False;modal=client=None
    def event(item):
        with (work/'controller.events.jsonl').open('a') as s:s.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-recording-analysis',client=client,environment_name='main',create_if_missing=True)
            creation=True
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id('im-A4IdpUIDiLlRbBha7sqzYy',client=client),
                gpu=None,cpu=(4,4),memory=(8192,8192),runtime='gvisor',timeout=10800,block_network=True,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'})
        record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'job-receipt.json',record);event({'stage':'sandbox','status':'created','worker_id':sandbox.object_id})
        for name,p in inputs.items():
            if verified_retry and name=='recording.mp4':continue
            event({'stage':'upload','status':'started','file':name,'bytes':p.stat().st_size})
            with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(p,'/work/'+name)
            event({'stage':'upload','status':'complete','file':name})
        if verified_retry:
            # Fail fast on metadata before transferring the full recording again.
            for name in ('head.bin','moov.bin','layout.json'):
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(ROOT/'artifacts/soccertrack-v2/probe'/name,'/work/'+name)
            build="from pathlib import Path; import json; w=Path('/work'); v=json.loads((w/'layout.json').read_text()); s=(w/'recording.mp4').open('wb'); s.write((w/'head.bin').read_bytes()); s.seek(v['moov_offset']); s.write((w/'moov.bin').read_bytes()); s.truncate(v['source_size']); s.close()"
            with redirect_stdout(Discard()),redirect_stderr(Discard()):
                sandbox.filesystem.write_text(json.dumps({'hashes':{k:v for k,v in hashes.items() if k!='recording.mp4'}}),'/work/job.json')
                p=sandbox.exec('python','-I','-c',build,timeout=30);p.wait()
                probe=sandbox.exec('python','-I','/work/worker.py','--probe-only',timeout=60,text=False)
                from touchline.repair_checks import read_worker_result
                probe_events=read_worker_result(probe.stdout);probe.wait()
                if p.returncode or probe.returncode:raise RuntimeError('Independent metadata probe rejected recording')
                sandbox.filesystem.copy_to_local('/work/source-probe.json',work/'source-probe.json')
            event({'stage':'source_probe','status':'verified_before_full_upload'})
            # Bounded chunks with remote SHA verification avoid opaque multi-GB stdin transfers.
            offset=0
            with video.open('rb') as stream:
                while chunk:=stream.read(128*1024**2):
                    digest=hashlib.sha256(chunk).hexdigest()
                    append="from pathlib import Path; import hashlib,sys; w=Path('/work'); b=(w/'upload-part.bin').read_bytes(); assert hashlib.sha256(b).hexdigest()==sys.argv[1]; p=w/'recording.mp4'; mode='wb' if int(sys.argv[2])==0 else 'ab'; assert int(sys.argv[2])==0 or p.stat().st_size==int(sys.argv[2]); s=p.open(mode); s.write(b); s.close()"
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        sandbox.filesystem.write_bytes(chunk,'/work/upload-part.bin')
                        transfer=sandbox.exec('python','-I','-c',append,digest,str(offset),timeout=60);transfer.wait()
                        if transfer.returncode:raise RuntimeError('Chunk integrity/offset failed')
                    offset+=len(chunk);event({'stage':'upload_verified','status':'progress','bytes':offset})
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.write_text(json.dumps({'hashes':hashes}),'/work/job.json')
            process=sandbox.exec('python','-I','/work/worker.py',timeout=9800,text=False,bufsize=-1)
        total=0;buffer=bytearray();complete=False
        for chunk in process.stdout:
            total+=len(chunk)
            if total>1024**2:raise ValueError('Worker log bound exceeded')
            buffer.extend(chunk)
            while b'\n' in buffer:
                line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
                if not line.strip():continue
                item=json.loads(line)
                if not isinstance(item,dict) or set(item)-{'stage','status','elapsed_s','coverage_rows','frames','width','height','fps','duration_s','error_category','error_line'}:raise ValueError('Unexpected worker event')
                event(item)
                if item.get('stage')=='complete' and item.get('status')=='complete':complete=True
        process.wait()
        if process.returncode or not complete or buffer.strip():raise RuntimeError('Recording worker failed')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/recording-results.zip').size>8*1024**3:raise ValueError('Export exceeds transfer bound')
            sandbox.filesystem.copy_to_local('/work/recording-results.zip',work/'recording-results.zip')
        extract(work/'recording-results.zip',work)
        receipt=json.loads((work/'recording-receipt.json').read_text())
        if receipt['input_hashes']!=hashes or receipt['status']!='complete':raise ValueError('Export lineage mismatch')
        record.update(status='complete',run_id=receipt['run_id'],decoded_frames=receipt['decoded_frames'],selected_frames=receipt['selected_frames'],observations=receipt['observations'])
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc));event({'stage':'controller','status':'failed',**safe_error(exc)})
        if sandbox is not None:
            for name in ('recording-failure.zip','source-probe.json','tests.log','artifacts.log'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/'+name).size<128*1024**2:sandbox.filesystem.copy_to_local('/work/'+name,work/('failure-'+name))
                except Exception:pass
    finally:
        if sandbox is None and creation:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox=modal.Sandbox.from_name('touchline-recording-analysis',work.name,client=client,environment_name='main')
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception:record['worker_terminated']='unknown'
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'job-receipt.json',record);ledger.reconcile(claim_id,finalized=False)
        if record['worker_terminated'] not in (True,'confirmed_no_worker'):ledger.block_pool('modal')
    return record
if __name__=='__main__':
    logging.disable(logging.CRITICAL);p=argparse.ArgumentParser();p.add_argument('--approved-existing-budget',action='store_true');p.add_argument('--verified-input-retry',action='store_true');a=p.parse_args()
    if not a.approved_existing_budget:raise SystemExit('Explicit selected recording/budget scope required')
    try:
        result=run(verified_retry=a.verified_input_retry);print(json.dumps({k:result.get(k) for k in ('status','worker_terminated','run_id','decoded_frames','selected_frames','observations')}))
        if result['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
