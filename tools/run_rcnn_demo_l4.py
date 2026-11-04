"""One three-minute demo on L4; shared $3 trial cap, reconnectable file status."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from decimal import Decimal
import hashlib,json,logging,math,shutil,sqlite3,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from run_modal_remote_checks import make_bundle
from touchline.data.manifest import sha256_file
from touchline.worker_status import parse_worker_status
from touchline.repair_checks import read_worker_result

def run():
    old=ROOT/'runs/modal-rcnn-full-gpu-20261103';replacement=ROOT/'runs/modal-rcnn-full-gpu-20261103-replacement'
    original=json.loads((old/'job-receipt.json').read_text());previous=json.loads((replacement/'job-receipt.json').read_text())
    if original['worker_terminated'] is not True or previous['worker_terminated'] is not True:raise ValueError('Both previous T4 workers must be confirmed terminated')
    # Append-only event-file creation predates allocation; final receipt follows
    # confirmed termination. Bound that second T4 conservatively by15 minutes.
    stat=(replacement/'events.jsonl').stat();started=getattr(stat,'st_birthtime',stat.st_ctime)
    if not 0<=(replacement/'job-receipt.json').stat().st_mtime-started<=900:raise ValueError('Previous allocation duration needs reconciliation')
    resource_bound=Decimal(2700)*(Decimal('.000164')+4*Decimal('.00003942')+8*Decimal('.00000667'))+Decimal(1500)*(Decimal('.000222')+4*Decimal('.00003942')+16*Decimal('.00000667'))+Decimal('.35')
    if resource_bound>3:raise ValueError('Shared trial cap exceeded')
    work=ROOT/'runs/modal-rcnn-demo-l4-20261103';work.mkdir(exist_ok=False);make_bundle(work/'code.zip');digest=sha256_file(work/'code.zip')
    with sqlite3.connect(ROOT/'runs/rcnn-gpu-owner-exception-20261103.sqlite3') as db:
        db.execute('INSERT INTO claims VALUES (?,?,?,?,?)',('owner-higher-GPU-three-minute-demo-20261103',0,53000000,53000000,digest))
    record={'status':'prepared','gpu':'L4','cpu':4,'memory_mib':16384,'timeout_s':1500,'shared_total_trial_cap_usd':'3.00','additional_reservation_usd':'0.00','project_retained_ceiling_usd':'53.00','resource_and_transfer_upper_bound_usd':str(resource_bound),'billing':'unreconciled','worker_terminated':'not_created','source_sha256':digest,'requested_seconds':180,'requested_frames':4320,'image_id':original['image_id'],'training':'not run'};save(work/'job-receipt.json',record)
    sandbox=None;client=None
    def event(item):
        with (work/'events.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-rcnn-video-analysis',client=client,environment_name='main',create_if_missing=True)
            sandbox=modal.Sandbox.create(app=app,client=client,image=modal.Image.from_id(original['image_id'],client=client),gpu='L4',cpu=(4,4),memory=(16384,16384),timeout=1500,runtime='gvisor',block_network=True,secrets=[],volumes={},workdir='/work',env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLCONFIGDIR':'/work/cache/matplotlib'})
        allocated=time.time();record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending',allocated_epoch=allocated);save(work/'job-receipt.json',record);event({'stage':'sandbox','status':'created','worker_id':sandbox.object_id})
        reference=old/'fp32-partial.zip';job={'code_sha256':digest,'weights_sha256':'dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf','video_sha256':'d6b4c64a1313ae46b1027214323abd995275bc6e8ed06ff86c22491785240d0a','source_sha256':'6210e20cd0cbabc1dc5c4f932be0db8c7fc3b7f9db3fb492f19d06ba4c2c0cdf','mode':'demo','detector_fps':6,'session_id_cap':24,'max_frames':4320,'decoder_threads':4,'fp32_reference_zip_sha256':sha256_file(reference)}
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.copy_from_local(reference,'/work/rcnn-fp32-reference.zip');sandbox.filesystem.copy_from_local(work/'code.zip','/work/rcnn-gpu-amp-code.zip');sandbox.filesystem.copy_from_local(ROOT/'tools/modal_rcnn_gpu_worker.py','/work/rcnn-l4-worker.py');sandbox.filesystem.write_text(json.dumps(job),'/work/rcnn-gpu-amp-job.json')
            process=sandbox.exec('python','-I','/work/rcnn-l4-worker.py','--amp-recovery',timeout=1350,text=False)
        sequence=0;last=None;complete=False;read_errors=0
        # Stdout is optional telemetry, never the worker's source of truth.
        while time.time()-allocated<1380:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):text=sandbox.filesystem.read_text('/work/rcnn-gpu-amp/worker-state.json')
                state=parse_worker_status(text);read_errors=0
                if state['event_seq']>sequence:
                    event(state);sequence=state['event_seq'];last=state
                if state['status']=='failed':raise RuntimeError('Worker reported a typed failure')
                if state['stage']=='complete' and state['status']=='complete':complete=True;break
            except RuntimeError:raise
            except Exception as exc:
                read_errors+=1;event({'stage':'status_read','status':'recovering','failure_type':type(exc).__name__,'attempt':read_errors})
                with redirect_stdout(Discard()),redirect_stderr(Discard()):exit_code=sandbox.poll()
                if exit_code is not None:raise RuntimeError('Worker exited before verified completion') from None
                if read_errors>6:raise RuntimeError('Status unavailable; preserve evidence before shutdown') from None
            time.sleep(10)
        if not complete:raise TimeoutError('Bounded demo deadline reached')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            summary=json.loads(sandbox.filesystem.read_text('/work/rcnn-gpu-amp/summary.json'))
            if summary['frames']!=4320 or summary['checkpoint_sha256']!=job['weights_sha256'] or summary['source_sha256']!=job['video_sha256']:raise ValueError('Demo result lineage mismatch')
            size=sandbox.filesystem.stat('/work/rcnn-gpu-amp/results.zip').size
            if not 0<size<=512*1024**2:raise ValueError('Demo result exceeds transfer bound')
            sandbox.filesystem.copy_to_local('/work/rcnn-gpu-amp/results.zip',work/'results.zip')
        with zipfile.ZipFile(work/'results.zip') as archive:
            entries=archive.infolist();names=set()
            if len(entries)>128 or sum(e.file_size for e in entries)>512*1024**2:raise ValueError('Expanded demo exceeds bound')
            for e in entries:
                target=(work/e.filename).resolve()
                if not target.is_relative_to(work) or target.exists() or e.filename in names or target.suffix not in {'.json','.jsonl','.csv','.jpg','.mp4','.html','.md','.log'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe demo export')
                names.add(e.filename)
            for e in entries:
                target=work/e.filename;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(e) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
        record.update(status='complete',run_id=summary['run_id'],frames=summary['frames'],result_sha256=sha256_file(work/'results.zip'))
    except BaseException as exc:
        record.update(status='failed',failure_type=type(exc).__name__);event({'stage':'controller','status':'failed','failure_type':type(exc).__name__})
        if sandbox is not None:
            for rel in ('worker-state.json','summary.json','precision-parity.json','checks.json','tests.log','artifacts.log','processing.log','raw_proposals.jsonl','observations.jsonl','coverage.jsonl'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/rcnn-gpu-amp/'+rel).size<128*1024**2:sandbox.filesystem.copy_to_local('/work/rcnn-gpu-amp/'+rel,work/('partial-'+rel))
                except Exception:pass
    finally:
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'job-receipt.json',record)
    return record
if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:
        record=run();print(json.dumps({k:record.get(k) for k in ('status','worker_terminated','run_id','frames')}))
        if record['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'blocked','failure_type':type(exc).__name__}));raise SystemExit(1) from None
