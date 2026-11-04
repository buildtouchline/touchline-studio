"""One owner-approved T4 full-video job; $3 exception above retained $50 holds."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import argparse,hashlib,json,logging,sqlite3,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from run_modal_remote_checks import make_bundle
from touchline.repair_checks import read_worker_result
from touchline.data.manifest import sha256_file
from probe_modal_gpu import safe_error

def run(*,replacement=False):
    work=ROOT/('runs/modal-rcnn-full-gpu-20261103-replacement' if replacement else 'runs/modal-rcnn-full-gpu-20261103');work.mkdir(exist_ok=False)
    parent=ROOT/'runs/modal-pxl-analysis-20261103-retry';cpu=json.loads((parent/'job-receipt.json').read_text())
    previous=json.loads((ROOT/'runs/modal-rcnn-full-gpu-20261103/job-receipt.json').read_text()) if replacement else None
    if replacement:
        if previous['worker_terminated'] is not True or previous['status']!='interrupted_for_owner_approved_replacement':raise ValueError('Confirmed previous T4 termination required')
    elif cpu['worker_terminated']!='pending':raise ValueError('Prepared live CPU worker required; no reupload fallback')
    make_bundle(work/'code.zip');source_hash=sha256_file(work/'code.zip')
    with sqlite3.connect(ROOT/'runs/rcnn-gpu-owner-exception-20261103.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY, reserved_micro_usd INTEGER, retained_prior_micro_usd INTEGER, ceiling_micro_usd INTEGER, source_sha256 TEXT)')
        db.execute('INSERT INTO claims VALUES (?,?,?,?,?)',('owner-T4-full-analysis-20261103-replacement' if replacement else 'owner-T4-full-analysis-20261103',0 if replacement else 3000000,53000000 if replacement else 50000000,53000000,source_hash))
    lifetime=5400 if replacement else 1800;remote_folder='rcnn-gpu-amp' if replacement else 'rcnn-gpu-full';archive_cap=2*1024**3 if replacement else 4*1024**3
    record={'status':'prepared','reserved_usd':'3.00','additional_reservation_usd':'0.00' if replacement else '3.00','shared_with_previous_trial':replacement,'prior_retained_usd':'50.00','exception_retained_ceiling_usd':'53.00','billing':'unreconciled','worker_terminated':'not_created','gpu':'T4','timeout_s':lifetime,'cpu':4,'memory_mib':8192,'source_sha256':source_hash,'training':'not run'};save(work/'job-receipt.json',record);sandbox=None
    def event(item):
        with (work/'events.jsonl').open('a') as log:log.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            if replacement:
                image=modal.Image.from_id(previous['image_id'],client=client)
            else:
                prepared=modal.Sandbox.from_id(cpu['worker_id'],client=client)
                prepared.filesystem.write_text(json.dumps({'status':'running','started_epoch':time.time(),'maximum_seconds':600}),'/work/auxiliary-review-active.json')
                event({'stage':'snapshot','status':'started'})
                image=prepared.snapshot_filesystem(timeout=240,ttl=86400)
            record['image_id']=image.object_id;record['image_ttl_s']=86400;save(work/'job-receipt.json',record)
        event({'stage':'snapshot','status':'saved','image_id':record['image_id']})
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            app=modal.App.lookup('touchline-rcnn-video-analysis',client=client,environment_name='main',create_if_missing=True)
            sandbox=modal.Sandbox.create(app=app,client=client,image=image,gpu='T4',cpu=(4,4),memory=(8192,8192),runtime='gvisor',timeout=lifetime,block_network=True,secrets=[],volumes={},workdir='/work',env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLCONFIGDIR':'/work/cache/matplotlib'})
        record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'job-receipt.json',record);event({'stage':'sandbox','status':'created','worker_id':sandbox.object_id})
        job={'code_sha256':source_hash,'weights_sha256':'dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf','video_sha256':'d6b4c64a1313ae46b1027214323abd995275bc6e8ed06ff86c22491785240d0a','source_sha256':cpu['hashes']['recording.mp4'],'cpu_parent_worker':cpu['worker_id'],'mode':'full','detector_fps':6,'session_id_cap':24}
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if replacement:
                reference=ROOT/'runs/modal-rcnn-full-gpu-20261103/fp32-partial.zip';job.update(fp32_reference_zip_sha256=sha256_file(reference),decoder_threads=4,precision='cuda-autocast-fp16',previous_t4_terminated=True)
                sandbox.filesystem.copy_from_local(reference,'/work/rcnn-fp32-reference.zip')
            sandbox.filesystem.copy_from_local(work/'code.zip','/work/rcnn-gpu-amp-code.zip' if replacement else '/work/rcnn-gpu-code.zip');sandbox.filesystem.copy_from_local(ROOT/'tools/modal_rcnn_gpu_worker.py','/work/rcnn-gpu-amp-worker.py' if replacement else '/work/rcnn-gpu-worker.py');sandbox.filesystem.write_text(json.dumps(job),'/work/rcnn-gpu-amp-job.json' if replacement else '/work/rcnn-gpu-job.json')
            p=sandbox.exec('python','-I','/work/rcnn-gpu-amp-worker.py' if replacement else '/work/rcnn-gpu-worker.py',*(['--amp-recovery'] if replacement else []),timeout=lifetime-180,text=False)
        complete=False;total=0;buffer=bytearray()
        for chunk in p.stdout:
            total+=len(chunk)
            if total>1024**2:raise ValueError('GPU event log exceeds bound')
            buffer.extend(chunk)
            while b'\n' in buffer:
                line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
                if not line.strip():continue
                item=json.loads(line)
                if not isinstance(item,dict) or set(item)-{'stage','status','frames','selected_frames','tests','ids','error_category','run_id','gpu_memory_mib'}:raise ValueError('Unexpected GPU event')
                event(item)
                if item.get('stage')=='complete' and item.get('status')=='complete':complete=True
        p.wait()
        if p.returncode or not complete:raise RuntimeError('GPU full analysis did not complete')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/'+remote_folder+'/results.zip').size>archive_cap:raise ValueError('GPU export exceeds bound')
            sandbox.filesystem.copy_to_local('/work/'+remote_folder+'/results.zip',work/'results.zip')
        with zipfile.ZipFile(work/'results.zip') as archive:
            entries=archive.infolist();names=set()
            if len(entries)>128 or sum(e.file_size for e in entries)>archive_cap:raise ValueError('Expanded GPU export exceeds bound')
            for e in entries:
                target=(work/e.filename).resolve()
                if not target.is_relative_to(work) or target.exists() or e.filename in names or target.suffix not in {'.json','.jsonl','.csv','.mp4','.jpg','.html','.md','.log'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe GPU result entry')
                names.add(e.filename)
            for e in entries:
                target=work/e.filename;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(e) as src,target.open('xb') as dst:
                    import shutil;shutil.copyfileobj(src,dst,1024**2)
        result=json.loads((work/'summary.json').read_text())
        if result['checkpoint_sha256']!=job['weights_sha256'] or result['source_sha256']!=job['video_sha256'] or result['frames']!=17830:raise ValueError('GPU result lineage mismatch')
        record.update(status='complete',run_id=result['run_id'],frames=result['frames'],result_sha256=sha256_file(work/'results.zip'))
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc));event({'stage':'controller','status':'failed',**{k:v for k,v in safe_error(exc).items() if k=='error_category'}})
        if sandbox is not None:
            for rel in ('summary.json','checks.json','precision-parity.json','raw_proposals.jsonl','observations.jsonl','coverage.jsonl','processing.log','tests.log','artifacts.log'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/'+remote_folder+'/'+rel).size<256*1024**2:sandbox.filesystem.copy_to_local('/work/'+remote_folder+'/'+rel,work/('failure-'+rel))
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
    logging.disable(logging.CRITICAL);parser=argparse.ArgumentParser();parser.add_argument('--owner-approved-t4-exception',action='store_true');parser.add_argument('--owner-approved-replacement',action='store_true');args=parser.parse_args()
    if args.owner_approved_t4_exception==args.owner_approved_replacement:raise SystemExit('Choose exactly one single-use authorization')
    try:
        r=run(replacement=args.owner_approved_replacement);print(json.dumps({k:r.get(k) for k in ('status','worker_terminated','run_id','frames')}))
        if r['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
