"""One same-T4 precision recovery within the original worker deadline and $3 hold."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import argparse,hashlib,json,logging,shutil,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from run_modal_remote_checks import make_bundle
from touchline.data.manifest import sha256_file
from probe_modal_gpu import safe_error
from touchline.repair_checks import read_worker_result

def run(*,fast=False):
    phase='fast' if fast else 'amp';target='rcnn-gpu-'+phase
    work=ROOT/'runs/modal-rcnn-full-gpu-20261103';record=json.loads((work/'job-receipt.json').read_text());adopt=work/(phase+'-adoption.json')
    if record['status']!='running' or record['worker_terminated']!='pending' or adopt.exists():raise ValueError('One existing-worker handoff only')
    # The claim predates Sandbox creation; its timestamp is a conservative deadline.
    deadline=json.loads((work/'amp-adoption.json').read_text())['deadline_epoch'] if fast else (ROOT/'runs/rcnn-gpu-owner-exception-20261103.sqlite3').stat().st_mtime+1800-60
    if deadline-time.time()<360:raise ValueError('Insufficient original lifetime for precision recovery')
    make_bundle(work/(phase+'-code.zip'));digest=sha256_file(work/(phase+'-code.zip'))
    save(adopt,{'status':'prepared','original_worker_id':record['worker_id'],'original_code_sha256':record['source_sha256'],'new_code_sha256':digest,'deadline_epoch':deadline,'new_worker_created':False,'extra_reservation_usd':'0.00','reason':'FP32 timing exceeded remaining full-video budget; retain numeric parity gate and stricter reviewed-anchor births'})
    sandbox=None
    def event(item):
        with (work/(phase+'-events.jsonl')).open('a') as log:log.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(record['worker_id'],client=client)
            stop="""from pathlib import Path
import json,os,signal,time,zipfile
w=Path('/work');pids=[]
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:args=(q/'cmdline').read_bytes().split(b'\\0')
 except OSError:continue
 if b'/work/rcnn-gpu-worker.py' in args:pids.append(int(q.name))
assert len(pids)==1
os.kill(pids[0],signal.SIGTERM);time.sleep(2)
(w/'rcnn-gpu-full/partial-receipt.json').write_text(json.dumps({'status':'interrupted_for_same_worker_precision_recovery','identity_accuracy':None,'complete':False}))
with zipfile.ZipFile(w/'rcnn-fp32-partial.zip','w',zipfile.ZIP_STORED) as z:
 for p in (w/'rcnn-gpu-full').iterdir():
  if p.is_file() and p.suffix in {'.json','.jsonl','.log'}:z.write(p,p.name)
print(json.dumps({'status':'stopped_fp32','processes':len(pids)}))
"""
            if fast:stop=stop.replace('/work/rcnn-gpu-worker.py','/work/rcnn-gpu-amp-worker.py').replace('rcnn-gpu-full','rcnn-gpu-amp').replace('rcnn-fp32-partial.zip','rcnn-amp-preserved.zip')
            p=sandbox.exec('python','-I','-c',stop,timeout=60,text=False);stopped=json.loads(read_worker_result(p.stdout));p.wait()
            if p.returncode or stopped.get('status')!='stopped_fp32':raise RuntimeError('Exact original inference handoff failed')
            preserved='/work/rcnn-amp-preserved.zip' if fast else '/work/rcnn-fp32-partial.zip'
            if sandbox.filesystem.stat(preserved).size>512*1024**2:raise ValueError('Partial evidence exceeds bound')
            sandbox.filesystem.copy_to_local(preserved,work/('amp-preserved.zip' if fast else 'fp32-partial.zip'))
            old=json.loads(sandbox.filesystem.read_text('/work/rcnn-gpu-job.json'))
            old.update(code_sha256=digest,precision='cuda-autocast-fp16',same_worker_recovery=True)
            sandbox.filesystem.copy_from_local(work/(phase+'-code.zip'),'/work/'+target+'-code.zip');sandbox.filesystem.copy_from_local(ROOT/'tools/modal_rcnn_gpu_worker.py','/work/'+target+'-worker.py');sandbox.filesystem.write_text(json.dumps(old),'/work/'+target+'-job.json')
            remaining=int(deadline-time.time())
            if remaining<300:raise ValueError('Original deadline too close')
            process=sandbox.exec('python','-I','/work/'+target+'-worker.py','--fast-recovery' if fast else '--amp-recovery',timeout=remaining,text=False)
        event({'stage':'handoff','status':'same_T4_worker','worker_id':record['worker_id']});total=0;buffer=bytearray();complete=False
        for chunk in process.stdout:
            total+=len(chunk)
            if total>1024**2:raise ValueError('Recovery events exceed bound')
            buffer.extend(chunk)
            while b'\n' in buffer:
                line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
                if not line.strip():continue
                item=json.loads(line)
                if not isinstance(item,dict) or set(item)-{'stage','status','frames','selected_frames','tests','ids','error_category','run_id','gpu_memory_mib'}:raise ValueError('Unexpected recovery event')
                event(item)
                if item.get('stage')=='complete' and item.get('status')=='complete':complete=True
        process.wait()
        if process.returncode or not complete:raise RuntimeError('Bounded full-video recovery incomplete')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/'+target+'/results.zip').size>4*1024**3:raise ValueError('Full export exceeds bound')
            sandbox.filesystem.copy_to_local('/work/'+target+'/results.zip',work/(phase+'-results.zip'))
        output=work/phase;output.mkdir()
        with zipfile.ZipFile(work/(phase+'-results.zip')) as archive:
            entries=archive.infolist();names=set()
            if len(entries)>128 or sum(e.file_size for e in entries)>4*1024**3:raise ValueError('Expanded results exceed bound')
            for e in entries:
                target=(output/e.filename).resolve()
                if not target.is_relative_to(output) or target.exists() or e.filename in names or target.suffix not in {'.json','.jsonl','.csv','.jpg','.mp4','.html','.md','.log'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe recovered member')
                names.add(e.filename)
            for e in entries:
                target=output/e.filename;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(e) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
        summary=json.loads((output/'summary.json').read_text())
        if summary['frames']!=17830 or summary['checkpoint_sha256']!=old['weights_sha256'] or summary['source_sha256']!=old['video_sha256']:raise ValueError('Recovered output lineage mismatch')
        record.update(status='complete',run_id=summary['run_id'],frames=summary['frames'],result_sha256=sha256_file(work/(phase+'-results.zip')),precision='cuda-autocast-fp16',same_worker_recovery=True,result_phase=phase)
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc));event({'stage':'controller','status':'failed','error_category':safe_error(exc)['error_category']})
        if sandbox is not None:
            try:
                backup="""from pathlib import Path
import zipfile,json
w=Path('/work/rcnn-gpu-amp')
with zipfile.ZipFile('/work/rcnn-amp-partial.zip','w',zipfile.ZIP_STORED) as z:
 for p in w.iterdir():
  if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.log'}:z.write(p,p.name)
print('saved')
"""
                with redirect_stdout(Discard()),redirect_stderr(Discard()):
                    if fast:backup=backup.replace('rcnn-gpu-amp','rcnn-gpu-fast')
                    p=sandbox.exec('python','-I','-c',backup,timeout=30,text=False);read_worker_result(p.stdout);p.wait()
                    if sandbox.filesystem.stat('/work/rcnn-amp-partial.zip').size<1024**3:sandbox.filesystem.copy_to_local('/work/rcnn-amp-partial.zip',work/(phase+'-partial.zip'))
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
    parser=argparse.ArgumentParser();parser.add_argument('--fast-recovery',action='store_true');args=parser.parse_args()
    try:print(json.dumps({k:v for k,v in run(fast=args.fast_recovery).items() if k in {'status','worker_terminated','run_id','frames'}}))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
