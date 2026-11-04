"""Run checked twenty-second R-CNN footage in the same reserved worker."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import argparse,hashlib,json,logging,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client,save
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error

def run(*,recover_cached=False):
    live=ROOT/'runs/modal-pxl-analysis-20261103-retry';worker=json.loads((live/'job-receipt.json').read_text())
    if worker['worker_terminated']!='pending':raise ValueError('Existing worker required')
    work=ROOT/('runs/modal-rcnn-preview-20261103-recovery' if recover_cached else 'runs/modal-rcnn-preview-20261103');work.mkdir(exist_ok=False);make_bundle(work/'code.zip')
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(worker['worker_id'],client=client)
        sandbox.filesystem.write_text(json.dumps({'status':'running','started_epoch':time.time(),'maximum_seconds':900}),'/work/auxiliary-review-active.json')
        sandbox.filesystem.copy_from_local(work/'code.zip','/work/rcnn-preview-code.zip')
        sandbox.filesystem.copy_from_local(ROOT/'tools/modal_rcnn_preview_worker.py','/work/rcnn-preview-worker.py')
        if recover_cached:
            check=sandbox.exec('python','-I','-c',"from pathlib import Path; import json; p=Path('/work/rcnn-preview20/run/manifest.json'); m=json.loads(p.read_text()); assert m['status']!='complete'; active=[]; [active.append(q.name) for q in Path('/proc').iterdir() if q.name.isdigit() and (q/'cmdline').exists() and b'/work/rcnn-preview-worker.py' in (q/'cmdline').read_bytes().split(b'\\0')]; assert not active; print('cached_recovery_ready')",timeout=30,text=False)
            ready=read_worker_result(check.stdout);check.wait()
            if check.returncode or ready.strip()!='cached_recovery_ready':raise ValueError('Prior inference still active or already complete')
        p=sandbox.exec('python','-I','/work/rcnn-preview-worker.py',*(['--recover-cached'] if recover_cached else []),timeout=900,text=False)
    complete=False;total=0;buffer=bytearray()
    try:
        for chunk in p.stdout:
            total+=len(chunk)
            if total>16384:raise ValueError('Preview events exceed bound')
            buffer.extend(chunk)
            while b'\n' in buffer:
                line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
                if not line.strip():continue
                item=json.loads(line)
                if not isinstance(item,dict) or set(item)-{'stage','status','tests','frames','observations','error_category'}:raise ValueError('Unexpected preview event')
                with (work/'events.jsonl').open('a') as log:log.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
                if item.get('stage')=='complete' and item.get('status')=='complete':complete=True
        p.wait()
        if p.returncode or not complete:raise RuntimeError('Remote R-CNN preview failed')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            remote='/work/'+('rcnn-preview20-recovery' if recover_cached else 'rcnn-preview20')+'/review.zip'
            if sandbox.filesystem.stat(remote).size>512*1024**2:raise ValueError('Preview archive exceeds bound')
            sandbox.filesystem.copy_to_local(remote,work/'review.zip')
        with zipfile.ZipFile(work/'review.zip') as archive:
            entries=archive.infolist();names=set()
            if len(entries)>100 or sum(e.file_size for e in entries)>512*1024**2:raise ValueError('Expanded preview exceeds bound')
            for e in entries:
                target=(work/e.filename).resolve()
                if not target.is_relative_to(work) or target.exists() or e.filename in names or target.suffix not in {'.json','.jsonl','.jpg','.png','.html','.md','.log','.mp4'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe preview archive')
                names.add(e.filename)
            for e in entries:
                target=work/e.filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(e))
        summary=json.loads((work/'summary.json').read_text());save(work/'controller-receipt.json',{**summary,'worker_id':worker['worker_id'],'new_worker_created':False,'code_sha256':hashlib.sha256((work/'code.zip').read_bytes()).hexdigest()});return summary
    finally:
        with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.write_text(json.dumps({'status':'finished'}),'/work/auxiliary-review-active.json')

if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser();parser.add_argument('--recover-cached',action='store_true');args=parser.parse_args()
    try:print(json.dumps(run(recover_cached=args.recover_cached)))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
