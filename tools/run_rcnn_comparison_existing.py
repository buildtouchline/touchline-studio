"""Attach one R-CNN comparison to the existing bounded recording Sandbox."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,modal_client,save
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error

def run(*,full_review=False):
    parent=ROOT/'runs/modal-pxl-analysis-20261103-retry';receipt=json.loads((parent/'job-receipt.json').read_text())
    if receipt['worker_terminated']!='pending':raise ValueError('Existing live worker required; no new worker authorized')
    work=ROOT/('runs/modal-rcnn-full-review-20261103' if full_review else 'runs/modal-rcnn-comparison-20261103-v2');work.mkdir(exist_ok=False)
    metadata=json.loads((ROOT/'artifacts/rcnn-comparison/weights.json').read_text())
    weights=ROOT/'artifacts/rcnn-comparison/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
    with weights.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=metadata['sha256']:raise ValueError('Pinned public weights changed')
    job={'weights_sha256':metadata['sha256'],'baseline_sha256':receipt['hashes']['candidate.pt'],'worker_source_sha256':hashlib.sha256((ROOT/'tools/modal_rcnn_comparison_worker.py').read_bytes()).hexdigest(),'new_worker_created':False,'gpu':None,'promotion':'not performed','scope':'full_recording_review' if full_review else 'first_twenty_seconds'}
    save(work/'job.json',job)
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
        sandbox=modal.Sandbox.from_id(receipt['worker_id'],client=client)
        sandbox.filesystem.copy_from_local(weights,'/work/rcnn-comparison-weights.pth')
        sandbox.filesystem.copy_from_local(ROOT/'tools/modal_rcnn_comparison_worker.py','/work/rcnn-comparison-worker.py')
        sandbox.filesystem.write_text(json.dumps(job),'/work/rcnn-comparison-job.json')
        check=sandbox.exec('python','-I','-c',"from pathlib import Path; p=Path('/work/rcnn-comparison-worker.py'); compile(p.read_text(),str(p),'exec'); print('syntax_ok')",timeout=30,text=False)
        check_output=read_worker_result(check.stdout);check.wait()
        if check.returncode or check_output.strip()!='syntax_ok':raise RuntimeError('Remote worker syntax check failed')
        p=sandbox.exec('python','-I','/work/rcnn-comparison-worker.py',timeout=900,text=False)
    complete=False;total=0;buffer=bytearray()
    for chunk in p.stdout:
        total+=len(chunk)
        if total>16384:raise ValueError('Comparison event log exceeds bound')
        buffer.extend(chunk)
        while b'\n' in buffer:
            line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
            if not line.strip():continue
            item=json.loads(line)
            if not isinstance(item,dict) or set(item)-{'stage','status','frame','candidate_person_boxes','frames','error_category'}:raise ValueError('Unexpected comparison event')
            with (work/'events.jsonl').open('a') as log:log.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
            if item.get('stage')=='complete' and item.get('status')=='complete':complete=True
    p.wait()
    if p.returncode or not complete:raise RuntimeError('R-CNN comparison failed; preserve baseline worker')
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        remote='/work/'+('rcnn-full-review' if full_review else 'rcnn-comparison')+'/review.zip'
        if sandbox.filesystem.stat(remote).size>64*1024**2:raise ValueError('Comparison export exceeds bound')
        sandbox.filesystem.copy_to_local(remote,work/'review.zip')
    with zipfile.ZipFile(work/'review.zip') as archive:
        entries=archive.infolist()
        if len(entries)>16 or sum(e.file_size for e in entries)>64*1024**2:raise ValueError('Expanded export exceeds bound')
        names=set()
        for entry in entries:
            target=(work/entry.filename).resolve()
            if target.parent!=work or target.exists() or entry.filename in names or target.suffix not in {'.json','.jsonl','.jpg','.html'} or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe result member')
            names.add(entry.filename)
        for entry in entries:(work/entry.filename).write_bytes(archive.read(entry))
    result=json.loads((work/'receipt.json').read_text())
    if result['checkpoint_sha256']!=job['weights_sha256'] or result['baseline_checkpoint_sha256']!=job['baseline_sha256'] or result['device']!='cpu':raise ValueError('Comparison lineage mismatch')
    save(work/'controller-receipt.json',{**result,'worker_id':receipt['worker_id'],'worker_source_sha256':job['worker_source_sha256'],'new_worker_created':False,'shares_existing_recording_cap_usd':'4.00'})
    return result

if __name__=='__main__':
    logging.disable(logging.CRITICAL);parser=argparse.ArgumentParser();parser.add_argument('--existing-worker-only',action='store_true');parser.add_argument('--full-review',action='store_true');args=parser.parse_args()
    if not args.existing_worker_only:raise SystemExit('Existing worker scope required')
    try:print(json.dumps(run(full_review=args.full_review)))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
