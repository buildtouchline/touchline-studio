"""Use the existing PXL CPU worker for a checked24-ID preview/full replay."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,json,logging,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client,save
from touchline.data.manifest import sha256_file
from probe_modal_gpu import safe_error
def run(scope):
    parent=ROOT/'runs/modal-pxl-analysis-20261103-retry';receipt=json.loads((parent/'job-receipt.json').read_text())
    if receipt['worker_terminated']!='pending':raise ValueError('Existing worker required; no new worker permitted')
    work=ROOT/('runs/modal-motion-refinement-20261103-'+scope);work.mkdir(exist_ok=False)
    code=work/'code.zip' if scope=='preview' else ROOT/'runs/modal-motion-refinement-20261103-preview/code.zip'
    if scope=='preview':make_bundle(code)
    else:
        previous=json.loads((ROOT/'runs/modal-motion-refinement-20261103-preview/summary.json').read_text())
        if previous['status']!='complete' or previous['session_id_cap']!=24 or previous['code_sha256']!=sha256_file(code):raise ValueError('Checked same-source preview required')
    job={'mode':scope,'code_sha256':sha256_file(code),'scope':'staged_review','session_id_cap':24,'promotion':'not performed'};save(work/'job.json',job)
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(receipt['worker_id'],client=client)
        if scope=='preview':
            sandbox.filesystem.copy_from_local(code,'/work/motion-code-v3.zip');sandbox.filesystem.copy_from_local(ROOT/'tools/modal_motion_refinement_worker.py','/work/motion-refinement-worker-v3.py')
        sandbox.filesystem.write_text(json.dumps(job),'/work/motion-refinement-'+scope+'.json')
        p=sandbox.exec('python','-I','/work/motion-refinement-worker-v3.py','/work/motion-refinement-'+scope+'.json',timeout=900 if scope=='preview' else 6000,text=False,bufsize=-1)
    total=0;buffer=bytearray();complete=False
    for chunk in p.stdout:
        total+=len(chunk)
        if total>1024**2:raise ValueError('Bounded motion event log exceeded')
        buffer.extend(chunk)
        while b'\n' in buffer:
            line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
            if not line.strip():continue
            item=json.loads(line)
            if not isinstance(item,dict) or set(item)-{'stage','status','frames','ids','tests','error_category','run_id'}:raise ValueError('Unexpected motion event')
            with (work/'events.jsonl').open('a') as s:s.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
            if item.get('stage')=='complete' and item.get('status')=='complete':complete=True
    p.wait()
    if p.returncode or not complete:raise RuntimeError('Motion refinement failed; retain existing worker receipts')
    if scope=='preview':
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/motion-preview20-v3/review.zip').size>256*1024**2:raise ValueError('Preview archive exceeds bound')
            sandbox.filesystem.copy_to_local('/work/motion-preview20-v3/review.zip',work/'review.zip')
        with zipfile.ZipFile(work/'review.zip') as z:
            if sum(e.file_size for e in z.infolist())>256*1024**2:raise ValueError('Expanded review exceeds bound')
            for e in z.infolist():
                target=(work/e.filename).resolve()
                if target.parent!=work or target.exists() or target.suffix not in {'.json','.jsonl','.csv','.jpg','.html','.md','.log','.mp4'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe review archive')
                target.write_bytes(z.read(e))
        result=json.loads((work/'summary.json').read_text())
    else:result={'status':'motion_export_ready','full_export_controller':'adopt_modal_recording.py','worker_id':receipt['worker_id']}
    save(work/'controller-receipt.json',result);return result
if __name__=='__main__':
    logging.disable(logging.CRITICAL);p=argparse.ArgumentParser();p.add_argument('--scope',choices=('preview','full'),required=True);a=p.parse_args()
    try:print(json.dumps(run(a.scope)))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
