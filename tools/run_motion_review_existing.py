"""One staged motion replay in the existing CPU Sandbox; no detector retraining."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import json,logging,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error
def run():
    work=ROOT/'runs/modal-motion-preview-20261103-v2';work.mkdir(exist_ok=False);make_bundle(work/'code.zip')
    parent=json.loads((ROOT/'runs/modal-pxl-analysis-20261103-retry/job-receipt.json').read_text())
    if parent['worker_terminated']!='pending':raise ValueError('Existing worker required')
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(parent['worker_id'],client=client)
        sandbox.filesystem.copy_from_local(work/'code.zip','/work/motion-code-v2.zip');sandbox.filesystem.copy_from_local(ROOT/'tools/modal_motion_review_worker.py','/work/motion-review-worker-v2.py')
        sandbox.filesystem.write_text(json.dumps({'scope':'review_only','model_inference':False}),'/work/motion-job-v2.json')
        p=sandbox.exec('python','-I','/work/motion-review-worker-v2.py',timeout=700,text=False);result=json.loads(read_worker_result(p.stdout));p.wait()
        if p.returncode or result['status']!='complete':raise RuntimeError('Motion replay failed')
        if sandbox.filesystem.stat('/work/motion-preview20-v2/motion-review.zip').size>256*1024**2:raise ValueError('Motion review exceeds bound')
        sandbox.filesystem.copy_to_local('/work/motion-preview20-v2/motion-review.zip',work/'motion-review.zip')
    with zipfile.ZipFile(work/'motion-review.zip') as z:
        if sum(e.file_size for e in z.infolist())>256*1024**2:raise ValueError('Expanded review exceeds bound')
        for e in z.infolist():
            p=(work/e.filename).resolve()
            if p.parent!=work or p.exists() or p.suffix not in {'.json','.jsonl','.html','.mp4'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe motion review entry')
            p.write_bytes(z.read(e))
    return result
if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:print(json.dumps(run()))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
