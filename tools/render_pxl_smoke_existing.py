"""User-requested PXL preview in the existing bounded CPU worker only."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,modal_client,save
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error
WORKER=r'''from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import hashlib,json,sys,zipfile
w=Path('/work/pxl-preview20');w.mkdir();sys.path.insert(0,'/work/recording-project/src')
from touchline.video.normalize import normalize_video
from touchline.cli import main as cli
from touchline.video.preview import render_run_preview
from touchline.diagnostics import diagnostic_session
import torch;torch.set_num_threads(4)
with (w/'preview.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(w/'diagnostics'):
 video=normalize_video(Path('/work/recording.mp4'),w/'source',max_seconds=20)
 if cli(['analyze',str(video),'--checkpoint','/work/candidate.pt','--device','cpu','--mode','fixed','--fps','6','--tracker','bytetrack','--min-size','1080','--max-size','4096','--record-proposals','--max-selected','120','--output',str(w/'run')]):raise RuntimeError('Preview inference failed')
 preview=render_run_preview(w/'run',w/'preview',source=video,max_seconds=20,hold_seconds=.17)
manifest=json.loads((w/'run/manifest.json').read_text())
if manifest['detector']['checkpoint']!='6340f81877408e693dcd4734d84214c29095011a184e4c8187a98139ef3b7d80' or manifest['detector_runtime']['model_parameter_device']!='cpu':raise ValueError('Preview checkpoint/device mismatch')
receipt={'status':'complete','scope':'first20seconds only','analysis_status':manifest['status'],'run_id':manifest['run_id'],'checkpoint':manifest['detector']['checkpoint'],'observations':manifest['observation_count'],'selected_frames':manifest['selected_frames'],'identity_accuracy':None,'promotion':'not performed','source':'PXL_20261101_045045814~2.mp4','gpu':None}
(w/'preview-receipt.json').write_text(json.dumps(receipt,indent=2))
(w/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>PXL: first20seconds, new candidate</h1><video controls width="1200" src="preview/overlay24.mp4"></video><p>Temporary tracks, not confirmed names/jersey numbers. SAMPLED = detector frame; HELD = visual carry-forward. Full recording is a separate running job. Detection validation AP50 does not establish identity accuracy on this footage.</p><a href="preview/players.jpg">Track gallery</a> | <a href="run/manifest.json">Manifest</a>')
with zipfile.ZipFile(w/'preview-results.zip','w',zipfile.ZIP_STORED) as z:
 for folder in ('run','preview','diagnostics','source'):
  for p in (w/folder).rglob('*'):
   if p.is_file():z.write(p,p.relative_to(w).as_posix())
 for name in ('preview-receipt.json','RESULTS.html','preview.log'):z.write(w/name,name)
print(json.dumps(receipt))
'''
def run():
    parent=ROOT/'runs/modal-pxl-analysis-20261103-retry';worker=json.loads((parent/'job-receipt.json').read_text())
    if worker['worker_terminated']!='pending':raise ValueError('Existing worker required; no new worker permitted')
    work=ROOT/'runs/modal-pxl-preview-20261103';work.mkdir(exist_ok=False)
    logging.disable(logging.CRITICAL)
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
        sandbox=modal.Sandbox.from_id(worker['worker_id'],client=client)
        sandbox.filesystem.write_text(WORKER,'/work/pxl-preview-worker.py')
        p=sandbox.exec('python','-I','/work/pxl-preview-worker.py',timeout=900,text=False)
        result=json.loads(read_worker_result(p.stdout));p.wait()
        if p.returncode or result['status']!='complete':raise RuntimeError('Remote PXL preview failed')
        if sandbox.filesystem.stat('/work/pxl-preview20/preview-results.zip').size>768*1024**2:raise ValueError('Preview export exceeds bound')
        sandbox.filesystem.copy_to_local('/work/pxl-preview20/preview-results.zip',work/'preview-results.zip')
    with zipfile.ZipFile(work/'preview-results.zip') as z:
        if sum(e.file_size for e in z.infolist())>768*1024**2:raise ValueError('Expanded preview exceeds bound')
        for e in z.infolist():
            path=(work/e.filename).resolve()
            if not path.is_relative_to(work) or path.exists() or Path(e.filename).parts[0] not in {'run','preview','diagnostics','source','preview-receipt.json','RESULTS.html','preview.log'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe preview archive')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(e))
    result.update(worker_id=worker['worker_id'],new_modal_worker_created=False,shares_recording_cap_usd='4.00',worker_source_sha256=hashlib.sha256(WORKER.encode()).hexdigest());save(work/'controller-receipt.json',result)
    return result
if __name__=='__main__':
    logging.disable(logging.CRITICAL);p=argparse.ArgumentParser();p.add_argument('--existing-worker-preview',action='store_true');a=p.parse_args()
    if not a.existing_worker_preview:raise SystemExit('Existing worker preview scope required')
    try:print(json.dumps(run()))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
