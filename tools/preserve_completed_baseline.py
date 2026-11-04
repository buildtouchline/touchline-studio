"""Preserve completed baseline media and evidence before its original CPU expiry."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import json,logging,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from touchline.settings import load_operator_settings
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error

def run():
    work=ROOT/'runs/modal-pxl-analysis-20261103-retry';r=json.loads((work/'job-receipt.json').read_text())
    if r['worker_terminated']!='pending':raise ValueError('Existing CPU worker required')
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(r['worker_id'],client=client)
        receipt=json.loads(sandbox.filesystem.read_text('/work/recording-receipt.json'))
        if receipt['status']!='complete' or receipt['input_hashes']!=r['hashes']:raise ValueError('Complete baseline lineage required')
        save(work/'verified-remote-recording-receipt.json',receipt)
        image=sandbox.snapshot_filesystem(timeout=90,ttl=86400)
        save(work/'completed-baseline-snapshot.json',{'status':'saved','image_id':image.object_id,'ttl_s':86400,'contents':'completed4K overlay, exact normalized source, original raw evidence and worker receipt','source_worker_id':r['worker_id'],'new_worker_created':False,'local_full_video_export':'still pending; snapshot is temporary not permanent shared storage'})
        code="""from pathlib import Path
import zipfile,json,shutil
w=Path('/work')
with zipfile.ZipFile(w/'baseline-evidence.zip','w',zipfile.ZIP_STORED) as z:
 for folder in ('run','profiles','diagnostics'):
  for p in (w/folder).rglob('*'):
   if p.is_file():
    with p.open('rb') as src,z.open(p.relative_to(w).as_posix(),'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,1024**2)
 for name in ('recording-receipt.json','normalized/normalization.json','normalized/frame-map.jsonl','preview/players.jpg','preview/track-index.json','preview/contact-sheet.jpg'):
  p=w/name
  if p.is_file():z.write(p,name)
print('saved')
"""
        p=sandbox.exec('python','-I','-c',code,timeout=60,text=False);output=read_worker_result(p.stdout);p.wait()
        if p.returncode or output.strip()!='saved':raise RuntimeError('Baseline evidence archive failed')
        if sandbox.filesystem.stat('/work/baseline-evidence.zip').size>256*1024**2:raise ValueError('Evidence export exceeds bound')
        sandbox.filesystem.copy_to_local('/work/baseline-evidence.zip',work/'baseline-evidence.zip')
    dest=work/'verified-evidence';dest.mkdir()
    with zipfile.ZipFile(work/'baseline-evidence.zip') as archive:
        entries=archive.infolist()
        if sum(e.file_size for e in entries)>256*1024**2:raise ValueError('Expanded evidence exceeds bound')
        names=set()
        for e in entries:
            p=(dest/e.filename).resolve()
            if not p.is_relative_to(dest) or p.exists() or e.filename in names or p.suffix not in {'.json','.jsonl','.jpg','.html','.md','.log'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe evidence member')
            names.add(e.filename)
        for e in entries:
            p=dest/e.filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
    return {'status':'baseline_snapshot_and_evidence_saved','image_id':image.object_id}
if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:print(json.dumps(run()))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
