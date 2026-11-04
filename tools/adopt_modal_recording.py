"""Reattach monitoring/export to the same bounded CPU worker; never creates one."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_recording import extract
from run_modal_video_trial import Discard,modal_client,save
from touchline.repair_checks import read_worker_result
from touchline.spending import SpendingLedger
from probe_modal_gpu import safe_error
PROBE="""from pathlib import Path
import hashlib,json,time,zipfile
w=Path('/work');p=w/'recording-results.zip';r={}
for name,rel in {'normalized_frames':'normalized/frame-map.jsonl','coverage_rows':'run/coverage.jsonl'}.items():
 q=w/rel;r[name]=sum(1 for _ in q.open('rb')) if q.exists() else 0
q=w/'preview/overlay24.mp4';r['overlay_bytes']=q.stat().st_size if q.exists() else 0
q=w/'recording-receipt.json';receipt=json.loads(q.read_text()) if q.exists() else {};r['worker_status']=receipt.get('status','running')
r['archive_ready']=False
if p.exists() and zipfile.is_zipfile(p):
 with zipfile.ZipFile(p) as z:r['archive_ready']=all(n in z.namelist() for n in ('recording-receipt.json','normalized/video24.mp4','preview/overlay24.mp4'))
 q=w/'auxiliary-review-active.json'
 if q.exists():
  aux=json.loads(q.read_text())
  # Retain the same lifetime/cap; give an active bounded review time to export.
  if aux.get('status')=='running' and 0<=time.time()-aux.get('started_epoch',0)<min(900,aux.get('maximum_seconds',900)):
   r['archive_ready']=False
 if r['archive_ready']:
  r['archive_bytes']=p.stat().st_size
  with p.open('rb') as s:r['archive_sha256']=hashlib.file_digest(s,'sha256').hexdigest()
print(json.dumps(r))
"""
def run():
    work=ROOT/'runs/modal-pxl-analysis-20261103-retry';record=json.loads((work/'job-receipt.json').read_text());handoff=json.loads((work/'controller-handoff.json').read_text())
    if record['status']!='running' or record['worker_terminated']!='pending' or handoff.get('original_controller_stopped') is not True or handoff['worker_id']!=record['worker_id']:raise ValueError('Explicit same-worker controller handoff required')
    ledger=SpendingLedger(ROOT/'runs/soccertrack-pxl-20261103-budget.sqlite3');sandbox=None;started=time.monotonic();complete=False
    def event(item):
        with (work/'adopted-controller.events.jsonl').open('a') as s:s.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            sandbox=modal.Sandbox.from_id(record['worker_id'],client=client)
        event({'stage':'controller_handoff','status':'attached_same_worker','worker_id':record['worker_id']})
        while time.monotonic()-started<10000:
            with redirect_stdout(Discard()),redirect_stderr(Discard()):
                p=sandbox.exec('python','-I','-c',PROBE,timeout=120,text=False);state=json.loads(read_worker_result(p.stdout));p.wait()
                if p.returncode:raise RuntimeError('Progress inspection failed')
            event({'stage':'recording_progress',**state})
            if state['worker_status']=='failed':raise RuntimeError('Recording worker failed')
            if state['archive_ready']:
                if not 0<state['archive_bytes']<=24*1024**3 or shutil.disk_usage(work).free<state['archive_bytes']*2+1024**3:raise ValueError('Export size/disk headroom exceeded')
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_to_local('/work/recording-results.zip',work/'recording-results.zip')
                with (work/'recording-results.zip').open('rb') as s:
                    if hashlib.file_digest(s,'sha256').hexdigest()!=state['archive_sha256']:raise ValueError('Downloaded archive hash mismatch')
                extract(work/'recording-results.zip',work,max_bytes=24*1024**3)
                receipt=json.loads((work/'recording-receipt.json').read_text())
                if receipt['input_hashes']!=record['hashes'] or receipt['status']!='complete':raise ValueError('Result lineage mismatch')
                record.update(status='complete',run_id=receipt['run_id'],decoded_frames=receipt['decoded_frames'],selected_frames=receipt['selected_frames'],observations=receipt['observations'],result_sha256=state['archive_sha256'],export_max_bytes=24*1024**3,controller_handoff=True)
                complete=True;break
            time.sleep(30)
        if not complete:raise RuntimeError('Controller monitoring deadline exceeded')
    except BaseException as exc:
        record.update(status='failed',**safe_error(exc));event({'stage':'controller_handoff','status':'failed',**safe_error(exc)})
        if sandbox is not None:
            for name in ('recording-failure.zip','source-probe.json','tests.log','artifacts.log'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/'+name).size<128*1024**2:sandbox.filesystem.copy_to_local('/work/'+name,work/('adopted-failure-'+name))
                except Exception:pass
    finally:
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'job-receipt.json',record);ledger.reconcile('pxl-full-cpu-20261103-retry',finalized=False)
        if record['worker_terminated'] is not True:ledger.block_pool('modal')
    return record
if __name__=='__main__':
    logging.disable(logging.CRITICAL);p=argparse.ArgumentParser();p.add_argument('--adopt-existing-worker',action='store_true');a=p.parse_args()
    if not a.adopt_existing_worker:raise SystemExit('Explicit same-worker handoff required')
    try:
        r=run();print(json.dumps({k:r.get(k) for k in ('status','worker_terminated','run_id','decoded_frames','selected_frames','observations')}))
        if r['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
