"""One source/ticket verification Sandbox within the approved panoramic allowance."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,sqlite3,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from run_modal_remote_checks import make_bundle
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error

RUNNER=r'''from pathlib import Path
import json,os,re,subprocess,sys,zipfile
w=Path('/work/publication-checks');w.mkdir()
for name,folder in [('code.zip','code'),('tickets.zip','tickets')]:
 target=w/folder;target.mkdir()
 with zipfile.ZipFile('/work/'+name) as archive:
  if sum(e.file_size for e in archive.infolist())>24*1024**2:raise ValueError('Source archive exceeds bound')
  for entry in archive.infolist():
   p=(target/entry.filename).resolve()
   if not p.is_relative_to(target) or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source member')
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(entry))
os.chdir(w/'code');results={}
for label,args in [('tests',['tools/run_offline_checks.py']),('artifacts',['tools/run_offline_checks.py','artifacts']),('tickets',['tools/check_ticket_pack.py',str(w/'tickets')])]:
 with (w/(label+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B',*args],stdout=log,stderr=subprocess.STDOUT,timeout=300)
 text=(w/(label+'.log')).read_text(errors='replace');counts=re.findall(r'Ran ([0-9]+) tests?',text);results[label]={'exit_code':r.returncode,'tests':int(counts[-1]) if counts else 0}
 if label!='tests':
  try:results[label]['detail']=json.loads(text)
  except ValueError:results[label]['detail']='see log'
results['status']='passed' if all(v['exit_code']==0 for v in results.values()) else 'failed'
(w/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
'''

def run():
    work=ROOT/'runs/modal-publication-checks-20261104';work.mkdir(exist_ok=False);make_bundle(work/'code.zip');tickets=ROOT.parent/'touchline-tickets-progress'
    with zipfile.ZipFile(work/'tickets.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in tickets.rglob('*'):
            if p.is_file() and '.git' not in p.parts and p.suffix in {'.md','.json','.csv'}:archive.write(p,p.relative_to(tickets).as_posix())
    with sqlite3.connect(ROOT/'runs/panorama-20261104-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (phase TEXT PRIMARY KEY, reserved_micro_usd INTEGER, total_micro_usd INTEGER)');db.execute('INSERT INTO claims VALUES (?,?,?)',('publication-checks',150000,2000000))
    record={'status':'prepared','reserved_usd':'.15','shared_panorama_cap_usd':'2.00','project_retained_ceiling_usd':'55.00','worker_terminated':'not_created','gpu':None,'source_sha256':hashlib.sha256((work/'code.zip').read_bytes()).hexdigest(),'tickets_sha256':hashlib.sha256((work/'tickets.zip').read_bytes()).hexdigest()};save(work/'receipt.json',record);sandbox=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);app=modal.App.lookup('touchline-publication-checks',client=client,environment_name='main',create_if_missing=True)
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id('im-A4IdpUIDiLlRbBha7sqzYy',client=client),cpu=(2,2),memory=(4096,4096),timeout=600,runtime='gvisor',block_network=True,secrets=[],volumes={},workdir='/work')
            record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'receipt.json',record)
            for name in ('code.zip','tickets.zip'):sandbox.filesystem.copy_from_local(work/name,'/work/'+name)
            sandbox.filesystem.write_text(RUNNER,'/work/publication-checks.py');p=sandbox.exec('python','-I','/work/publication-checks.py',timeout=500,text=False)
            result=json.loads(read_worker_result(p.stdout));p.wait()
            for name in ('summary.json','tests.log','artifacts.log','tickets.log'):sandbox.filesystem.copy_to_local('/work/publication-checks/'+name,work/name)
            if p.returncode:raise RuntimeError('Verification worker failed')
        record.update(status=result['status'],result=result)
    except Exception as exc:record.update(status='failed',**safe_error(exc))
    finally:
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'receipt.json',record)
    return record

if __name__=='__main__':
    logging.disable(logging.CRITICAL);parser=argparse.ArgumentParser();parser.add_argument('--execute-reviewed',action='store_true');args=parser.parse_args()
    if not args.execute_reviewed:raise SystemExit('Explicit approved publication-check scope required')
    try:
        r=run();print(json.dumps({k:r.get(k) for k in ('status','worker_terminated','result')}))
        if r['status']!='passed':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
