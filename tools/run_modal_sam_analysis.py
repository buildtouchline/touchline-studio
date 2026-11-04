"""Bounded real SAM 3.1 second pass through a trusted broker; CPU workers have no keys."""
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
from pathlib import Path
import argparse,base64,hashlib,json,logging,re,sqlite3,sys,zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.modal_trial import parse_sam_boxes
from touchline.data.manifest import sha256_file
from touchline.repair_checks import read_worker_result
from run_modal_remote_checks import make_bundle
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error


def run():
    parent=ROOT/'runs/modal-candidate-analysis-20261102'
    manifest=json.loads((parent/'run/manifest.json').read_text())
    work=ROOT/'runs/modal-sam-analysis-20261103';work.mkdir(exist_ok=False)
    make_bundle(work/'code.zip')
    inputs={'code.zip':work/'code.zip','worker.py':ROOT/'tools/modal_sam_analysis_worker.py',
        'video24.mp4':ROOT/'runs/modal-cpu-full-video-20261102/source/video24.mp4','raw_proposals.jsonl':parent/'run/raw_proposals.jsonl'}
    hashes={n:sha256_file(p) for n,p in inputs.items()}
    if hashes['video24.mp4']!=manifest['source']['id']:raise ValueError('Analysis/source hash mismatch')
    with sqlite3.connect(ROOT/'runs/modal-sam-analysis-20261103-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (provider TEXT PRIMARY KEY, cap_micro_usd INTEGER)')
        db.executemany('INSERT INTO claims VALUES (?,?)',[('modal',1000000),('meta',2000000)])
    record={'status':'prepared','created_utc':datetime.now(timezone.utc).isoformat(),'model':'sam-3.1',
        'parent_run_id':manifest['run_id'],'checkpoint_sha256':manifest['detector']['checkpoint'],
        'hashes':hashes,'modal_cap_usd':'1.00','meta_cap_usd':'2.00','billing':'unreconciled',
        'worker_terminated':'not_created','requests':[],'training_eligible':False}
    save(work/'sam-analysis.json',record);sandbox=None;creation=False;modal=client=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            key=settings.credentials.get('META_API_KEY')
            if not key:raise ValueError('Meta credential unavailable')
            app=modal.App.lookup('touchline-sam-analysis',client=client,environment_name='main',create_if_missing=True);creation=True
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id('im-A4IdpUIDiLlRbBha7sqzYy',client=client),
                runtime='gvisor',gpu=None,cpu=(2,2),memory=(4096,4096),timeout=1200,block_network=True,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'})
        record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'sam-analysis.json',record)
        print(json.dumps({'stage':'sandbox','status':'created','worker_id':sandbox.object_id}),flush=True)
        for n,p in inputs.items():
            with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(p,'/work/'+n)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.write_text(json.dumps({'hashes':hashes}),'/work/job.json')
            process=sandbox.exec('python','-I','/work/worker.py','prepare',timeout=700,text=False)
        preparation=json.loads(read_worker_result(process.stdout));process.wait()
        if process.returncode or preparation.get('stage')!='prepared':raise RuntimeError('Frame preparation/checks failed')
        print(json.dumps(preparation),flush=True)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.copy_to_local('/work/prepared.json',work/'prepared.json')
            for name in ('tests.log','artifacts.log'):sandbox.filesystem.copy_to_local('/work/'+name,work/name)
        prepared=json.loads((work/'prepared.json').read_text());responses={};frames=work/'frames';frames.mkdir()
        import httpx
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=60) as http:
            for f in prepared['frames']:
                index=f['index'];name=f'{index:06d}.jpg'
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_to_local('/work/frames/'+name,frames/name)
                payload={'model':'sam-3.1','input':[{'type':'message','role':'user','content':[
                    {'type':'input_text','text':'person'},
                    {'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode((frames/name).read_bytes()).decode()}]}],
                    'stream':False,'metadata':{'mask_encoding':'one_bit','include_confidence':'true'}}
                entry={'frame_index':index,'state':'dispatched','image_sha256':sha256_file(frames/name)}
                record['requests'].append(entry);save(work/'sam-analysis.json',record)
                with http.stream('POST','https://api.meta.ai/v1/responses',headers={'Authorization':'Bearer '+key},json=payload) as response:
                    entry['http_status']=response.status_code
                    if response.status_code!=200:raise RuntimeError('SAM request rejected; no retry')
                    body=bytearray()
                    for chunk in response.iter_bytes():
                        if len(body)+len(chunk)>8*1024**2:raise ValueError('SAM response exceeds limit')
                        body.extend(chunk)
                    data=json.loads(body)
                if data.get('status')!='completed':raise ValueError('Incomplete SAM output')
                text=''.join(part['text'] for msg in data.get('output',[]) for part in msg.get('content',[]) if part.get('type')=='output_text')
                objects=parse_sam_boxes(text,f['width'],f['height'])
                usage=data.get('usage',{});usage={k:v for k,v in usage.items() if k in {'input_tokens','output_tokens','total_tokens'} and type(v) is int and 0<=v<=100000000} if isinstance(usage,dict) else {}
                response_id=data.get('id')
                if isinstance(response_id,str) and re.fullmatch('[a-zA-Z0-9_-]{1,200}',response_id):entry['response_id']=response_id
                entry.update(state='completed',objects=len(objects),usage=usage)
                responses[str(index)]={'objects':objects,'usage':usage}
                save(work/'sam.json',responses);save(work/'sam-analysis.json',record)
                print(json.dumps({'stage':'meta.sam','frame_index':index,'http_status':200,'objects':len(objects),'usage':usage}),flush=True)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.copy_from_local(work/'sam.json','/work/sam.json')
            process=sandbox.exec('python','-I','/work/worker.py','render',timeout=120,text=False)
        rendered=json.loads(read_worker_result(process.stdout));process.wait()
        if process.returncode or rendered.get('stage')!='rendered':raise RuntimeError('SAM review render failed')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            if sandbox.filesystem.stat('/work/sam-review.zip').size>128*1024**2:raise ValueError('Review archive too large')
            sandbox.filesystem.copy_to_local('/work/sam-review.zip',work/'sam-review.zip')
        with zipfile.ZipFile(work/'sam-review.zip') as z:
            for e in z.infolist():
                p=(work/e.filename).resolve()
                if not p.is_relative_to(work/'review') or p.exists() or p.suffix not in {'.jpg','.json','.html'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe review artifact')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
        record.update(status='complete',review=rendered)
    except BaseException as exc:record.update(status='failed',**safe_error(exc))
    finally:
        if sandbox is None and creation:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox=modal.Sandbox.from_name('touchline-sam-analysis',work.name,client=client,environment_name='main')
            except modal.exception.NotFoundError:record['worker_terminated']='confirmed_no_worker'
            except Exception:record['worker_terminated']='unknown'
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'sam-analysis.json',record)
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--approved-meta-analysis',action='store_true');args=p.parse_args()
    if not args.approved_meta_analysis:raise SystemExit('Explicit bounded Meta analysis required')
    try:
        r=run();print(json.dumps({'status':r['status'],'worker_terminated':r['worker_terminated'],'completed_requests':sum(v['state']=='completed' for v in r['requests'])}))
        if r['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
