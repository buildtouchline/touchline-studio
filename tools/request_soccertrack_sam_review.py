"""Eight bounded SAM proposals for omitted people; human review still required."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import argparse,base64,hashlib,json,logging,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.spending import SpendingLedger
from touchline.data.manifest import sha256_file
from touchline.modal_trial import parse_sam_boxes
from touchline.sam_analysis import compare_boxes
from run_modal_video_trial import Discard,save,modal_client
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error
def run():
    review=ROOT/'runs/modal-soccertrack-audit-20261103/review';intake=json.loads((review/'intake-review.json').read_text())
    frames=[f for f in intake['frames'] if f['match_id']=='118575']
    if len(frames)!=8:raise ValueError('Exactly eight frozen review frames required')
    work=ROOT/'runs/soccertrack-sam-review-20261103';work.mkdir(exist_ok=False)
    ledger=SpendingLedger(ROOT/'runs/soccertrack-pxl-20261103-budget.sqlite3')
    ledger.verify_pool('meta',prior_spend_usd='5.00',available_credit_usd='70.00')
    fingerprint=hashlib.sha256(json.dumps(frames,sort_keys=True).encode()).hexdigest()
    if not ledger.reserve('meta','soccertrack-sam-review-20261103',fingerprint=fingerprint,worst_case_usd='2.00')['created']:raise ValueError('Consumed SAM review claim')
    record={'status':'running','model':'sam-3.1','reserved_usd':'2.00','requests':[],'training_eligible':False,'human_review':'required','billing':'unreconciled'};save(work/'receipt.json',record);proposals=[]
    try:
        import httpx
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):settings=load_operator_settings(ROOT/'.env',environ={})
        key=settings.credentials.get('META_API_KEY')
        if not key:raise ValueError('Meta key missing')
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=60) as http:
            for f in frames:
                p=(review/f['raw_image']).resolve()
                if not p.is_relative_to(review.resolve()) or p.suffix!='.jpg':raise ValueError('Invalid review image')
                entry={'frame_index':f['source_frame_index'],'image_sha256':sha256_file(p),'state':'dispatched'};record['requests'].append(entry);save(work/'receipt.json',record)
                payload={'model':'sam-3.1','input':[{'type':'message','role':'user','content':[{'type':'input_text','text':'person'},{'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()}]}],'stream':False,'metadata':{'mask_encoding':'one_bit','include_confidence':'true'}}
                with http.stream('POST','https://api.meta.ai/v1/responses',headers={'Authorization':'Bearer '+key},json=payload) as response:
                    entry['http_status']=response.status_code
                    if response.status_code!=200:raise RuntimeError('SAM rejected request; no retry')
                    body=bytearray()
                    for chunk in response.iter_bytes():
                        if len(body)+len(chunk)>8*1024**2:raise ValueError('SAM response exceeds bound')
                        body.extend(chunk)
                    data=json.loads(body)
                if data.get('status')!='completed':raise ValueError('Incomplete SAM result')
                text=''.join(p['text'] for msg in data.get('output',[]) for p in msg.get('content',[]) if p.get('type')=='output_text')
                objects=parse_sam_boxes(text,f['width'],f['height']);usage=data.get('usage',{});usage={k:v for k,v in usage.items() if k in {'input_tokens','output_tokens','total_tokens'} and type(v) is int and 0<=v<=10**8} if isinstance(usage,dict) else {}
                response_id=data.get('id');entry.update(state='completed',objects=len(objects),usage=usage)
                if isinstance(response_id,str) and re.fullmatch('[a-zA-Z0-9_-]{1,200}',response_id):entry['response_id']=response_id
                proposals.append({**f,'sam_objects':objects,'comparison':compare_boxes(f['objects'],objects,f['width'],f['height'])});save(work/'proposals.json',proposals);save(work/'receipt.json',record)
                print(json.dumps({'stage':'meta.sam','frame_index':f['source_frame_index'],'http_status':200,'objects':len(objects),'sam_only':len(proposals[-1]['comparison']['sam_only'])}),flush=True)
        # Rendering shares the already-running CPU worker's fixed resource/time cap.
        live=ROOT/'runs/modal-pxl-analysis-20261103-retry';worker=json.loads((live/'job-receipt.json').read_text())
        if worker['worker_terminated']!='pending':raise ValueError('Existing review-render worker unavailable; do not create another')
        render="""from pathlib import Path
import json,zipfile
from PIL import Image,ImageDraw
w=Path('/work/soccertrack-sam-review');rows=json.loads((w/'proposals.json').read_text());html='<!doctype html><meta charset="utf-8"><h1>SoccerTrack person-label review</h1><p>Green=upstream player/GK. Orange=SAM-only candidate, not truth. Review and add omitted people before training.</p>'
for f in rows:
 im=Image.open(w/f['raw_image']).convert('RGB');d=ImageDraw.Draw(im)
 for o in f['objects']:d.rectangle(o['bbox_xyxy'],outline='#20ff70',width=3)
 for i in f['comparison']['sam_only']:d.rectangle(f['sam_objects'][i]['bbox_xyxy'],outline='#ffb000',width=4)
 name=str(f['source_frame_index']).zfill(6)+'.jpg';im.save(w/name,quality=92);html+='<h2>Frame'+str(f['source_frame_index'])+'</h2><img style="width:100%" src="'+name+'">'
(w/'index.html').write_text(html)
with zipfile.ZipFile(w/'review.zip','w',zipfile.ZIP_STORED) as z:
 for p in w.iterdir():
  if p.name!='review.zip' and p.is_file():z.write(p,p.name)
print(json.dumps({'status':'rendered','images':len(rows)}))
"""
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(worker['worker_id'],client=client)
            for f in frames:sandbox.filesystem.copy_from_local(review/f['raw_image'],'/work/soccertrack-sam-review/'+f['raw_image'])
            sandbox.filesystem.copy_from_local(work/'proposals.json','/work/soccertrack-sam-review/proposals.json')
            p=sandbox.exec('python','-I','-c',render,timeout=90,text=False);rendered=json.loads(read_worker_result(p.stdout));p.wait()
            if p.returncode or rendered['status']!='rendered':raise RuntimeError('Remote review render failed')
            if sandbox.filesystem.stat('/work/soccertrack-sam-review/review.zip').size>64*1024**2:raise ValueError('Review export exceeds bound')
            sandbox.filesystem.copy_to_local('/work/soccertrack-sam-review/review.zip',work/'review.zip')
        dest=work/'review';dest.mkdir()
        with zipfile.ZipFile(work/'review.zip') as z:
            for e in z.infolist():
                p=(dest/e.filename).resolve()
                if not p.is_relative_to(dest) or p.exists() or p.suffix not in {'.jpg','.json','.html'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe review export')
                p.write_bytes(z.read(e))
        record.update(status='complete',render_worker_id=worker['worker_id'],new_modal_worker_created=False)
    except BaseException as exc:record.update(status='failed',**safe_error(exc))
    finally:save(work/'receipt.json',record);ledger.reconcile('soccertrack-sam-review-20261103',finalized=False)
    return record
if __name__=='__main__':
    logging.disable(logging.CRITICAL);p=argparse.ArgumentParser();p.add_argument('--approved-existing-meta-budget',action='store_true');a=p.parse_args()
    if not a.approved_existing_meta_budget:raise SystemExit('Explicit bounded SAM label-review scope required')
    try:
        r=run();print(json.dumps({'status':r['status'],'completed_requests':sum(x['state']=='completed' for x in r['requests'])}))
        if r['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
