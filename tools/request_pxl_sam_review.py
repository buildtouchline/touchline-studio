"""Six bounded SAM calls on exact PXL frames; proposals are not reviewed labels."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import base64,hashlib,json,logging,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from touchline.repair_checks import read_worker_result
from touchline.spending import SpendingLedger
from touchline.modal_trial import parse_sam_boxes
from touchline.data.manifest import sha256_file
from probe_modal_gpu import safe_error

def run():
    parent=ROOT/'runs/modal-pxl-analysis-20261103-retry';worker=json.loads((parent/'job-receipt.json').read_text())
    if worker['worker_terminated']!='pending':raise ValueError('Existing worker required')
    work=ROOT/'runs/pxl-sam-review-20261103';work.mkdir(exist_ok=False)
    frames=[0,96,192,288,384,472]
    extract="""from pathlib import Path
import av,json,zipfile
w=Path('/work/pxl-sam-review');w.mkdir();indices={0,96,192,288,384,472};rows=[]
with av.open('/work/normalized/video24.mp4') as reader:
 for i,f in enumerate(reader.decode(video=0)):
  if i>max(indices):break
  if i in indices:
   image=f.to_image().convert('RGB');image.save(w/f'{i:06d}.jpg',quality=92);rows.append({'frame_index':i,'width':image.width,'height':image.height})
(w/'frames.json').write_text(json.dumps(rows))
with zipfile.ZipFile(w/'frames.zip','w',zipfile.ZIP_STORED) as z:
 for p in w.iterdir():
  if p.suffix in {'.jpg','.json'}:z.write(p,p.name)
print(json.dumps({'status':'complete','frames':len(rows)}))
"""
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(worker['worker_id'],client=client)
        p=sandbox.exec('python','-I','-c',extract,timeout=120,text=False);state=json.loads(read_worker_result(p.stdout));p.wait()
        if p.returncode or state!={'status':'complete','frames':6}:raise RuntimeError('SAM frame staging failed')
        if sandbox.filesystem.stat('/work/pxl-sam-review/frames.zip').size>32*1024**2:raise ValueError('Frame export exceeds bound')
        sandbox.filesystem.copy_to_local('/work/pxl-sam-review/frames.zip',work/'frames.zip')
    with zipfile.ZipFile(work/'frames.zip') as archive:
        entries=archive.infolist()
        if len(entries)!=7 or sum(e.file_size for e in entries)>32*1024**2:raise ValueError('Frame archive exceeds bound')
        for e in entries:
            target=(work/e.filename).resolve()
            if target.parent!=work or target.exists() or target.name not in {'frames.json',*(f'{i:06d}.jpg' for i in frames)} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe frame entry')
            target.write_bytes(archive.read(e))
    rows=json.loads((work/'frames.json').read_text())
    for row in rows:row['image_sha256']=sha256_file(work/f"{row['frame_index']:06d}.jpg")
    fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest();claim='pxl-sam-review-20261103'
    ledger=SpendingLedger(ROOT/'runs/soccertrack-pxl-20261103-budget.sqlite3')
    if not ledger.reserve('meta',claim,fingerprint=fingerprint,worst_case_usd='2.00')['created']:raise ValueError('Consumed SAM claim')
    record={'status':'running','model':'sam-3.1','reserved_usd':'2.00','requests':[],'training_eligible':False,'billing':'unreconciled','worker_id':worker['worker_id'],'new_modal_worker_created':False};save(work/'receipt.json',record);proposals=[]
    try:
        import httpx
        key=settings.credentials.get('META_API_KEY')
        if not key:raise ValueError('Meta key missing')
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=60) as http:
            for row in rows:
                image=work/f"{row['frame_index']:06d}.jpg";entry={**row,'state':'dispatched'};record['requests'].append(entry);save(work/'receipt.json',record)
                payload={'model':'sam-3.1','input':[{'type':'message','role':'user','content':[{'type':'input_text','text':'person'},{'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(image.read_bytes()).decode()}]}],'stream':False,'metadata':{'mask_encoding':'one_bit','include_confidence':'true'}}
                with http.stream('POST','https://api.meta.ai/v1/responses',headers={'Authorization':'Bearer '+key},json=payload) as response:
                    entry['http_status']=response.status_code
                    if response.status_code!=200:raise RuntimeError('SAM request rejected; no retry')
                    body=bytearray()
                    for chunk in response.iter_bytes():
                        if len(body)+len(chunk)>8*1024**2:raise ValueError('SAM response exceeds bound')
                        body.extend(chunk)
                    data=json.loads(body)
                if data.get('status')!='completed':raise ValueError('Incomplete SAM response')
                text=''.join(c['text'] for msg in data.get('output',[]) for c in msg.get('content',[]) if c.get('type')=='output_text')
                objects=parse_sam_boxes(text,row['width'],row['height'])
                entry.update(state='completed',objects=len(objects))
                if isinstance(data.get('id'),str) and re.fullmatch('[a-zA-Z0-9_-]{1,200}',data['id']):entry['response_id']=data['id']
                usage=data.get('usage',{})
                entry['usage']={k:v for k,v in usage.items() if k in {'input_tokens','output_tokens','total_tokens'} and type(v) is int and 0<=v<=10**8} if isinstance(usage,dict) else {}
                proposals.append({**row,'sam_objects':objects});save(work/'proposals.json',proposals);save(work/'receipt.json',record)
                print(json.dumps({'stage':'meta.sam','frame_index':row['frame_index'],'http_status':200,'objects':len(objects)}),flush=True)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.filesystem.copy_from_local(work/'proposals.json','/work/pxl-sam-review/proposals.json')
        record['status']='complete'
    except BaseException as exc:record.update(status='failed',**safe_error(exc))
    finally:save(work/'receipt.json',record);ledger.reconcile(claim,finalized=False)
    return record

if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:
        r=run();print(json.dumps({'status':r['status'],'completed_requests':sum(x['state']=='completed' for x in r['requests'])}))
        if r['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
