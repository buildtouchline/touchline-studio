"""Fetch a closed20-second review video while the same GPU analysis continues."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import json,logging,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from touchline.settings import load_operator_settings
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error

WORKER=r'''from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import av,json,zipfile,time
from PIL import Image,ImageDraw,ImageFont
w=Path('/work/rcnn-live-preview');w.mkdir();base=Path('/work/rcnn-gpu-amp');deadline=time.monotonic()+90
while True:
 coverage=[]
 if (base/'coverage.jsonl').exists():
  with (base/'coverage.jsonl').open() as s:
   for line in s:
    if line.endswith('\n'):
     r=json.loads(line)
     if r['frame_index']<480:coverage.append(r)
 if len(coverage)==120:break
 if time.monotonic()>deadline:raise TimeoutError('First twenty seconds not ready')
 time.sleep(3)
observations=defaultdict(list)
with (base/'observations.jsonl').open() as s:
 for line in s:
  if not line.endswith('\n'):continue
  r=json.loads(line)
  if r['frame_index']<480:observations[r['frame_index']].append(r)
  else:break
job=json.loads(Path('/work/rcnn-gpu-amp-job.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
gallery=Image.new('RGB',(960,1800),'#171717');sheet=Image.new('RGB',(1920,1080),'black');seen=set();last=[];trails=defaultdict(list);stats=defaultdict(lambda:{'first_frame':None,'anchor_frames':0});frames=0
with av.open('/work/normalized/video24.mp4') as reader,av.open(str(w/'overlay24.mp4'),'w') as encoded:
 source=reader.streams.video[0];source.thread_type='AUTO';source.codec_context.thread_count=4
 stream=encoded.add_stream('libx264',rate=24);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'ultrafast'};stream.codec_context.thread_count=2
 for f in reader.decode(video=0):
  index=round(f.pts*f.time_base*24)
  if index>=480:break
  if index!=frames:raise ValueError('Unexpected source timing')
  image=f.reformat(width=1920,height=1080,format='rgb24').to_image();draw=ImageDraw.Draw(image);anchor=index%4==0
  if anchor:last=observations.get(index,[])
  for row in last:
   tid=row['track_id'];box=[v*.5 for v in row['bbox_xyxy']];color=((tid*73)%190+60,(tid*131)%190+60,(tid*47)%190+60)
   draw.rectangle(box,outline=color,width=4);draw.text((max(0,box[0]),max(72,box[1]-28)),f'T{tid} '+('OBS' if anchor and row['evidence']=='detector_anchor' else 'PRED'),font=font,fill='white',stroke_width=2,stroke_fill='black')
   p=row['footpoint_xy'];trails[tid].append((p[0]*.5,p[1]*.5));trails[tid]=trails[tid][-48:]
   if len(trails[tid])>1:draw.line(trails[tid],fill=color,width=3)
   if anchor:stats[tid]['anchor_frames']+=row['evidence']=='detector_anchor'
   if tid not in seen:
    seen.add(tid);stats[tid]['first_frame']=index;crop=image.crop(tuple(map(int,box)));crop.thumbnail((220,225));slot=tid-1;x=slot%4*240;y=slot//4*300;gallery.paste(crop,(x+(240-crop.width)//2,y+5));gd=ImageDraw.Draw(gallery);gd.text((x+8,y+235),f'T{tid}',font=font,fill=color);gd.text((x+8,y+268),f'{index/24:.2f}s',font=font,fill='white')
  draw.rectangle((0,0,1920,68),fill='black');draw.text((8,2),f'{index/24:.2f}s | ResNet50 on T4 | temporary IDs and source-pixel trails',font=font,fill='white');draw.text((8,35),'OBS=detector anchor; PRED=estimated/held. T numbers are not jersey numbers.',font=font,fill='white')
  if index in {96,192,288,384}:
   slot=[96,192,288,384].index(index);thumb=image.resize((960,540));sheet.paste(thumb,(slot%2*960,slot//2*540))
  out=av.VideoFrame.from_image(image);out.pts=index;out.time_base=Fraction(1,24)
  for packet in stream.encode(out):encoded.mux(packet)
  frames+=1
 for packet in stream.encode():encoded.mux(packet)
if frames!=480:raise ValueError('Complete twenty-second preview required')
gallery.save(w/'players.jpg',quality=92);sheet.save(w/'contact-sheet.jpg',quality=92)
receipt={'status':'complete','scope':'first20seconds only; full recording is separate','frames':480,'fps':24,'detector_fps':6,'source_geometry':[3840,2160],'display_geometry':[1920,1080],'checkpoint_sha256':job['weights_sha256'],'source_sha256':job['video_sha256'],'temporary_ids':sorted(seen),'tracks':stats,'identity_accuracy':None,'tracking_idf1':None,'model_device':'cuda','render_device':'cpu','birth_policy':'SAM-supported anchors; ambiguous reidentification stays unassigned'}
(w/'summary.json').write_text(json.dumps(receipt,indent=2));(w/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>ResNet early video review</h1><video controls width="1200" src="overlay24.mp4"></video><p>First20seconds. Temporary IDs, not jersey numbers. Full-recording result is separate. No measured identity accuracy.</p><a href="players.jpg">Track gallery</a>')
with zipfile.ZipFile(w/'preview.zip','w',zipfile.ZIP_STORED) as archive:
 for p in w.iterdir():
  if p.is_file() and p.name!='preview.zip':archive.write(p,p.name)
print(json.dumps({'status':'complete','frames':frames,'ids':len(seen)}))
'''

def run():
    parent=ROOT/'runs/modal-rcnn-full-gpu-20261103-replacement';r=json.loads((parent/'job-receipt.json').read_text())
    if r['status']!='running' or r['worker_terminated']!='pending':raise ValueError('Existing replacement required')
    work=ROOT/'runs/modal-rcnn-live-preview-20261103';work.mkdir(exist_ok=False)
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);sandbox=modal.Sandbox.from_id(r['worker_id'],client=client)
        sandbox.filesystem.write_text(WORKER,'/work/rcnn-live-preview-worker.py');p=sandbox.exec('python','-I','/work/rcnn-live-preview-worker.py',timeout=240,text=False)
        result=json.loads(read_worker_result(p.stdout));p.wait()
        if p.returncode or result['status']!='complete':raise RuntimeError('Live review render failed')
        if sandbox.filesystem.stat('/work/rcnn-live-preview/preview.zip').size>128*1024**2:raise ValueError('Preview export exceeds bound')
        sandbox.filesystem.copy_to_local('/work/rcnn-live-preview/preview.zip',work/'preview.zip')
    with zipfile.ZipFile(work/'preview.zip') as archive:
        entries=archive.infolist()
        if len(entries)!=5 or sum(e.file_size for e in entries)>128*1024**2:raise ValueError('Expanded preview exceeds bound')
        names=set()
        for e in entries:
            target=(work/e.filename).resolve()
            if target.parent!=work or target.exists() or e.filename in names or target.suffix not in {'.mp4','.json','.jpg','.html'} or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe preview member')
            names.add(e.filename)
        for e in entries:(work/e.filename).write_bytes(archive.read(e))
    result.update(worker_id=r['worker_id'],new_worker_created=False);save(work/'controller-receipt.json',result);return result
if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:print(json.dumps(run()))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
