"""Render three real minutes from preserved GPU ResNet results in Modal CPU."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,logging,sqlite3,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from touchline.data.manifest import sha256_file
from probe_modal_gpu import safe_error

WORKER=r'''from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import av,hashlib,json,time,zipfile,shutil,math
from PIL import Image,ImageDraw,ImageFont
w=Path('/work/cached-resnet-demo');w.mkdir();job=json.loads(Path('/work/cached-demo-job.json').read_text())
with Path('/work/resnet-preserved.zip').open('rb') as stream:
 if hashlib.file_digest(stream,'sha256').hexdigest()!=job['cache_sha256']:raise ValueError('Cached analysis changed')
rows=defaultdict(list);proposals=defaultdict(list);coverage=set()
with zipfile.ZipFile('/work/resnet-preserved.zip') as archive:
 for name,target in [('observations.jsonl',rows),('raw_proposals.jsonl',proposals),('coverage.jsonl',None)]:
  entry=archive.getinfo(name)
  if entry.file_size>128*1024**2:raise ValueError('Cached evidence exceeds bound')
  with archive.open(entry) as stream:
   for line in stream:
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    index=r['frame_index']
    if type(index) is not int or index<0:raise ValueError('Invalid cached frame')
    if index>=4320:continue
    if abs(r['timestamp_s']-index/24)>1e-4:raise ValueError('Cached frame timing mismatch')
    if target is None:coverage.add(index)
    else:target[index].append(r)
if len(coverage)<540 or max(coverage)<4312:raise ValueError('Complete three-minute cache required')
video=Path('/work/normalized/video24.mp4')
with video.open('rb') as stream:
 if hashlib.file_digest(stream,'sha256').hexdigest()!=job['video_sha256']:raise ValueError('Exact analyzed video required')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',23);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
gallery=Image.new('RGB',(960,1800),'#141a1f');seen=set();stats=defaultdict(lambda:{'first_frame':None,'last_frame':0,'visible_frames':0});trails=defaultdict(list);sheet=Image.new('RGB',(1920,1620),'black');snapshot_frames={120:0,960:1,1800:2,2640:3,3480:4,4200:5};last=[];last_proposals=[];frames=0;start=time.monotonic()
with av.open(str(video)) as reader,av.open(str(w/'Touchline-ResNet-3min-demo.mp4'),'w',options={'movflags':'+faststart'}) as encoded:
 source=reader.streams.video[0];source.thread_type='AUTO';source.codec_context.thread_count=4
 if source.average_rate!=24 or source.width!=3840 or source.height!=2160:raise ValueError('Expected24FPS analyzed geometry')
 stream=encoded.add_stream('libx264',rate=24);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'21','preset':'veryfast','tune':'zerolatency'};stream.codec_context.thread_count=2
 for f in reader.decode(video=0):
  index=round(f.pts*f.time_base*24)
  if index>=4320:break
  if index!=frames:raise ValueError('Decoded frame correspondence changed')
  image=f.reformat(width=1920,height=1080,format='rgb24').to_image();draw=ImageDraw.Draw(image);anchor=index in coverage
  if anchor:last=rows.get(index,[]);last_proposals=proposals.get(index,[])
  for row in last:
   tid=row['track_id'];box=[v*.5 for v in row['bbox_xyxy']]
   if type(tid) is not int or not 1<=tid<=24 or not all(math.isfinite(v) for v in box):raise ValueError('Invalid cached track')
   color=((tid*73)%190+60,(tid*131)%190+60,(tid*47)%190+60);draw.rectangle(box,outline=color,width=4)
   text=f'T{tid}'+('' if anchor and row['evidence']=='detector_anchor' else ' held');x=max(0,min(1800,box[0]));y=max(82,box[1]-31)
   extent=draw.textbbox((x,y),text,font=font);draw.rectangle((extent[0]-4,extent[1]-3,extent[2]+4,extent[3]+3),fill='#12181e');draw.text((x,y),text,font=font,fill='white')
   p=row['footpoint_xy'];trails[tid].append((p[0]*.5,p[1]*.5));trails[tid]=trails[tid][-36:]
   if len(trails[tid])>1:draw.line(trails[tid],fill=color,width=3)
   stats[tid]['last_frame']=index;stats[tid]['visible_frames']+=1
   if tid not in seen:
    seen.add(tid);stats[tid]['first_frame']=index;crop=image.crop(tuple(map(int,box)));crop.thumbnail((220,225));slot=tid-1;xx=slot%4*240;yy=slot//4*300;gallery.paste(crop,(xx+(240-crop.width)//2,yy+5));gd=ImageDraw.Draw(gallery);gd.text((xx+8,yy+238),f'T{tid}',font=font,fill=color);gd.text((xx+8,yy+270),f'{index/24:.1f}s',font=small,fill='white')
  draw.rectangle((0,0,1920,78),fill='#111820');draw.text((20,8),'TOUCHLINE  |  Football session review',font=font,fill='white');draw.text((20,43),f'{index//1440:02d}:{(index//24)%60:02d}  |  ResNet detections and temporary track IDs  |  First 3 minutes',font=small,fill='#d4dee9');draw.text((1160,11),'T labels identify tracks, not jersey numbers',font=small,fill='#d4dee9')
  draw.rectangle((0,1048,1920,1080),fill='#111820');draw.text((20,1053),'Experimental tracking. Held labels are visual estimates between detector samples. Review uncertain identities.',font=small,fill='#d4dee9')
  if index in snapshot_frames:
   slot=snapshot_frames[index];sheet.paste(image.resize((960,540)),(slot%2*960,slot//2*540))
  frame=av.VideoFrame.from_image(image);frame.pts=index;frame.time_base=Fraction(1,24)
  for packet in stream.encode(frame):encoded.mux(packet)
  frames+=1
  if frames%240==0:print(json.dumps({'stage':'render','status':'progress','frames':frames}),flush=True)
 for packet in stream.encode():encoded.mux(packet)
if frames!=4320:raise ValueError('Full180-second output required')
gallery.save(w/'players.jpg',quality=92);sheet.save(w/'contact-sheet.jpg',quality=92)
with av.open(str(w/'Touchline-ResNet-3min-demo.mp4')) as check:
 stream=check.streams.video[0]
 if stream.frames!=4320 or stream.average_rate!=24 or (stream.width,stream.height)!=(1920,1080) or abs(float(stream.duration*stream.time_base)-180)>.001:raise ValueError('Export video timing validation failed')
with (w/'Touchline-ResNet-3min-demo.mp4').open('rb') as stream:video_hash=hashlib.file_digest(stream,'sha256').hexdigest()
summary={'status':'complete','scope':'first180seconds of PXL_20261101_045045814~2.mp4','frames':4320,'duration_s':180,'video_fps':24,'display_geometry':[1920,1080],'source_geometry':[3840,2160],'source_sha256':job['video_sha256'],'checkpoint_sha256':job['model_sha256'],'model':'fasterrcnn_resnet50_fpn_v2','model_execution':'previous preserved T4 GPU attempt; no new model inference','render_execution':'Modal CPU Sandbox','cached_detector_sampling':'adaptive3–6Hz; all4320 source frames rendered','inference_resize':[720,1280],'cache_sha256':job['cache_sha256'],'selected_frames':len(coverage),'visible_ids':sorted(seen),'tracks':stats,'tracking_idf1':None,'detection_ap50':None,'video_sha256':video_hash,'render_elapsed_s':round(time.monotonic()-start,2),'limitations':'Temporary IDs may fragment or switch; not jersey identities. A complete demo clip does not establish tracking accuracy or full-recording completion.'}
(w/'summary.json').write_text(json.dumps(summary,indent=2));(w/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>Touchline ResNet three minute demo</h1><video controls width="1200" src="Touchline-ResNet-3min-demo.mp4"></video><p>Real recording and preserved GPU ResNet observations. Temporary IDs, not jersey numbers. No measured tracking accuracy.</p><a href="players.jpg">Track gallery</a>')
with zipfile.ZipFile(w/'demo.zip','w',zipfile.ZIP_STORED) as archive:
 for p in w.iterdir():
  if p.is_file() and p.name!='demo.zip':
   with p.open('rb') as src,archive.open(p.name,'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,1024**2)
print(json.dumps({'stage':'complete','status':'complete','frames':frames,'duration_s':180}),flush=True)
'''

def run():
    work=ROOT/'runs/modal-cached-resnet-demo-20261104-export';work.mkdir(exist_ok=False);cache=ROOT/'runs/modal-rcnn-full-gpu-20261103/fast-preserved.zip'
    with sqlite3.connect(ROOT/'runs/rcnn-gpu-owner-exception-20261103.sqlite3') as db:db.execute('INSERT INTO claims VALUES (?,?,?,?,?)',('cached-ResNet-demo-export-20261104',0,53000000,53000000,sha256_file(cache)))
    record={'status':'prepared','gpu':None,'cpu':4,'memory_mib':8192,'timeout_s':1200,'additional_reservation_usd':'0.00','shared_total_trial_cap_usd':'3.00','combined_resource_upper_bound_usd':'2.60','worker_terminated':'not_created','source':'preserved real GPU ResNet results; no new inference'};save(work/'job-receipt.json',record);sandbox=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work);app=modal.App.lookup('touchline-demo-render',client=client,environment_name='main',create_if_missing=True)
            sandbox=modal.Sandbox.create(app=app,client=client,image=modal.Image.from_id('im-01M6H7NX4NQ9HWJ6Y22WDEVPRX',client=client),cpu=(4,4),memory=(8192,8192),runtime='gvisor',timeout=1200,block_network=True,secrets=[],volumes={},workdir='/work')
        record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'job-receipt.json',record);print(json.dumps({'stage':'sandbox','status':'created','worker_id':sandbox.object_id}),flush=True)
        job={'cache_sha256':sha256_file(cache),'video_sha256':'d6b4c64a1313ae46b1027214323abd995275bc6e8ed06ff86c22491785240d0a','model_sha256':'dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'}
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            sandbox.filesystem.copy_from_local(cache,'/work/resnet-preserved.zip');sandbox.filesystem.write_text(json.dumps(job),'/work/cached-demo-job.json');sandbox.filesystem.write_text(WORKER,'/work/cached-demo-worker.py');process=sandbox.exec('python','-I','/work/cached-demo-worker.py',timeout=1050,text=False)
        buffer=bytearray();complete=False
        for chunk in process.stdout:
            buffer.extend(chunk)
            if len(buffer)>16384:raise ValueError('Render log exceeds bound')
            while b'\n' in buffer:
                line,_,tail=buffer.partition(b'\n');buffer=bytearray(tail)
                if not line.strip():continue
                item=json.loads(line)
                with (work/'events.jsonl').open('a') as log:log.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
                if item.get('stage')=='complete':complete=True
        process.wait()
        if process.returncode or not complete:raise RuntimeError('Cached demo render incomplete')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            summary=json.loads(sandbox.filesystem.read_text('/work/cached-resnet-demo/summary.json'))
            if summary['frames']!=4320 or summary['duration_s']!=180 or summary['video_fps']!=24 or summary['cache_sha256']!=job['cache_sha256']:raise ValueError('Demo metadata mismatch')
            save(work/'summary.json',summary)
            name='Touchline-ResNet-3min-demo.mp4'
            if not 0<sandbox.filesystem.stat('/work/cached-resnet-demo/'+name).size<=2*1024**3:raise ValueError('Demo video exceeds bounded transfer size')
            sandbox.filesystem.copy_to_local('/work/cached-resnet-demo/'+name,work/name)
            if sha256_file(work/name)!=summary['video_sha256']:raise ValueError('Downloaded video hash mismatch')
            for name in ('players.jpg','contact-sheet.jpg','RESULTS.html'):
                if sandbox.filesystem.stat('/work/cached-resnet-demo/'+name).size>32*1024**2:raise ValueError('Review artifact exceeds bound')
                sandbox.filesystem.copy_to_local('/work/cached-resnet-demo/'+name,work/name)
        record.update(status='complete',frames=summary['frames'],video_sha256=summary['video_sha256'])
    except BaseException as exc:record.update(status='failed',failure_type=type(exc).__name__)
    finally:
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'job-receipt.json',record)
    return record
if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:
        r=run();print(json.dumps(r))
        if r['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
