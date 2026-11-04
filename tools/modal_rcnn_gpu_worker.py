"""Owner-selected full R-CNN GPU analysis with bounded24-ID motion association."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
from contextlib import redirect_stdout,redirect_stderr
import csv,hashlib,json,os,re,subprocess,sys,threading,time,uuid,zipfile
W=Path('/work');DONE=threading.Event();STATE={'stage':'input','frames':0,'selected_frames':0};STATUS_PATH=None;STATUS_LOCK=threading.Lock();EVENT_SEQ=0
def emit(stage,**values):
    global EVENT_SEQ
    with STATUS_LOCK:
        if STATE.get('terminal') and values.get('status')=='heartbeat':return
        STATE['stage']=stage
        if stage=='complete' and values.get('status')=='complete' or values.get('status')=='failed':STATE['terminal']=True
        EVENT_SEQ+=1
        item={'stage':stage,'frames':STATE['frames'],'selected_frames':STATE['selected_frames'],**values,'event_seq':EVENT_SEQ,'worker_epoch':time.time()}
        if STATUS_PATH is not None:
            temporary=STATUS_PATH.with_suffix('.tmp');temporary.write_text(json.dumps(item));temporary.replace(STATUS_PATH)
        sys.__stdout__.write(json.dumps(item)+'\n');sys.__stdout__.flush()

def main():
    global STATUS_PATH
    fast='--fast-recovery' in sys.argv;amp=fast or '--amp-recovery' in sys.argv
    folder='rcnn-gpu-fast' if fast else 'rcnn-gpu-amp' if amp else 'rcnn-gpu-full'
    work=W/folder;work.mkdir();job=json.loads((W/(folder+'-job.json' if amp else 'rcnn-gpu-job.json')).read_text())
    STATUS_PATH=work/'worker-state.json';emit('input',status='started')
    def pulse():
        while not DONE.wait(30):emit(STATE['stage'],status='heartbeat',frames=STATE['frames'],selected_frames=STATE['selected_frames'])
    threading.Thread(target=pulse,daemon=True).start()
    code=W/(folder+'-code.zip' if amp else 'rcnn-gpu-code.zip')
    for p,digest in [(W/'normalized/video24.mp4',job['video_sha256']),(W/'rcnn-comparison-weights.pth',job['weights_sha256']),(code,job['code_sha256'])]:
        with p.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Frozen GPU input mismatch')
    if job.get('fp32_reference_zip_sha256'):
        reference_zip=W/'rcnn-fp32-reference.zip'
        with reference_zip.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=job['fp32_reference_zip_sha256']:raise ValueError('FP32 reference archive changed')
        target=W/'rcnn-gpu-full';target.mkdir()
        with zipfile.ZipFile(reference_zip) as archive:
            entry=archive.getinfo('raw_proposals.jsonl')
            if entry.file_size>256*1024**2:raise ValueError('Reference proposals exceed bound')
            with archive.open(entry) as src,(target/'raw_proposals.jsonl').open('xb') as dst:
                import shutil;shutil.copyfileobj(src,dst,1024**2)
    project=work/'project';project.mkdir()
    with zipfile.ZipFile(code) as archive:
        for e in archive.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'));checks={}
    for mode,args in [('tests',[]),('artifacts',['artifacts'])]:
        with (work/(mode+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        text=(work/(mode+'.log')).read_text();counts=re.findall(r'Ran (\d+) tests?',text);checks[mode]={'exit_code':r.returncode,'tests':int(counts[-1]) if counts else 0}
        if r.returncode:raise RuntimeError('GPU job source checks failed')
    (work/'checks.json').write_text(json.dumps(checks));emit('checks',status='passed',tests=checks['tests']['tests'])
    import av,cv2,numpy as np,torch
    from PIL import Image,ImageDraw,ImageFont
    from torchvision.transforms.functional import to_tensor
    from torchvision.ops import nms
    from touchline.adapters.rcnn_comparison import RcnnComparisonDetector
    from touchline.motion_tracking import MotionTracker,estimate_motion,appearance,iou
    from touchline.pitch_scope import estimate_pitch_mask,footpoint_scope
    from touchline.video.decode import decode_video,probe_video
    from touchline.data.manifest import sha256_file
    from touchline.diagnostics import diagnostic_session
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('Explicit T4 CUDA unavailable')
    nms(torch.tensor([[0.,0.,10.,10.]],device='cuda'),torch.tensor([.9],device='cuda'),.5)
    detector=RcnnComparisonDetector(W/'rcnn-comparison-weights.pth',device='cuda',min_size=720 if fast else 1080,max_size=1280 if fast else 4096);model=detector.model
    torch.backends.cudnn.benchmark=amp
    if not detector.actual_device.startswith('cuda'):raise RuntimeError('Model is not on GPU')
    source=W/'normalized/video24.mp4';info=probe_video(source)
    if info['average_fps']!=24 or info['frame_count']!=17830:raise ValueError('Exact24FPS full source required')
    analysis_frames=job.get('max_frames',info['frame_count'])
    if type(analysis_frames) is not int or not 24<=analysis_frames<=info['frame_count']:raise ValueError('Bounded requested frame range required')
    sam=json.loads((W/'pxl-sam-review/proposals.json').read_text()) if (W/'pxl-sam-review/proposals.json').exists() else []
    sam={r['frame_index']:r['sam_objects'] for r in sam};tracker=MotionTracker(info['width'],info['height'],max_session_ids=24,retain_s=900.)
    def predict(frame,force_fp32=False):
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp and not force_fp32):
            image=frame.image.reformat(width=1280,height=720,format='rgb24').to_image() if fast else frame.image.to_image().convert('RGB')
            tensor=to_tensor(image).to('cuda');images,_=model.transform([tensor]);features=model.backbone(images.tensors);rpn,_=model.rpn(images,features,None);pred,_=model.roi_heads(features,rpn,images.image_sizes,None)
            prediction=model.transform.postprocess(pred,images.image_sizes,[(frame.height,frame.width)])[0]
            keep=torch.where((prediction['labels']==1)&(prediction['scores']>=.1))[0];boxes=prediction['boxes'][keep];scores=prediction['scores'][keep]
            scaled=boxes.clone();scaled[:,[0,2]]*=images.image_sizes[0][1]/frame.width;scaled[:,[1,3]]*=images.image_sizes[0][0]/frame.height
            pooled=model.roi_heads.box_roi_pool(features,[scaled],images.image_sizes);embeddings=model.roi_heads.box_head(pooled).float().clamp_min(0);embeddings=embeddings/embeddings.sum(dim=1,keepdim=True).clamp_min(1e-9)
            return boxes.float().cpu().tolist(),scores.float().cpu().tolist(),embeddings.cpu().numpy()
    parity=[]
    if amp:
        wanted={0,96,192,288,384,472};reference=defaultdict(list)
        with (W/'rcnn-gpu-full/raw_proposals.jsonl').open() as stream:
            for line in stream:
                if not line.endswith('\n'):continue
                row=json.loads(line)
                if row['frame_index'] in wanted and row['score']>=.5:reference[row['frame_index']].append(row)
        for frame in decode_video(source,job['video_sha256'],decoder_threads=job.get('decoder_threads',4 if fast else None)):
            if frame.index>max(wanted):break
            if frame.index not in wanted:continue
            boxes,scores,_=predict(frame);candidate=[b for b,s in zip(boxes,scores) if s>=.5];old=reference[frame.index]
            if fast:
                ref_boxes,ref_scores,_=predict(frame,force_fp32=True);old=[{'bbox_xyxy':b} for b,s in zip(ref_boxes,ref_scores) if s>=.5]
            matched=sum(max((iou(r['bbox_xyxy'],b) for b in candidate),default=0)>=.85 for r in old)
            record={'frame_index':frame.index,'fp32_boxes':len(old),'amp_boxes':len(candidate),'fp32_boxes_matched_iou85':matched};parity.append(record)
            if not old or abs(len(old)-len(candidate))>2 or matched/len(old)<.8:raise ValueError('FP16 geometry parity gate failed')
        (work/'precision-parity.json').write_text(json.dumps(parity,indent=2));emit('precision_parity',status='passed',frames=len(parity))
    props=defaultdict(list);observations=defaultdict(list);previous=None;sample_count=0;scoped_out=unknown_scope=0;unassigned_total=0;runtime=time.monotonic();seen=set();run_id=str(uuid.uuid4());fps=3 if fast else 6;burst_until=-1.;burst_budget=640
    scale=info['width']/960;small_h=round(info['height']/scale);baseline=job['weights_sha256']
    emit('analysis',status='started',gpu_memory_mib=round(torch.cuda.get_device_properties(0).total_memory/1024**2))
    with (work/'processing.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(work/'diagnostics'),(work/'raw_proposals.jsonl').open('w') as rawlog,(work/'observations.jsonl').open('w') as obslog,(work/'coverage.jsonl').open('w') as covlog:
        for frame in decode_video(source,job['video_sha256'],decoder_threads=job.get('decoder_threads',4 if fast else None)):
            if frame.index>=analysis_frames:break
            STATE['frames']=frame.index+1
            if abs(frame.timestamp_s-frame.index/24)>1e-4:raise ValueError('Analysis frame timestamp differs from exact24FPS cadence')
            if fast:
                if frame.index%8:
                    if frame.index%4 or frame.timestamp_s>burst_until or burst_budget<=0:continue
                    burst_budget-=1
            elif frame.index%4:continue
            rgb=frame.image.reformat(width=960,height=small_h,format='rgb24').to_ndarray();gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            camera=np.array([[1.,0.,0.],[0.,1.,0.]]);valid=False;flow={}
            if previous is not None:camera,valid,_,flow=estimate_motion(previous,gray,{k:t.box for k,t in tracker.tracks.items() if frame.timestamp_s-t.last_anchor<=tracker.max_gap},info['width'])
            if fast and any(np.linalg.norm(np.asarray(v)-camera[:,2])>.2*max(8.,tracker.tracks[k].box[3]-tracker.tracks[k].box[1]) for k,v in flow.items() if k in tracker.tracks):burst_until=frame.timestamp_s+.5
            previous=gray;pitch=estimate_pitch_mask(rgb)
            boxes,scores,embeddings=predict(frame)
            candidates=[];frame_props=[]
            for box,score,feature in zip(boxes,scores,embeddings):
                x1,y1,x2,y2=box;box=[max(0,x1),max(0,y1),min(frame.width,x2),min(frame.height,y2)]
                if not np.isfinite(box).all() or not box[0]<box[2] or not box[1]<box[3]:continue
                scope=footpoint_scope(box,frame.width,frame.height,pitch);scoped_out+=scope=='outside_estimated_pitch';unknown_scope+=scope=='unknown'
                proposal={'frame_index':frame.index,'timestamp_s':frame.timestamp_s,'bbox_xyxy':box,'score':score,'pitch_scope':scope};rawlog.write(json.dumps(proposal)+'\n');frame_props.append(proposal)
                if scope=='outside_estimated_pitch':continue
                color=appearance(rgb,np.array(box)/scale)
                if color is None:color=np.zeros(208)
                descriptor=np.r_[.6*feature,.4*color]
                semantic=not sam or frame.index in sam and any(iou(box,o['bbox_xyxy'])>=.25 for o in sam[frame.index])
                candidates.append({'bbox_xyxy':box,'score':score,'feature':descriptor,'new_track_allowed':semantic and scope=='inside_estimated_pitch'})
            rows=tracker.update(frame.timestamp_s,candidates,camera=camera,camera_valid=valid,foreground=flow);unassigned_total+=len(tracker.unassigned)
            for row in rows:
                seen.add(row['track_id']);obslog.write(json.dumps({'run_id':run_id,'frame_index':frame.index,'timestamp_s':frame.timestamp_s,**row})+'\n')
            observations[frame.index]={'tracks':rows,'unassigned':tracker.unassigned};props[frame.index]=frame_props;sample_count+=1;STATE['selected_frames']=sample_count
            covlog.write(json.dumps({'frame_index':frame.index,'timestamp_s':frame.timestamp_s,'detections':len(frame_props),'in_pitch_candidates':len(candidates),'assigned_tracks':len(rows),'scope_known':pitch is not None})+'\n')
            if sample_count%120==0:emit('analysis',status='progress',frames=frame.index+1,selected_frames=sample_count,ids=tracker.next_id-1)
    emit('analysis',status='complete',frames=STATE['frames'],selected_frames=sample_count,ids=tracker.next_id-1)
    # Render at1080p while preserving4K source-pixel coordinates in all records.
    display_width=1280 if fast else 1920
    emit('render',status='started');ds=display_width/info['width'];height=round(info['height']*ds)//2*2;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18 if fast else 22);gallery=Image.new('RGB',(960,1800),'#161616');gallery_seen=set();sheet=Image.new('RGB',(960,840),'black');sheet_frames={round(i*(analysis_frames-1)/5):i for i in range(6)};last={};trails=defaultdict(list)
    with av.open(str(work/'overlay24.mp4'),'w') as encoded,(work/'pixel-trajectories.csv').open('w',newline='') as csvfile:
        writer=csv.writer(csvfile);writer.writerow(['frame','timestamp_s','track_id','evidence','foot_x_source_px','foot_y_source_px']);stream=encoded.add_stream('libx264',rate=24);stream.width=display_width;stream.height=height;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'ultrafast' if fast else 'veryfast'};stream.codec_context.thread_count=2
        for frame in decode_video(source,job['video_sha256'],decoder_threads=job.get('decoder_threads',4 if fast else None)):
            if frame.index>=analysis_frames:break
            image=frame.image.reformat(width=display_width,height=height,format='rgb24').to_image();draw=ImageDraw.Draw(image)
            if frame.index in observations:last=observations[frame.index]
            for row in last.get('tracks',[]):
                tid=row['track_id'];color=((tid*73)%190+60,(tid*131)%190+60,(tid*47)%190+60);box=[v*ds for v in row['bbox_xyxy']];draw.rectangle(box,outline=color,width=3);anchor=frame.index in observations
                draw.text((max(0,box[0]),max(70,box[1]-26)),f"T{tid} {'OBS' if anchor and row['evidence']=='detector_anchor' else 'PRED'}",font=font,fill='white',stroke_width=2,stroke_fill='black')
                point=row['footpoint_xy'];trails[tid].append((point[0]*ds,point[1]*ds));trails[tid]=trails[tid][-48:]
                if len(trails[tid])>1:draw.line(trails[tid],fill=color,width=3)
                writer.writerow([frame.index,frame.timestamp_s,tid,'detector_anchor' if anchor and row['evidence']=='detector_anchor' else 'visual_hold_prediction',*point])
                if tid not in gallery_seen:
                    crop=image.crop(tuple(map(int,box)));crop.thumbnail((220,230));slot=tid-1;x=slot%4*240;y=slot//4*300;gallery.paste(crop,(x+(240-crop.width)//2,y+5));gd=ImageDraw.Draw(gallery);gd.text((x+8,y+240),f'T{tid}',font=font,fill=color);gd.text((x+8,y+270),f'{frame.timestamp_s:.2f}s',font=font,fill='white');gallery_seen.add(tid)
            for u in last.get('unassigned',[]):draw.rectangle([v*ds for v in u['bbox_xyxy']],outline='#ffb000',width=2)
            draw.rectangle((0,0,display_width,66),fill='black');draw.text((8,2),f'{frame.timestamp_s:.2f}s | ResNet50 R-CNN on T4 | IDs {tracker.next_id-1}/24',font=font,fill='white');draw.text((8,34),'OBS=detector anchor; PRED=estimated/held; orange=unassigned. IDs not jersey numbers.',font=font,fill='white')
            if frame.index in sheet_frames:
                slot=sheet_frames[frame.index];thumb=image.copy();thumb.thumbnail((480,270));sheet.paste(thumb,(slot%2*480,slot//2*280))
            output=av.VideoFrame.from_image(image);output.pts=frame.index;output.time_base=Fraction(1,24)
            for packet in stream.encode(output):encoded.mux(packet)
            if frame.index%240==0:emit('render',status='progress',frames=frame.index+1)
        for packet in stream.encode():encoded.mux(packet)
    gallery.save(work/'players.jpg',quality=92);sheet.save(work/'contact-sheet.jpg',quality=92)
    summary={'status':'complete','run_id':run_id,'architecture':detector.architecture,'checkpoint_sha256':baseline,'source_sha256':job['video_sha256'],'original_source_sha256':job['source_sha256'],'frames':info['frame_count'],'selected_frames':sample_count,'session_id_cap':24,'allocated_session_ids':tracker.next_id-1,'visible_ids':len(seen),'unassigned_detections':unassigned_total,'outside_estimated_pitch_proposals':scoped_out,'unknown_pitch_proposals':unknown_scope,'device':detector.actual_device,'gpu_name':torch.cuda.get_device_name(0),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'detector_fps':6,'source_fps':24,'source_geometry':[info['width'],info['height']],'display_geometry':[1920,height],'checks':checks,'sam_review_frames':sorted(sam),'reid':'generic nonnegative R-CNN ROI descriptor plus color; not trained person ReID','tracking_idf1':None,'ap50':None,'physical_distance_m':None,'elapsed_s':round(time.monotonic()-runtime,2),'training':'not run','selection':'owner selected ResNet for analysis','limitations':'Estimated pitch scope and temporary IDs require review.24-slot limit never recycles IDs. Extra/unconfirmed detections remain unassigned.24FPS display includes tagged held estimates between6Hz anchors; no per-frame detector claim.'}
    summary.update(frames=analysis_frames,source_frames=info['frame_count'],full_recording_analyzed=analysis_frames==info['frame_count'],analyzed_duration_s=analysis_frames/24,precision='cuda-autocast-fp16' if amp else 'fp32',precision_parity=parity,decoder_threads=job.get('decoder_threads',4 if fast else None),detector_fps=fps,detector_max_fps=6,adaptive_burst_limit=640 if fast else None,actual_detector_samples=sample_count,display_geometry=[display_width,height],inference_min_size=720 if fast else 1080,inference_max_size=1280 if fast else 4096,input_pre_resize=[1280,720] if fast else None,new_identity_policy='SAM-supported births only when SAM review frames are supplied; extras remain unassigned')
    (work/'summary.json').write_text(json.dumps(summary,indent=2));(work/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>Full video ResNet R-CNN review</h1><video controls width="1200" src="overlay24.mp4"></video><p>Experimental24-ID motion association and automatic grass-pitch scope. Generic ROI appearance is not trained person ReID. Orange means unassigned. Physical metrics/AP50/IDF1 unknown.</p><a href="players.jpg">Player gallery</a> | <a href="summary.json">Receipt</a>')
    with zipfile.ZipFile(work/'results.zip','w',zipfile.ZIP_STORED) as archive:
        for p in work.iterdir():
            if p.is_file() and p.name!='results.zip':
                with p.open('rb') as src,archive.open(p.name,'w',force_zip64=True) as dst:
                    import shutil;shutil.copyfileobj(src,dst,1024**2)
        for p in (work/'diagnostics').rglob('*'):
            if p.is_file():
                with p.open('rb') as src,archive.open(p.relative_to(work).as_posix(),'w',force_zip64=True) as dst:
                    import shutil;shutil.copyfileobj(src,dst,1024**2)
    emit('complete',status='complete',run_id=run_id,frames=analysis_frames,ids=tracker.next_id-1)

if __name__=='__main__':
    try:main()
    except BaseException as exc:emit(STATE['stage'],status='failed',error_category=type(exc).__name__);raise SystemExit(1) from None
    finally:DONE.set()
