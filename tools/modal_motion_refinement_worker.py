"""Staged24-ID motion replay and complete pixel-trajectory video export."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import csv,hashlib,json,os,re,subprocess,sys,time,uuid,zipfile
W=Path('/work')
def emit(stage,**values):print(json.dumps({'stage':stage,**values}),flush=True)
def main():
    job=json.loads(Path(sys.argv[1]).read_text());full=job['mode']=='full'
    if job['mode'] not in ('preview','full'):raise ValueError('Explicit replay scope required')
    name='motion-full-v3' if full else 'motion-preview20-v3';work=W/name;work.mkdir();project=work/'project';project.mkdir()
    code=W/'motion-code-v3.zip'
    with code.open('rb') as s:
        if hashlib.file_digest(s,'sha256').hexdigest()!=job['code_sha256']:raise ValueError('Motion source changed')
    with zipfile.ZipFile(code) as z:
        for e in z.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    checks={}
    if not full:
        for mode,args in (('tests',[]),('artifacts',['artifacts'])):
            with (work/(mode+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
            text=(work/(mode+'.log')).read_text(errors='replace');count=re.findall(r'Ran (\d+) tests?',text);checks[mode]={'exit_code':r.returncode,'tests':int(count[-1]) if count else 0}
            if r.returncode:raise RuntimeError('Motion checks failed')
        (work/'checks.json').write_text(json.dumps({'code_sha256':job['code_sha256'],'checks':checks}));emit('checks',status='passed',tests=checks['tests']['tests'])
    else:
        passed=json.loads((W/'motion-preview20-v3/checks.json').read_text())
        if passed['code_sha256']!=job['code_sha256'] or any(v['exit_code'] for v in passed['checks'].values()):raise ValueError('Same-source remote checks required')
        start=time.monotonic()
        while not (W/'raw-analysis-pause.json').exists():
            if time.monotonic()-start>3600:raise TimeoutError('Frozen raw analysis not ready')
            time.sleep(2)
        checks=passed['checks']
    import av,cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from touchline.motion_tracking import MotionTracker,appearance,estimate_motion
    from touchline.video.decode import decode_video,probe_video
    from touchline.data.manifest import sha256_file
    cv2.setNumThreads(2)
    source=W/'normalized/video24.mp4' if full else W/'pxl-preview20/source/video24.mp4';base=W/'run' if full else W/'pxl-preview20/run';manifest=json.loads((base/'manifest.json').read_text())
    if full and manifest['status']!='complete':raise ValueError('Complete baseline required')
    if sha256_file(source)!=manifest['source']['id']:raise ValueError('Exact analyzed source required')
    info=probe_video(source)
    if info['average_fps']!=24 or info['is_rotated']:raise ValueError('Verified24FPS copy required')
    proposals=defaultdict(list)
    with (base/'raw_proposals.jsonl').open() as s:
        for line in s:r=json.loads(line);proposals[r['frame']['index']].append(r)
    tracker=MotionTracker(info['width'],info['height'],max_session_ids=24);scale=info['width']/960;small_h=round(info['height']/scale);display_w=1920;display_h=round(info['height']*display_w/info['width']);display_h-=display_h%2;ds=display_w/info['width']
    previous=None;seen=set();frames=anchors=verified=unassigned_count=0;points=defaultdict(list);stats={};started=time.monotonic();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22);gallery=Image.new('RGB',(960,1800),'#151515');gallery_seen=set();sheet=Image.new('RGB',(960,840),'black');sheet_frames={round(i*(info['frame_count']-1)/5):i for i in range(6)}
    with av.open(str(work/'overlay24.mp4'),'w') as encoded,(work/'pixel-trajectories.jsonl').open('w') as trace,(work/'pixel-trajectories.csv').open('w',newline='') as csvfile:
        writer=csv.writer(csvfile);writer.writerow(['frame','timestamp_s','track_id','evidence','foot_x_source_px','foot_y_source_px','camera_adjusted_dx_px','camera_adjusted_dy_px','camera_valid','foreground_flow_valid'])
        stream=encoded.add_stream('libx264',rate=24);stream.width=display_w;stream.height=display_h;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'veryfast'};stream.codec_context.thread_count=2
        for frame in decode_video(source,'motion-refinement'):
            rgb=frame.image.reformat(width=960,height=small_h,format='rgb24').to_ndarray();gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);camera=np.array([[1.,0.,0.],[0.,1.,0.]]);valid=False;quality=0.;flow={}
            if previous is not None:camera,valid,quality,flow=estimate_motion(previous,gray,{i:t.box for i,t in tracker.tracks.items() if frame.timestamp_s-t.last_anchor<=tracker.max_gap},info['width'])
            detections=[{'bbox_xyxy':r['bbox_xyxy'],'score':r['score'],'feature':appearance(rgb,np.asarray(r['bbox_xyxy'])/scale)} for r in proposals.get(frame.index,[])]
            rows=tracker.update(frame.timestamp_s,detections,camera=camera,camera_valid=valid,foreground=flow);previous=gray;image=frame.image.reformat(width=display_w,height=display_h,format='rgb24').to_image();draw=ImageDraw.Draw(image);raw=None
            for row in rows:
                tid=row['track_id'];seen.add(tid);color=((tid*73)%190+60,(tid*131)%190+60,(tid*47)%190+60);box=[v*ds for v in row['bbox_xyxy']];draw.rectangle(box,outline=color,width=3)
                label='T'+str(tid)+(' OBS' if row['evidence']=='detector_anchor' else ' FLOW' if row['evidence']=='optical_flow_prediction' else ' PRED');draw.text((max(0,box[0]),max(72,box[1]-28)),label,font=font,fill='white',stroke_width=2,stroke_fill='black')
                trail=[(p[0]*ds,p[1]*ds) for p in row['trail_xy']]
                if len(trail)>1:draw.line(trail,fill=color,width=3)
                delta=row['camera_compensated_delta_px'];p=row['footpoint_xy'];writer.writerow([frame.index,frame.timestamp_s,tid,row['evidence'],*p,*(delta if delta is not None else ('','')),valid,row['foreground_flow_valid']]);anchors+=row['evidence']=='detector_anchor'
                if tid not in stats:stats[tid]={'first_frame':frame.index,'last_frame':frame.index,'visible_frames':0,'anchor_frames':0,'camera_adjusted_path_px':0.}
                stats[tid]['last_frame']=frame.index;stats[tid]['visible_frames']+=1;stats[tid]['anchor_frames']+=row['evidence']=='detector_anchor'
                if delta is not None:stats[tid]['camera_adjusted_path_px']+=float(np.linalg.norm(delta))
                if not full:points[tid].append({'frame':frame.index,'timestamp_s':frame.timestamp_s,'footpoint_xy':p,'evidence':row['evidence']})
                if tid not in gallery_seen:
                    raw=frame.image.to_image().convert('RGB') if raw is None else raw;crop=raw.crop(tuple(map(int,row['bbox_xyxy'])));crop.thumbnail((220,230));slot=tid-1;x=(slot%4)*240;y=(slot//4)*300;gallery.paste(crop,(x+(240-crop.width)//2,y+5));gd=ImageDraw.Draw(gallery);gd.text((x+8,y+240),'T'+str(tid),fill=color,font=font);gd.text((x+8,y+270),f'{frame.timestamp_s:.2f}s',fill='white',font=font);gallery_seen.add(tid)
            for u in tracker.unassigned:
                draw.rectangle([v*ds for v in u['bbox_xyxy']],outline='#ffb000',width=3)
            unassigned_count+=len(tracker.unassigned);draw.rectangle((0,0,display_w,68),fill='black');draw.text((8,3),f'{frame.timestamp_s:.3f}s | {len(rows)} visible | session IDs {tracker.next_id-1}/24 | camera-adjusted pixel trails',font=font,fill='white');draw.text((8,35),'OBS=detector; FLOW/PRED=propagated. Orange=unassigned. IDs and physical metrics unconfirmed.',font=font,fill='white')
            trace.write(json.dumps({'frame_index':frame.index,'timestamp_s':frame.timestamp_s,'camera_affine':camera.tolist(),'camera_valid':valid,'camera_inlier_fraction':quality,'tracks':[{k:v for k,v in r.items() if k!='trail_xy'} for r in rows],'unassigned':tracker.unassigned})+'\n')
            if frame.index in sheet_frames:
                slot=sheet_frames[frame.index];thumb=image.copy();thumb.thumbnail((480,270));x=slot%2*480;y=slot//2*280;sheet.paste(thumb,(x,y));ImageDraw.Draw(sheet).text((x+3,y+265),f'{frame.timestamp_s:.2f}s',fill='white')
            output=av.VideoFrame.from_image(image);output.pts=frame.index;output.time_base=Fraction(1,24)
            for packet in stream.encode(output):encoded.mux(packet)
            frames+=1;verified+=valid
            if frames%240==0:emit('motion_replay',status='progress',frames=frames,ids=tracker.next_id-1)
        for packet in stream.encode():encoded.mux(packet)
    gallery.save(work/'players.jpg',quality=92);sheet.save(work/'contact-sheet.jpg',quality=92)
    baseline_ids=set()
    with (base/'observations.jsonl').open() as s:
        for line in s:baseline_ids.add(json.loads(line)['local_track_id'])
    summary={'status':'complete','run_id':str(uuid.uuid4()),'baseline_run_id':manifest['run_id'],'checkpoint':manifest['detector']['checkpoint'],'source_sha256':manifest['source']['id'],'frames':frames,'detector_samples':manifest['selected_frames'],'detector_anchor_matches':anchors,'baseline_temporary_ids':len(baseline_ids),'motion_temporary_ids':len(seen),'session_id_cap':24,'allocated_session_ids':tracker.next_id-1,'unassigned_detections':unassigned_count,'camera_verified_frames':verified,'tracks':stats,'coordinate_units':'source image pixels; bbox footpoint proxy, including explicitly tagged propagated estimates','physical_distance_m':None,'identity_accuracy':None,'tracking_idf1':None,'model_inference':'not rerun; identical saved proposals','promotion':'not performed','source_geometry':[info['width'],info['height']],'display_geometry':[display_w,display_h],'code_sha256':job['code_sha256'],'checks':checks,'duration_s':round(time.monotonic()-started,2),'limitations':'The24-ID cap is a capacity constraint, not accuracy evidence. Similar appearances and long occlusions remain uncertain; unmatched extra people are unassigned, not merged.'}
    (work/'summary.json').write_text(json.dumps(summary,indent=2))
    if not full:(work/'track-points.json').write_text(json.dumps(points))
    report='# Pixel motion review\n\nTemporary session IDs only. OBS=detector, FLOW/PRED=propagated. Paths are image-plane pixel proxies and include estimates; calibration, confirmed identity and coach review are absent. No speed in metres/s, approved scores or IDF1 claim.\n\n| ID | Visible frames | Detector anchors | Camera-adjusted path px |\n| --- | --- | --- | --- |\n'
    for tid,s in stats.items():report+=f'| T{tid} | {s["visible_frames"]} | {s["anchor_frames"]} | {s["camera_adjusted_path_px"]:.1f} |\n'
    (work/'review.md').write_text(report)
    (work/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>Motion-assisted player review</h1><video controls width="1200" src="'+('preview/' if full else '')+'overlay24.mp4"></video><p>Session IDs≤24, not confirmed names or jersey numbers. OBS=detector; FLOW/PRED=propagation; orange=unassigned. Pixel trails are camera-adjusted image-plane proxies, not metres or measured identity accuracy.</p>')
    if full:
        parent_job=json.loads((W/'job.json').read_text());norm=json.loads((W/'normalized/normalization.json').read_text())
        receipt={'status':'complete','run_id':summary['run_id'],'baseline_detector_run_id':manifest['run_id'],'input_hashes':parent_job['hashes'],'normalization':norm,'decoded_frames':manifest['decoded_frames'],'selected_frames':manifest['selected_frames'],'observations':anchors,'detector_runtime':manifest['detector_runtime'],'overlay_sha256':sha256_file(work/'overlay24.mp4'),'tracking_refinement':summary,'training':'not run','identity_accuracy':None,'baseline_render':'intentionally paused after complete raw inference; new staged motion view exported'}
        (W/'recording-receipt.json').write_text(json.dumps(receipt,indent=2))
        with zipfile.ZipFile(W/'recording-results.zip','w',zipfile.ZIP_STORED) as z:
            for folder in ('run','normalized','diagnostics'):
                for p in (W/folder).rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(W).as_posix())
            for p in work.iterdir():
                if p.is_file():z.write(p,'RESULTS.html' if p.name=='RESULTS.html' else 'profiles/review.md' if p.name=='review.md' else 'preview/'+p.name)
            z.write(W/'recording-receipt.json','recording-receipt.json')
    else:
        with zipfile.ZipFile(work/'review.zip','w',zipfile.ZIP_STORED) as z:
            for p in work.iterdir():
                if p.is_file() and p.name!='review.zip':z.write(p,p.name)
    emit('complete',status='complete',run_id=summary['run_id'],frames=frames,ids=tracker.next_id-1)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        emit('failed',error_category=type(exc).__name__)
        if len(sys.argv)>1 and json.loads(Path(sys.argv[1]).read_text()).get('mode')=='full':(W/'recording-receipt.json').write_text(json.dumps({'status':'failed','error_category':type(exc).__name__,'stage':'motion_refinement'}))
        raise SystemExit(1) from None
