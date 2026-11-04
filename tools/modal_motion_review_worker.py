"""Replay saved proposals with motion/camera cues and explicit pixel trajectories."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from collections import defaultdict
from fractions import Fraction
import hashlib,json,math,sys,time,uuid,zipfile
W=Path('/work')
def main():
    work=W/'motion-preview20-v2';work.mkdir();job=json.loads((W/'motion-job-v2.json').read_text());project=W/'motion-project-v2';project.mkdir()
    with zipfile.ZipFile(W/'motion-code-v2.zip') as z:
        for e in z.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
    sys.path.insert(0,str(project/'src'))
    import av,cv2,numpy as np
    from PIL import ImageDraw,ImageFont
    from touchline.motion_tracking import MotionTracker,appearance,estimate_motion
    from touchline.video.decode import decode_video,probe_video
    from touchline.data.manifest import sha256_file
    cv2.setNumThreads(2)
    source=W/'pxl-preview20/source/video24.mp4';old=W/'pxl-preview20/run';manifest=json.loads((old/'manifest.json').read_text())
    if sha256_file(source)!=manifest['source']['id']:raise ValueError('Exact source changed')
    proposals=defaultdict(list)
    for line in (old/'raw_proposals.jsonl').read_text().splitlines():
        row=json.loads(line);proposals[row['frame']['index']].append(row)
    info=probe_video(source);tracker=MotionTracker(info['width'],info['height']);previous=None;seen=set();records=0;anchored=0;camera_verified=0;points=defaultdict(list);started=time.monotonic()
    scale=info['width']/960;small_height=round(info['height']/scale);display_width=1920;display_height=round(info['height']*display_width/info['width']);display_height-=display_height%2;display_scale=display_width/info['width']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
    with av.open(str(work/'motion-overlay24.mp4'),'w') as encoded,(work/'pixel-trajectories.jsonl').open('w') as trace:
        stream=encoded.add_stream('libx264',rate=24);stream.width=display_width;stream.height=display_height;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'veryfast'};stream.codec_context.thread_count=2
        for frame in decode_video(source,'motion-review'):
            rgb=frame.image.reformat(width=960,height=small_height,format='rgb24').to_ndarray();gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            camera=np.array([[1.,0.,0.],[0.,1.,0.]]);valid=False;quality=0.;flow={}
            if previous is not None:camera,valid,quality,flow=estimate_motion(previous,gray,{i:t.box for i,t in tracker.tracks.items() if frame.timestamp_s-t.last_anchor<=tracker.max_gap},info['width'])
            detections=[{'bbox_xyxy':r['bbox_xyxy'],'score':r['score'],'feature':appearance(rgb,np.asarray(r['bbox_xyxy'])/scale)} for r in proposals.get(frame.index,[])]
            rows=tracker.update(frame.timestamp_s,detections,camera=camera,camera_valid=valid,foreground=flow);previous=gray
            image=frame.image.reformat(width=display_width,height=display_height,format='rgb24').to_image();draw=ImageDraw.Draw(image)
            for row in rows:
                tid=row['track_id'];seen.add(tid);color=((tid*73)%190+60,(tid*131)%190+60,(tid*47)%190+60);box=[x*display_scale for x in row['bbox_xyxy']];draw.rectangle(box,outline=color,width=3)
                label='T'+str(tid)+(' OBS' if row['evidence']=='detector_anchor' else ' FLOW' if row['evidence']=='optical_flow_prediction' else ' PRED')
                x=max(0,box[0]);y=max(74,box[1]-30);draw.text((x,y),label,font=font,fill='white',stroke_width=2,stroke_fill='black')
                trail=[(p[0]*display_scale,p[1]*display_scale) for p in row['trail_xy']]
                if len(trail)>1:draw.line(trail,fill=color,width=3)
                points[tid].append({'frame':frame.index,'timestamp_s':frame.timestamp_s,'footpoint_xy':row['footpoint_xy'],'evidence':row['evidence']});anchored+=row['evidence']=='detector_anchor'
            draw.rectangle((0,0,display_width,70),fill='black');draw.text((8,4),f'{frame.timestamp_s:.3f}s | Motion-assisted temporary tracks | source pixels {info["width"]}x{info["height"]}',font=font,fill='white');draw.text((8,36),'OBS=detector; FLOW/PRED=propagated, not new detection. Identity accuracy unknown.',font=font,fill='white')
            trace.write(json.dumps({'frame_index':frame.index,'timestamp_s':frame.timestamp_s,'camera_affine':camera.tolist(),'camera_valid':valid,'camera_inlier_fraction':quality,'tracks':[{k:v for k,v in row.items() if k!='trail_xy'} for row in rows]})+'\n')
            output=av.VideoFrame.from_image(image);output.pts=frame.index;output.time_base=Fraction(1,24)
            for packet in stream.encode(output):encoded.mux(packet)
            records+=1;camera_verified+=valid
        for packet in stream.encode():encoded.mux(packet)
    baseline_ids=set()
    for line in (old/'observations.jsonl').read_text().splitlines():baseline_ids.add(json.loads(line)['local_track_id'])
    summary={'status':'complete','run_id':str(uuid.uuid4()),'baseline_run_id':manifest['run_id'],'checkpoint':manifest['detector']['checkpoint'],'source_sha256':manifest['source']['id'],
        'frames':records,'detector_samples':manifest['selected_frames'],'baseline_temporary_ids':len(baseline_ids),'motion_temporary_ids':len(seen),'detector_anchor_matches':anchored,'camera_verified_frames':camera_verified,
        'source_geometry':[info['width'],info['height']],'display_geometry':[display_width,display_height],'coordinate_units':'original source pixels; camera-compensated delta is an image-plane proxy, not metres',
        'identity_accuracy':None,'tracking_idf1':None,'promotion':'not performed','model_inference':'not rerun; same saved proposals','code_sha256':sha256_file(W/'motion-code-v2.zip'),'duration_s':round(time.monotonic()-started,2),
        'limitations':'Fewer IDs do not prove correct identities. Appearance cues can confuse teammates. Missing/ambiguous tracks are hidden after0.8s; no permanent identity is forced.'}
    (work/'summary.json').write_text(json.dumps(summary,indent=2));(work/'track-points.json').write_text(json.dumps(points))
    (work/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>PXL motion review</h1><video controls width="1200" src="motion-overlay24.mp4"></video><p>Temporary IDs; OBS=detector anchor; FLOW/PRED=propagation. Trails use source-pixel positions adjusted for estimated camera motion. IDF1 unknown.</p><a href="summary.json">Summary</a> | <a href="pixel-trajectories.jsonl">Pixel motion per frame</a>')
    with zipfile.ZipFile(work/'motion-review.zip','w',zipfile.ZIP_STORED) as z:
        for p in work.iterdir():
            if p.name!='motion-review.zip':z.write(p,p.name)
    print(json.dumps(summary))
if __name__=='__main__':
    try:main()
    except BaseException as exc:print(json.dumps({'status':'failed','error_category':type(exc).__name__}));raise SystemExit(1) from None
