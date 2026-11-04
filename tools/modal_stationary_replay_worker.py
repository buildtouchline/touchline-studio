"""CPU-only replay of frozen native ResNet proposals; no detector or training."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import hashlib,json,os,subprocess,sys,time,zipfile

W=Path('/work/replay')

def emit(item):
    temporary=W/'state.tmp';temporary.write_text(json.dumps(item));temporary.replace(W/'worker-state.json')
    print(json.dumps(item),flush=True)


def main():
    W.mkdir();job=json.loads(Path('/work/replay-job.json').read_text());project=W/'project';project.mkdir()
    for name,digest in job['inputs'].items():
        with (Path('/work')/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Frozen replay input changed')
    with zipfile.ZipFile('/work/code.zip') as archive:
        for e in archive.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    for name,args in [('tests',[]),('artifacts',['artifacts'])]:
        with (W/(name+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if r.returncode:raise RuntimeError('Replay regression gate failed')
    import av,cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from touchline.stationary_tracking import StationaryTracker
    from touchline.motion_tracking import appearance,iou,estimate_motion
    from touchline.byte_tracking import ByteTrackTracker
    from touchline.contracts import Box,Detection
    from touchline.video.decode import decode_video,probe_video
    cv2.setNumThreads(1)
    info=probe_video('/work/video24.mp4')
    if info['frame_count']!=5760 or info['average_fps']!=24 or [info['width'],info['height']]!=[4096,1080]:raise ValueError('Exact reviewed footage required')
    proposals=defaultdict(list)
    with Path('/work/raw_proposals.jsonl').open() as f:
        for line in f:
            r=json.loads(line);proposals[r['frame_index']].append(r)
    if set(proposals)!=set(range(0,5760,4)):raise ValueError('Complete six-Hz proposals required')
    teacher=json.loads(Path('/work/sam-proposals.json').read_text());teacher={r['frame_index']:r['sam_objects'] for r in teacher}
    trackers={'motion':StationaryTracker(4096,1080),'sam_motion':StationaryTracker(4096,1080,teacher_births_only=True)}
    control=ByteTrackTracker(6,activation_threshold=.35,max_gap_s=3.)
    streams={name:(W/(name+'-observations.jsonl')).open('w') for name in [*trackers,'bytetrack_control']}
    frames={name:{} for name in streams};unassigned=defaultdict(int);frame_metadata={};start=time.monotonic();previous=None
    for frame in decode_video('/work/video24.mp4',job['video_sha256'],decoder_threads=4):
        data=[]
        small=frame.image.reformat(width=2048,height=540,format='rgb24').to_ndarray()
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        motion={}
        if previous is not None:
            for name,tracker in trackers.items():
                boxes={t.session_id:tracker._box(t.state) for t in tracker.tracks.values() if t.session_id is not None and frame.timestamp_s-t.last_anchor<=2.}
                _,_,_,motion[name]=estimate_motion(previous,gray,boxes,4096)
        previous=gray
        if frame.index%4==0:
            rgb=frame.image.to_ndarray(format='rgb24')
            closest=min(teacher,key=lambda k:abs(k-frame.index));near=abs(closest-frame.index)<=12
            for r in proposals[frame.index]:
                if r['pitch_scope']=='outside_estimated_pitch':continue
                feature=appearance(rgb,r['bbox_xyxy'])
                data.append({'bbox_xyxy':r['bbox_xyxy'],'score':r['score'],'feature':feature,
                    'teacher_supported':near and any(iou(r['bbox_xyxy'],p['bbox_xyxy'])>=.25 for p in teacher[closest])})
            control_rows=control.update(frame.timestamp_s,[Detection(Box(*d['bbox_xyxy']),d['score']) for d in data])
            rows=[{'track_id':r['local_track_id'],'bbox_xyxy':r['bbox_xyxy'],'evidence':'detector_anchor','identity_confirmed':False} for r in control_rows]
            frames['bytetrack_control'][frame.index]=rows
            for r in rows:streams['bytetrack_control'].write(json.dumps({'frame_index':frame.index,'timestamp_s':frame.timestamp_s,**r})+'\n')
            frame_metadata[frame.index]=frame.metadata()
        for name,tracker in trackers.items():
            rows=tracker.update(frame.timestamp_s,data,foreground=motion.get(name))
            frames[name][frame.index]=rows
            if frame.index%4==0:
                for r in rows:streams[name].write(json.dumps({'frame_index':frame.index,'timestamp_s':frame.timestamp_s,**r})+'\n')
                unassigned[name]+=len(tracker.unassigned)
        if frame.index%480==0:
            emit({'stage':'replay','frames':frame.index+1,'ids':{k:t.next_id-1 for k,t in trackers.items()}})
    for stream in streams.values():stream.close()
    # The reference is read only AFTER both fixed recipes finish. It is never a
    # tracker input, birth seed, appearance feature or identity assignment.
    with (W/'comparison.log').open('wb') as log:
        p=subprocess.run([sys.executable,'-I','/work/evaluate-replay.py'],stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if p.returncode:raise RuntimeError('Independent provisional comparison failed')
    report=json.loads(Path('/work/replay-comparison.json').read_text())
    candidates={k:report['results'][k]['tracking_at_common_6hz_frames'] for k in trackers}
    selected=max(candidates,key=lambda k:candidates[k]['global_identity_f1_reference_scope'])
    baseline=report['results']['current_resnet']['tracking_at_common_6hz_frames']
    quality=candidates[selected]
    improved=quality['global_identity_f1_reference_scope']>baseline['global_identity_f1_reference_scope']+.02 and quality['recall_reference_scope']>=baseline['recall_reference_scope']
    # Evidence review selection on this same development clip, not promotion or
    # an independently held-out test result. Persist every recipe's score.
    summary={'status':'complete','selection':selected,'improved_on_development_clip':improved,
             'selection_scope':'Two predeclared development recipes, same upstream reference; not held-out acceptance',
             'quality_gate_accepted':False,'source_sha256':job['video_sha256'],'frames':5760,
             'detector_inference':'not run; cached frozen ResNet proposals','training':'not run','gpu':None,
             'unassigned_events':dict(unassigned),'temporary_session_ids':{k:t.next_id-1 for k,t in trackers.items()},
             'bytetrack_control_note':'Diagnostic control may allocate more than24 internal IDs; never selected as capped session output',
             'metrics':quality,'elapsed_s':round(time.monotonic()-start,2)}
    (W/'summary.json').write_text(json.dumps(summary,indent=2));(W/'comparison.json').write_text(json.dumps(report,indent=2))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',21)
    samples={round(i*5759/5):i for i in range(6)};sheet=Image.new('RGB',(1920,870),'black');gallery=Image.new('RGB',(1440,1440),'#151515');seen=set();last=[]
    with av.open(str(W/'overlay24.mp4'),'w') as encoded,(W/'canonical-observations.jsonl').open('w') as canonical:
        stream=encoded.add_stream('libx264',rate=24);stream.width=4096;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'21','preset':'veryfast'};stream.codec_context.thread_count=2
        for frame in decode_video('/work/video24.mp4',job['video_sha256'],decoder_threads=4):
            image=frame.image.to_image().convert('RGB');draw=ImageDraw.Draw(image)
            anchor=frame.index in frames[selected]
            if anchor:last=frames[selected][frame.index]
            for row in last:
                tid=row['track_id'];box=row['bbox_xyxy'];observed=anchor and row['evidence']=='detector_anchor';color=((tid*73)%190+60,(tid*131)%190+60,(tid*47)%190+60)
                draw.rectangle(box,outline=color,width=3 if observed else 2);draw.text((box[0],max(65,box[1]-25)),f'T{tid} {"OBS" if observed else "PRED"}',font=font,fill=color,stroke_width=2,stroke_fill='black')
                if observed:
                    canonical.write(json.dumps({'schema_version':'touchline.observation.v1','run_id':job['attempt_id'],'frame':frame.metadata(),'synthetic':False,'local_track_id':tid,'bbox_xyxy':box,'detector_score':row['detector_score'],'association_state':'tentative','player_id':None})+'\n')
                if tid not in seen and observed:
                    crop=image.crop(tuple(map(int,box)));crop.thumbnail((220,270));slot=tid-1;x=slot%6*240;y=slot//6*360;gallery.paste(crop,(x+(240-crop.width)//2,y));ImageDraw.Draw(gallery).text((x+5,y+290),f'T{tid}',font=font,fill=color);seen.add(tid)
            draw.rectangle((0,0,4096,61),fill='black');draw.text((10,0),f'{frame.timestamp_s:.2f}s | CPU tracking repair: {selected} | frozen ResNet detections; no model rerun',font=font,fill='white');draw.text((10,30),'OBS=measured detector anchor; PRED=prediction/held overlay. Temporary IDs are not jersey numbers. Development review only.',font=font,fill='white')
            if frame.index in samples:
                slot=samples[frame.index];thumb=image.copy();thumb.thumbnail((960,270));sheet.paste(thumb,(slot%2*960,slot//2*290))
            output=av.VideoFrame.from_image(image);output.pts=frame.index;output.time_base=Fraction(1,24)
            for packet in stream.encode(output):encoded.mux(packet)
            if frame.index%960==0:emit({'stage':'render','frames':frame.index+1})
        for packet in stream.encode():encoded.mux(packet)
    gallery.save(W/'players.jpg',quality=92);sheet.save(W/'contact-sheet.jpg',quality=92)
    with zipfile.ZipFile(W/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in W.iterdir():
            if p.is_file() and p.suffix in {'.json','.jsonl','.jpg','.log'}:z.write(p,p.name)
    emit({'stage':'complete','status':'complete','selected':selected,'improved':improved})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if W.exists():(W/'failure.json').write_text(json.dumps({'error_category':type(exc).__name__,'message':str(exc)[:1000]}))
        if W.exists():emit({'stage':'failed','error_category':type(exc).__name__})
        raise SystemExit(1) from None
