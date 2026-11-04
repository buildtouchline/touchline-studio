"""Quarantine-only MOT audit and review frames; never marks labels complete."""
from pathlib import Path
import csv,hashlib,json,math,sys,zipfile
W=Path('/work')
def main():
    job=json.loads((W/'job.json').read_text());out=W/'review';out.mkdir()
    for name,digest in job['hashes'].items():
        with (W/name).open('rb') as s:
            if hashlib.file_digest(s,'sha256').hexdigest()!=digest:raise ValueError('Input hash mismatch')
    project=W/'audit-project';project.mkdir()
    with zipfile.ZipFile(W/'code.zip') as z:
        for e in z.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
    sys.path.insert(0,str(project/'src'))
    from touchline.video.decode import probe_video
    import av
    from PIL import ImageDraw
    av.logging.set_level(av.logging.PANIC)
    layout=json.loads((W/'layout.json').read_text())
    with (W/'sparse-recording.mp4').open('wb') as s:
        s.write((W/'head.bin').read_bytes());s.seek(layout['moov_offset']);s.write((W/'moov.bin').read_bytes());s.truncate(layout['source_size'])
    probe=probe_video(W/'sparse-recording.mp4');(out/'pxl-probe.json').write_text(json.dumps(probe,indent=2))
    print(json.dumps({'stage':'source_probe','status':'complete'}),flush=True)
    summaries=[];frames=[]
    for match in ('118575','118577','117093'):
        video=W/(match+'.mp4');info=probe_video(video);rows={};seen=set();invalid=0;total=0;scores=set()
        with (W/(match+'.txt')).open() as s:
            for line,row in enumerate(csv.reader(s),1):
                if len(row)!=10:raise ValueError('Unexpected MOT columns')
                frame,track=int(row[0]),int(row[1]);x,y,w,h,score=map(float,row[2:7]);total+=1;scores.add(score)
                if frame<0 or track<1 or (frame,track) in seen or not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0:raise ValueError('Invalid MOT row')
                seen.add((frame,track));outside=not(0<=x<x+w<=info['width'] and 0<=y<y+h<=info['height']);invalid+=outside
                rows.setdefault(frame,[]).append({'track_id':track,'bbox_xyxy':[x,y,x+w,y+h],'class_id':1,'outside_image':outside,'annotation_status':'upstream_player_gt_pending_review'})
        if min(rows)!=0 or max(rows)+1!=info['frame_count']:raise ValueError('Source-frame alignment mismatch')
        selected=[int(i*(len(rows)-1)/7) for i in range(8)]
        with av.open(str(video)) as container:
            stream=container.streams.video[0];origin=next(container.decode(stream)).pts
            for index in selected:
                target=origin+round(index/float(stream.average_rate)/float(stream.time_base));container.seek(target,stream=stream,backward=True)
                image=None
                for f in container.decode(stream):
                    n=round(float((f.pts-origin)*stream.time_base)*float(stream.average_rate))
                    if n==index:image=f.to_image().convert('RGB');break
                    if n>index:raise ValueError('Exact source frame missing')
                if image is None:raise ValueError('Source frame missing')
                name=match+'-'+str(index).zfill(6);raw=out/(name+'-raw.jpg');image.save(raw,quality=95)
                draw=ImageDraw.Draw(image)
                for obj in rows[index]:
                    box=obj['bbox_xyxy'];draw.rectangle(box,outline='red' if obj['outside_image'] else '#20ff70',width=3);draw.text((max(0,box[0]),max(0,box[1]-14)),str(obj['track_id']),fill='white',stroke_width=2,stroke_fill='black')
                image.save(out/(name+'-labels.jpg'),quality=92)
                frames.append({'match_id':match,'source_frame_index':index,'width':info['width'],'height':info['height'],'raw_image':raw.name,'labeled_image':name+'-labels.jpg','objects':rows[index],'split':'val' if match=='117093' else 'train','training_eligible':False,'person_labels_complete':False})
        summaries.append({'match':match,'video':info,'rows':total,'frames':len(rows),'ids':len({track for _,track in seen}),'outside_image_boxes':invalid,'score_values':sorted(scores),'training_eligible':False})
        print(json.dumps({'stage':'audit_'+match,'status':'complete','images':8}),flush=True)
    result={'status':'quarantined','source_revision':job['source_revision'],'matches':summaries,'frames':frames,'coverage':'players and goalkeepers only; referees/unlabeled people are not guaranteed','training_eligible':False,'human_review':'required','validation_policy':'Existing570-image SoccerNet validation unchanged; SoccerTrack117093 withheld, not train'}
    (out/'intake-review.json').write_text(json.dumps(result,indent=2))
    html='<!doctype html><meta charset="utf-8"><title>SoccerTrack intake review</title><style>body{background:#111;color:white;font:18px system-ui}img{width:100%}section{margin:35px 0}a{color:skyblue}</style><h1>SoccerTrack: incomplete person labels</h1><p>Green boxes are upstream player/GK labels. Look for omitted referees, sideline people, clipped/occluded players and box alignment. These frames are not training eligible. Source frames remain25FPS; no24FPS index conversion is applied to labels.</p>'
    for f in frames:html+='<section><h2>'+f['match_id']+' frame '+str(f['source_frame_index'])+' ('+f['split']+')</h2><img src="'+f['labeled_image']+'"><a href="'+f['raw_image']+'">Raw image for correction</a></section>'
    (out/'index.html').write_text(html)
    with zipfile.ZipFile(W/'gpu-results.zip','w',zipfile.ZIP_STORED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(W).as_posix())
    print(json.dumps({'stage':'complete','status':'complete'}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:print(json.dumps({'stage':'failed','error_category':type(exc).__name__}));raise SystemExit(1) from None
