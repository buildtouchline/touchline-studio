"""CPU-only frame preparation and visual SAM disagreement review; no secrets."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,os,re,subprocess,sys,zipfile


def main(mode):
    w=Path('/work')
    if mode=='prepare':
        job=json.loads((w/'job.json').read_text())
        for name,digest in job['hashes'].items():
            with (w/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Input hash mismatch')
        p=w/'sam-project';p.mkdir()
        with zipfile.ZipFile(w/'code.zip') as z:
            for e in z.infolist():
                t=(p/e.filename).resolve()
                if not t.is_relative_to(p) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
                t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(z.read(e))
        os.chdir(p);sys.path.insert(0,str(p/'src'))
        counts={}
        for label,args in (('tests',[]),('artifacts',['artifacts'])):
            with (w/(label+'.log')).open('wb') as log:
                r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
            text=(w/(label+'.log')).read_text(errors='replace');found=re.findall(r'Ran (\d+) tests?',text)
            counts[label]={'exit_code':r.returncode,'tests':int(found[-1]) if found else 0}
            if r.returncode:raise RuntimeError('Remote checks failed')
        from touchline.sam_analysis import FRAME_INDICES
        from touchline.video.decode import decode_video
        student={i:[] for i in FRAME_INDICES}
        for line in (w/'raw_proposals.jsonl').read_text().splitlines():
            row=json.loads(line);index=row['frame']['index']
            if index in student and row['score']>=.5:student[index].append(row)
        frames=[];out=w/'frames';out.mkdir()
        for frame in decode_video(w/'video24.mp4','sam-analysis'):
            if frame.index in student:
                image=frame.image.to_image().convert('RGB');image.save(out/f'{frame.index:06d}.jpg',quality=95)
                frames.append({'index':frame.index,'timestamp_s':frame.timestamp_s,'width':frame.width,'height':frame.height,'student':student[frame.index]})
            if frame.index>=max(FRAME_INDICES):break
        if len(frames)!=8:raise ValueError('Missing exact review frames')
        (w/'prepared.json').write_text(json.dumps({'frames':frames,'checks':counts}))
        print(json.dumps({'stage':'prepared','frames':8,'checks':counts}))
    elif mode=='render':
        sys.path.insert(0,str(w/'sam-project/src'))
        from touchline.sam_analysis import compare_boxes
        from PIL import Image,ImageDraw
        prepared=json.loads((w/'prepared.json').read_text());sam=json.loads((w/'sam.json').read_text());out=w/'review';out.mkdir()
        rows=[];thumbs=[]
        for f in prepared['frames']:
            key=str(f['index'])
            if key not in sam:continue
            teacher=sam[key]['objects'];r=compare_boxes(f['student'],teacher,f['width'],f['height'])
            image=Image.open(w/'frames'/f'{f["index"]:06d}.jpg').convert('RGB');draw=ImageDraw.Draw(image)
            for obj in f['student']:draw.rectangle(obj['bbox_xyxy'],outline='lime',width=3)
            for j in r['sam_only']:draw.rectangle(teacher[j]['bbox_xyxy'],outline='orange',width=4)
            draw.text((20,20),f'{f["timestamp_s"]:.1f}s | GREEN=detector >=0.5 ORANGE=SAM-only proposal | HUMAN REVIEW REQUIRED',fill='white',stroke_width=2,stroke_fill='black')
            name=f'{f["index"]:06d}-comparison.jpg';image.save(out/name,quality=94)
            image.thumbnail((900,300));thumbs.append(image.copy())
            rows.append({'frame_index':f['index'],'timestamp_s':f['timestamp_s'],'student_count':len(f['student']),
                'sam_count':len(teacher),'comparison':r,'image':name,'provider_usage':sam[key].get('usage')})
        sheet=Image.new('RGB',(1800,300*((len(thumbs)+1)//2)),'black')
        for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*900,(i//2)*300))
        sheet.save(out/'contact-sheet.jpg',quality=94)
        summary={'status':'complete' if len(rows)==8 else 'partial','frames':rows,'checks':prepared['checks'],
            'sam_only_candidates':sum(len(r['comparison']['sam_only']) for r in rows),
            'training_eligible':False,'identity_accuracy':None,'billing':'unreconciled','promotion':'not performed'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SAM 3.1 second-pass analysis</h1><p>Orange boxes are SAM-only candidates, not confirmed missed players. Green boxes are detector proposals at score ≥0.5. No training, identity assignment or promotion.</p>'+''.join('<h2>'+str(r['timestamp_s'])+' seconds</h2><img width="1400" src="'+r['image']+'">' for r in rows))
        with zipfile.ZipFile(w/'sam-review.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(w).as_posix())
        print(json.dumps({'stage':'rendered','frames':len(rows),'sam_only_candidates':summary['sam_only_candidates']}))
    else:raise ValueError('Unknown phase')


if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException as exc:print(json.dumps({'stage':'failed','error_category':type(exc).__name__}));raise SystemExit(1) from None
