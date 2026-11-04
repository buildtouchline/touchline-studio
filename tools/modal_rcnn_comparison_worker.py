"""Same-frame R-CNN comparison template; explicit offline weights, no promotion."""
from pathlib import Path
import hashlib,json,time,zipfile

def main():
    import av,torch,torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.transforms.functional import to_tensor
    from PIL import Image,ImageDraw
    root=Path('/work');job=json.loads((root/'rcnn-comparison-job.json').read_text())
    full=job.get('scope')=='full_recording_review'
    work=root/('rcnn-full-review' if full else 'rcnn-comparison');work.mkdir(exist_ok=False)
    weights=root/'rcnn-comparison-weights.pth'
    with weights.open('rb') as src:digest=hashlib.file_digest(src,'sha256').hexdigest()
    if digest!=job['weights_sha256'] or not digest.startswith('dd69338a'):raise ValueError('Weight digest mismatch')
    manifest=json.loads((root/'run/manifest.json').read_text())
    if manifest['status']!='complete' or manifest['detector']['checkpoint']!=job['baseline_sha256']:raise ValueError('Complete original baseline required')
    torch.set_num_threads(4)
    model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,num_classes=91,min_size=1080,max_size=4096,box_score_thresh=.1)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.eval()
    indices={int((manifest['decoded_frames']-1)*i/7)//4*4 for i in range(8)} if full else {0,60,120,180,240,300,360,420};baseline={}
    with (root/'run/raw_proposals.jsonl').open() as stream:
        for line in stream:
            item=json.loads(line);index=item['frame']['index']
            if index in indices:baseline.setdefault(index,[]).append(item)
            if index>max(indices):break
    rows=[];panels=[];start=time.monotonic()
    with av.open(str(root/'normalized/video24.mp4')) as reader:
        stream=reader.streams.video[0]
        from fractions import Fraction
        for index in sorted(indices):
            reader.seek(int(Fraction(index,24)/stream.time_base),stream=stream,backward=True,any_frame=False)
            frame=None
            for candidate in reader.decode(video=0):
                actual=round(candidate.pts*candidate.time_base*24)
                if actual==index:frame=candidate;break
                if actual>index:break
            if frame is None:raise ValueError('Exact selected frame not found')
            image=frame.to_image().convert('RGB');then=time.monotonic()
            with torch.inference_mode():prediction=model([to_tensor(image)])[0]
            proposals=[{'bbox_xyxy':b,'score':s,'label':l} for b,s,l in zip(prediction['boxes'].tolist(),prediction['scores'].tolist(),prediction['labels'].tolist()) if l==1]
            old=baseline.get(index,[])
            # Coverage uses the same original full-resolution pixel coordinates.
            baseline_boxes=old
            record={'frame_index':index,'timestamp_s':index/24,'r50_proposals':proposals,'baseline_proposals':baseline_boxes,'r50_inference_s':round(time.monotonic()-then,3)}
            rows.append(record);parts=[]
            for title,boxes in [('Trained MobileNet baseline',baseline_boxes),('ResNet50 COCO candidate',proposals)]:
                thumb=image.resize((960,540));canvas=Image.new('RGB',(960,574),'white');canvas.paste(thumb,(0,34));draw=ImageDraw.Draw(canvas)
                visible=[b for b in boxes if b['score']>=.5]
                draw.text((12,10),f'{title} | t={index/24:.2f}s | person >=0.5: {len(visible)}',fill='black')
                for b in visible:
                    x1,y1,x2,y2=b['bbox_xyxy'];box=(x1*960/image.width,y1*540/image.height+34,x2*960/image.width,y2*540/image.height+34)
                    draw.rectangle(box,outline='lime',width=2);draw.text((box[0],max(34,box[1]-12)),f"{b['score']:.2f}",fill='yellow')
                parts.append(canvas)
            pair=Image.new('RGB',(1920,574));pair.paste(parts[0],(0,0));pair.paste(parts[1],(960,0));pair.save(work/f'comparison-{index:06d}.jpg',quality=92);panels.append(pair)
            print(json.dumps({'stage':'rcnn_comparison','frame':index,'candidate_person_boxes':sum(p['score']>=.5 for p in proposals)}),flush=True)
    if len(rows)!=8 or any(not r['baseline_proposals'] for r in rows):raise ValueError('Expected eight paired detector frames')
    sheet=Image.new('RGB',(1920,574*4),'white')
    for i,panel in enumerate(panels[::2]):sheet.paste(panel,(0,i*574))
    sheet.save(work/'comparison-sheet.jpg',quality=90)
    with (work/'proposals.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    receipt={'status':'complete','architecture':'fasterrcnn_resnet50_fpn_v2','checkpoint_sha256':digest,'baseline_checkpoint_sha256':manifest['detector']['checkpoint'],'source_sha256':manifest['source']['id'],'frames':len(rows),'frame_indices':sorted(indices),'device':str(next(model.parameters()).device),'torch':torch.__version__,'torchvision':torchvision.__version__,'min_size':1080,'max_size':4096,'elapsed_s':round(time.monotonic()-start,2),'ap50':None,'idf1':None,'promotion':'not performed','training':'not run','scope':'eight matched frames spread across full recording; review only' if full else 'eight matched frames in first twenty seconds; review only'}
    (work/'receipt.json').write_text(json.dumps(receipt,indent=2))
    (work/'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>R-CNN same-frame comparison</h1><p>Left trained MobileNet; right generic COCO ResNet50. Boxes >=0.5. Detection counts are not accuracy. No identity or model promotion claim.</p>'+''.join(f'<img width="1600" src="comparison-{i:06d}.jpg">' for i in sorted(indices)))
    with zipfile.ZipFile(work/'review.zip','w',zipfile.ZIP_STORED) as archive:
        for p in work.iterdir():
            if p.is_file() and p.name!='review.zip':archive.write(p,p.name)
    print(json.dumps({'stage':'complete','status':'complete','frames':len(rows)}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'stage':'rcnn_comparison','status':'failed','error_category':type(exc).__name__}),flush=True)
        raise SystemExit(1) from None
