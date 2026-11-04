"""Explicit SoccerNet MOT-to-COCO staging; never infer class from MOT placeholder columns."""
from ..diagnostics import diagnosed
import csv
import json
import math
from pathlib import Path
from .manifest import sha256_file


@diagnosed("data.stage")
def stage_soccernet(config_path, output):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not config.get('sequences'):
        raise ValueError('explicit sequences required')
    if config.get('annotations_reviewed') is not True or config.get('person_labels_complete') is not True or config.get('training_eligible') is not True:
        raise ValueError('review, completeness and intended-use eligibility must be confirmed')
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    imports=[];groups={};splits={};lineage=[];ids=set()
    for sequence in config['sequences']:
        sid=sequence['id']; group=sequence['capture_group_id']; split=sequence['split']
        if not sid or sid in ids or split not in ('train','val','test') or not group:
            raise ValueError('unique sequence ID, real capture group and explicit train/val/test split required')
        ids.add(sid)
        if group in splits and splits[group]!=split:
            raise ValueError('same match/capture cannot appear across splits')
        splits[group]=split
        images_root=(config_path.parent/sequence['images_root']).resolve()
        annotation_path=(config_path.parent/sequence['annotations']).resolve()
        # Source track_classes is inspected from that sequence's metadata, never global track IDs.
        track_classes=sequence['track_classes']
        origin=sequence.get('bbox_origin')
        if type(origin) is not int or origin not in (0,1):
            raise ValueError('bbox_origin must explicitly declare source pixel origin 0 or 1')
        kept={str(v) for v in sequence['person_track_classes']}
        ignored={str(v) for v in sequence.get('ignored_track_classes',[])}
        if not kept or kept&ignored:
            raise ValueError('explicit nonoverlapping person and ignored source classes required')
        stride=sequence.get('frame_stride',12)
        if type(stride) is not int or stride<=0:raise ValueError('positive frame_stride required')
        by_frame={};seen=set();unselected_rows=0;unselected_invalid_box_rows=0;unselected_invalid_box_examples=[]
        with annotation_path.open(encoding='utf-8-sig',newline='') as handle:
            for line_number,row in enumerate(csv.reader(handle),1):
                if not row: continue
                if len(row)!=10:raise ValueError('expected ten-column SoccerNet MOT ground truth')
                frame,track=int(row[0]),int(row[1]);x,y,w,h,score=map(float,row[2:7])
                if frame<1 or track<0 or score!=1:
                    raise ValueError(f'{sid}, line {line_number}, frame={frame}, track={track}: '
                                     f'positive frame, nonnegative track and GT score=1 required; row={row}')
                if (frame,track) in seen:raise ValueError('duplicate track in frame')
                seen.add((frame,track));name=track_classes.get(str(track))
                if name not in kept|ignored:raise ValueError(f'{sid}: explicit source class decision missing for track {track}')
                valid_box=all(math.isfinite(v) for v in (x,y,w,h)) and w>0 and h>0
                # Exclude the entire unselected frame before geometry validation.
                # An invalid unused row must not become a missing label in a used image.
                if (frame-1)%stride:
                    unselected_rows+=1
                    if not valid_box:
                        unselected_invalid_box_rows+=1
                        if len(unselected_invalid_box_examples)<20:
                            unselected_invalid_box_examples.append({'line':line_number,'frame':frame,'track':track,'row':row})
                    continue
                if not valid_box:
                    raise ValueError(f'{sid}, line {line_number}, frame={frame}, track={track}: '
                                     f'selected frame has nonfinite/nonpositive box geometry; row={row}')
                by_frame.setdefault(frame,[])
                if name in kept:by_frame[frame].append([x-origin,y-origin,w,h])
        if not by_frame:raise ValueError('stride selected no annotated frames')
        coco={'images':[],'annotations':[],'categories':[{'id':1,'name':'person'}]};aid=0
        for frame in sorted(by_frame):
            relative=sequence.get('image_pattern','{frame:06d}.jpg').format(frame=frame)
            path=(images_root/relative).resolve()
            if not path.is_relative_to(images_root) or not path.is_file():raise ValueError('frame image missing or outside image root')
            from PIL import Image
            with Image.open(path) as image:width,height=image.size
            coco['images'].append({'id':frame,'file_name':relative,'width':width,'height':height})
            groups[sid+'/'+relative]=group
            for box in by_frame[frame]:
                aid+=1;coco['annotations'].append({'id':aid,'image_id':frame,'category_id':1,'bbox':box,'source_frame_id':frame})
        if not coco['images']:raise ValueError('stride selected no annotated frames')
        # File name derives from enumeration, not a potentially unsafe source ID.
        target=output/f'annotations-{len(imports):03d}.json'
        target.write_text(json.dumps(coco)+'\n',encoding='utf-8')
        imports.append({'id':sid,'format':'coco_boxes','images_root':str(images_root),'annotations':str(target)})
        lineage.append({'sequence':sid,'capture_group_id':group,'split':split,'annotation_sha256':sha256_file(annotation_path),
            'selected_images':len(coco['images']),'source_frame_ids':[i['id'] for i in coco['images']],
            'frame_stride':stride,'unselected_rows':unselected_rows,
            'unselected_invalid_box_rows':unselected_invalid_box_rows,
            'unselected_invalid_box_examples':unselected_invalid_box_examples})
    etl={k:config[k] for k in ('release_id','dataset_version','reviewer_id','annotations_reviewed','person_labels_complete','training_eligible')}
    etl.update(fixture_only=False,class_aliases={'person':1},ignored_classes=[],capture_groups=groups,split_assignments=splits,inputs=imports)
    (output/'etl-config.json').write_text(json.dumps(etl,indent=2)+'\n',encoding='utf-8')
    (output/'staging-report.json').write_text(json.dumps({'source_config_sha256':sha256_file(config_path),'sequences':lineage,
        'notes':'Frames without a GT row are omitted, never assumed negative. Track ID is sequence-local. No timestamps invented.'},indent=2)+'\n',encoding='utf-8')
    return output/'etl-config.json'
