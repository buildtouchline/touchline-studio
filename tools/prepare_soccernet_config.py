"""Draft a reviewed-input config from the supplied SoccerNet Sequence INI format.

Standalone: may be uploaded to an existing Colab runtime without reinstalling
Touchline. No downloads/training or automatic review attestations are performed.
"""
import argparse
from collections import Counter
import configparser
import csv
import hashlib
import json
from pathlib import Path
import re


HUMAN_ROLES = {'player', 'goalkeeper', 'goalkeepers', 'referee', 'staff'}


def _sequence(path):
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with Path(path).open(encoding='utf-8-sig') as handle:
        parser.read_file(handle)
    return dict(parser['Sequence'])


def prepare_config(root, output, bbox_origin=0, frame_stride=25, split_seed=42):
    """Create a draft with explicit metadata mappings; review flags stay false.

    bbox_origin=0 preserves coordinates in the inspected SoccerNet training export.
    A generic MOT-format statement is not evidence that this export needs a shift.
    Check each source's bounds/overlays. frame_stride=25 selects about 1 FPS at 25 FPS.
    """
    root = Path(root).resolve(); output = Path(output).resolve()
    if type(bbox_origin) is not int or bbox_origin not in (0, 1):
        raise ValueError('bbox_origin must be 0 or 1')
    if type(frame_stride) is not int or frame_stride <= 0:
        raise ValueError('positive integer frame_stride required')
    if output.exists():
        raise FileExistsError(f'Config already exists; reuse it or choose a new path: {output}')
    files = sorted(root.rglob('seqinfo.ini'))
    if not files:
        raise ValueError(f'No seqinfo.ini found under {root}; extract train.zip first')
    entries=[]; summaries=[]; seen=set(); classes=set()
    for seq_file in files:
        folder=seq_file.parent
        # This helper is intentionally for train.zip only, not official test/challenge.
        if 'train' not in folder.relative_to(root.parent).parts:
            raise ValueError(f'Expected an extracted train/ sequence, got {folder}')
        seq=_sequence(seq_file); game=_sequence(folder/'gameinfo.ini')
        sid=seq['name']; gid=game['gameID'].strip()
        if not gid or sid in seen or game.get('name') != sid:
            raise ValueError(f'Invalid/mismatched sequence or game metadata: {folder}')
        seen.add(sid)
        images=(folder/seq['imDir']).resolve(); gt=folder/'gt/gt.txt'
        if not images.is_relative_to(folder.resolve()) or not images.is_dir() or not gt.is_file():
            raise ValueError(f'Missing or out-of-root images/ground truth: {folder}')
        extension=seq['imExt']
        if extension.lower() not in ('.jpg','.jpeg','.png'):
            raise ValueError(f'Unsupported image extension: {extension}')
        frame_rate=float(seq['frameRate']); width=int(seq['imWidth']); height=int(seq['imHeight'])
        if frame_rate<=0 or width<=0 or height<=0:
            raise ValueError(f'Invalid source timing/geometry: {sid}')
        track_classes={};kept=set();ignored=set()
        for key,value in game.items():
            match=re.fullmatch(r'trackletID_(\d+)',key,re.IGNORECASE)
            if not match:continue
            label=value.split(';',1)[0].strip().casefold()
            if not label:raise ValueError(f'Empty tracklet class in {sid}')
            if label=='ball':ignored.add(label)
            # SoccerNet-Tracking paper, section 3: "other" includes medical
            # staff/coaches entering the field, so keep it for generic person.
            # Do not broaden this to arbitrary labels starting with "other".
            elif label=='other' or label.split()[0] in HUMAN_ROLES:kept.add(label)
            else:raise ValueError(f'Unhandled class {label!r} in {sid}; inspect before mapping')
            track_classes[match.group(1)]=label
        if not kept:raise ValueError(f'No person classes in {sid}')
        if int(game['num_tracklets'])!=len(track_classes):
            raise ValueError(f'Tracklet metadata count mismatch in {sid}')
        observed=set()
        with gt.open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.reader(handle):
                if not row:continue
                if len(row)!=10:raise ValueError(f'Expected ten-column GT in {sid}')
                observed.add(str(int(row[1])))
        if observed-set(track_classes):
            raise ValueError(f'GT track IDs have no metadata in {sid}: {observed-set(track_classes)}')
        group=f'SoccerNet-SN-Tracking-2023-game-{gid}'
        entries.append({'id':sid,'capture_group_id':group,'split':None,
            'images_root':str(images),'annotations':str(gt),'image_pattern':'{frame:06d}'+extension,
            'frame_stride':frame_stride,'bbox_origin':bbox_origin,'track_classes':track_classes,
            'person_track_classes':sorted(kept),'ignored_track_classes':sorted(ignored)})
        summaries.append({'sequence':sid,'game_id':gid,'frame_rate':frame_rate,
            'width':width,'height':height,'gt_track_ids':len(observed)})
        classes.update(kept|ignored)
    groups=sorted({e['capture_group_id'] for e in entries},
        key=lambda g:hashlib.sha256(f'{split_seed}:{g}'.encode()).hexdigest())
    if len(groups)<3:raise ValueError('At least three actual gameID groups are needed; do not split one game across partitions')
    holdout=max(1,len(groups)//5)
    assignments={g:('test' if i<holdout else 'val' if i<2*holdout else 'train') for i,g in enumerate(groups)}
    for entry in entries:entry['split']=assignments[entry['capture_group_id']]
    config={'release_id':'soccernet-person-internal-v1','dataset_version':'1','reviewer_id':'',
        'annotations_reviewed':False,'person_labels_complete':False,'training_eligible':False,
        'split_description':'Custom match-group holdouts from train.zip; not the official SoccerNet test/challenge benchmark.',
        'bbox_origin_basis':('Preserve zero-based pixels in the inspected SoccerNet export; confirm bounds/overlays.'
            if bbox_origin==0 else 'Caller explicitly selected one-based input; verify against this source export.'),
        'sequences':entries}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    report={'sequences':len(entries),'games':len(groups),'classes':sorted(classes),
        'source_fps_counts':dict(Counter(s['frame_rate'] for s in summaries)),
        'sequences_per_split':dict(Counter(e['split'] for e in entries)),
        'game_split_assignments':assignments,'source_metadata':summaries,
        'review_flags':'False until actual source/overlay/completeness/eligibility review.'}
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return config,report


def preview_config(path, max_games=3):
    """Return raw GT overlays for distinct games; never changes review flags."""
    from PIL import Image,ImageDraw
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    results=[];seen=set()
    for seq in config['sequences']:
        if seq['capture_group_id'] in seen:continue
        seen.add(seq['capture_group_id'])
        # Inspect several moments, not only the first frame of the first clip.
        for frame in (1,251,501):
            image_path=Path(seq['images_root'])/seq['image_pattern'].format(frame=frame)
            if not image_path.is_file():continue
            with Image.open(image_path) as raw:image=raw.convert('RGB')
            draw=ImageDraw.Draw(image);origin=seq['bbox_origin']
            with Path(seq['annotations']).open(encoding='utf-8-sig',newline='') as handle:
                for row in csv.reader(handle):
                    if not row or int(row[0])!=frame:continue
                    if seq['track_classes'][str(int(row[1]))] not in seq['person_track_classes']:continue
                    x,y,w,h=map(float,row[2:6]);draw.rectangle([x-origin,y-origin,x-origin+w,y-origin+h],outline='red',width=3)
            image.thumbnail((960,540));results.append((f"{seq['id']} / {seq['capture_group_id']} / frame {frame}",image))
        if len(seen)>=max_games:break
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bbox-origin',type=int,choices=[0,1],default=0)
    parser.add_argument('--frame-stride',type=int,default=25)
    args=parser.parse_args()
    _,report=prepare_config(args.root,args.output,args.bbox_origin,args.frame_stride)
    print(json.dumps(report,indent=2))
