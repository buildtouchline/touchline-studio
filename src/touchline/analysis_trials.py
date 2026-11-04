"""Review-only tracking recipes. Diagnostic proxies never establish identity accuracy."""
from collections import defaultdict
import json
import math
from pathlib import Path

RECIPES=(
    {'name':'baseline','activation_threshold':.5,'max_gap_s':1.5},
    {'name':'lower_activation','activation_threshold':.3,'max_gap_s':1.5},
    {'name':'longer_recovery','activation_threshold':.4,'max_gap_s':3.0},
)


def validate_recipe(value):
    if not isinstance(value,dict) or set(value)!={'name','activation_threshold','max_gap_s'}:
        raise ValueError('Exact tracking recipe schema required')
    if value['name'] not in {r['name'] for r in RECIPES}:
        raise ValueError('Registered recipe name required')
    for name,low,high in (('activation_threshold',.3,.5),('max_gap_s',1.5,3.0)):
        n=value[name]
        if type(n) not in (int,float) or not math.isfinite(n) or not low<=n<=high:
            raise ValueError('Tracking trial parameter outside reviewed bounds')
    return dict(value)


def summarize_run(run_dir):
    """Count fragmentation proxies; no identity truth or physical metrics inferred."""
    run_dir=Path(run_dir)
    manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    tracks=defaultdict(list);frames=defaultdict(set);count=0
    with (run_dir/'observations.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            tid=row['local_track_id'];frame=row['frame']['index']
            if tid in frames[frame]:raise ValueError('Duplicate track ID in frame')
            frames[frame].add(tid);tracks[tid].append(row['frame']['timestamp_s']);count+=1
    lengths=sorted(len(rows) for rows in tracks.values())
    return {'run_id':manifest['run_id'],'observations':count,'temporary_tracks':len(tracks),
        'short_tracks_under_1s':sum(max(rows)-min(rows)<1 for rows in tracks.values()),
        'median_observations_per_track':lengths[len(lengths)//2] if lengths else 0,
        'frames_with_tracks':len(frames),'decoded_frames':manifest['decoded_frames'],
        'selected_frames':manifest['selected_frames'],'identity_accuracy':None,
        'requires_human_review':True,'promotion':'not performed'}
