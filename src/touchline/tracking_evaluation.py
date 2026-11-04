"""Explicit reviewed-frame identity evaluation, never an unlabeled accuracy estimate."""
import json
from pathlib import Path
from collections import defaultdict

from .contracts import Box
from .data.manifest import sha256_file


def evaluate_tracking(run_dir,reference_path):
    """Global one-to-one IDF1 at IoU 0.5 on all explicitly reviewed sampled frames.

    Implements identity-count optimization, not TrackEval dataset preprocessing.
    Evaluated frame IDs must be identical when comparing runs at different rates.
    Unsampled frames are rejected, not silently omitted. Empty reviewed frames
    are retained and penalize false positives. Roster identity is never inferred.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    run_dir,reference_path=Path(run_dir),Path(reference_path)
    manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    if reference.get('schema')!='touchline.tracking-reference.v1' or reference.get('reviewed') is not True or not isinstance(reference.get('reviewer_id'),str) or not reference['reviewer_id'].strip():
        raise ValueError('A genuinely reviewed tracking reference with reviewer ID is required')
    if reference.get('source_sha256')!=manifest['source']['id']:
        raise ValueError('Reference must match the exact analyzed video hash and normalized frame numbering')
    frames=reference.get('frames')
    if not isinstance(frames,list) or not frames or len(frames)>10000:
        raise ValueError('Provide 1 to 10000 explicitly reviewed frames')
    gt={}
    for frame in frames:
        index=frame['frame_index']
        if type(index) is not int or index<0 or index in gt or not isinstance(frame.get('people'),list):
            raise ValueError('Unique nonnegative reviewed frame indices and people lists required')
        people={}
        for row in frame['people']:
            identity=row.get('reference_id')
            if not isinstance(identity,str) or not identity.strip() or identity in people:
                raise ValueError('Unique nonempty person reference IDs required within each frame')
            people[identity]=Box(*row['bbox_xyxy'])
        gt[index]=people
    coverage={}
    with (run_dir/'coverage.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            item=json.loads(line);index=item['frame']['index']
            if index in gt:coverage[index]=item
    for index,people in gt.items():
        if index not in coverage or coverage[index]['state']=='unsampled_by_sampler':
            raise ValueError('Every reviewed frame must be an actually sampled frame; select a common schedule explicitly')
        f=coverage[index]['frame']
        if any(not box.fits(f['width'],f['height']) for box in people.values()):
            raise ValueError('Reference boxes must use valid normalized-video source pixel coordinates')
    predicted=defaultdict(dict)
    with (run_dir/'observations.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);index=row['frame']['index']
            if index not in gt:continue
            tid=row['local_track_id']
            if type(tid) is not int or tid<0 or tid in predicted[index]:
                raise ValueError('Predicted track IDs must be unique per frame')
            box=Box(*row['bbox_xyxy']);f=coverage[index]['frame']
            if not box.fits(f['width'],f['height']):raise ValueError('Invalid prediction box')
            predicted[index][tid]=box
    gids=sorted({identity for people in gt.values() for identity in people})
    pids=sorted({tid for people in predicted.values() for tid in people})
    if len(gids)+len(pids)>2000:raise ValueError('Use a bounded reviewed clip for identity evaluation')
    gi={k:i for i,k in enumerate(gids)};pi={k:i for i,k in enumerate(pids)}
    potential=np.zeros((len(gids),len(pids)),dtype=np.int64)
    gt_count=pred_count=spatial_tp=changes=0
    last_assignment={};matched_ids=defaultdict(set)
    for index in sorted(gt):
        truth=list(gt[index].items());prediction=list(predicted[index].items())
        gt_count+=len(truth);pred_count+=len(prediction)
        ious=np.asarray([[a.iou(b) for _,b in prediction] for _,a in truth],dtype=float).reshape(len(truth),len(prediction))
        for a,b in zip(*np.nonzero(ious>=.5)):
            potential[gi[truth[a][0]],pi[prediction[b][0]]]+=1
        if truth and prediction:
            costs=np.where(ious>=.5,1-ious,1e6)
            rows,cols=linear_sum_assignment(costs)
            for a,b in zip(rows,cols):
                if ious[a,b]<.5:continue
                spatial_tp+=1;identity=truth[a][0];tid=prediction[b][0]
                if identity in last_assignment and last_assignment[identity]!=tid:changes+=1
                last_assignment[identity]=tid;matched_ids[identity].add(tid)
    idtp=0
    if potential.size:
        rows,cols=linear_sum_assignment(potential,maximize=True)
        idtp=int(potential[rows,cols].sum())
    idfn=gt_count-idtp;idfp=pred_count-idtp
    idf1=2*idtp/(gt_count+pred_count) if gt_count+pred_count else None
    return {'schema':'touchline.tracking-evaluation.v1','run_id':manifest['run_id'],
        'reference_sha256':sha256_file(reference_path),'source_sha256':manifest['source']['id'],
        'checkpoint':manifest['detector'].get('checkpoint'),'tracker':manifest.get('tracker'),
        'scope':'explicitly-reviewed-sampled-frames','frame_indices':sorted(gt),'iou_threshold':.5,
        'gt_detections':gt_count,'predicted_detections':pred_count,'idtp':idtp,'idfp':idfp,'idfn':idfn,'idf1':idf1,
        'detection_precision':spatial_tp/pred_count if pred_count else None,
        'detection_recall':spatial_tp/gt_count if gt_count else None,
        'identity_changes_among_reviewed_matches':changes,
        'predicted_ids_per_reference':{identity:sorted(matched_ids[identity]) for identity in gids},
        'run_status':manifest['status'],'accuracy_on_unreviewed_frames':'unknown',
        'minimum_idf1_target':.75,'desired_idf1_target':.80,
        'synthetic':bool(manifest.get('synthetic',False)),
        'minimum_target_met_on_reviewed_scope':not manifest.get('synthetic',False) and idf1 is not None and idf1>=.75,
        'desired_target_met_on_reviewed_scope':not manifest.get('synthetic',False) and idf1 is not None and idf1>=.80,
        'full_video_accuracy_verified':False,'roster_identity_verified':False}
