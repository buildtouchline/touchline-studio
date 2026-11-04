"""Sandbox-only provisional comparison on upstream player-only reference labels.

Copied unchanged metric functions from the initial same-recording comparison.
Not the reviewed-reference evaluator, TrackEval preprocessing or an acceptance gate.
The replay worker invokes this only after the frozen tracking recipes complete.
"""
from pathlib import Path
from collections import defaultdict,Counter
from contextlib import redirect_stdout
import csv,json,time,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
w=Path('/work/panorama'); deadline=time.monotonic()+300
while not (w/'summary.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Full recording not ready')
 time.sleep(2)
summary=json.loads((w/'summary.json').read_text())
if summary['frames']!=5760:raise ValueError('Full source required')
frames=list(range(0,5760,4));mapping={}
with Path('/work/frame-map.jsonl').open() as f:
 for line in f:
  r=json.loads(line);mapping[r['output_frame_index']]=r['source_frame_index']
gt=defaultdict(list)
with Path('/work/compare-reference.txt').open() as f:
 for r in csv.reader(f):
  fi,tid=int(r[0]),int(r[1]);x,y,bw,bh=map(float,r[2:6]);box=[x,y,x+bw,y+bh]
  if not 0<=x<x+bw<=4096 or not 0<=y<y+bh<=1080:raise ValueError('Invalid reference box; no silent repair')
  gt[fi].append((tid,box))
truth={fi:gt[mapping[fi]] for fi in frames}
if any(len(r)!=22 for r in truth.values()):raise ValueError('Expected22 upstream player/GK labels')
def load(path,tracks=False):
 rows=defaultdict(list)
 with path.open() as f:
  for line in f:
   r=json.loads(line);index=r['frame_index'] if 'frame_index' in r else r['frame']['index']
   if index%4 or index>=5760:continue
   score=r.get('score',r.get('detector_score',1.))
   rows[index].append((r.get('track_id',r.get('local_track_id')),r['bbox_xyxy'],score))
 return rows
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
 return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def evaluate(pred,tracking=False,threshold=.5):
 gids=sorted({tid for rows in truth.values() for tid,_ in rows});pids=sorted({tid for rows in pred.values() for tid,_,_ in rows}) if tracking else []
 gi={k:i for i,k in enumerate(gids)};pi={k:i for i,k in enumerate(pids)};potential=np.zeros((len(gids),len(pids)),dtype=np.int64)
 tp=ng=npred=changes=0;last={};windows=defaultdict(lambda:[0,0,0])
 for fi in frames:
  a=truth[fi];b=[(tid,box,s) for tid,box,s in pred[fi] if tracking or s>=threshold];ng+=len(a);npred+=len(b)
  costs=np.asarray([[iou(x,y) for _,y,_ in b] for _,x in a]).reshape(len(a),len(b));matched=0
  if tracking:
   for r,c in zip(*np.nonzero(costs>=.5)):potential[gi[a[r][0]],pi[b[c][0]]]+=1
  if len(a) and len(b):
   rr,cc=linear_sum_assignment(np.where(costs>=.5,1-costs,1e6))
   for r,c in zip(rr,cc):
    if costs[r,c]<.5:continue
    tp+=1;matched+=1
    if tracking:
     tid=a[r][0];pid=b[c][0]
     if tid in last and last[tid]!=pid:changes+=1
     last[tid]=pid
  window=min(3,fi//1440);windows[window][0]+=matched;windows[window][1]+=len(a);windows[window][2]+=len(b)
 result={'iou_threshold':.5,'score_threshold':None if tracking else threshold,'reference_boxes':ng,'predicted_boxes':npred,'spatial_matches':tp,'precision_reference_scope':tp/npred if npred else None,'recall_reference_scope':tp/ng if ng else None,'f1_spatial':2*tp/(ng+npred) if ng+npred else None,'minute_windows':{str(k):{'matches':v[0],'reference':v[1],'predicted':v[2],'recall':v[0]/v[1],'precision':v[0]/v[2] if v[2] else None} for k,v in windows.items()}}
 if tracking:
  rr,cc=linear_sum_assignment(potential,maximize=True);idtp=int(potential[rr,cc].sum()) if potential.size else 0
  result.update(global_identity_f1_reference_scope=2*idtp/(ng+npred),idtp=idtp,idfn=ng-idtp,idfp=npred-idtp,predicted_unique_ids=len(pids),reference_unique_ids=len(gids),identity_changes_among_spatial_matches=changes)
 return result
def ap50(pred):
 data={'images':[{'id':fi+1,'width':4096,'height':1080} for fi in frames],'annotations':[],'categories':[{'id':1,'name':'upstream_player_or_goalkeeper'}],'info':{}}
 for fi in frames:
  for _,box in truth[fi]:data['annotations'].append({'id':len(data['annotations'])+1,'image_id':fi+1,'category_id':1,'bbox':[box[0],box[1],box[2]-box[0],box[3]-box[1]],'area':(box[2]-box[0])*(box[3]-box[1]),'iscrowd':0})
 dt=[{'image_id':fi+1,'category_id':1,'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'score':s} for fi in frames for _,b,s in pred[fi]]
 with Path('/work/comparison-coco.log').open('a') as log,redirect_stdout(log):
  coco=COCO();coco.dataset=data;coco.createIndex();res=coco.loadRes(dt);ev=COCOeval(coco,res,'bbox');ev.params.imgIds=[fi+1 for fi in frames];ev.params.iouThrs=np.array([.5]);ev.evaluate();ev.accumulate();precision=ev.eval['precision'][0,:,:,0,2];valid=precision[precision>-1];return float(valid.mean())
results={}
for name,raw,obs in [('current_resnet',w/'raw_proposals.jsonl',w/'observations.jsonl'),('motion',w/'raw_proposals.jsonl',Path('/work/replay/motion-observations.jsonl')),('sam_motion',w/'raw_proposals.jsonl',Path('/work/replay/sam_motion-observations.jsonl')),('bytetrack_control',w/'raw_proposals.jsonl',Path('/work/replay/bytetrack_control-observations.jsonl'))]:
 raw=load(raw);tracks=load(obs,True);results[name]={'detection_at_score_05':evaluate(raw),'detection_at_score_035':evaluate(raw,threshold=.35),'detection_ap50_upstream_player_scope':ap50(raw),'tracking_at_common_6hz_frames':evaluate(tracks,True)}
report={'schema':'touchline.provisional-panorama-comparison.v1','status':'complete','frames':len(frames),'source_frames':5760,'duration_s':240,'normalization_sha256':summary['source_sha256'],'reference_sha256':hashlib.sha256(Path('/work/compare-reference.txt').read_bytes()).hexdigest(),'reference_origin':'SoccerTrack-v2 MOT118577; zero-based source25FPS via preserved24FPS frame map','reference_review_status':'upstream annotations; no local human completeness/identity review','reference_scope':'22 players/goalkeepers only; referees/other people omitted','metric_limits':'Same-frame exploratory comparison. Unannotated valid people may count as false positives. Not full all-person AP50, independently reviewed tracking acceptance or jersey identity accuracy. Global identity-count optimization is separate from TrackEval preprocessing. No labels fed to inference.','results':results,'accepted_quality_gate':False}
Path('/work/replay-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':'complete','frames':len(frames)}))
