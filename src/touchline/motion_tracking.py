"""Staged image-motion tracks; detector anchors and propagated boxes stay distinct."""
from collections import deque
from dataclasses import dataclass,field
import math
import numpy as np

def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=w*h
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def foot(box):return np.array([(box[0]+box[2])/2,box[3]],dtype=float)
def transform_point(p,m):return m[:,:2]@np.asarray(p)+m[:,2]
def bound(box,width,height):
    x1,y1,x2,y2=box
    return np.array([max(0,min(width-1,x1)),max(0,min(height-1,y1)),max(1,min(width,x2)),max(1,min(height,y2))],dtype=float)
def appearance(rgb,box):
    import cv2
    x1,y1,x2,y2=map(int,box);h,w=rgb.shape[:2];x1=max(0,x1);x2=min(w,x2);y1=max(0,y1);y2=min(h,y2)
    if x2-x1<4 or y2-y1<8:return None
    crop=rgb[y1:y2,x1:x2];crop=crop[int(.12*crop.shape[0]):max(2,int(.88*crop.shape[0])),int(.20*crop.shape[1]):max(2,int(.80*crop.shape[1]))]
    if not crop.size:return None
    hsv=cv2.cvtColor(cv2.resize(crop,(24,48)),cv2.COLOR_RGB2HSV);parts=[]
    for region in (hsv[:28],hsv[28:]):
        hist=cv2.calcHist([region],[0,1],None,[12,8],[0,180,0,256]).reshape(-1);parts.append(hist)
        parts.append(cv2.calcHist([region],[2],None,[8],[0,256]).reshape(-1))
    vector=np.concatenate(parts).astype(float);return vector/max(vector.sum(),1e-9)
def appearance_distance(a,b):
    if a is None or b is None:return .5
    return float(np.clip(1-np.sqrt(a*b).sum(),0,1))

def estimate_motion(previous,current,boxes,source_width):
    """Robust background affine plus independent foreground point displacements."""
    import cv2
    scale=source_width/previous.shape[1];identity=np.array([[1.,0.,0.],[0.,1.,0.]])
    mask=np.full(previous.shape,255,dtype=np.uint8)
    for box in boxes.values():
        x1,y1,x2,y2=(np.asarray(box)/scale).astype(int)
        if x2<0 or y2<0 or x1>=mask.shape[1] or y1>=mask.shape[0]:continue
        cv2.rectangle(mask,(max(0,x1-4),max(0,y1-4)),(min(mask.shape[1]-1,x2+4),min(mask.shape[0]-1,y2+4)),0,-1)
    def flow(points,indices=False):
        empty=(None,None,None) if indices else (None,None)
        if points is None or len(points)<4:return empty
        nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
        if nxt is None:return empty
        back,backstatus,_=cv2.calcOpticalFlowPyrLK(current,previous,nxt,None,winSize=(21,21),maxLevel=3)
        if back is None:return empty
        good=(status.reshape(-1)>0)&(backstatus.reshape(-1)>0)&(np.linalg.norm(back.reshape(-1,2)-points.reshape(-1,2),axis=1)<1.5)
        result=(points.reshape(-1,2)[good],nxt.reshape(-1,2)[good])
        return (*result,np.flatnonzero(good)) if indices else result
    points=cv2.goodFeaturesToTrack(previous,maxCorners=300,qualityLevel=.01,minDistance=10,mask=mask)
    a,b=flow(points);matrix=None;quality=0.
    if a is not None and len(a)>=12:
        matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.)
        quality=float(inliers.mean()) if inliers is not None else 0.
    valid=matrix is not None and quality>=.5 and .8<=np.linalg.det(matrix[:,:2])<=1.25
    if valid:matrix=matrix.astype(float);matrix[:,2]*=scale
    else:matrix=identity
    foreground={};all_points=[];owners=[]
    for tid,box in boxes.items():
        x1,y1,x2,y2=np.asarray(box)/scale;width=x2-x1;height=y2-y1
        roi=np.zeros(previous.shape,dtype=np.uint8);xx1=max(0,int(x1+.22*width));xx2=min(roi.shape[1],int(x2-.22*width));yy1=max(0,int(y1+.12*height));yy2=min(roi.shape[0],int(y2-.15*height))
        if xx2<=xx1 or yy2<=yy1:continue
        roi[yy1:yy2,xx1:xx2]=255;pts=cv2.goodFeaturesToTrack(previous,maxCorners=24,qualityLevel=.015,minDistance=3,mask=roi)
        if pts is not None and len(pts)>=4:all_points.append(pts);owners.extend([tid]*len(pts))
    # One foreground pyramid/flow call for every player, rather than one per box.
    if all_points:
        p,q,indices=flow(np.concatenate(all_points),True)
        if p is not None:
            ids=np.asarray(owners)[indices]
            for tid in boxes:
                keep=ids==tid
                if keep.sum()<4:continue
                delta=q[keep]-p[keep];median=np.median(delta,axis=0);good=np.linalg.norm(delta-median,axis=1)<3.
                if good.sum()>=4:foreground[tid]=np.median(delta[good],axis=0)*scale
    return matrix,bool(valid),quality,foreground

@dataclass
class Track:
    tid:int
    box:np.ndarray
    last_anchor:float
    feature:object=None
    hits:int=1
    history:deque=field(default_factory=lambda:deque(maxlen=48))
    evidence:str='detector_anchor'
    raw_detection:object=None
    flow_valid:bool=False
    flow_rejection_reason:object=None

class MotionTracker:
    """No permanent identity through unseen/ambiguous gaps. No model training."""
    def __init__(self,width,height,*,max_gap_s=.8,retain_s=900.,max_session_ids=24):
        if type(max_session_ids) is not int or not 1<=max_session_ids<=24:raise ValueError('Session IDs must be capped at24')
        self.width=width;self.height=height;self.max_gap=max_gap_s;self.retain=retain_s;self.max_ids=max_session_ids;self.tracks={};self.next_id=1;self.last_time=None;self.unassigned=[]
    def update(self,t,detections,*,camera=None,camera_valid=False,foreground=None):
        from scipy.optimize import linear_sum_assignment
        if not math.isfinite(t) or self.last_time is not None and t<=self.last_time:raise ValueError('Increasing finite timestamps required')
        self.last_time=t;self.unassigned=[];camera=np.array([[1.,0.,0.],[0.,1.,0.]]) if camera is None else np.asarray(camera,dtype=float);foreground={} if foreground is None else foreground
        if camera.shape!=(2,3) or not np.isfinite(camera).all():raise ValueError('Finite camera affine required')
        self.tracks={k:v for k,v in self.tracks.items() if t-v.last_anchor<=self.retain};previous={}
        for tid,track in self.tracks.items():
            if t-track.last_anchor<=self.max_gap:previous[tid]=foot(track.box)
            track.raw_detection=None;track.flow_valid=False;track.flow_rejection_reason=None
            flow_delta=None
            if tid in foreground:
                try:flow_delta=np.asarray(foreground[tid],dtype=float)
                except (ValueError,TypeError):flow_delta=None
                if flow_delta is None or flow_delta.shape!=(2,) or not np.isfinite(flow_delta).all():
                    flow_delta=None;track.flow_rejection_reason='invalid_displacement'
                elif np.linalg.norm(flow_delta)>max(8.,1.5*np.linalg.norm(track.box[2:]-track.box[:2])):
                    flow_delta=None;track.flow_rejection_reason='implausible_displacement'
                else:track.flow_valid=True
            while track.history and t-track.history[0][1]>2.:track.history.popleft()
            for p in track.history:p[0][:]=transform_point(p[0],camera)
            if track.flow_valid:
                track.box=track.box+np.tile(flow_delta,2);track.evidence='optical_flow_prediction'
            else:
                box=track.box;corners=np.array([[box[0],box[1]],[box[0],box[3]],[box[2],box[1]],[box[2],box[3]]]);projected=corners@camera[:,:2].T+camera[:,2];track.box=np.r_[projected.min(axis=0),projected.max(axis=0)];track.evidence='camera_prediction'
        clean=[]
        for item in detections:
            box=np.asarray(item['bbox_xyxy'],dtype=float);score=item['score']
            if box.shape!=(4,) or not np.isfinite(box).all() or not 0<=box[0]<box[2]<=self.width or not 0<=box[1]<box[3]<=self.height or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Valid source-pixel detections required')
            if score>=.1:clean.append({**item,'bbox_xyxy':box})
        unmatched=set(self.tracks);used=set()
        # Currently visible tracks have priority; a dormant identity cannot steal
        # their detections merely because its appearance looks similar.
        for high,dormant in ((True,False),(False,False),(True,True)):
            ids=sorted(tid for tid in unmatched if (t-self.tracks[tid].last_anchor>self.max_gap)==dormant);indices=[i for i,d in enumerate(clean) if i not in used and (d['score']>=.35)==high]
            if not ids or not indices:continue
            costs=np.full((len(ids),len(indices)),1e6)
            for a,tid in enumerate(ids):
                track=self.tracks[tid];center=(track.box[:2]+track.box[2:])/2;diag=max(10,np.linalg.norm(track.box[2:]-track.box[:2]))
                for b,i in enumerate(indices):
                    d=clean[i];box=d['bbox_xyxy'];overlap=iou(track.box,box);distance=np.linalg.norm(center-(box[:2]+box[2:])/2)/diag;app=appearance_distance(track.feature,d.get('feature'))
                    age=t-track.last_anchor
                    if age>self.max_gap:
                        ratio=np.prod(box[2:]-box[:2])/max(1,np.prod(track.box[2:]-track.box[:2]))
                        if app>.15 or not .3<=ratio<=3.3 or distance>1+age*1.5:continue
                        costs[a,b]=.25*(1-overlap)+.6*app+.15*min(distance/max(1,age*1.5),1)
                    else:
                        if distance>1.3 and overlap<.01 and app>.15:continue
                        if app>.65 and overlap<.05 and distance>.5:continue
                        # View angle/brightness must not veto a strong motion match.
                        if track.flow_valid and overlap>.15 and distance<.6:app=min(app,.35)
                        costs[a,b]=.65*(1-overlap)+.2*app+.15*min(distance,2)
            rows,cols=linear_sum_assignment(costs)
            for a,b in zip(rows,cols):
                if costs[a,b]>.7:continue
                tid=ids[a];i=indices[b];track=self.tracks[tid];d=clean[i];box=d['bbox_xyxy']
                alternatives=np.sort(costs[a]);ambiguous=len(alternatives)>1 and alternatives[1]-alternatives[0]<.04 and iou(track.box,box)<.15
                if t-track.last_anchor>self.max_gap:
                    column=np.sort(costs[:,b]);ambiguous=ambiguous or len(alternatives)>1 and alternatives[1]-costs[a,b]<.08 or len(column)>1 and column[1]-costs[a,b]<.08
                if ambiguous:continue
                center=.35*(track.box[:2]+track.box[2:])/2+.65*(box[:2]+box[2:])/2
                size=.7*(track.box[2:]-track.box[:2])+.3*(box[2:]-box[:2]);track.box=bound(np.r_[center-size/2,center+size/2],self.width,self.height)
                track.last_anchor=t;track.hits+=1;track.evidence='detector_anchor';track.raw_detection=box.tolist()
                if high and d.get('feature') is not None:track.feature=d['feature'] if track.feature is None else .85*track.feature+.15*d['feature']
                unmatched.discard(tid);used.add(i)
        for i,d in enumerate(clean):
            if i in used or d['score']<.6:continue
            if d.get('new_track_allowed',True) is not True:
                self.unassigned.append({'bbox_xyxy':d['bbox_xyxy'].tolist(),'score':d['score'],'reason':'semantic_confirmation_required','identity_confirmed':False});continue
            # A rejected dormant association is uncertainty, not evidence that
            # this is a new person. Preserve the box without consuming an ID.
            possible_dormant=False
            for track in self.tracks.values():
                age=t-track.last_anchor
                if age<=self.max_gap:continue
                diag=max(10.,np.linalg.norm(track.box[2:]-track.box[:2]))
                distance=np.linalg.norm(foot(track.box)-foot(d['bbox_xyxy']))/diag
                if distance<=1+min(age,4)*.5 and appearance_distance(track.feature,d.get('feature'))<.35:
                    possible_dormant=True;break
            if possible_dormant:
                self.unassigned.append({'bbox_xyxy':d['bbox_xyxy'].tolist(),'score':d['score'],'reason':'ambiguous_reidentification','identity_confirmed':False});continue
            # Keep a recently propagated same-location track rather than duplicate it.
            duplicate=False
            for track in self.tracks.values():
                box=d['bbox_xyxy'];a=track.box;inter=max(0,min(a[2],box[2])-max(a[0],box[0]))*max(0,min(a[3],box[3])-max(a[1],box[1]));small=max(1,min(np.prod(a[2:]-a[:2]),np.prod(box[2:]-box[:2])))
                if t-track.last_anchor<=self.max_gap and (iou(a,box)>.55 or inter/small>.7 and appearance_distance(track.feature,d.get('feature'))<.35):duplicate=True;break
            if duplicate:continue
            if self.next_id>self.max_ids:
                self.unassigned.append({'bbox_xyxy':d['bbox_xyxy'].tolist(),'score':d['score'],'reason':'session_id_capacity_reached','identity_confirmed':False});continue
            tid=self.next_id;self.next_id+=1;self.tracks[tid]=Track(tid,d['bbox_xyxy'].copy(),t,d.get('feature'),raw_detection=d['bbox_xyxy'].tolist())
        output=[]
        for tid,track in self.tracks.items():
            if track.hits<2 or t-track.last_anchor>self.max_gap:continue
            if track.box[2]<=0 or track.box[3]<=0 or track.box[0]>=self.width or track.box[1]>=self.height:continue
            visible=bound(track.box,self.width,self.height)
            if visible[2]-visible[0]<4 or visible[3]-visible[1]<8:continue
            clipped=not np.allclose(visible,track.box);point=foot(visible);before=previous.get(tid);relative=point-transform_point(before,camera) if before is not None and camera_valid and not clipped and 0<=before[0]<=self.width and 0<=before[1]<=self.height else None
            track.history.append((point.copy(),t))
            output.append({'track_id':tid,'bbox_xyxy':visible.tolist(),'box_clipped':clipped,'footpoint_xy':point.tolist(),'evidence':track.evidence,'detector_bbox_xyxy':track.raw_detection,
                'camera_compensated_delta_px':None if relative is None else relative.tolist(),'camera_motion_verified':bool(camera_valid),'foreground_flow_valid':track.flow_valid,
                'since_detector_anchor_s':t-track.last_anchor,'trail_xy':[p.tolist() for p,_ in track.history],
                'foreground_flow_rejection_reason':track.flow_rejection_reason,
                'identity_confirmed':False,'physical_distance_m':None})
        return output
