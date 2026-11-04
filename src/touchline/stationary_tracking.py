"""Experimental stationary-camera motion/appearance association, without model calls.

Tentative tracklets do not consume session slots. Allocated session IDs are never
recycled or deleted; lost IDs can be reacquired conservatively. An ID is never a
confirmed player identity. Detector anchors retain measured geometry exactly.
"""
from dataclasses import dataclass
import math
import numpy as np
from .motion_tracking import appearance_distance, foot, iou


@dataclass
class _Track:
    token: int
    box: object
    state: object
    covariance: object
    feature: object
    last_anchor: float
    hits: int = 1
    session_id: object = None
    teacher_supported: bool = False


class StationaryTracker:
    def __init__(self, width, height, *, max_session_ids=24, recovery_s=8., prediction_s=.5, teacher_births_only=False):
        if type(max_session_ids) is not int or not 1 <= max_session_ids <= 24:
            raise ValueError('At most24 non-recycled session IDs required')
        if width <= 0 or height <= 0 or not 0 <= prediction_s <= 1 or not 1 <= recovery_s <= 15:
            raise ValueError('Bounded geometry and recovery policy required')
        self.width, self.height = width, height
        self.max_ids, self.recovery, self.prediction = max_session_ids, recovery_s, prediction_s
        self.teacher_births_only = teacher_births_only
        self.tracks = {}; self.next_token = 1; self.next_id = 1; self.last_time = None
        self.unassigned = []; self.events = []

    @staticmethod
    def _measurement(box):
        return np.array([*(foot(box)), box[2]-box[0], box[3]-box[1]], dtype=float)

    @staticmethod
    def _box(state):
        x, y, w, h = state[:4]
        return np.array([x-w/2, y-h, x+w/2, y])

    def update(self, t, detections, *, foreground=None):
        from scipy.optimize import linear_sum_assignment
        if not math.isfinite(t) or t < 0 or self.last_time is not None and t <= self.last_time:
            raise ValueError('Increasing finite timestamps required')
        dt = 0 if self.last_time is None else t-self.last_time
        self.last_time = t; self.unassigned = []; self.events = []
        clean = []
        for row in detections:
            b = np.asarray(row['bbox_xyxy'], dtype=float); score = row['score']
            if (b.shape != (4,) or not np.isfinite(b).all() or not 0 <= b[0] < b[2] <= self.width
                    or not 0 <= b[1] < b[3] <= self.height or type(score) not in (int,float)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Valid source-pixel detections required')
            feature = row.get('feature')
            if feature is not None:
                feature = np.asarray(feature, dtype=float)
                if feature.ndim != 1 or not np.isfinite(feature).all() or (feature < 0).any() or feature.sum() <= 0:
                    raise ValueError('Finite nonnegative appearance distribution required')
                feature = feature / feature.sum()
            if score >= .1:
                clean.append({**row, 'bbox_xyxy': b, 'feature': feature})
        # Session records survive even after the short recovery window. They are
        # not emitted while unseen and never silently reused for another person.
        self.tracks = {k: v for k,v in self.tracks.items() if v.session_id is not None or t-v.last_anchor <= 1.}
        foreground={} if foreground is None else foreground
        predicted = {}; flow_valid=set()
        for token, track in self.tracks.items():
            if t-track.last_anchor > self.recovery:
                continue
            f = np.eye(8); f[:4,4:] = np.eye(4)*dt
            scale = max(10., track.state[3]); q = np.diag([.02,.02,.005,.005,.08,.08,.01,.01])*scale**2*max(dt,.001)
            prior=track.state[:2].copy()
            track.state = f@track.state
            delta=foreground.get(track.session_id)
            if delta is not None:
                delta=np.asarray(delta,dtype=float)
                if delta.shape==(2,) and np.isfinite(delta).all() and np.linalg.norm(delta)<=max(8.,.5*scale) and dt>0:
                    track.state[:2]=prior+delta
                    track.state[4:6]=.8*track.state[4:6]+.2*delta/dt
                    flow_valid.add(token)
            track.state[2:4] = np.maximum(track.state[2:4], 4.)
            track.covariance = f@track.covariance@f.T+q
            predicted[token] = self._box(track.state)
        unmatched = set(predicted); used = set(); anchors = {}
        # High then low confidence visible association, followed by conservative
        # recovery. Dormant tracks never take detections already matched active.
        # Confirmed session tracks own association priority. Otherwise a fresh
        # unconfirmed duplicate can repeatedly steal a known person's boxes,
        # leaving the established ID dormant and exhausting all roster slots.
        stages=((True,False,True),(False,False,True),(True,True,True),
                (True,False,False),(False,False,False))
        for high, dormant, confirmed in stages:
            tokens = sorted(k for k in unmatched if (t-self.tracks[k].last_anchor > .6) == dormant
                            and (self.tracks[k].session_id is not None) == confirmed)
            indices = [i for i,d in enumerate(clean) if i not in used and (d['score'] >= .35) == high]
            if not tokens or not indices:continue
            cost = np.full((len(tokens),len(indices)),1e6)
            for a, token in enumerate(tokens):
                track = self.tracks[token]; box = predicted[token]; age = t-track.last_anchor
                height = max(12., (box[3]-box[1])); expected = foot(box)
                for b, index in enumerate(indices):
                    det = clean[index]; measured = det['bbox_xyxy']; dist = np.linalg.norm(foot(measured)-expected)
                    ratio = (measured[3]-measured[1])/height
                    app = appearance_distance(track.feature,det['feature'])
                    gate = height*(.6 if not dormant else .8)+min(age,2.)*35.
                    if dist > gate or not .45 <= ratio <= 2.2:continue
                    flow_supported=token in flow_valid and iou(box,measured)>.15 and age<=2.
                    if flow_supported:app=min(app,.25)
                    if dormant and (app > .3 or dist > height+min(age,2.)*25):continue
                    cost[a,b] = .35*(1-iou(box,measured))+.40*min(dist/max(height,.1),1.5)+.25*app
            rows, cols = linear_sum_assignment(cost)
            for a,b in zip(rows,cols):
                if cost[a,b] >= .8:continue
                token, index = tokens[a], indices[b]; track = self.tracks[token]; det = clean[index]
                # Refuse uncertain long-gap joins instead of manufacturing a
                # persistent identity when same-kit people converge.
                if dormant:
                    alternatives = np.sort(cost[a]); column = np.sort(cost[:,b])
                    if (len(alternatives)>1 and alternatives[1]-cost[a,b]<.08
                            or len(column)>1 and column[1]-cost[a,b]<.08):continue
                z = self._measurement(det['bbox_xyxy']); h = np.eye(4,8)
                r = np.diag([.015,.015,.03,.03])*max(10.,z[3])**2
                innovation = z-h@track.state; s = h@track.covariance@h.T+r
                k = np.linalg.solve(s,(track.covariance@h.T).T).T
                track.state += k@innovation
                identity = np.eye(8)-k@h
                track.covariance = identity@track.covariance@identity.T+k@r@k.T
                # Keep measured anchor geometry; smoothing belongs to predictions.
                # Keep the Kalman posterior consistent with its covariance.
                # Exact observed geometry is stored separately for the overlay.
                track.state[2:4] = np.maximum(track.state[2:4],4.)
                track.box = det['bbox_xyxy'].copy()
                track.last_anchor = t; track.hits += 1
                track.teacher_supported |= bool(det.get('teacher_supported',False))
                if high and det['feature'] is not None:
                    track.feature = det['feature'] if track.feature is None else .9*track.feature+.1*det['feature']
                anchors[token] = det; used.add(index); unmatched.discard(token)
        for i, det in enumerate(clean):
            if i in used or det['score'] < .65:continue
            token = self.next_token; self.next_token += 1; z = self._measurement(det['bbox_xyxy'])
            scale = max(10.,z[3]); state = np.r_[z,np.zeros(4)]
            covariance = np.diag([.1,.1,.05,.05,1.,1.,.1,.1])*scale**2
            self.tracks[token] = _Track(token,det['bbox_xyxy'].copy(),state,covariance,det['feature'],t,
                                        teacher_supported=bool(det.get('teacher_supported',False)))
            anchors[token] = det
        output = []
        for token, track in self.tracks.items():
            det = anchors.get(token)
            required_hits = 3 if track.teacher_supported else 6
            if (track.session_id is None and track.hits >= required_hits and det is not None
                    and (not self.teacher_births_only or track.teacher_supported)):
                if self.next_id <= self.max_ids:
                    track.session_id = self.next_id; self.next_id += 1
                    self.events.append({'event':'temporary_session_id_allocated','tracklet':token,'track_id':track.session_id})
            if track.session_id is None:
                if det is not None:self.unassigned.append({'bbox_xyxy':det['bbox_xyxy'].tolist(),'score':det['score'],
                    'reason':'tentative_confirmation' if self.next_id<=self.max_ids else 'session_capacity_requires_review'})
                continue
            age = t-track.last_anchor
            if age > self.prediction:continue
            box = track.box if det is not None else self._box(track.state)
            box = np.array([max(0,box[0]),max(0,box[1]),min(self.width,box[2]),min(self.height,box[3])])
            if not box[0] < box[2] or not box[1] < box[3]:continue
            output.append({'track_id':track.session_id,'tracklet_id':token,'bbox_xyxy':box.tolist(),
                'footpoint_xy':foot(box).tolist(),'evidence':'detector_anchor' if det is not None else 'optical_flow_prediction' if token in flow_valid else 'kalman_prediction',
                'foreground_flow_valid':token in flow_valid,
                'detector_bbox_xyxy':None if det is None else det['bbox_xyxy'].tolist(),
                'detector_score':None if det is None else det['score'],
                'since_detector_anchor_s':age,'identity_confirmed':False,'physical_distance_m':None})
        return output
