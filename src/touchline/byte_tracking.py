"""Optional pinned ByteTrack association for uniform-rate stationary-view footage."""
import math
from time import perf_counter
from importlib.metadata import version

from .contracts import Box


class ByteTrackTracker:
    def __init__(self, fps=12, *, activation_threshold=.5, max_gap_s=1.5,prediction_device='cpu'):
        if prediction_device not in ('cpu','cuda'):
            raise ValueError('ByteTrack prediction device must be cpu or cuda')
        if type(fps) not in (int,float) or fps not in (2,3,4,6,8,12,24):
            raise ValueError('ByteTrack requires a uniform supported sample rate dividing 24 FPS')
        if not math.isfinite(activation_threshold) or not .1 < activation_threshold < .9:
            raise ValueError('ByteTrack activation threshold must be between 0.1 and 0.9')
        if not math.isfinite(max_gap_s) or not 0 < max_gap_s <= 3:
            raise ValueError('ByteTrack lost-track window must be positive and at most 3 seconds')
        try:
            import numpy as np
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError('Install the optional tracking extra before selecting ByteTrack') from exc
        if version('supervision') != '0.26.0':
            raise RuntimeError('Use the pinned supervision==0.26.0 tracking extra')
        self.np,self.sv=np,sv
        self.fps=int(fps);self.activation_threshold=activation_threshold
        self.max_gap_s=max_gap_s;self.last_time=None
        self._wall_seconds=0.0
        # Supervision scales this 30-Hz reference buffer by the update rate.
        self.buffer_reference_frames=math.ceil(max_gap_s*30)
        self.backend=sv.ByteTrack(track_activation_threshold=activation_threshold,
            lost_track_buffer=self.buffer_reference_frames, frame_rate=self.fps,
            minimum_matching_threshold=.8,minimum_consecutive_frames=2)
        self.prediction_device=prediction_device
        if prediction_device=='cuda':
            from .cuda_motion import CudaMotionPredictor
            self.backend.shared_kalman=CudaMotionPredictor(self.backend.shared_kalman)

    def provenance(self):
        return {'type':'bytetrack-supervision','supervision':'0.26.0',
            'motion_prediction_device':self.prediction_device,'association_device':'cpu',
            'kalman_correction_device':'cpu','full_gpu_tracker':False,
            'update_fps':self.fps,'activation_threshold':self.activation_threshold,
            'new_track_threshold':self.activation_threshold+.1,'input_score_floor':.1,
            'lost_track_window_s':self.max_gap_s,'lost_buffer_reference_frames':self.buffer_reference_frames,
            'minimum_consecutive_frames':2,'matching_cost_threshold':.8,
            'motion_model':'Kalman constant velocity in image coordinates',
            'appearance_reid':False,'resume_supported':False}

    def runtime_stats(self):
        result=self.backend.shared_kalman.runtime_stats() if self.prediction_device=='cuda' else {
            'motion_prediction_device':'cpu','association_device':'cpu',
            'kalman_correction_device':'cpu','cuda_prediction_batches':0,'full_gpu_tracker':False}
        return {**result,'total_updates':self.backend.frame_id,'tracking_wall_seconds_including_transfers':self._wall_seconds}

    def update(self,timestamp_s,detections):
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError('Valid relative tracker timestamp required')
        if self.last_time is not None and abs(timestamp_s-self.last_time-1/self.fps)>1e-5:
            raise ValueError('ByteTrack updates must follow the configured uniform timestamp cadence')
        self.last_time=timestamp_s
        started=perf_counter()
        np=self.np
        data=self.sv.Detections(xyxy=np.asarray([d.box.values() for d in detections],dtype=float).reshape(-1,4),
            confidence=np.asarray([d.score for d in detections],dtype=float),
            class_id=np.ones(len(detections),dtype=int))
        assigned=self.backend.update_with_detections(data)
        results=[]
        for box,score,tid in zip(assigned.xyxy,assigned.confidence,assigned.tracker_id):
            b=Box(*[float(v) for v in box])
            ambiguous=any(b.iou(other.box)>.3 and tuple(b.values())!=tuple(other.box.values()) for other in detections)
            results.append({'local_track_id':int(tid),'bbox_xyxy':b.values(),
                'detector_score':float(score),'association_state':'ambiguous' if ambiguous else 'tentative',
                'player_id':None})
        self._wall_seconds+=perf_counter()-started
        return results
