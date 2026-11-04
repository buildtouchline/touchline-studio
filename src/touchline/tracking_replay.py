"""Re-associate saved proposals without loading weights, reading video, or GPU work."""
import itertools
import json
from pathlib import Path

from .byte_tracking import ByteTrackTracker
from .contracts import Box, Detection, Frame
from .data.manifest import sha256_file
from .pipeline import run_pipeline, write_json
from .video.sampling import FixedRateSampler, Anchor24Sampler


def _rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


class _SavedDetector:
    def __init__(self,run_dir,manifest):
        self.run_dir=run_dir;self.manifest=manifest
        self.reader=itertools.groupby(_rows(run_dir/'raw_proposals.jsonl'),key=lambda r:r['frame']['index'])
        self.pending=next(self.reader,None)
        self.previous=-1

    def provenance(self):
        original=self.manifest['detector']
        return {**original,'device':'saved-proposal-replay-cpu', 'model_parameter_device':None,
            'original_detector_device':original.get('device'),
            'original_detector_threshold':original.get('threshold'),'threshold':.1,
            'replayed_from_run_id':self.manifest['run_id'],'inference_was_run':False}

    def runtime_stats(self):
        return {'execution':'saved-proposal-replay-cpu','inference_calls':0,'model_loaded':False}

    def detect_with_proposals(self,frame):
        if self.pending is not None and self.pending[0] < frame.index:
            raise ValueError('Proposals are unordered or do not match sampled coverage')
        raw=[]
        if self.pending is not None and self.pending[0] == frame.index:
            if frame.index <= self.previous:raise ValueError('Repeated proposal frame')
            self.previous=frame.index
            for row in self.pending[1]:
                if row['frame'] != frame.metadata():raise ValueError('Proposal/source timing or geometry mismatch')
                box=Box(*row['bbox_xyxy']);score=row['score']
                Detection(box,score)  # finite-score and box validation, including discarded proposals
                if not box.fits(frame.width,frame.height) or row.get('label') != 1:
                    raise ValueError('Invalid person proposal')
                raw.append({'bbox_xyxy':box.values(),'score':score,'label':1})
            self.pending=next(self.reader,None)
        return [Detection(Box(*r['bbox_xyxy']),r['score']) for r in raw if r['score']>=.1],raw

    def detect(self,frame):return self.detect_with_proposals(frame)[0]


def retrack_run(run_dir,output,*,prediction_device='cpu',activation_threshold=.5,max_gap_s=1.5):
    """Fresh ByteTrack run over the original selected frames; preserve partial status."""
    run_dir,output=Path(run_dir),Path(output)
    parent=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    if parent['status'] not in ('partial','complete') or parent['selected_frames'] <= 0:
        raise ValueError('Replay requires a completed or explicitly capped analyzed run with sampled coverage')
    if not (run_dir/'raw_proposals.jsonl').is_file():
        raise ValueError('Raw proposals are missing; rerun inference with --record-proposals')
    if parent['sampler']=='FixedRateSampler':
        sampler=FixedRateSampler(parent['sampler_config']['fps']);fps=sampler.fps
    elif parent['sampler']=='Anchor24Sampler':
        sampler=Anchor24Sampler(parent['sampler_config'].get('tolerance_s',.0001));fps=2
    else:raise ValueError('Replay supports only recorded uniform fixed/anchor sampling')
    tracker=ByteTrackTracker(fps,prediction_device=prediction_device,
        activation_threshold=activation_threshold,max_gap_s=max_gap_s)
    detector=_SavedDetector(run_dir,parent)
    parent_hashes={name:sha256_file(run_dir/name) for name in ('manifest.json','coverage.jsonl','raw_proposals.jsonl')}
    def frames():
        previous=-1
        for row in _rows(run_dir/'coverage.jsonl'):
            f=row['frame']
            if f['index'] != previous+1:raise ValueError('Replay coverage must be contiguous from frame zero')
            previous=f['index']
            yield Frame(f['source_id'],f['index'],f['pts'],f['time_base_num'],f['time_base_den'],
                f['timestamp_s'],f['width'],f['height'])
    result=run_pipeline(frames(),detector,sampler,output,parent['source'],synthetic=parent.get('synthetic',False),
        max_selected=parent['selected_frames'] if parent['status']=='partial' else None,
        record_proposals=True,tracker=tracker)
    if result['decoded_frames'] != parent['decoded_frames'] or result['selected_frames'] != parent['selected_frames'] or detector.pending is not None:
        result['status']='failed';write_json(output/'manifest.json',result)
        raise ValueError('Replay coverage/proposals differ from the parent run')
    if any(sha256_file(run_dir/name)!=digest for name,digest in parent_hashes.items()):
        result['status']='failed';write_json(output/'manifest.json',result)
        raise ValueError('Parent artifacts changed during replay')
    result['replay']={'parent_run_id':parent['run_id'],'parent_sha256':parent_hashes,
        'inference_was_run':False,'video_was_decoded':False,'physical_calibration_inherited':False,
        'original_observation_count':parent['observation_count']}
    write_json(output/'manifest.json',result)
    return result
