"""Isolated R-CNN video preview through the existing pipeline; no baseline edits."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,os,re,subprocess,sys,time,zipfile

def main():
    recovery='--recover-cached' in sys.argv
    w=Path('/work/rcnn-preview20-recovery' if recovery else '/work/rcnn-preview20');w.mkdir()
    project=w/'project';project.mkdir()
    with zipfile.ZipFile('/work/rcnn-preview-code.zip') as archive:
        for e in archive.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe code archive')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'));checks={}
    for mode,args in [('tests',[]),('artifacts',['artifacts'])]:
        with (w/(mode+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        text=(w/(mode+'.log')).read_text();counts=re.findall(r'Ran (\d+) tests?',text);checks[mode]={'exit_code':r.returncode,'tests':int(counts[-1]) if counts else 0}
        if r.returncode:raise RuntimeError('Remote R-CNN checks failed')
    print(json.dumps({'stage':'checks','status':'passed','tests':checks['tests']['tests']}),flush=True)
    import torch;torch.set_num_threads(4)
    from touchline.adapters.rcnn_comparison import RcnnComparisonDetector
    from touchline.byte_tracking import ByteTrackTracker
    from touchline.pipeline import run_pipeline
    from touchline.video.decode import decode_video
    from touchline.video.preview import render_run_preview
    from touchline.analysis_policy import preflight_analysis_video
    from touchline.data.manifest import sha256_file
    from touchline.diagnostics import diagnostic_session
    from touchline.video.sampling import FixedRateSampler
    source=Path('/work/pxl-preview20/source/video24.mp4');source_hash=sha256_file(source);start=time.monotonic()
    with (w/'processing.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(w/'diagnostics'):
        detector=RcnnComparisonDetector('/work/rcnn-comparison-weights.pth')
        cached={};cache_receipt={'reused_frames':0,'new_forward_calls':0}
        if recovery:
            old=Path('/work/rcnn-preview20/run');previous=json.loads((old/'manifest.json').read_text())
            if previous['source']['id']!=source_hash or previous['detector']['checkpoint']!=detector.provenance()['checkpoint']:raise ValueError('Cached detector/source mismatch')
            with (old/'raw_proposals.jsonl').open() as stream:
                for line in stream:
                    if not line.endswith('\n'):continue
                    row=json.loads(line)
                    if row['frame']['source_id']!=source_hash:raise ValueError('Cached frame lineage mismatch')
                    cached.setdefault(row['frame']['index'],[]).append(row)
            # The final frame may have only partially flushed proposals.
            if cached:cached.pop(max(cached))
            original=detector.detect_with_proposals
            from touchline.contracts import Box,Detection
            def cached_detect(frame):
                props=cached.get(frame.index)
                if props is None:
                    cache_receipt['new_forward_calls']+=1
                    return original(frame)
                if any(r['frame']['index']!=frame.index or abs(r['frame']['timestamp_s']-frame.timestamp_s)>1e-8 for r in props):raise ValueError('Cached timestamp mismatch')
                clean=[{'bbox_xyxy':p['bbox_xyxy'],'score':p['score'],'label':p['label']} for p in props]
                detections=[Detection(Box(*p['bbox_xyxy']),p['score']) for p in clean if p['label']==1 and p['score']>=.1]
                cache_receipt['reused_frames']+=1
                return detections,clean
            detector.detect_with_proposals=cached_detect
        result=run_pipeline(decode_video(source,source_hash),detector,FixedRateSampler(6),w/'run',{'id':source_hash,'path':str(source),'stream_index':0,'analysis_policy':preflight_analysis_video(source)},record_proposals=True,tracker=ByteTrackTracker(6))
        preview=render_run_preview(w/'run',w/'preview',source=source,max_seconds=20,hold_seconds=.17)
    m=json.loads((w/'run/manifest.json').read_text())
    if m['status']!='complete' or m['decoded_frames']!=480:raise ValueError('Complete20-second run required')
    summary={'status':'complete','run_id':m['run_id'],'architecture':detector.architecture,'checkpoint_sha256':m['detector']['checkpoint'],'source_sha256':source_hash,'frames':480,'detector_samples':m['selected_frames'],'observations':m['observation_count'],'checks':checks,'elapsed_s':round(time.monotonic()-start,2),'tracking_idf1':None,'ap50':None,'promotion':'not performed','tracking':'unchanged baseline ByteTrack to isolate detector effect','cache_recovery':cache_receipt if recovery else None}
    (w/'summary.json').write_text(json.dumps(summary,indent=2))
    (w/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>ResNet50 R-CNN video comparison</h1><video width="1200" controls src="preview/overlay24.mp4"></video><p>Experimental generic COCO model. Same baseline ByteTrack; raw temporary IDs may exceed24. Not the accepted24-session-ID output. No measured AP50/IDF1 or promotion.</p><a href="preview/players.jpg">Track gallery</a>')
    with zipfile.ZipFile(w/'review.zip','w',zipfile.ZIP_STORED) as archive:
        for folder in ('run','preview','diagnostics'):
            for p in (w/folder).rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(w).as_posix())
        for name in ('summary.json','RESULTS.html','tests.log','artifacts.log','processing.log'):archive.write(w/name,name)
    print(json.dumps({'stage':'complete','status':'complete','frames':480,'observations':m['observation_count']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'stage':'rcnn_preview','status':'failed','error_category':type(exc).__name__}),flush=True)
        raise SystemExit(1) from None
