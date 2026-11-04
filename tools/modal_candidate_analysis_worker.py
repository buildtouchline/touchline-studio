"""A CPU-only 20-second visual smoke run of a provisional trained candidate."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import hashlib,json,sys,zipfile,threading,time

DONE=threading.Event()


def main():
    work=Path('/work');job=json.loads((work/'job.json').read_text())
    start=time.monotonic()
    def pulse():
        while not DONE.wait(20):sys.__stdout__.write(json.dumps({'stage':'analysis','status':'heartbeat','elapsed_s':round(time.monotonic()-start,1)})+'\n');sys.__stdout__.flush()
    threading.Thread(target=pulse,daemon=True).start()
    for name,digest in job['hashes'].items():
        with (work/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Candidate input changed')
    native=job.get('native',False)
    fresh_source=native or job.get('fresh_source',False)
    if fresh_source:
        with zipfile.ZipFile(work/'code.zip') as z:
            for e in z.infolist():
                p=(work/'sam-project'/e.filename).resolve()
                if not p.is_relative_to(work/'sam-project') or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
    sys.path.insert(0,str(work/('sam-project/src' if fresh_source else 'project/src')))
    from touchline.cli import main as cli
    from touchline.diagnostics import diagnostic_session
    from touchline.video.preview import render_run_preview
    with (work/'training.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(work/'diagnostics'):
        code=cli(['analyze',str(work/'video24.mp4'),'--checkpoint',str(work/'candidate.pt'),'--device','cpu',
            '--mode','fixed','--fps','6','--tracker','bytetrack','--min-size','1080','--max-size','4096' if native else '1920',
            '--record-proposals','--max-selected','120','--output',str(work/'run')])
        if code:raise RuntimeError('Candidate analysis failed')
        if cli(['player-analysis',str(work/'run'),'--output',str(work/'profiles')]):raise RuntimeError('Candidate profiles failed')
        render_run_preview(work/'run',work/'preview',source=work/'video24.mp4',max_seconds=20,hold_seconds=.17)
        if native and job.get('sam_review',True):
            from touchline.sam_analysis import FRAME_INDICES
            from touchline.video.decode import decode_video
            student={i:[] for i in FRAME_INDICES};frames=[];(work/'frames').mkdir()
            for line in (work/'run/raw_proposals.jsonl').read_text().splitlines():
                row=json.loads(line);index=row['frame']['index']
                if index in student and row['score']>=.5:student[index].append(row)
            for f in decode_video(work/'video24.mp4','native-sam-review'):
                if f.index in student:
                    f.image.to_image().convert('RGB').save(work/'frames'/f'{f.index:06d}.jpg',quality=95)
                    frames.append({'index':f.index,'timestamp_s':f.timestamp_s,'width':f.width,'height':f.height,'student':student[f.index]})
                if f.index>=max(FRAME_INDICES):break
            (work/'prepared.json').write_text(json.dumps({'frames':frames,'checks':{'execution':'prior remote429-test pass; native inference exercised here'}}))
            import importlib.util
            spec=importlib.util.spec_from_file_location('sam_review_renderer',work/'sam_worker.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main('render')
    manifest=json.loads((work/'run/manifest.json').read_text())
    if manifest['detector']['checkpoint']!=job['hashes']['candidate.pt'] or manifest['detector_runtime']['model_parameter_device']!='cpu':raise ValueError('Candidate/device mismatch')
    baseline_count=0;baseline_tracks=set()
    for line in (work/'baseline-observations.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['frame']['index']<480:baseline_count+=1;baseline_tracks.add(row['local_track_id'])
    receipt={'status':'complete','analysis_status':manifest['status'],'run_id':manifest['run_id'],
        'selected_frames':manifest['selected_frames'],'observations':manifest['observation_count'],
        'original_first20s_observations':baseline_count,'original_first20s_temporary_tracks':len(baseline_tracks),
        'input_hashes':job['hashes'],'detector_runtime':manifest['detector_runtime'],
        'selection':job['selection'],'identity_accuracy':None,'promotion':'not performed',
        'limitations':'Capped visual candidate test; counts are diagnostic proxies, not accuracy. Physical metrics require calibration.'}
    if native and job.get('sam_review',True):receipt['sam_review']=json.loads((work/'review/summary.json').read_text())
    (work/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2))
    (work/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>Provisional trained candidate: CPU video smoke test</h1><p>No model promotion or measured identity accuracy.</p><video controls width="1100" src="preview/overlay24.mp4"></video><p><a href="preview/players.jpg">Track gallery</a> | <a href="profiles/review.md">Profiles</a> | <a href="analysis-receipt.json">Receipt</a></p>')
    with zipfile.ZipFile(work/'gpu-results.zip','w',zipfile.ZIP_STORED) as archive:
        for folder in ('run','profiles','preview','diagnostics',*(['review'] if native and job.get('sam_review',True) else [])):
            for p in (work/folder).rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(work).as_posix())
        for name in ('analysis-receipt.json','RESULTS.html','training.log'):archive.write(work/name,name)
    print(json.dumps({'stage':'complete','status':'complete'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:print(json.dumps({'stage':'failed','failure_code':'INTERNAL_ERROR','error_category':type(exc).__name__}),flush=True);raise SystemExit(1) from None
    finally:DONE.set()
