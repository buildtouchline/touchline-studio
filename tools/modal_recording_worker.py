"""Full recording normalization/CPU analysis; no training or provider secrets."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import hashlib,json,os,subprocess,sys,threading,time,traceback,zipfile
W=Path('/work');DONE=threading.Event();STAGE='input';START=time.monotonic()
def emit(status,**values):
    sys.__stdout__.write(json.dumps({'stage':STAGE,'status':status,'elapsed_s':round(time.monotonic()-START,1),**values})+'\n');sys.__stdout__.flush()
def bundle(name,*,failure=False):
    with zipfile.ZipFile(W/name,'w',zipfile.ZIP_STORED) as z:
        for folder in ('run','profiles','preview','diagnostics',*(() if failure else ('normalized',))):
            for p in (W/folder).rglob('*'):
                if p.is_file() and (not failure or p.suffix in {'.json','.jsonl'}):z.write(p,p.relative_to(W).as_posix())
        for name in ('recording-receipt.json','RESULTS.html','checks.json','processing.log','tests.log','artifacts.log'):
            if (W/name).is_file():z.write(W/name,name)
def main(*,probe_only=False):
    global STAGE
    job=json.loads((W/'job.json').read_text())
    def pulse():
        while not DONE.wait(30):
            values={}
            p=W/'run/coverage.jsonl'
            if p.exists():
                with p.open('rb') as s:values['coverage_rows']=sum(1 for _ in s)
            emit('heartbeat',**values)
    threading.Thread(target=pulse,daemon=True).start()
    for name,digest in job['hashes'].items():
        STAGE='verify_'+name
        with (W/name).open('rb') as s:
            if hashlib.file_digest(s,'sha256').hexdigest()!=digest:raise ValueError('Input hash mismatch')
        emit('complete')
    if not probe_only and 'recording.mp4' not in job['hashes']:raise ValueError('Full-source digest required')
    STAGE='extract_source'
    project=W/('probe-project' if probe_only else 'recording-project');project.mkdir()
    with zipfile.ZipFile(W/'code.zip') as z:
        for e in z.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    from touchline.video.decode import probe_video
    info=probe_video(W/'recording.mp4')
    duration=info.get('duration_s')
    (W/'source-probe.json').write_text(json.dumps(info,indent=2))
    if duration is None or not 2<=duration<=900 or info['width']<info['height'] or info['is_rotated'] or max(info['width'],info['height'])>8192:raise ValueError('Recording outside reviewed 15-minute landscape bounds')
    STAGE='source_probe';emit('complete',width=info['width'],height=info['height'],fps=info['average_fps'],duration_s=duration)
    if probe_only:return
    STAGE='checks';checks={}
    for mode,args in (('tests',[]),('artifacts',['artifacts'])):
        with (W/(mode+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
        checks[mode]={'exit_code':r.returncode}
        if r.returncode:raise RuntimeError('Remote checks failed')
    (W/'checks.json').write_text(json.dumps(checks));emit('complete')
    import torch
    torch.set_num_threads(4)
    from touchline.video.normalize import normalize_video
    from touchline.cli import main as cli
    from touchline.diagnostics import diagnostic_session
    STAGE='normalization';emit('started')
    with (W/'processing.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(W/'diagnostics'):
        video=normalize_video(W/'recording.mp4',W/'normalized')
        norm=json.loads((W/'normalized/normalization.json').read_text())
        if norm['source_sha256']!=job['hashes']['recording.mp4']:raise ValueError('Normalized source mismatch')
        STAGE='normalization';emit('saved',frames=norm['output_frames'])
        STAGE='analysis';emit('started')
        if cli(['analyze',str(video),'--checkpoint',str(W/'candidate.pt'),'--device','cpu','--mode','fixed','--fps','6','--tracker','bytetrack','--min-size','1080','--max-size','4096','--record-proposals','--output',str(W/'run')]):raise RuntimeError('Analysis failed')
        m=json.loads((W/'run/manifest.json').read_text())
        if m['status']!='complete' or m['decoded_frames']!=norm['output_frames'] or m['detector']['checkpoint']!=job['hashes']['candidate.pt'] or m['detector_runtime']['model_parameter_device']!='cpu':raise ValueError('Run completeness or model provenance mismatch')
        STAGE='profiles';emit('started')
        if cli(['player-analysis',str(W/'run'),'--output',str(W/'profiles')]):raise RuntimeError('Profiles failed')
        STAGE='render';emit('started')
        from touchline.video.preview import render_run_preview
        preview=render_run_preview(W/'run',W/'preview',source=video,max_seconds=None,full_run=True,hold_seconds=.17)
    receipt={'status':'complete','run_id':m['run_id'],'input_hashes':job['hashes'],'source_metadata':info,'normalization':norm,
        'decoded_frames':m['decoded_frames'],'selected_frames':m['selected_frames'],'observations':m['observation_count'],
        'detector_runtime':m['detector_runtime'],'overlay_sha256':preview['overlay_sha256'],'training':'not run',
        'identity_accuracy':None,'cpu_threads':torch.get_num_threads(),'checks':checks,'duration_s':round(time.monotonic()-START,2)}
    (W/'recording-receipt.json').write_text(json.dumps(receipt,indent=2))
    (W/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>Full recording: CPU analysis</h1><video controls width="1200" src="preview/overlay24.mp4"></video><p>T = temporary tracker ID, not a confirmed player or jersey. SAMPLED = detector evidence; HELD = visual carry-forward. No measured identity accuracy, calibrated physical metrics or approved scores.</p><p><a href="preview/players.jpg">Track gallery</a> | <a href="preview/track-index.json">Track index</a> | <a href="profiles/review.md">Profiles</a> | <a href="recording-receipt.json">Receipt</a></p>')
    STAGE='export';emit('started');bundle('recording-results.zip');STAGE='complete';emit('complete')
if __name__=='__main__':
    try:main(probe_only='--probe-only' in sys.argv)
    except BaseException as exc:
        locations=[f.lineno for f in traceback.extract_tb(exc.__traceback__) if Path(f.filename).name=='worker.py']
        emit('failed',error_category=type(exc).__name__,error_line=locations[-1] if locations else 0)
        try:(W/'recording-receipt.json').write_text(json.dumps({'status':'failed','stage':STAGE,'error_category':type(exc).__name__}));bundle('recording-failure.zip',failure=True)
        except Exception:pass
        raise SystemExit(1) from None
    finally:DONE.set()
