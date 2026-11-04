"""Explicit local CPU full-video analysis, profiles and complete overlay; no keys."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from touchline.data.manifest import sha256_file
from touchline.video.decode import probe_video
from touchline.video.normalize import normalize_video
from touchline.diagnostics import diagnostic_session


def command(video,checkpoint,output):
    return ['analyze',str(video),'--checkpoint',str(checkpoint),'--device','cpu','--mode','fixed',
        '--fps','6','--tracker','bytetrack','--min-size','1080','--max-size','1920',
        '--record-proposals','--output',str(output)]


def cpu_environment(cache):
    cache=Path(cache).resolve()
    env={k:os.environ[k] for k in ('SYSTEMROOT','WINDIR','PATH','PATHEXT','TEMP','TMP','SYSTEMDRIVE') if k in os.environ}
    for key,name in {'MPLCONFIGDIR':'matplotlib','TORCH_HOME':'torch','XDG_CACHE_HOME':'cache',
                     'USERPROFILE':'home','APPDATA':'appdata','LOCALAPPDATA':'localappdata'}.items():
        path=cache/name;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    env.update(PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
    return env


def main(video,checkpoint,output):
    video=Path(video).resolve();checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Keep generated results under project runs')
    if not video.is_file() or not checkpoint.is_file():raise ValueError('Select video and trained weights')
    info=probe_video(video)
    if info.get('duration_s') is None or not 0<info['duration_s']<=300:raise ValueError('This bounded CPU job supports up to five minutes')
    if shutil.disk_usage(output.parent).free<max(1024**3,video.stat().st_size*3):raise ValueError('Insufficient output headroom')
    metadata=json.loads(checkpoint.with_name('result.json').read_text(encoding='utf-8'))
    digest=sha256_file(checkpoint)
    if digest!=metadata['checkpoint_sha256']:raise ValueError('Checkpoint selection changed')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();stage='initializing';done=threading.Event()
    def write_status(state,**values):
        record={'schema':'touchline.cpu-analysis.v1','utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_s':round(time.monotonic()-started,2),'stage':stage,'status':state,
            'device':'cpu','checkpoint_sha256':digest,'training':'not run','provider_calls':False,**values}
        temporary=output/'status.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        temporary.replace(output/'status.json')
        return record
    def heartbeat():
        while not done.wait(30):
            coverage=output/'run/coverage.jsonl'
            try:
                with coverage.open('rb') as stream:rows=sum(1 for _ in stream)
            except OSError:rows=0
            item={'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
                  'elapsed_s':round(time.monotonic()-started,1),'coverage_rows_written':rows}
            with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
    monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
    try:
        stage='cpu_runtime_preflight';print(json.dumps(write_status('running')),flush=True)
        env=cpu_environment(output/'cache')
        preflight='import torch,torchvision; from torchvision.ops import nms; assert nms(torch.tensor([[0.,0.,10.,10.]],device="cpu"),torch.tensor([.9],device="cpu"),.5).tolist()==[0]'
        with (output/'runtime.stdout.log').open('wb') as stdout,(output/'runtime.stderr.log').open('wb') as stderr:
            subprocess.run([sys.executable,'-B','-c',preflight],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=90,check=True)
        stage='normalizing_full_video';print(json.dumps(write_status('running')),flush=True)
        with diagnostic_session(output/'diagnostics'):
            normalized=normalize_video(video,output/'normalized')
        stage='cpu_detector_and_tracking';print(json.dumps(write_status('running')),flush=True)
        # The child never loads .env, imports provider SDKs or downloads weights.
        runner='import sys,torch; torch.set_num_threads(4); from touchline.cli import main; raise SystemExit(main(sys.argv[1:]))'
        with (output/'analysis.stdout.log').open('wb') as stdout,(output/'analysis.stderr.log').open('wb') as stderr:
            subprocess.run([sys.executable,'-B','-c',runner,*command(normalized,checkpoint,output/'run')],
                cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=7200,check=True)
        manifest=json.loads((output/'run/manifest.json').read_text(encoding='utf-8'))
        if manifest['status']!='complete' or manifest['detector']['checkpoint']!=digest:
            raise ValueError('Full run or checkpoint validation failed')
        if manifest['detector_runtime']['model_parameter_device']!='cpu':raise ValueError('Unexpected inference device')
        stage='player_profiles';print(json.dumps(write_status('running',run_id=manifest['run_id'])),flush=True)
        from touchline.cli import main as cli
        with diagnostic_session(output/'diagnostics'):
            if cli(['player-analysis',str(output/'run'),'--output',str(output/'profiles')])!=0:raise RuntimeError('Profiles failed')
        stage='full_video_overlay';print(json.dumps(write_status('running',run_id=manifest['run_id'])),flush=True)
        from touchline.video.preview import render_run_preview
        preview=render_run_preview(output/'run',output/'preview',source=normalized,max_seconds=None,full_run=True,
            hold_seconds=.17,show_proposals=False)
        (output/'preview-result.json').write_text(json.dumps(preview,indent=2)+'\n',encoding='utf-8')
        stage='complete'
        result=write_status('complete',run_id=manifest['run_id'],decoded_frames=manifest['decoded_frames'],
            selected_frames=manifest['selected_frames'],observation_count=manifest['observation_count'],
            overlay='preview/overlay24.mp4')
        (output/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><title>CPU video analysis</title>'
            '<h1>Full CPU analysis</h1><video controls width="1100" src="preview/overlay24.mp4"></video>'
            '<p>Temporary tracks are not jersey numbers. Tracking accuracy remains unmeasured.</p>'
            '<p><a href="preview/players.jpg">Track gallery</a> | <a href="preview/track-index.json">Track index</a> | '
            '<a href="profiles/review.md">Profile review</a> | <a href="run/manifest.json">Manifest</a> | '
            '<a href="status.json">CPU receipt</a> | <a href="progress.jsonl">Progress log</a></p>',encoding='utf-8')
        return result
    except BaseException as exc:
        write_status('failed',error_category=type(exc).__name__)
        raise
    finally:done.set();monitor.join(timeout=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:print(json.dumps(main(args.video,args.checkpoint,args.output),indent=2))
    except BaseException as exc:
        print(json.dumps({'status':'failed','error_category':type(exc).__name__}));raise SystemExit(1) from None
