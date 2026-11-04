"""Fixed CPU-only analysis worker for a no-secret Modal Sandbox."""
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

START=time.monotonic();STAGE='input';DONE=threading.Event()


def emit(status,**counts):
    item={'schema':'touchline.cpu-worker-event.v1','stage':STAGE,'status':status,
        'elapsed_s':round(time.monotonic()-START,2),**counts}
    sys.__stdout__.write(json.dumps(item)+'\n');sys.__stdout__.flush()


class Discard:
    def write(self,value):return len(value)
    def flush(self):pass


def main():
    global STAGE
    work=Path('/work');config=json.loads((work/'job.json').read_text(encoding='utf-8'))
    for name,digest in config['hashes'].items():
        with (work/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Frozen input changed')
    with zipfile.ZipFile(work/'code.zip') as archive:
        for entry in archive.infolist():
            target=(work/'code'/entry.filename).resolve()
            if (not target.is_relative_to(work/'code') or not entry.filename.startswith('src/touchline/') or
                '..' in Path(entry.filename).parts or not entry.filename.endswith('.py') or
                (entry.external_attr>>16)&0o170000==0o120000):raise ValueError('Unsafe code bundle')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(entry))
    sys.path.insert(0,str(work/'code/src'))
    for folder in ('cache/matplotlib','cache/torch','diagnostics'):(work/folder).mkdir(parents=True,exist_ok=True)
    STAGE='cpu_runtime';emit('started')
    import torch
    import torchvision
    from torchvision.ops import nms
    torch.set_num_threads(4)
    if nms(torch.tensor([[0.,0.,10.,10.]],device='cpu'),torch.tensor([.9],device='cpu'),.5).tolist()!=[0]:raise RuntimeError('CPU NMS failed')
    def progress():
        while not DONE.wait(30):
            try:
                with (work/'run/coverage.jsonl').open('rb') as stream:rows=sum(1 for _ in stream)
            except OSError:rows=0
            emit('progress',coverage_rows=rows)
    pulse=threading.Thread(target=progress,daemon=True);pulse.start()
    from touchline.cli import main as cli
    from touchline.diagnostics import diagnostic_session
    STAGE='cpu_analysis';emit('started')
    with diagnostic_session(work/'diagnostics'):
        code=cli(['analyze',str(work/'video24.mp4'),'--checkpoint',str(work/'candidate.pt'),'--device','cpu',
            '--mode','fixed','--fps','6','--tracker','bytetrack','--min-size','1080','--max-size','1920',
            '--record-proposals','--output',str(work/'run')])
    if code!=0:raise RuntimeError('CPU inference failed')
    manifest=json.loads((work/'run/manifest.json').read_text(encoding='utf-8'))
    runtime=manifest.get('detector_runtime',{})
    if (manifest['status']!='complete' or manifest['detector']['checkpoint']!=config['hashes']['candidate.pt'] or
        manifest['decoded_frames']!=5760 or manifest['selected_frames']!=1440 or
        runtime.get('model_parameter_device')!='cpu' or runtime.get('last_input_device')!='cpu'):
        raise ValueError('CPU run completeness or provenance mismatch')
    STAGE='profiles';emit('started')
    with diagnostic_session(work/'diagnostics'):
        if cli(['player-analysis',str(work/'run'),'--output',str(work/'profiles')])!=0:raise RuntimeError('Profiles failed')
    STAGE='full_overlay';emit('started')
    from touchline.video.preview import render_run_preview
    preview=render_run_preview(work/'run',work/'preview',source=work/'video24.mp4',max_seconds=None,full_run=True,hold_seconds=.17)
    receipt={'schema':'touchline.modal-cpu-result.v1','status':'complete','run_id':manifest['run_id'],
        'decoded_frames':manifest['decoded_frames'],'selected_frames':manifest['selected_frames'],
        'observation_count':manifest['observation_count'],'detector_runtime':runtime,
        'input_hashes':config['hashes'],'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
        'cpu_threads':torch.get_num_threads(),'training':'not run','gpu_requested':False,
        'overlay_sha256':preview['overlay_sha256'],'duration_s':round(time.monotonic()-START,2)}
    (work/'cpu-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (work/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><title>Modal CPU analysis</title>'
        '<h1>Modal CPU analysis</h1><video controls width="1100" src="preview/overlay24.mp4"></video>'
        '<p>Temporary track IDs are not jersey numbers. Accuracy remains unmeasured.</p>'
        '<p><a href="preview/players.jpg">Track gallery</a> | <a href="preview/track-index.json">Track index</a> | '
        '<a href="profiles/review.md">Profiles</a> | <a href="run/manifest.json">Manifest</a> | '
        '<a href="cpu-receipt.json">CPU receipt</a></p>',encoding='utf-8')
    STAGE='export';emit('started')
    with zipfile.ZipFile(work/'cpu-results.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for directory in ('run','profiles','preview','diagnostics'):
            for path in (work/directory).rglob('*'):
                if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.html','.md','.mp4','.jpg','.png'}:
                    archive.write(path,path.relative_to(work).as_posix())
        for name in ('cpu-receipt.json','RESULTS.html'):archive.write(work/name,name)
    STAGE='complete';emit('complete',decoded_frames=5760,selected_frames=1440,observations=manifest['observation_count'])


if __name__=='__main__':
    os.chdir('/work')
    try:
        with redirect_stdout(Discard()),redirect_stderr(Discard()):main()
    except BaseException as exc:
        emit('failed',failure_code='RESOURCE_EXHAUSTED' if isinstance(exc,MemoryError) else 'INPUT_INVALID' if isinstance(exc,ValueError) else 'INTERNAL_ERROR')
        try:
            work=Path('/work')
            (work/'cpu-receipt.json').write_text(json.dumps({'status':'failed','stage':STAGE,'error_category':type(exc).__name__,'gpu_requested':False}),encoding='utf-8')
            with zipfile.ZipFile(work/'cpu-failure.zip','w',zipfile.ZIP_DEFLATED) as archive:
                archive.write(work/'cpu-receipt.json','cpu-receipt.json')
                for directory in ('run','diagnostics'):
                    for path in (work/directory).glob('*.json*'):
                        if path.is_file():archive.write(path,path.relative_to(work).as_posix())
        except Exception:pass
        raise SystemExit(1) from None
    finally:DONE.set()
