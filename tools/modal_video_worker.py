"""Fixed no-secret GPU analysis worker. It never trains on uploaded video."""
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import runpy
from pathlib import Path
import sys
import zipfile
import os
import time

STAGE='input'
STARTED=time.monotonic()


def stage(value):
    global STAGE
    STAGE=value
    os.write(2,('TOUCHLINE_EVENT '+json.dumps({'stage':value,'elapsed_s':round(time.monotonic()-STARTED,3)})+'\n').encode())


class Discard:
    def write(self,value):return len(value)
    def flush(self):pass


def main():
    global STAGE
    work=Path('/work');config=json.loads((work/'job.json').read_text())
    for name,expected in config['hashes'].items():
        with (work/name).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=expected:raise ValueError('Input hash mismatch')
    with zipfile.ZipFile(work/'code.zip') as archive:
        for entry in archive.infolist():
            path=(work/'code'/entry.filename).resolve()
            if not path.is_relative_to(work/'code') or not entry.filename.startswith('src/touchline/') or not entry.filename.endswith('.py'):
                raise ValueError('Unsafe code bundle')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(entry))
    sys.path.insert(0,str(work/'code'/'src'))
    stage('cuda')
    import torch
    from torchvision.ops import nms
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is disabled')
    if nms(torch.tensor([[0.,0.,10.,10.]],device='cuda'),torch.tensor([.9],device='cuda'),.5).tolist()!=[0]:raise RuntimeError('CUDA NMS failed')
    stage('parity')
    runpy.run_path('/work/check_cuda_tracking.py',run_name='__main__')
    from touchline.cli import main as cli
    from touchline.diagnostics import diagnostic_session
    stage('analysis')
    with diagnostic_session(work/'run-diagnostics'):
        status=cli(['analyze','/work/video24.mp4','--checkpoint','/work/candidate.pt',
            '--device','cuda','--mode','fixed','--fps','6','--tracker','bytetrack-cuda',
            '--min-size','1080','--max-size','4096','--record-proposals','--output','/work/run'])
    if status!=0:raise RuntimeError('GPU pipeline failed')
    stage('export')
    manifest=json.loads((work/'run'/'manifest.json').read_text())
    if manifest['detector']['checkpoint']!=config['hashes']['candidate.pt']:raise ValueError('Checkpoint lineage mismatch')
    receipt={'status':manifest['status'],'run_id':manifest['run_id'],'gpu_name':torch.cuda.get_device_name(0),
        'cuda':torch.version.cuda,'torch':torch.__version__,'decoded_frames':manifest['decoded_frames'],
        'selected_frames':manifest['selected_frames'],'observation_count':manifest['observation_count'],
        'training':'not run; reviewed labels required','code_repair':'armed; no code failure detected'}
    receipt['synthetic_cuda_parity']='passed'
    receipt['cuda_nms']='passed'
    receipt['input_hashes']=config['hashes']
    (work/'run'/'gpu-receipt.json').write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(work/'result.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in (work/'run').iterdir():
            if path.is_file() and path.suffix in ('.json','.jsonl','.html'):archive.write(path,'run/'+path.name)
        for path in (work/'run-diagnostics').glob('*.jsonl'):
            archive.write(path,'run/diagnostics/'+path.name)
    return receipt


if __name__=='__main__':
    try:
        with redirect_stdout(Discard()),redirect_stderr(Discard()):result=main()
        print(json.dumps(result))
    except BaseException as exc:
        # No traceback, raw exception or source/player data leaves the worker.
        code=('RESOURCE_EXHAUSTED' if 'out of memory' in str(exc).lower() else
              'DEPENDENCY_MISSING' if isinstance(exc,ImportError) else
              'INPUT_INVALID' if isinstance(exc,ValueError) else 'INTERNAL_ERROR')
        print(json.dumps({'status':'failed','failure_code':code,'stage':STAGE}))
        raise SystemExit(1) from None
