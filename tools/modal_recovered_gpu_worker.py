"""GPU-only baseline/minor trial from an immutable reviewed CPU data snapshot."""
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,sys,threading,time,zipfile
from pathlib import Path

DONE=threading.Event();STAGE='preflight'


def emit(stage,**values):
    global STAGE
    STAGE=stage;sys.__stdout__.write(json.dumps({'stage':stage,**values})+'\n');sys.__stdout__.flush()


def export(work,name):
    with zipfile.ZipFile(work/name,'w',zipfile.ZIP_STORED) as archive:
        for folder in ('candidate','candidate-val','baseline-val','diagnostics'):
            if (work/folder).exists():
                for p in (work/folder).rglob('*'):
                    if p.is_file():archive.write(p,p.relative_to(work).as_posix())
        for n in ('training-receipt.json','minor-trial-receipt.json','training.log'):
            if (work/n).is_file():archive.write(work/n,n)


def main():
    work=Path('/work');job=json.loads((work/'job.json').read_text());start=time.monotonic()
    def pulse():
        while not DONE.wait(30):emit(STAGE,status='heartbeat',elapsed_s=round(time.monotonic()-start,1))
    threading.Thread(target=pulse,daemon=True).start()
    for name,digest in job['hashes'].items():
        with (work/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('INPUT_HASH_MISMATCH')
    if not (work/'project/src').exists():
        with zipfile.ZipFile(work/'checks-code.zip') as archive:
            for e in archive.infolist():
                p=(work/'project'/e.filename).resolve()
                if not p.is_relative_to(work/'project') or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('UNSAFE_SOURCE')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
    if not (work/'dataset/release.json').exists():
        import shutil
        with zipfile.ZipFile(work/'dataset.zip') as archive:
            total=0
            for e in archive.infolist():
                p=(work/e.filename).resolve();total+=e.file_size
                if total>4*1024**3 or not e.filename.startswith('dataset/') or not p.is_relative_to(work/'dataset') or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('UNSAFE_DATASET')
                p.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(e) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
    sys.path.insert(0,str(work/'project/src'))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import torch
    from torchvision.ops import nms
    if not torch.cuda.is_available() or nms(torch.tensor([[0.,0.,10.,10.]],device='cuda'),torch.tensor([.9],device='cuda'),.5).tolist()!=[0]:raise RuntimeError('CUDA_UNAVAILABLE')
    from touchline.data.manifest import read_manifest,sha256_file
    from touchline.training.runner import train_detector
    from touchline.training.evaluate import evaluate_detector
    from touchline.training_trials import admit_minor_trial
    from touchline.diagnostics import diagnostic_session
    data_receipt=json.loads((work/'data-receipt.json').read_text());manifest=work/'dataset/release.json'
    read_manifest(manifest,training=True)
    if sha256_file(manifest)!=data_receipt['release_sha256']:raise ValueError('DATA_HASH_MISMATCH')
    advice=json.loads((work/'advice.json').read_text())
    if job['mode']=='minor':
        baseline=json.loads((work/'baseline-receipt.json').read_text());evaluation=json.loads((work/'baseline-evaluation.json').read_text())
        plan=admit_minor_trial(advice['proposal'],advice['baseline'],baseline,evaluation)
    elif job['mode']=='baseline':plan={'learning_rate':.001,'weight_decay':.0005}
    else:raise ValueError('UNKNOWN_TRIAL_MODE')
    emit('training',status='started',learning_rate=plan['learning_rate'],gpu=torch.cuda.get_device_name(0))
    with (work/'training.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(work/'diagnostics'):
        trained=train_detector(manifest,work/'candidate',epochs=3,max_steps=1200,max_seconds=1800,device='cuda',
            initial_checkpoint=work/'candidate.pt',learning_rate=plan['learning_rate'],weight_decay=plan['weight_decay'],
            execution_backend='modal',parent_run_id=job['parent_run_id'])
    emit('checkpoint',status='saved',steps=trained['steps'],checkpoint_sha256=trained['checkpoint_sha256'])
    results={}
    entries=(('baseline',work/'candidate.pt'),('candidate',work/'candidate/candidate.pt')) if job['mode']=='baseline' else (('candidate',work/'candidate/candidate.pt'),)
    for label,checkpoint in entries:
        emit('validation',model=label,status='started')
        with (work/'training.log').open('a') as log,redirect_stdout(log),redirect_stderr(log):
            result=evaluate_detector(manifest,work/(label+'-val'),checkpoint,'val','cuda',None,False)
        results[label]=result['metrics'];emit('validation',model=label,images=result['images'],ap50=result['metrics']['ap50'])
    if sha256_file(work/'candidate.pt')!=advice['baseline']['checkpoint_sha256']:raise ValueError('ROLLBACK_CHECKPOINT_CHANGED')
    common={'status':'complete','training':trained,'input_hashes':job['hashes'],'restored_images':1710,
        'release_sha256':data_receipt['release_sha256'],'gpu':torch.cuda.get_device_name(0),'validation':results,
        'final_test':'not evaluated','promotion':'not performed','duration_s':round(time.monotonic()-start,2),
        'rollback_checkpoint_unchanged':True,'checkpoint_reload':'passed via full-validation reload'}
    if job['mode']=='baseline':name='training-receipt.json'
    else:
        name='minor-trial-receipt.json';common.update(plan=plan,baseline_ap50=plan['baseline_ap50'],
            candidate_ap50=results['candidate']['ap50'],validation_images=570)
    (work/name).write_text(json.dumps(common,indent=2));export(work,'gpu-results.zip');emit('complete',status='complete')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        code='DEPENDENCY_MISSING' if isinstance(exc,ModuleNotFoundError) else ('RESOURCE_EXHAUSTED' if type(exc).__name__=='OutOfMemoryError' else 'INTERNAL_ERROR')
        emit('failed',failure_code=code,error_category=type(exc).__name__)
        try:export(Path('/work'),'gpu-failure.zip')
        except Exception:pass
        raise SystemExit(1) from None
    finally:DONE.set()
