"""Execute one accepted minor proposal on frozen restored data; no network."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,os,re,subprocess,sys,threading,time,zipfile

DONE=threading.Event()


def emit(stage,**values):
    sys.__stdout__.write(json.dumps({'stage':stage,**values})+'\n');sys.__stdout__.flush()


def main():
    work=Path('/work');job=json.loads((work/'job.json').read_text());start=time.monotonic()
    def pulse():
        while not DONE.wait(30):emit('minor-trial',status='heartbeat',elapsed_s=round(time.monotonic()-start,1))
    threading.Thread(target=pulse,daemon=True).start()
    for name,digest in job['hashes'].items():
        with (work/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Frozen input mismatch')
    project=work/'project';project.mkdir()
    for archive_name,folder in (('checks-code.zip',project),('dataset.zip',work)):
        with zipfile.ZipFile(work/archive_name) as archive:
            total=0
            for entry in archive.infolist():
                if archive_name=='dataset.zip' and not entry.filename.startswith('dataset/'):continue
                target=(folder/entry.filename).resolve();total+=entry.file_size
                if total>4*1024**3 or not target.is_relative_to(folder) or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe trial archive')
                if target.exists():raise ValueError('Duplicate trial input')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(entry) as src,target.open('xb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    for mode,args in (('tests',[]),('artifacts',['artifacts'])):
        with (work/(mode+'.log')).open('wb') as log:
            result=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
        text=(work/(mode+'.log')).read_text(errors='replace');counts=re.findall(r'Ran (\d+) tests?',text)
        emit('checks',mode=mode,exit_code=result.returncode,tests=int(counts[-1]) if counts else 0)
        if result.returncode:raise RuntimeError('Fixed checks failed')
    from touchline.training_trials import admit_minor_trial
    from touchline.data.manifest import read_manifest,sha256_file
    advice=json.loads((work/'advice.json').read_text());receipt=json.loads((work/'baseline-receipt.json').read_text())
    evaluation=json.loads((work/'baseline-evaluation.json').read_text())
    plan=admit_minor_trial(advice['proposal'],advice['baseline'],receipt,evaluation)
    manifest=work/'dataset/release.json';read_manifest(manifest,training=True)
    if sha256_file(manifest)!=receipt['release_sha256'] or sha256_file(work/'candidate.pt')!=advice['baseline']['checkpoint_sha256']:raise ValueError('Trial lineage changed')
    import torch
    from torchvision.ops import nms
    if not torch.cuda.is_available() or nms(torch.tensor([[0.,0.,10.,10.]],device='cuda'),torch.tensor([.9],device='cuda'),.5).tolist()!=[0]:raise RuntimeError('CUDA/NMS unavailable')
    from touchline.training.runner import train_detector
    from touchline.training.evaluate import evaluate_detector
    from touchline.diagnostics import diagnostic_session
    emit('training',status='started',learning_rate=plan['learning_rate'],weight_decay=plan['weight_decay'])
    with (work/'training.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(work/'diagnostics'):
        trained=train_detector(manifest,work/'candidate',epochs=3,max_steps=1200,max_seconds=1800,device='cuda',
            initial_checkpoint=work/'candidate.pt',learning_rate=plan['learning_rate'],weight_decay=plan['weight_decay'],
            execution_backend='modal',parent_run_id=receipt['training']['run_id'])
        validated=evaluate_detector(manifest,work/'candidate-val',work/'candidate/candidate.pt','val','cuda',None,False)
    if sha256_file(work/'candidate.pt')!=advice['baseline']['checkpoint_sha256']:raise ValueError('Rollback checkpoint changed')
    result={'status':'complete','input_hashes':job['hashes'],'plan':plan,'training':trained,
        'baseline_ap50':plan['baseline_ap50'],'candidate_ap50':validated['metrics']['ap50'],
        'validation_images':validated['images'],'final_test':'not evaluated','promotion':'not performed',
        'checkpoint_reload':'passed via independent evaluation','rollback_checkpoint_unchanged':True,
        'gpu':torch.cuda.get_device_name(0),'duration_s':round(time.monotonic()-start,2)}
    (work/'minor-trial-receipt.json').write_text(json.dumps(result,indent=2))
    with zipfile.ZipFile(work/'minor-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('candidate','candidate-val','diagnostics'):
            for p in (work/folder).rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(work).as_posix())
        for name in ('minor-trial-receipt.json','tests.log','artifacts.log','training.log'):archive.write(work/name,name)
    emit('complete',status='complete',steps=trained['steps'],candidate_ap50=result['candidate_ap50'])


if __name__=='__main__':
    try:main()
    except BaseException as exc:emit('failed',error_category=type(exc).__name__);raise SystemExit(1) from None
    finally:DONE.set()
