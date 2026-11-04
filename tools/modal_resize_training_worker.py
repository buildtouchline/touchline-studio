"""Owner-reviewed high-resolution heads trial on frozen existing data only."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,threading,time,zipfile
W=Path('/work');DONE=threading.Event();STAGE='preflight';START=time.monotonic()
def emit(stage,**values):
    global STAGE
    STAGE=stage;sys.__stdout__.write(json.dumps({'stage':stage,**values})+'\n');sys.__stdout__.flush()
def export(name):
    with zipfile.ZipFile(W/name,'w',zipfile.ZIP_STORED) as z:
        for folder in ('candidate','candidate-val','diagnostics'):
            for p in (W/folder).rglob('*'):
                if p.is_file():z.write(p,p.relative_to(W).as_posix())
        for name in ('training-receipt.json','training.log','checks.json'):
            if (W/name).exists():z.write(W/name,name)
def main():
    job=json.loads((W/'job.json').read_text())
    def pulse():
        while not DONE.wait(30):emit(STAGE,status='heartbeat',elapsed_s=round(time.monotonic()-START,1))
    threading.Thread(target=pulse,daemon=True).start()
    for name,digest in job['hashes'].items():
        with (W/name).open('rb') as s:
            if hashlib.file_digest(s,'sha256').hexdigest()!=digest:raise ValueError('Frozen input mismatch')
    project=W/'resize-project';project.mkdir()
    with zipfile.ZipFile(W/'code.zip') as z:
        for e in z.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'));checks={}
    for mode,args in (('tests',[]),('artifacts',['artifacts'])):
        with (W/(mode+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
        text=(W/(mode+'.log')).read_text(errors='replace');counts=re.findall(r'Ran (\d+) tests?',text)
        checks[mode]={'exit_code':r.returncode,'tests':int(counts[-1]) if counts else 0};emit('checks',mode=mode,**checks[mode])
        if r.returncode:raise RuntimeError('Remote checks failed')
    from touchline.training.recipes import validate_resize_recipe
    from touchline.data.manifest import read_manifest,sha256_file
    doc=json.loads((W/'recipe-review.json').read_text());plan=validate_resize_recipe(doc,job['hashes']['candidate.pt'])
    reference=json.loads((W/'reference.json').read_text());manifest=W/'dataset/release.json';data=read_manifest(manifest,training=True)
    if sha256_file(manifest)!=reference['manifest_sha256'] or reference['split']!='val' or reference['subset'] or reference['images']!=570 or sum(s['split']=='train' for s in data['samples'])!=600:raise ValueError('Frozen data mismatch')
    import torch
    from torchvision.ops import nms
    from pycocotools.cocoeval import COCOeval
    if not torch.cuda.is_available() or nms(torch.tensor([[0.,0.,10.,10.]],device='cuda'),torch.tensor([.9],device='cuda'),.5).tolist()!=[0]:raise RuntimeError('CUDA NMS unavailable')
    from touchline.training.runner import train_detector
    from touchline.training.evaluate import evaluate_detector
    from touchline.diagnostics import diagnostic_session
    emit('training',status='started',gpu=torch.cuda.get_device_name(0))
    with (W/'training.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(W/'diagnostics'):
        trained=train_detector(manifest,W/'candidate',epochs=plan['epochs'],max_steps=plan['max_steps'],max_seconds=plan['max_seconds'],device='cuda',initial_checkpoint=W/'candidate.pt',learning_rate=plan['learning_rate'],weight_decay=plan['weight_decay'],execution_backend='modal',parent_run_id=job['parent_run_id'],reviewed_recipe=doc)
    emit('checkpoint',status='saved',steps=trained['steps'],checkpoint_sha256=trained['checkpoint_sha256'])
    emit('validation',status='started')
    with (W/'training.log').open('a') as log,redirect_stdout(log),redirect_stderr(log):
        evaluated=evaluate_detector(manifest,W/'candidate-val',W/'candidate/candidate.pt','val','cuda',None,False,min_size=1080,max_size=4096)
    if evaluated['sample_ids']!=reference['sample_ids'] or sha256_file(W/'candidate.pt')!=job['hashes']['candidate.pt']:raise ValueError('Validation/rollback lineage mismatch')
    result={'status':'complete','training':trained,'input_hashes':job['hashes'],'validation':evaluated['metrics'],'validation_images':570,
        'reference_native_ap50':job['reference_native_ap50'],'candidate_native_ap50':evaluated['metrics']['ap50'],
        'delta_percentage_points':100*(evaluated['metrics']['ap50']-job['reference_native_ap50']),
        'reaches_70_ap50':evaluated['metrics']['ap50']>=.7,'dataset':'existing reviewed SoccerNet only; SoccerTrack pending completeness review',
        'proposal_sha256':doc['proposal_sha256'],'checks':checks,'rollback_checkpoint_unchanged':True,'final_test':'not evaluated','promotion':'not performed','duration_s':round(time.monotonic()-START,2)}
    (W/'training-receipt.json').write_text(json.dumps(result,indent=2));(W/'checks.json').write_text(json.dumps(checks));export('gpu-results.zip')
    emit('validation',status='complete',images=570,ap50=evaluated['metrics']['ap50']);emit('complete',status='complete')
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        emit('failed',error_category=type(exc).__name__)
        try:export('gpu-failure.zip')
        except Exception:pass
        raise SystemExit(1) from None
    finally:DONE.set()
