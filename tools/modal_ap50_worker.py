"""Read-only full-validation comparison of declared inference resize settings."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,threading,time,zipfile
DONE=threading.Event()

def emit(**values):sys.__stdout__.write(json.dumps(values)+'\n');sys.__stdout__.flush()

def main():
    w=Path('/work');job=json.loads((w/'job.json').read_text());start=time.monotonic()
    def pulse():
        while not DONE.wait(20):emit(stage='evaluation',status='heartbeat',elapsed_s=round(time.monotonic()-start,1))
    threading.Thread(target=pulse,daemon=True).start()
    for name,digest in job['hashes'].items():
        with (w/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Input hash mismatch')
    project=w/'evaluation-project';project.mkdir()
    with zipfile.ZipFile(w/'code.zip') as z:
        for e in z.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe code')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'));checks={}
    for mode,args in (('tests',[]),('artifacts',['artifacts'])):
        with (w/(mode+'.log')).open('wb') as log:r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
        text=(w/(mode+'.log')).read_text(errors='replace');counts=re.findall(r'Ran (\d+) tests?',text)
        checks[mode]={'exit_code':r.returncode,'tests':int(counts[-1]) if counts else 0};emit(stage='checks',mode=mode,**checks[mode])
        if r.returncode:raise RuntimeError('Remote checks failed')
    from touchline.training.evaluate import evaluate_detector
    from touchline.data.manifest import sha256_file
    reference=json.loads((w/'reference.json').read_text());manifest=w/'dataset/release.json'
    if reference['split']!='val' or reference['subset'] or reference['images']!=570 or sha256_file(manifest)!=reference['manifest_sha256']:raise ValueError('Frozen validation mismatch')
    reports={}
    for name,options in (('default',{}),('native',{'min_size':1080,'max_size':4096})):
        emit(stage='evaluation',model=name,status='started')
        with (w/'training.log').open('a') as log,redirect_stdout(log),redirect_stderr(log):
            r=evaluate_detector(manifest,w/('baseline-val' if name=='default' else 'candidate-val'),w/'candidate.pt','val','cpu',None,False,**options)
        if r['sample_ids']!=reference['sample_ids'] or r['model']['checkpoint']!=job['hashes']['candidate.pt']:raise ValueError('Evaluation lineage mismatch')
        reports[name]=r;emit(stage='evaluation',model=name,images=r['images'],ap50=r['metrics']['ap50'])
    result={'status':'complete','images':570,'split':'val','input_hashes':job['hashes'],
        'reference_gpu_ap50':reference['metrics']['ap50'],'default_cpu_ap50':reports['default']['metrics']['ap50'],
        'native_cpu_ap50':reports['native']['metrics']['ap50'],
        'delta_percentage_points':100*(reports['native']['metrics']['ap50']-reports['default']['metrics']['ap50']),
        'native_reaches_70':reports['native']['metrics']['ap50']>=.7,'inference_transforms':{n:r['inference_transform'] for n,r in reports.items()},
        'validation_manifest_sha256':reference['manifest_sha256'],'sample_ids_unchanged':True,
        'final_test':'not evaluated','training':'not run','promotion':'not performed','checks':checks,
        'duration_s':round(time.monotonic()-start,2),'limitation':'SoccerNet person detection, not own-camera identity/tracking accuracy.'}
    (w/'evaluation-comparison.json').write_text(json.dumps(result,indent=2));(w/'checks.json').write_text(json.dumps(checks))
    with zipfile.ZipFile(w/'gpu-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('baseline-val','candidate-val'):
            for p in (w/folder).rglob('*'):
                if p.is_file():z.write(p,p.relative_to(w).as_posix())
        for name in ('evaluation-comparison.json','checks.json','training.log'):z.write(w/name,name)
    emit(stage='complete',status='complete')

if __name__=='__main__':
    try:main()
    except BaseException as exc:emit(stage='failed',error_category=type(exc).__name__);raise SystemExit(1) from None
    finally:DONE.set()
