"""CPU restoration, dependency preflight and immutable dataset snapshot preparation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,re,subprocess,sys,threading,time,urllib.request,zipfile
from pathlib import Path
from urllib.error import HTTPError,URLError

DONE=threading.Event();STAGE='input'


def emit(stage,**values):
    global STAGE
    STAGE=stage;sys.__stdout__.write(json.dumps({'stage':stage,**values})+'\n');sys.__stdout__.flush()


def main():
    work=Path('/work');job=json.loads((work/'job.json').read_text());start=time.monotonic()
    def pulse():
        while not DONE.wait(30):emit(STAGE,status='heartbeat',elapsed_s=round(time.monotonic()-start,1))
    threading.Thread(target=pulse,daemon=True).start()
    for name,digest in job['hashes'].items():
        with (work/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('INPUT_HASH_MISMATCH')
    project=work/'project';project.mkdir()
    with zipfile.ZipFile(work/'checks-code.zip') as archive:
        for e in archive.infolist():
            p=(project/e.filename).resolve()
            if not p.is_relative_to(project) or (e.external_attr>>16)&0o170000==0o120000:raise ValueError('UNSAFE_SOURCE')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(e))
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    # Import every evaluation dependency before download, restoration or training.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import importlib.metadata
    emit('evaluation.preflight',status='passed',pycocotools=importlib.metadata.version('pycocotools'))
    for mode,args in (('tests',[]),('artifacts',['artifacts'])):
        with (work/(mode+'.log')).open('wb') as log:
            r=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
        text=(work/(mode+'.log')).read_text(errors='replace');counts=re.findall(r'Ran (\d+) tests?',text)
        emit('checks',mode=mode,exit_code=r.returncode,tests=int(counts[-1]) if counts else 0)
        if r.returncode:raise RuntimeError('CODE_CHECK_FAILED')
    from touchline.data.manifest import read_manifest,sha256_file
    from touchline.data.recovery import restore_image
    release=work/'dataset';release.mkdir();(release/'release.json').write_bytes((work/'release.json').read_bytes())
    data=read_manifest(release/'release.json',training=False)
    if any(s.get('training_eligible') is not True for s in data['samples']):raise ValueError('DATA_NOT_REVIEWED')
    url='https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/3777f456d908570e10b19e7354744640e77c351e/train.zip'
    size=9575523385;archive_path=work/'train.zip';digest=hashlib.sha256();count=0;t=time.monotonic()
    emit('restore.download',expected_bytes=size)
    with urllib.request.urlopen(url,timeout=60) as response,archive_path.open('xb') as stream:
        if int(response.headers.get('Content-Length',0))!=size:raise ValueError('ARCHIVE_SIZE_CHANGED')
        while chunk:=response.read(4*1024**2):
            count+=len(chunk)
            if count>size or time.monotonic()-t>1200:raise ValueError('DOWNLOAD_CAP_EXCEEDED')
            digest.update(chunk);stream.write(chunk)
            if count//1024**3!=(count-len(chunk))//1024**3:emit('restore.download',bytes=count)
    if count!=size or digest.hexdigest()!='1a185032e2abf6a3babda48179535144d4b557a7fcc457324cce3cb05735db5b':raise ValueError('ARCHIVE_HASH_MISMATCH')
    lock=threading.Lock();stop=threading.Event();done=0;expanded=0
    with zipfile.ZipFile(archive_path) as archive:
        mapping={}
        for name in archive.namelist():
            parts=name.split('/')
            if len(parts)>=3 and parts[-2]=='img1':mapping.setdefault(parts[-3]+'/'+parts[-1],[]).append(name)
        def restore(sample):
            nonlocal done,expanded
            if stop.is_set():raise ValueError('RESTORE_STOPPED')
            try:
                names=mapping.get(sample['source_image_ref'],[])
                if len(names)!=1 or archive.getinfo(names[0]).file_size>20*1024**2:raise ValueError('SOURCE_IMAGE_MISSING')
                n=restore_image(archive.read(names[0]),sample,release)
                with lock:
                    done+=1;expanded+=n
                    if expanded>4*1024**3:raise ValueError('IMAGE_CAP_EXCEEDED')
                    if done%100==0:emit('restore.images',verified=done)
            except BaseException:stop.set();raise
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(restore,data['samples']))
    archive_path.unlink();read_manifest(release/'release.json',training=True)
    receipt={'status':'complete','restored_images':done,'restored_bytes':expanded,
        'release_sha256':sha256_file(release/'release.json'),'input_hashes':job['hashes'],
        'duration_s':round(time.monotonic()-start,2),'training':'not run','gpu_requested':False}
    (work/'data-receipt.json').write_text(json.dumps(receipt,indent=2));emit('complete',images=done,status='complete')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        code='DEPENDENCY_MISSING' if isinstance(exc,ModuleNotFoundError) else ('PROVIDER_UNAVAILABLE' if isinstance(exc,(HTTPError,URLError)) else ('DATA_INVALID' if isinstance(exc,ValueError) else 'INTERNAL_ERROR'))
        emit('failed',error_category=type(exc).__name__,failure_code=code);raise SystemExit(1) from None
    finally:DONE.set()
