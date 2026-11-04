"""Restore exactly reviewed SoccerNet images, then one bounded CUDA warm-start."""
from contextlib import redirect_stdout,redirect_stderr
import hashlib,io,json,os,re,subprocess,sys,time,urllib.request,zipfile,threading
from pathlib import Path

ARCHIVE_URL='https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/3777f456d908570e10b19e7354744640e77c351e/train.zip'
ARCHIVE_SIZE=9575523385
ARCHIVE_SHA='1a185032e2abf6a3babda48179535144d4b557a7fcc457324cce3cb05735db5b'


STAGE='input';DONE=threading.Event()


def event(stage,**values):
    global STAGE
    STAGE=stage
    sys.__stdout__.write(json.dumps({'stage':stage,**values})+'\n');sys.__stdout__.flush()


def main():
    work=Path('/work');start=time.monotonic();job=json.loads((work/'job.json').read_text())
    def heartbeat():
        while not DONE.wait(30):
            sys.__stdout__.write(json.dumps({'stage':STAGE,'status':'heartbeat','elapsed_s':round(time.monotonic()-start,1)})+'\n');sys.__stdout__.flush()
    threading.Thread(target=heartbeat,daemon=True).start()
    for name,digest in job['hashes'].items():
        with (work/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Frozen input mismatch')
    project=work/'project';project.mkdir()
    with zipfile.ZipFile(work/'checks-code.zip') as archive:
        for entry in archive.infolist():
            p=(project/entry.filename).resolve()
            if not p.is_relative_to(project) or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(entry))
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    for mode,args in (('tests',[]),('artifacts',['artifacts'])):
        with (work/(mode+'.log')).open('wb') as log:
            result=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
        text=(work/(mode+'.log')).read_text(errors='replace');counts=re.findall(r'Ran (\d+) tests?',text)
        event('checks',mode=mode,exit_code=result.returncode,tests=int(counts[-1]) if counts else 0)
        if result.returncode:raise RuntimeError('Fixed checks failed')
    import torch
    from torchvision.ops import nms
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if nms(torch.tensor([[0.,0.,10.,10.]],device='cuda'),torch.tensor([.9],device='cuda'),.5).tolist()!=[0]:raise RuntimeError('CUDA NMS failed')
    event('cuda',status='verified',gpu=torch.cuda.get_device_name(0))
    release=work/'dataset';release.mkdir();(release/'images').mkdir()
    (release/'release.json').write_bytes((work/'release.json').read_bytes())
    from touchline.data.manifest import read_manifest,sha256_file
    data=read_manifest(release/'release.json',training=False)
    if any(s.get('training_eligible') is not True or s.get('review_status')!='human_reviewed' for s in data['samples']):raise ValueError('Reviewed release required')
    archive_path=work/'train.zip';count=0;digest=hashlib.sha256();download_start=time.monotonic()
    event('restore.download',status='started',expected_bytes=ARCHIVE_SIZE)
    with urllib.request.urlopen(ARCHIVE_URL,timeout=60) as response,archive_path.open('xb') as out:
        if int(response.headers.get('Content-Length','0'))!=ARCHIVE_SIZE:raise ValueError('Archive size changed')
        while chunk:=response.read(4*1024**2):
            count+=len(chunk)
            if count>ARCHIVE_SIZE or time.monotonic()-download_start>1200:raise ValueError('Download cap exceeded')
            digest.update(chunk);out.write(chunk)
            if count//1024**3!=(count-len(chunk))//1024**3:event('restore.download',bytes=count)
    if count!=ARCHIVE_SIZE or digest.hexdigest()!=ARCHIVE_SHA:raise ValueError('Archive checksum changed')
    from PIL import Image
    size=0
    with zipfile.ZipFile(archive_path) as archive:
        names=archive.namelist()
        for index,s in enumerate(data['samples']):
            seq,frame=s['source_image_ref'].split('/')
            matches=[name for name in names if name.endswith('/'+seq+'/img1/'+frame) or name==seq+'/img1/'+frame]
            if len(matches)!=1:raise ValueError('Unique original image missing')
            entry=archive.getinfo(matches[0])
            if entry.file_size>20*1024**2:raise ValueError('Unexpected image size')
            raw=archive.read(entry)
            if hashlib.sha256(raw).hexdigest()!=s['source_sha256']:raise ValueError('Original image checksum mismatch')
            target=(release/s['image_path']).resolve()
            if not target.is_relative_to(release/'images') or target.exists():raise ValueError('Unsafe restored image path')
            with Image.open(io.BytesIO(raw)) as image:image.convert('RGB').save(target)
            if sha256_file(target)!=s['image_sha256']:raise ValueError('Normalized PNG checksum mismatch; no automatic rehash')
            size+=target.stat().st_size
            if size>4*1024**3:raise ValueError('Restored image budget exceeded')
            if index%100==0:event('restore.images',verified=index+1)
    archive_path.unlink();read_manifest(release/'release.json',training=True)
    event('restore.complete',images=len(data['samples']),bytes=size)
    from touchline.training.runner import train_detector
    from touchline.training.evaluate import evaluate_detector
    from touchline.diagnostics import diagnostic_session
    with (work/'training.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),diagnostic_session(work/'diagnostics'):
        event_data=train_detector(release/'release.json',work/'candidate',epochs=3,max_steps=1200,max_seconds=1800,
            device='cuda',initial_checkpoint=work/'candidate.pt',execution_backend='modal',parent_run_id=job['parent_run_id'])
    event('training.complete',steps=event_data['steps'],checkpoint_sha256=event_data['checkpoint_sha256'])
    results={}
    for label,checkpoint in (('baseline',work/'candidate.pt'),('candidate',work/'candidate/candidate.pt')):
        with (work/'evaluation.log').open('a') as log,redirect_stdout(log),redirect_stderr(log):
            result=evaluate_detector(release/'release.json',work/(label+'-val'),checkpoint,'val','cuda',None,False)
        results[label]=result['metrics'];event('validation',model=label,images=result['images'],ap50=result['metrics']['ap50'])
    receipt={'status':'complete','input_hashes':job['hashes'],'restored_images':len(data['samples']),
        'release_sha256':sha256_file(release/'release.json'),'gpu':torch.cuda.get_device_name(0),
        'training':event_data,'validation':results,'final_test':'not evaluated','promotion':'not performed',
        'optimizer_resume':False,'duration_s':round(time.monotonic()-start,2)}
    (work/'training-receipt.json').write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(work/'training-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('dataset','candidate','baseline-val','candidate-val','diagnostics'):
            for p in (work/folder).rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(work).as_posix())
        for name in ('training-receipt.json','tests.log','artifacts.log','training.log','evaluation.log'):archive.write(work/name,name)
    event('complete',status='complete')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        messages={'Normalized PNG checksum mismatch; no automatic rehash':'NORMALIZED_IMAGE_HASH_MISMATCH',
            'Original image checksum mismatch':'SOURCE_IMAGE_HASH_MISMATCH',
            'Unique original image missing':'SOURCE_IMAGE_MISSING','Fixed checks failed':'CODE_CHECK_FAILED',
            'CUDA unavailable':'CUDA_UNAVAILABLE','Archive checksum changed':'ARCHIVE_HASH_MISMATCH'}
        event('failed',error_category=type(exc).__name__,failure_code=messages.get(str(exc),'TRAINING_PIPELINE_FAILED'))
        raise SystemExit(1) from None
    finally:DONE.set()
