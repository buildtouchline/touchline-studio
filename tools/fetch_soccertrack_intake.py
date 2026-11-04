"""Trusted public-source intake broker. Never sends HF credentials to workers."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,shutil,sys,urllib.request,urllib.error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,save
from touchline.data.manifest import sha256_file

REPO='atomscott/soccertrack-v2'
def fetch(url,token='',limit=8*1024**2,byte_range=None):
    headers={'User-Agent':'Touchline-dataset-intake/1'}
    if token:headers['Authorization']='Bearer '+token
    if byte_range is not None:headers['Range']='bytes='+str(byte_range[0])+'-'+str(byte_range[1])
    # urllib strips Authorization on cross-host redirects only with our handler.
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            result=super().redirect_request(req,fp,code,msg,headers,newurl)
            if result is not None:
                from urllib.parse import urlsplit
                if urlsplit(req.full_url).netloc!=urlsplit(newurl).netloc:result.remove_header('Authorization')
            return result
    with urllib.request.build_opener(Redirect()).open(urllib.request.Request(url,headers=headers),timeout=90) as r:
        data=r.read(limit+1)
        if len(data)>limit:raise ValueError('Source metadata exceeds bound')
        return data

def run(*,clips=False,schema_sample=False):
    target=ROOT/'artifacts/soccertrack-v2';target.mkdir(parents=True,exist_ok=True)
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):settings=load_operator_settings(ROOT/'.env',environ={})
    token=settings.credentials.get('HF_TOKEN')
    if not token:return {'status':'HF_TOKEN_missing','credentials':'not displayed'}
    metadata=json.loads(fetch('https://huggingface.co/api/datasets/'+REPO+'?blobs=true',token))
    revision=metadata['sha'];files=[{'name':x['rfilename'],'size':x.get('size'),'lfs_sha256':x.get('lfs',{}).get('sha256')} for x in metadata['siblings']]
    if (target/'source-metadata.json').exists() and json.loads((target/'source-metadata.json').read_text())['revision']!=revision:raise ValueError('Source revision changed; review a fresh intake directory')
    save(target/'source-metadata.json',{'repo':REPO,'revision':revision,'gated':metadata.get('gated'),'files':files,'training_eligible':False})
    allowed=['README.md','mot/README.md','CHANGES_v1_2.md',*['mot/'+m+'.txt' for m in ('117092','117093','118575','118577')]]
    downloads=[]
    for name in allowed:
        if not any(f['name']==name for f in files):continue
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            payload=fetch('https://huggingface.co/datasets/'+REPO+'/resolve/'+revision+'/'+name,token,12*1024**2)
            temp=p.with_suffix(p.suffix+'.tmp');temp.write_bytes(payload);temp.replace(p)
        downloads.append({'file':name,'sha256':sha256_file(p),'bytes':p.stat().st_size})
    save(target/'intake.json',{'status':'quarantined','revision':revision,'downloads':downloads,'labels_reviewed':False,'training_eligible':False})
    if schema_sample:
        name='gsr/118575/118575_1st.json';entry=next(f for f in files if f['name']==name);ranges=[(0,65535),(entry['size']-131072,entry['size']-1)];samples=[]
        for label,span in zip(('head','tail'),ranges):
            p=target/('gsr-schema-'+label+'.json-fragment')
            if not p.exists():p.write_bytes(fetch('https://huggingface.co/datasets/'+REPO+'/resolve/'+revision+'/'+name,token,span[1]-span[0]+1,span))
            samples.append({'file':p.name,'range_inclusive':list(span),'sha256':sha256_file(p),'bytes':p.stat().st_size})
        save(target/'gsr-schema-sampling.json',{'source':name,'source_size':entry['size'],'revision':revision,'samples':samples,'complete_file_downloaded':False,'training_eligible':False})
    if clips:
        for match in ('117092','117093','118575','118577'):
            name='mot/clips/'+match+'.mp4';entry=next(f for f in files if f['name']==name);p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                if p.stat().st_size!=entry['size'] or sha256_file(p)!=entry['lfs_sha256']:raise ValueError('Existing clip differs from pinned source')
                continue
            if shutil.disk_usage(target).free<entry['size']*3:raise ValueError('Insufficient intake disk space')
            original=ROOT/'artifacts/videos'/p.name
            if original.is_file() and original.stat().st_size==entry['size'] and sha256_file(original)==entry['lfs_sha256']:
                shutil.copyfile(original,p)
            else:
                # The credential goes only to HF; signed CDN redirects omit it.
                class ClipRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self,req,fp,code,msg,headers,newurl):
                        result=super().redirect_request(req,fp,code,msg,headers,newurl)
                        if result is not None:
                            from urllib.parse import urlsplit
                            if urlsplit(req.full_url).netloc!=urlsplit(newurl).netloc:result.remove_header('Authorization')
                        return result
                url='https://huggingface.co/datasets/'+REPO+'/resolve/'+revision+'/'+name
                request=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'User-Agent':'Touchline-dataset-intake/1'})
                tmp=p.with_suffix('.mp4.tmp');received=0
                print(json.dumps({'stage':'clip_download','match':match,'status':'started','expected_bytes':entry['size']}),flush=True)
                with urllib.request.build_opener(ClipRedirect()).open(request,timeout=180) as response,tmp.open('xb') as out:
                    while chunk:=response.read(1024**2):
                        received+=len(chunk)
                        if received>entry['size']:raise ValueError('Clip exceeds pinned size')
                        out.write(chunk)
                if received!=entry['size'] or sha256_file(tmp)!=entry['lfs_sha256']:raise ValueError('Downloaded clip hash mismatch')
                tmp.replace(p)
            print(json.dumps({'stage':'clip_download','match':match,'status':'verified','bytes':p.stat().st_size}),flush=True)
    return {'status':'quarantined','revision':revision,'downloaded_files':len(downloads),'mot_files':[f for f in files if f['name'].startswith('mot/')]}

if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser();parser.add_argument('--clips',action='store_true');parser.add_argument('--gsr-schema-sample',action='store_true');args=parser.parse_args()
    try:print(json.dumps(run(clips=args.clips,schema_sample=args.gsr_schema_sample),indent=2))
    except urllib.error.HTTPError as exc:print(json.dumps({'status':'download_blocked','http_status':exc.code,'secrets':'not displayed'}));raise SystemExit(1) from None
    except Exception as exc:print(json.dumps({'status':'intake_failed','error_category':type(exc).__name__}));raise SystemExit(1) from None
