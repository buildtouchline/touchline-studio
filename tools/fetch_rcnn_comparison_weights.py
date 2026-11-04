"""Fetch official Torchvision weights without loading/running a model locally."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
URL='https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'

def main():
    out=ROOT/'artifacts/rcnn-comparison';out.mkdir(parents=True,exist_ok=True)
    target=out/URL.rsplit('/',1)[1]
    if not target.exists():
        partial=target.with_suffix('.download')
        if partial.exists():raise FileExistsError('Review previous partial download')
        with urllib.request.urlopen(URL,timeout=60) as response,partial.open('xb') as dst:
            total=0
            while chunk:=response.read(1024**2):
                total+=len(chunk)
                if total>200*1024**2:raise ValueError('Official weight download exceeds bound')
                dst.write(chunk)
        with partial.open('rb') as src:digest=hashlib.file_digest(src,'sha256').hexdigest()
        if not digest.startswith('dd69338a'):raise ValueError('Official filename checksum mismatch')
        partial.replace(target)
    with target.open('rb') as src:digest=hashlib.file_digest(src,'sha256').hexdigest()
    if not digest.startswith('dd69338a'):raise ValueError('Official filename checksum mismatch')
    record={'architecture':'fasterrcnn_resnet50_fpn_v2','source':URL,'sha256':digest,'bytes':target.stat().st_size,'training_source':'Torchvision COCO_V1','trained_on_touchline':False,'local_inference':False}
    (out/'weights.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
if __name__=='__main__':main()
