"""Pin the public upstream baseline for review; never execute downloaded code."""
from pathlib import Path
import hashlib,json,re,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
FILES=('LICENSE','baselines/mot/train.py','baselines/mot/eval.py','baselines/mot/config.yaml','baselines/gsr/train.py','src/data_utils/create_yolo_dataset.py','src/evaluation/mot_hota.py','pyproject.toml')
def fetch(url,limit=256*1024):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Touchline-source-audit/1'}),timeout=60) as r:
        data=r.read(limit+1)
        if len(data)>limit:raise ValueError('Unexpected source size')
        return data
def main():
    out=ROOT/'artifacts/soccertrack-upstream';out.mkdir(exist_ok=False)
    commit=json.loads(fetch('https://api.github.com/repos/AtomScott/SoccerTrack-v2/commits/main'))['sha']
    if not re.fullmatch('[a-f0-9]{40}',commit):raise ValueError('Invalid upstream commit')
    records=[]
    for name in FILES:
        data=fetch('https://raw.githubusercontent.com/AtomScott/SoccerTrack-v2/'+commit+'/'+name);p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        records.append({'file':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    (out/'source-receipt.json').write_text(json.dumps({'git_commit':commit,'repository':'https://github.com/AtomScott/SoccerTrack-v2','files':records,'downloaded_code_executed':False,'credentials_used':False},indent=2))
    print(json.dumps({'status':'pinned_for_adaptation','git_commit':commit,'files':len(records),'downloaded_code_executed':False}))
if __name__=='__main__':
    try:main()
    except Exception as exc:print(json.dumps({'status':'failed','error_category':type(exc).__name__}));raise SystemExit(1) from None
