"""Trusted, bounded log retrieval for an existing analysis App; no job dispatch."""
from datetime import datetime,timezone
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import save,modal_client,Discard
from probe_modal_gpu import safe_error


def summarize(text):
    patterns={'missing_module':r'No module named','memory_error':r'MemoryError|out of memory|OOMKilled',
        'startup_timeout':r'startup.{0,80}(?:timeout|timed out)',
        'image_error':r'image.{0,80}(?:failed|error)|pull.{0,80}failed',
        'permission_error':r'PermissionError|Permission denied',
        'file_not_found':r'FileNotFoundError|No such file or directory',
        'disk_exhausted':r'No space left|disk quota',
        'serialization_error':r'(?:serializ|pickle).{0,80}(?:error|failed)|DeserializationError',
        'internal_failure':r'InternalFailure|internal failure',
        'network_error':r'Network is unreachable|ConnectionError|Connection refused|ConnectTimeout',
        'container_crash':r'container.{0,80}(?:crash|exit|failed)',
        'initialization_error':r'initializ.{0,80}(?:error|failed)',
        'traceback':r'Traceback \(most recent call last\)'}
    flags={k:len(re.findall(v,text,re.IGNORECASE)) for k,v in patterns.items()}
    events=[]
    for line in text.splitlines():
        if 'TOUCHLINE_EVENT ' not in line:continue
        try:item=json.loads(line.split('TOUCHLINE_EVENT ',1)[1])
        except Exception:continue
        if not isinstance(item,dict) or set(item)!={'stage','elapsed_s'}:continue
        if item['stage'] not in {'inputs_verified','worker_started','worker_alive','worker_finished','cuda','parity','analysis','export'}:continue
        elapsed=item['elapsed_s']
        if type(elapsed) not in (int,float) or not 0<=elapsed<=600:continue
        events.append({'stage':item['stage'],'elapsed_s':elapsed})
    modules=('touchline','torch','torchvision','supervision','numpy','av','modal_analysis_function')
    missing=[n for n in modules if re.search(r"No module named ['\"]"+re.escape(n)+r"['\"]",text)]
    return {'flags':{k:v for k,v in flags.items() if v},'known_missing_modules':missing,'events':events,
            'raw_lines':len(text.splitlines()),'raw_retained':False}


def fetch(work,settings,*,live_snapshot=False):
    record=json.loads((work/'modal-function.json').read_text(encoding='utf-8'))
    app_id=record.get('app_id','')
    if not re.fullmatch(r'ap-[a-zA-Z0-9_-]{1,100}',app_id):raise ValueError('Existing App ID required')
    absent=work/'no-modal-log-profile.toml'
    if absent.exists():raise ValueError('Reserved profile path must be absent')
    # Only the trusted child gets this explicitly selected token pair. Never log env.
    env={k:v for k,v in os.environ.items() if k.upper() in {'SYSTEMROOT','WINDIR','PATH','PATHEXT','TEMP','TMP','SYSTEMDRIVE'}}
    env.update(MODAL_CONFIG_PATH=str(absent),MODAL_SERVER_URL='https://api.modal.com',MODAL_LOGLEVEL='CRITICAL',
        MODAL_TOKEN_ID=settings.credentials.get('MODAL_TOKEN_ID'),MODAL_TOKEN_SECRET=settings.credentials.get('MODAL_TOKEN_SECRET'),
        MODAL_ENVIRONMENT='main',NO_COLOR='1',PYTHONIOENCODING='utf-8')
    process=subprocess.Popen([sys.executable,'-m','modal','app','logs',app_id,'--env','main','--tail','300','--timestamps'],
        env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    timer=threading.Timer(45,process.kill);timer.start();raw=bytearray();truncated=False
    try:
        while True:
            chunk=process.stdout.read(4096)
            if not chunk:break
            if len(chunk)>1024*1024-len(raw):truncated=True;process.kill();break
            raw.extend(chunk)
        process.wait(timeout=5)
    finally:
        timer.cancel()
        if process.poll() is None:process.kill();process.wait()
        process.stdout.close()
    report={'schema':'touchline.modal-log-summary.v1','app_id':app_id,'attempt_id':record['attempt_id'],
        'fetched_utc':datetime.now(timezone.utc).isoformat(),'fetch_succeeded':process.returncode==0 and not truncated,
        'truncated':truncated,'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),
        **summarize(raw.decode('utf-8',errors='replace'))}
    report['call_graph_best_effort']=True
    if re.fullmatch(r'fc-[a-zA-Z0-9_-]{1,100}',record.get('function_call_id','')):
        try:
            with redirect_stdout(Discard()),redirect_stderr(Discard()):
                modal,client=modal_client(settings,work)
                graph=modal.FunctionCall.from_id(record['function_call_id'],client=client).get_call_graph()
            nodes=list(graph);states=[]
            while nodes and len(states)<128:
                node=nodes.pop();nodes.extend(node.children)
                name=node.status.name
                states.append({'status':name if name in {'PENDING','SUCCESS','FAILURE','INIT_FAILURE','TERMINATED','TIMEOUT'} else 'UNKNOWN',
                    'task_assigned':bool(node.task_id)})
            report['input_states']=states
        except Exception as exc:report['call_graph_error']=safe_error(exc)
    suffix='-live' if live_snapshot is True else ''
    save(work/f'provider-log-summary{suffix}.json',report)
    with (work/f'provider-events{suffix}.jsonl').open('w',encoding='utf-8') as stream:
        for event in report['events']:stream.write(json.dumps({'attempt_id':record['attempt_id'],**event})+'\n')
    return report


if __name__=='__main__':
    import argparse
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=ROOT/'runs/modal-analysis-20261102')
    parser.add_argument('--live-snapshot',action='store_true')
    args=parser.parse_args()
    from touchline.settings import load_operator_settings
    try:
        work=args.work.resolve()
        if not work.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Project run directory required')
        print(json.dumps(fetch(work,load_operator_settings(ROOT/'.env',environ={}),live_snapshot=args.live_snapshot),indent=2))
    except Exception as exc:
        print(json.dumps({'fetch_succeeded':False,**safe_error(exc)}));raise SystemExit(1) from None
