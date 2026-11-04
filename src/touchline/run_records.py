"""Portable run identity and explicit content-restricted local support exports."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import re
import uuid
from .diagnostics import diagnosed,diagnostic_summary


def new_record(kind,*,backend='python',run_id=None,parent_run_id=None):
    if kind not in ('training','analysis','repair') or backend not in ('python','colab','modal'):
        raise ValueError('Unknown run kind/backend')
    identifier=run_id or str(uuid.uuid4())
    if not isinstance(identifier,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,120}',identifier):raise ValueError('Invalid run identifier')
    if parent_run_id is not None and (not isinstance(parent_run_id,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,120}',parent_run_id)):
        raise ValueError('Invalid parent run identifier')
    return {'schema':'touchline.run-record.v1','run_id':identifier,'parent_run_id':parent_run_id,
        'kind':kind,'backend':backend,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'preparing'}


def write_record(directory,record):
    path=Path(directory)/'run-record.json';temporary=Path(directory)/'run-record.json.tmp'
    if path.is_symlink() or temporary.is_symlink():raise ValueError('Unsafe run record path')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def _owned_file(root,name):
    root=Path(root).absolute();path=root/name
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe run artifact path')
    return path


@diagnosed('run.export')
def export_run_logs(run_dir,output,*,diagnostics_dir=None):
    root=Path(run_dir);output=Path(output)
    record_path=_owned_file(root,'run-record.json');manifest_path=_owned_file(root,'manifest.json')
    if record_path.is_file():record=json.loads(record_path.read_text(encoding='utf-8'))
    elif manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        record={'run_id':manifest.get('run_id'),'kind':'analysis','backend':'python',
            'status':manifest.get('status'),'diagnostic_trace_id':manifest.get('diagnostic_trace_id')}
    else:raise ValueError('No run record; preserve legacy files and identify the original run explicitly')
    run_id=record.get('run_id')
    if not isinstance(run_id,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,120}',run_id):raise ValueError('Run ID missing or invalid')
    status=record.get('status');kind=record.get('kind');backend=record.get('backend')
    if status not in ('preparing','running','completed','complete','failed','partial','capped','cancelled') or kind not in ('training','analysis','repair') or backend not in ('python','colab','modal'):
        raise ValueError('Unsupported run record state')
    safe={'schema':'touchline.run-support.v1','run_id':run_id,'kind':kind,'backend':backend,'status':status,
        'exported_utc':datetime.now(timezone.utc).isoformat(),'trace_available':False}
    trace_id=record.get('diagnostic_trace_id')
    if trace_id is not None and (not isinstance(trace_id,str) or not re.fullmatch('[a-f0-9]{32}',trace_id)):
        raise ValueError('Invalid diagnostic trace identifier')
    if trace_id and diagnostics_dir is not None:
        safe['diagnostic_summary']=diagnostic_summary(Path(diagnostics_dir)/(trace_id+'.jsonl'));safe['trace_available']=True
    loss_path=_owned_file(root,'losses.jsonl');losses=[]
    if loss_path.is_file():
        if loss_path.stat().st_size>16*1024*1024:raise ValueError('Loss log needs a bounded export window')
        with loss_path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line)
                if not isinstance(row,dict) or any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) or row[k]<0 for k in ('step','epoch','loss')):
                    raise ValueError('Invalid numeric loss record')
                losses.append({k:row[k] for k in ('step','epoch','loss')})
    safe['loss_record_count']=len(losses)
    if any(p.is_symlink() for p in (output,*output.absolute().parents)):raise ValueError('Unsafe support export location')
    output.mkdir(parents=True,exist_ok=False)
    (output/'support.json').write_text(json.dumps(safe,indent=2)+'\n',encoding='utf-8')
    with (output/'losses.jsonl').open('w',encoding='utf-8') as stream:
        for row in losses:stream.write(json.dumps(row)+'\n')
    return safe
