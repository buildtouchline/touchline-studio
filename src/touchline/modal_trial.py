"""Contracts for an explicit, single, bounded video trial; no credential access."""
from contextlib import closing
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import zipfile

CAPS={'modal':'2.00','meta':'0.75','baseten':'0.25'}
AUTHORIZATION='video-trial-20261101-three-dollars'
SAM_FRAMES=(0,120,240,360)
GPU='T4'
GPU_TIMEOUT=600


def sandbox_options(gpu=GPU):
    if gpu not in ('T4','L4'):raise ValueError('Only explicitly approved affordable GPU types are allowed')
    return {'runtime':'gvisor','gpu':gpu,'cpu':(2,2),'memory':(8192,8192),
        'timeout':GPU_TIMEOUT,'block_network':True,'secrets':[],'volumes':{},'env':{},'workdir':'/work'}


class TrialBudget:
    """Keep full provider holds, including unknown billing; never auto-release."""
    def __init__(self,path):
        self.path=Path(path)
        if self.path.is_symlink():raise ValueError('Unsafe allowance path')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS allocations (authorization TEXT, provider TEXT, run_id TEXT, cap TEXT, PRIMARY KEY(authorization,provider))')

    def reserve(self,provider,run_id):
        if provider not in CAPS or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',run_id):raise ValueError('Invalid trial allocation')
        with closing(sqlite3.connect(self.path,timeout=10)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO allocations VALUES (?,?,?,?)',(AUTHORIZATION,provider,run_id,CAPS[provider]))
        return {'provider':provider,'reserved_usd':CAPS[provider],'billing':'unreconciled'}


def make_code_bundle(source_root,destination):
    """Only Python implementation files, never checkout/environment/media directories."""
    source_root=Path(source_root).resolve();destination=Path(destination)
    records=[]
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((source_root/'src'/'touchline').rglob('*.py')):
            if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Linked source is not allowed')
            data=path.read_bytes()
            if len(data)>1024*1024:raise ValueError('Unexpected large source file')
            relative=path.relative_to(source_root).as_posix()
            archive.writestr(relative,data)
            records.append({'path':relative,'sha256':hashlib.sha256(data).hexdigest()})
    return records


def parse_sam_boxes(text,width,height):
    """Read documented image-box tokens only; masks remain opaque, never executable.

    SAM inclusive endpoints are converted to this project's half-open XYXY.
    A source-pixel image must retain exact w/h; reject unexplained transforms.
    """
    if not isinstance(text,str) or len(text.encode())>8*1024*1024:raise ValueError('Oversized SAM text')
    if not text.strip():return []
    if not text.startswith('<0f>'):raise ValueError('Expected image frame-zero output')
    tokens=re.findall(r'<\|box;([^|]*)\|>',text)
    if text.count('<|box;')!=len(tokens) or len(tokens)>256:raise ValueError('Malformed SAM box output')
    if not tokens:raise ValueError('Nonempty SAM response has no supported box tokens')
    boxes=[]
    for token in tokens:
        parts=token.split(';');fields={}
        for part in parts:
            key,value=part.split('=',1)
            if key in fields:raise ValueError('Duplicate SAM field')
            fields[key]=value
        if set(fields) not in ({'x1','y1','x2','y2','w','h'},{'x1','y1','x2','y2','w','h','c'}):raise ValueError('Unsupported SAM box schema')
        values={k:int(fields[k]) for k in ('x1','y1','x2','y2','w','h')}
        if (values['w'],values['h'])!=(width,height):raise ValueError('SAM source geometry changed')
        x1,y1,x2,y2=(values[k] for k in ('x1','y1','x2','y2'))
        if not (0<=x1<=x2<width and 0<=y1<=y2<height):raise ValueError('SAM box outside source image')
        confidence=Decimal(fields['c']) if 'c' in fields else None
        if confidence is not None and (not confidence.is_finite() or not 0<=confidence<=1):raise ValueError('Invalid SAM confidence')
        boxes.append({'class_id':1,'bbox_xyxy':[x1,y1,x2+1,y2+1],
            'proposal_score':float(confidence) if confidence is not None else None,
            'annotation_status':'model_proposal','review_status':'pending','training_eligible':False})
    return boxes


def parse_sam_proposals(text, width, height):
    """Read documented image-box and polygon tokens from SAM 3.1 output (TL-033)."""
    boxes = parse_sam_boxes(text, width, height)
    poly_tokens = re.findall(r'<\|poly;([^|]*)\|>', text)
    if text.count('<|poly;') != len(poly_tokens) or poly_tokens and len(poly_tokens) != len(boxes):
        raise ValueError('SAM polygon/box cardinality mismatch')
    for box, token in zip(boxes, poly_tokens):
        fields = {}
        for part in token.split(';'):
            key, value = part.split('=', 1)
            if key in fields:raise ValueError('Duplicate polygon field')
            fields[key] = value
        if set(fields) not in ({'points'}, {'points','w','h'}):raise ValueError('Unsupported polygon schema')
        if 'w' in fields and (int(fields['w']),int(fields['h'])) != (width,height):raise ValueError('Polygon geometry changed')
        try:coords = [int(c.strip()) for c in fields['points'].split(',')]
        except ValueError:raise ValueError('Integer polygon coordinates required') from None
        if not 6 <= len(coords) <= 2048 or len(coords) % 2:
            raise ValueError('Bounded polygon vertex list required')
        points = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
        if any(not 0 <= x < width or not 0 <= y < height for x, y in points):
            raise ValueError('SAM polygon outside source image')
        box['polygon'] = points
    return boxes


def classify_repair(code):
    from .repair import failure_action
    action=failure_action(code)
    return {'enabled':True,'failure_code':code,'action':action,
        'patch_execution':'requires_registered_reproducer_and_independent_checks',
        'source_changed':False,'automatic_promotion':False}


def extract_run_bundle(archive_path,output):
    output=Path(output).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        entries=archive.infolist()
        if sum(e.file_size for e in entries)>64*1024*1024:raise ValueError('Run bundle exceeds limit')
        names=set()
        for entry in entries:
            name=entry.filename
            target=(output/name).resolve()
            if name in names or not name.startswith('run/') or not target.is_relative_to(output):raise ValueError('Unsafe output bundle')
            if (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Linked output')
            if target.exists():raise ValueError('Output overwrite refused')
            if Path(name).suffix not in ('.json','.jsonl','.html'):raise ValueError('Unexpected result type')
            names.add(name)
        for entry in entries:
            target=output/entry.filename;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.read(entry))
