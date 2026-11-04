"""Frozen analysis-job contracts and single-use authorization; no provider imports."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3

FILES=frozenset({'code.zip','candidate.pt','video24.mp4','worker.py','check_cuda_tracking.py'})
MAX_INPUT_BYTES=256*1024*1024
MAX_OUTPUT_BYTES=64*1024*1024
AUTHORIZATION='modal-analysis-20261102-one-dollar'


def load_frozen_inputs(specification,root,work):
    root=Path(root).resolve();work=Path(work).resolve()
    expected={'code.zip':work/'code.zip','candidate.pt':root/'artifacts/models/football-person/candidate.pt',
        'video24.mp4':root/'runs/video-trial-20261101/normalized/video24.mp4',
        'worker.py':root/'tools/modal_video_worker.py','check_cuda_tracking.py':root/'tools/check_cuda_tracking.py'}
    paths=specification.get('input_paths')
    if not isinstance(paths,dict) or set(paths)!=FILES:raise ValueError('Exact allowlisted files required')
    for name,path in expected.items():
        if Path(paths[name]).absolute()!=path.absolute() or not path.is_file():raise ValueError('Input path is not allowlisted')
        if any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root)):
            raise ValueError('Linked input refused')
    if sum(p.stat().st_size for p in expected.values())>MAX_INPUT_BYTES:raise ValueError('Input byte budget exceeded')
    files={n:p.read_bytes() for n,p in expected.items()}
    fingerprint=validate_payload(files,specification['hashes'])
    return files,fingerprint


def validate_payload(files,hashes):
    if not isinstance(files,dict) or set(files)!=FILES or not isinstance(hashes,dict) or set(hashes)!=FILES:
        raise ValueError('Exact frozen analysis inputs required')
    if any(type(v) is not bytes for v in files.values()) or sum(len(v) for v in files.values())>MAX_INPUT_BYTES:
        raise ValueError('Input byte budget exceeded')
    for name,data in files.items():
        digest=hashes[name]
        if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest) or hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('Frozen input changed')
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class AnalysisAllowance:
    """Hold the explicit additional $1 without releasing earlier trial holds."""
    def __init__(self,path):
        path=Path(path)
        if path.is_symlink():raise ValueError('Linked allowance refused')
        path.parent.mkdir(parents=True,exist_ok=True);self.path=path
        with closing(sqlite3.connect(path)) as db,db:
            db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT, reserved_micro_usd INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS bounded_retry (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT, parent_attempt TEXT)')

    def reserve(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):
            raise ValueError('Exact run and payload fingerprint required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO claims VALUES (?,?,?,?)',(AUTHORIZATION,run_id,fingerprint,1000000))
        return {'authorization':AUTHORIZATION,'reserved_usd':'1.00','billing':'unreconciled',
                'scope':'one Modal analysis attempt; no Meta/Baseten call or training'}

    def reserve_retry(self,run_id,fingerprint,parent_attempt):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint) or not re.fullmatch('[a-f0-9]{32}',parent_attempt):
            raise ValueError('Bounded retry lineage required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT reserved_micro_usd FROM claims WHERE authorization=?',(AUTHORIZATION,)).fetchone()!=(1000000,):
                raise ValueError('Original one-dollar hold required')
            db.execute('INSERT INTO bounded_retry VALUES (?,?,?,?)',(AUTHORIZATION,run_id,fingerprint,parent_attempt))
        return {'authorization':AUTHORIZATION,'reserved_usd':'1.00','billing':'unreconciled',
            'scope':'one explicitly approved logged retry; shares original dollar hold, not an additional dollar'}
