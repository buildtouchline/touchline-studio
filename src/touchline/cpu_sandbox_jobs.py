"""Single approved CPU Sandbox analysis; no secrets or provider imports."""
from contextlib import closing
from pathlib import Path
import re
import sqlite3
import zipfile

AUTHORIZATION='modal-cpu-analysis-20261102-ten-dollars'


def sandbox_options():
    return {'runtime':'gvisor','gpu':None,'cpu':(4,4),'memory':(8192,8192),'timeout':7200,
        'block_network':True,'secrets':[],'volumes':{},'workdir':'/work',
        'env':{'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLCONFIGDIR':'/work/cache/matplotlib','TORCH_HOME':'/work/cache/torch'}}


class CpuSandboxAllowance:
    def __init__(self,path):
        self.path=Path(path)
        if self.path.is_symlink():raise ValueError('Linked allowance refused')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db,db:
            db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT, reserved_micro_usd INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS remote_checks (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS remote_checks_retry (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS tracking_trial (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS event_checks (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS final_checks (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS handover_checks (authorization TEXT PRIMARY KEY, run_id TEXT, fingerprint TEXT)')

    def reserve(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):raise ValueError('Exact CPU job lineage required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO claims VALUES (?,?,?,?)',(AUTHORIZATION,run_id,fingerprint,10000000))
        return {'authorization':AUTHORIZATION,'reserved_usd':'10.00','billing':'unreconciled','gpu':None}

    def reserve_remote_checks(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):raise ValueError('Frozen remote-check lineage required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT reserved_micro_usd FROM claims WHERE authorization=?',(AUTHORIZATION,)).fetchone()!=(10000000,):
                raise ValueError('Original CPU reservation required')
            db.execute('INSERT INTO remote_checks VALUES (?,?,?)',(AUTHORIZATION,run_id,fingerprint))
        return {'authorization':AUTHORIZATION,'reserved_usd':'10.00','shares_original_hold':True,'gpu':None,'billing':'unreconciled'}

    def reserve_remote_checks_retry(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):raise ValueError('Frozen check retry required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT COUNT(*) FROM remote_checks WHERE authorization=?',(AUTHORIZATION,)).fetchone()!=(1,):raise ValueError('Original remote checks required')
            db.execute('INSERT INTO remote_checks_retry VALUES (?,?,?)',(AUTHORIZATION,run_id,fingerprint))
        return {'authorization':AUTHORIZATION,'reserved_usd':'10.00','shares_original_hold':True,'gpu':None,'billing':'unreconciled'}

    def reserve_tracking_trial(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):raise ValueError('Frozen tracking trial required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT reserved_micro_usd FROM claims WHERE authorization=?',(AUTHORIZATION,)).fetchone()!=(10000000,):
                raise ValueError('Original CPU reservation required')
            db.execute('INSERT INTO tracking_trial VALUES (?,?,?)',(AUTHORIZATION,run_id,fingerprint))
        return {'authorization':AUTHORIZATION,'reserved_usd':'10.00','shares_original_hold':True,'gpu':None,'billing':'unreconciled'}

    def reserve_event_checks(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):raise ValueError('Frozen event-check lineage required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT reserved_micro_usd FROM claims WHERE authorization=?',(AUTHORIZATION,)).fetchone()!=(10000000,):raise ValueError('Original CPU hold required')
            db.execute('INSERT INTO event_checks VALUES (?,?,?)',(AUTHORIZATION,run_id,fingerprint))
        return {'authorization':AUTHORIZATION,'reserved_usd':'10.00','shares_original_hold':True,'gpu':None,'billing':'unreconciled'}

    def reserve_final_checks(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):raise ValueError('Frozen final-check lineage required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT reserved_micro_usd FROM claims WHERE authorization=?',(AUTHORIZATION,)).fetchone()!=(10000000,):raise ValueError('Original CPU hold required')
            db.execute('INSERT INTO final_checks VALUES (?,?,?)',(AUTHORIZATION,run_id,fingerprint))
        return {'authorization':AUTHORIZATION,'reserved_usd':'10.00','shares_original_hold':True,'gpu':None,'billing':'unreconciled'}

    def reserve_handover_checks(self,run_id,fingerprint):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id) or not re.fullmatch('[a-f0-9]{64}',fingerprint):raise ValueError('Frozen handover-check lineage required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT reserved_micro_usd FROM claims WHERE authorization=?',(AUTHORIZATION,)).fetchone()!=(10000000,):raise ValueError('Original CPU hold required')
            db.execute('INSERT INTO handover_checks VALUES (?,?,?)',(AUTHORIZATION,run_id,fingerprint))
        return {'authorization':AUTHORIZATION,'reserved_usd':'10.00','shares_original_hold':True,'gpu':None,'billing':'unreconciled'}


def extract_results(archive_path,output):
    output=Path(output).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        entries=archive.infolist();names=set()
        if len(entries)>10000 or sum(e.file_size for e in entries)>1024**3:raise ValueError('CPU results exceed output budget')
        for entry in entries:
            p=Path(entry.filename);target=(output/p).resolve()
            if (entry.filename in names or not p.parts or p.parts[0] not in {'run','profiles','preview','diagnostics','cpu-receipt.json','RESULTS.html'} or
                not target.is_relative_to(output) or target.exists() or (entry.external_attr>>16)&0o170000==0o120000 or
                p.suffix.lower() not in {'.json','.jsonl','.html','.md','.mp4','.jpg','.png'}):raise ValueError('Unsafe CPU result entry')
            names.add(entry.filename)
        for entry in entries:
            p=output/entry.filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.read(entry))
