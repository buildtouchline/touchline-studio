"""Bounded code-repair preparation/staging. Candidate code is never executed locally."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from contextlib import contextmanager
from .diagnostics import diagnosed

TARGET='src/touchline/video/sampling.py'
CHECK_ID='sampling-contract-v1'
MAX_SOURCE_BYTES=32768
MAX_ATTEMPTS=2


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def failure_action(code):
    actions={'INTERNAL_ERROR':'code_proposal','RESOURCE_EXHAUSTED':'parameter_review',
        'TRAINING_NONFINITE':'data_and_recipe_review','DATA_INVALID':'data_review',
        'MEDIA_UNSUPPORTED':'input_review','AUTH_REJECTED':'operator_access_review',
        'RATE_LIMITED':'reconcile_then_bounded_backoff','PROVIDER_UNAVAILABLE':'provider_reconciliation',
        'TIMEOUT':'worker_and_billing_reconciliation','BUDGET_REJECTED':'budget_review',
        'BUDGET_OVERRUN':'stop_and_reconcile','DEPENDENCY_MISSING':'review_pinned_image',
        'INPUT_INVALID':'input_review','STORAGE_UNAVAILABLE':'storage_review','REVIEW_REQUIRED':'coach_review'}
    if code not in actions:raise ValueError('Unknown repair failure category')
    return actions[code]


def _safe_file(root,relative):
    root=Path(root).absolute();path=root/relative
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Unsafe repair input path')
    if not path.is_file() or path.stat().st_size>MAX_SOURCE_BYTES:raise ValueError('Missing or oversized repair source')
    return path


def validate_packet(packet):
    if not isinstance(packet,dict) or set(packet)!={'schema','failure_code','check_id','source','baseline_id','case_id'}:
        raise ValueError('Invalid repair case schema')
    source=packet['source']
    if packet['schema']!='touchline.repair-case.v1' or packet['check_id']!=CHECK_ID or packet['failure_code']!='INTERNAL_ERROR':
        raise ValueError('This failure/target has no automatic code-repair capability')
    if not isinstance(source,dict) or set(source)!={'path','sha256','content'} or source['path']!=TARGET or not isinstance(source['content'],str):
        raise ValueError('Repair target is not registered')
    if len(source['content'].encode())>MAX_SOURCE_BYTES or hashlib.sha256(source['content'].encode()).hexdigest()!=source['sha256']:
        raise ValueError('Repair source hash mismatch')
    baseline=canonical_hash({'path':TARGET,'sha256':source['sha256'],'check_id':CHECK_ID})
    if packet['baseline_id']!=baseline or packet['case_id']!=canonical_hash({'baseline_id':baseline,'failure_code':packet['failure_code']}):
        raise ValueError('Repair case lineage mismatch')
    return packet


@diagnosed('repair.prepare')
def prepare_case(source_root,*,failure_code='INTERNAL_ERROR',target=TARGET):
    if target!=TARGET:raise ValueError('Only the registered sampling target has independent repair checks')
    if failure_action(failure_code)!='code_proposal':raise ValueError('Failure requires review/reconciliation, not an automatic code edit')
    content=_safe_file(source_root,TARGET).read_bytes().decode('utf-8')
    source={'path':TARGET,'sha256':hashlib.sha256(content.encode()).hexdigest(),'content':content}
    baseline=canonical_hash({'path':TARGET,'sha256':source['sha256'],'check_id':CHECK_ID})
    return {'schema':'touchline.repair-case.v1','failure_code':failure_code,'check_id':CHECK_ID,'source':source,
        'baseline_id':baseline,'case_id':canonical_hash({'baseline_id':baseline,'failure_code':failure_code})}


def _python_policy(content):
    if not isinstance(content,str) or len(content.encode())>MAX_SOURCE_BYTES or '\x00' in content:
        raise ValueError('Invalid replacement source')
    tree=ast.parse(content)
    forbidden={'eval','exec','compile','open','__import__','getattr','setattr','delattr','globals','locals','vars','breakpoint','input'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if isinstance(node,ast.ImportFrom) and node.level:raise ValueError('Relative imports are not permitted in this repair target')
            if any(n.split('.')[0] not in ('math','dataclasses','typing','collections','statistics') for n in names):
                raise ValueError('Unapproved import in repair proposal')
        if isinstance(node,ast.Name) and (node.id in forbidden or node.id.startswith('__')):
            raise ValueError('Unsafe dynamic capability in repair proposal')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('__') or node.attr in {'open','write','write_text','write_bytes','unlink','remove','rmdir','rename','system','popen'}):
            raise ValueError('Unsafe attribute capability in repair proposal')
    # This filter reduces accidental/deliberate scope expansion; it is not a Python sandbox.


@diagnosed('repair.validate')
def validate_proposal(packet,proposal):
    validate_packet(packet)
    if not isinstance(proposal,dict) or set(proposal)!={'schema','baseline_id','check_id','replacement'}:
        raise ValueError('Unexpected proposal fields or schema')
    if proposal['schema']!='touchline.patch-proposal.v1' or proposal['baseline_id']!=packet['baseline_id'] or proposal['check_id']!=CHECK_ID:
        raise ValueError('Stale or incompatible repair proposal')
    replacement=proposal['replacement']
    if not isinstance(replacement,dict) or set(replacement)!={'path','before_sha256','content'}:
        raise ValueError('Invalid replacement envelope')
    if replacement['path']!=TARGET or replacement['before_sha256']!=packet['source']['sha256']:
        raise ValueError('Protected path or stale source hash')
    if replacement['content']==packet['source']['content']:raise ValueError('No-op repair is not a candidate')
    _python_policy(replacement['content'])
    return {'schema':'touchline.validated-patch.v1','case_id':packet['case_id'],'baseline_id':packet['baseline_id'],
        'candidate_id':canonical_hash(proposal),'check_id':CHECK_ID,'proposal':proposal,'promotion_state':'not_promoted'}


@diagnosed('repair.stage')
def stage_proposal(packet,proposal,output,*,source_root=None):
    validated=validate_proposal(packet,proposal)
    if source_root is not None:
        current=_safe_file(source_root,TARGET).read_bytes().decode('utf-8')
        if hashlib.sha256(current.encode()).hexdigest()!=packet['source']['sha256']:
            raise ValueError('Source changed after case preparation; create a new case')
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Unsafe repair output path')
    if source_root is not None:
        root=Path(source_root).resolve()
        if output.is_relative_to(root) and output.relative_to(root).parts[:1]!=('runs',):
            raise ValueError('Stage inside runs or outside the source checkout; never overwrite project code')
    output.mkdir(parents=True,exist_ok=False)
    (output/'baseline.py').write_text(packet['source']['content'],encoding='utf-8',newline='\n')
    (output/'candidate.py').write_text(proposal['replacement']['content'],encoding='utf-8',newline='\n')
    (output/'case.json').write_text(json.dumps(packet,indent=2)+'\n',encoding='utf-8')
    (output/'validation.json').write_text(json.dumps(validated,indent=2)+'\n',encoding='utf-8')
    return validated


class RepairAttempts:
    """One trusted local controller; at most two claimed attempts per immutable case."""
    def __init__(self,path):
        self.path=str(path)
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS repair_attempts (case_id TEXT, attempt_id TEXT PRIMARY KEY, state TEXT, candidate_id TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS repair_reconciliations (attempt_id TEXT PRIMARY KEY, reviewer_id TEXT NOT NULL, evidence_sha256 TEXT NOT NULL, reviewed_utc TEXT NOT NULL)')
    @contextmanager
    def transaction(self):
        db=sqlite3.connect(self.path,timeout=10)
        try:
            db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except Exception:db.rollback();raise
        finally:db.close()
    def claim(self,case_id,attempt_id):
        if not re.fullmatch('[a-f0-9]{64}',case_id) or not re.fullmatch('[a-zA-Z0-9_-]{1,80}',attempt_id):
            raise ValueError('Invalid case/attempt identifier')
        with self.transaction() as db:
            previous=db.execute('SELECT case_id FROM repair_attempts WHERE attempt_id=?',(attempt_id,)).fetchone()
            if previous:
                if previous[0]!=case_id:raise ValueError('Attempt ID belongs to another case')
                return False
            states=[r[0] for r in db.execute('SELECT state FROM repair_attempts WHERE case_id=?',(case_id,))]
            if len(states)>=MAX_ATTEMPTS or any(s in ('claimed','unknown') for s in states):
                raise ValueError('Repair cap reached or previous work needs reconciliation')
            db.execute('INSERT INTO repair_attempts VALUES (?,?,?,NULL)',(case_id,attempt_id,'claimed'))
            return True
    def finish(self,attempt_id,state,*,candidate_id=None):
        if state not in ('proposed','rejected','checked','unknown'):raise ValueError('Invalid repair state')
        if candidate_id is not None and not re.fullmatch('[a-f0-9]{64}',candidate_id):raise ValueError('Invalid candidate identifier')
        with self.transaction() as db:
            row=db.execute('SELECT state FROM repair_attempts WHERE attempt_id=?',(attempt_id,)).fetchone()
            if row is None or row[0] not in ('claimed','proposed'):raise ValueError('Repair attempt cannot be rewritten')
            db.execute('UPDATE repair_attempts SET state=?,candidate_id=? WHERE attempt_id=?',(state,candidate_id,attempt_id))

    def reconcile_unknown(self,attempt_id,*,reviewer_id,evidence_sha256,
                          worker_terminated=False,billing_finalized=False):
        """Trusted operator attestation, never a provider query or model success claim.

        Retire an uncertain attempt as rejected, not checked. The evidence receipt
        must cover termination/no active work and final billing. Attempt counts,
        reservations and provider blocks are never reset by this operation.
        """
        if not isinstance(attempt_id,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,80}',attempt_id):
            raise ValueError('Invalid attempt ID')
        if not isinstance(reviewer_id,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,80}',reviewer_id):
            raise ValueError('Use a non-personal operator reviewer ID')
        if not isinstance(evidence_sha256,str) or not re.fullmatch('[a-f0-9]{64}',evidence_sha256):
            raise ValueError('Reconciliation evidence SHA256 is required')
        if worker_terminated is not True or billing_finalized is not True:
            raise ValueError('Confirm provider termination and finalized billing before reconciliation')
        with self.transaction() as db:
            prior=db.execute('SELECT reviewer_id,evidence_sha256 FROM repair_reconciliations WHERE attempt_id=?',(attempt_id,)).fetchone()
            if prior:
                if prior!=(reviewer_id,evidence_sha256):raise ValueError('Reconciliation evidence cannot be rewritten')
                return {'attempt_id':attempt_id,'state':'rejected','created':False}
            row=db.execute('SELECT state FROM repair_attempts WHERE attempt_id=?',(attempt_id,)).fetchone()
            if row is None or row[0]!='unknown':raise ValueError('Only unknown attempts may be reconciled')
            db.execute('INSERT INTO repair_reconciliations VALUES (?,?,?,?)',
                (attempt_id,reviewer_id,evidence_sha256,datetime.now(timezone.utc).isoformat()))
            db.execute("UPDATE repair_attempts SET state='rejected' WHERE attempt_id=?",(attempt_id,))
            return {'attempt_id':attempt_id,'state':'rejected','created':True}


@diagnosed('repair.check')
def run_repair_cycle(source_root,output,*,adviser,checker,attempt_id,quote,check_worst_case_usd,
                     enabled=False,source_reviewed_for_model=False,check_quote_verified=False,expected_baseline_id=None):
    """Optional end-to-end candidate automation; it never changes the live source tree."""
    if enabled is not True or source_reviewed_for_model is not True or check_quote_verified is not True:
        raise ValueError('Repair execution is disabled until the operator approves source, access and allocations')
    packet=prepare_case(source_root)
    if expected_baseline_id is not None and packet['baseline_id']!=expected_baseline_id:
        raise ValueError('Event baseline changed before adviser dispatch')
    proposal=adviser.propose(packet,attempt_id=attempt_id,quote=quote,enabled=True,source_reviewed_for_model=True)
    try:
        validated=stage_proposal(packet,proposal,output,source_root=source_root)
        receipt=checker.check(packet,proposal,attempt_id=attempt_id,worst_case_usd=check_worst_case_usd,
            enabled=True,quote_verified=True,run_dir=Path(output)/'check-run')
        if (not isinstance(receipt,dict) or receipt.get('candidate_id')!=validated['candidate_id'] or
            receipt.get('case_id')!=packet['case_id'] or receipt.get('check_id')!=CHECK_ID or
            type(receipt.get('passed')) is not bool or receipt.get('worker_terminated') is not True):
            raise ValueError('Stale/unverified checker receipt; never trust a model success message')
        adviser.attempts.finish(attempt_id,'checked' if receipt['passed'] else 'rejected',candidate_id=validated['candidate_id'])
        result={k:receipt.get(k) for k in ('schema','run_id','case_id','candidate_id','check_id','passed','worker_id','image_id','worker_terminated','backend')}
        result.update(promotion_state='human_review_required' if receipt['passed'] else 'rejected',live_source_changed=False)
        (Path(output)/'check-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
    except Exception:
        # A proposed attempt cannot be replayed blindly after uncertain worker state.
        try:adviser.attempts.finish(attempt_id,'unknown')
        except ValueError:pass
        raise
