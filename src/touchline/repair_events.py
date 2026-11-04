"""Durable failure events and an explicit trusted-controller repair dispatcher."""
from contextlib import closing
import json
from pathlib import Path,PurePosixPath
import re,sqlite3,uuid
from .repair import TARGET,CHECK_ID,canonical_hash,failure_action
from .data.manifest import sha256_file

STAGES={'analysis','tracking','training','normalization','provider','input','restore','export','checks'}


def failure_event(run_id,attempt_id,stage,failure_code,*,source=None):
    for value in (run_id,attempt_id):
        if not isinstance(value,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,100}',value):raise ValueError('Bounded run/attempt IDs required')
    if stage not in STAGES:raise ValueError('Registered failure stage required')
    failure_action(failure_code)
    if source is not None:
        if not isinstance(source,dict) or set(source)!={'path','sha256','check_id'}:raise ValueError('Exact source lineage required')
        if not isinstance(source['path'],str) or not isinstance(source['sha256'],str):raise ValueError('Text source lineage required')
        p=PurePosixPath(source['path'])
        if p.is_absolute() or '..' in p.parts or '\\' in source['path'] or not source['path'].startswith(('src/touchline/','tools/')) or p.suffix!='.py':raise ValueError('Unsafe event source path')
        if not re.fullmatch('[a-f0-9]{64}',source['sha256']) or source['check_id'] not in (None,CHECK_ID):raise ValueError('Invalid source/check lineage')
    body={'schema':'touchline.repair-event.v1','run_id':run_id,'attempt_id':attempt_id,'stage':stage,
        'failure_code':failure_code,'source':source}
    return {**body,'event_id':canonical_hash(body)}


def validate_event(event):
    if not isinstance(event,dict) or set(event)!={'schema','run_id','attempt_id','stage','failure_code','source','event_id'}:raise ValueError('Unexpected failure event fields')
    expected=failure_event(event['run_id'],event['attempt_id'],event['stage'],event['failure_code'],source=event['source'])
    if event!=expected:raise ValueError('Stale or modified event')
    return event


def publish_job_failure(work,record,*,queue_path,stage):
    """Job controllers publish after their final termination receipt is saved.

    Unknown worker state never becomes an automatic patch. No raw exceptions,
    HTTP payloads, credentials or private paths are put into the event.
    """
    if record.get('status')!='failed':return None
    work=Path(work);code='PROVIDER_UNAVAILABLE' if stage=='provider' else 'INTERNAL_ERROR'
    if record.get('worker_terminated') is not True:code='PROVIDER_UNAVAILABLE'
    journal=work/'worker.events.jsonl'
    if journal.is_file() and journal.stat().st_size<=1024**2:
        events=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines() if line.strip()]
        failures=[v for v in events if v.get('stage')=='failed' or v.get('status')=='failed']
        if failures:
            fixed=failures[-1].get('failure_code')
            if isinstance(fixed,str):
                if fixed in {'SOURCE_IMAGE_HASH_MISMATCH','NORMALIZED_IMAGE_HASH_MISMATCH','SOURCE_IMAGE_MISSING','ARCHIVE_HASH_MISMATCH'}:code='DATA_INVALID'
                elif fixed=='CUDA_UNAVAILABLE':code='DEPENDENCY_MISSING'
                else:
                    try:failure_action(fixed);code=fixed
                    except ValueError:pass
        elif stage=='provider':code='PROVIDER_UNAVAILABLE'
    event=failure_event(work.name,record.get('attempt_id',work.name),stage,code)
    result=RepairEventQueue(queue_path).enqueue(event)
    (work/'repair-event.json').write_text(json.dumps(event,indent=2)+'\n',encoding='utf-8')
    return result


def registered_code_handler(source_root,output_root,*,adviser,checker,quote,check_worst_case_usd):
    """Connect a trusted event dispatcher to the existing staged repair cycle.

    Caller supplies verified quotes, reviewed source access and budgeted clients.
    The checker's receipt is independently validated by run_repair_cycle.
    """
    from .repair import run_repair_cycle,prepare_case
    def handle(event):
        validate_event(event)
        if event['failure_code']!='INTERNAL_ERROR' or event['source'] is None or event['source']['path']!=TARGET or event['source']['check_id']!=CHECK_ID:
            raise ValueError('Unregistered repair event')
        packet=prepare_case(source_root)
        if packet['source']['sha256']!=event['source']['sha256']:raise ValueError('Stale event source')
        return run_repair_cycle(source_root,Path(output_root)/event['event_id'],adviser=adviser,checker=checker,
            attempt_id='event-'+event['event_id'][:48],quote=quote,check_worst_case_usd=check_worst_case_usd,
            enabled=True,source_reviewed_for_model=True,check_quote_verified=True,expected_baseline_id=packet['baseline_id'])
    return handle


class RepairEventQueue:
    """Single trusted controller; unknown leases require explicit reconciliation."""
    def __init__(self,path):
        self.path=Path(path)
        if self.path.is_symlink():raise ValueError('Linked queue refused')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db,db:
            db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL, lease TEXT, result TEXT)')

    def enqueue(self,event):
        validate_event(event)
        with closing(sqlite3.connect(self.path)) as db,db:
            row=db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,NULL,NULL)',(event['event_id'],json.dumps(event,sort_keys=True),'queued'))
            return {'event_id':event['event_id'],'created':row.rowcount==1}

    def claim(self,event_id):
        if not re.fullmatch('[a-f0-9]{64}',event_id):raise ValueError('Event ID required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT payload,state FROM events WHERE id=?',(event_id,)).fetchone()
            if row is None or row[1]!='queued':return None
            lease=uuid.uuid4().hex
            db.execute("UPDATE events SET state='claimed',lease=? WHERE id=?",(lease,event_id))
            return {'event':validate_event(json.loads(row[0])),'lease':lease}

    def finish(self,event_id,lease,result):
        if result.get('state') not in {'diagnosed','checked','blocked','major_approval_required','provider_reconciliation','review_required','unknown'}:raise ValueError('Fixed event result state required')
        with closing(sqlite3.connect(self.path,timeout=10)) as db,db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("UPDATE events SET state=?,result=? WHERE id=? AND lease=? AND state='claimed'",
                (result['state'],json.dumps(result,sort_keys=True),event_id,lease))
            if row.rowcount!=1:raise ValueError('Unknown/stale event lease')


def dispatch_event(queue,event_id,*,handlers,source_root=None,enabled=False,worker_terminated=False,failure_reproduced=False):
    """Handlers are trusted budgeted adapters, never model-returned functions.

    This consumes one event, not a background service. No lease expires into an
    automatic retry. Existing RepairAttempts/SpendingLedger enforce paid attempts.
    """
    if enabled is not True:raise ValueError('Event dispatch requires explicit enablement')
    if worker_terminated is not True:return {'state':'blocked','reason':'worker_reconciliation_required'}
    claimed=queue.claim(event_id)
    if claimed is None:return {'state':'blocked','reason':'duplicate_or_unknown_event'}
    event=claimed['event'];source=event['source'];action=failure_action(event['failure_code'])
    result={'state':'review_required','action':action,'event_id':event_id,'live_source_changed':False,'promotion':'not performed'}
    try:
        if source is not None:
            if source_root is None:raise ValueError('Source root required')
            root=Path(source_root).resolve();p=(root/source['path']).resolve()
            if not p.is_relative_to(root) or any(v.is_symlink() for v in (root/source['path'],*(root/source['path']).parents)) or sha256_file(p)!=source['sha256']:
                raise ValueError('Stale event source')
        if action=='code_proposal':
            registered=source is not None and source['path']==TARGET and source['check_id']==CHECK_ID and failure_reproduced is True
            name='code_repair' if registered else 'diagnosis'
            handler=handlers.get(name)
            if handler is None:result.update(state='major_approval_required',reason='missing_registered_reproducer_or_handler')
            else:
                receipt=handler(event)
                if not isinstance(receipt,dict) or receipt.get('live_source_changed') is not False:raise ValueError('Unsafe handler receipt')
                if registered:
                    if receipt.get('worker_terminated') is not True or type(receipt.get('passed')) is not bool:raise ValueError('Independent check receipt required')
                    result.update(state='checked' if receipt['passed'] else 'blocked',passed=receipt['passed'])
                else:result.update(state='major_approval_required',diagnosis_available=True)
        elif action in {'parameter_review','data_and_recipe_review'}:
            handler=handlers.get('recipe_diagnosis')
            if handler is not None:
                receipt=handler(event)
                if not isinstance(receipt,dict) or receipt.get('live_source_changed') is not False:raise ValueError('Unsafe recipe receipt')
                result.update(state='diagnosed',candidate_trial='separate reviewed-data and recipe policy gates required')
        elif action in {'provider_reconciliation','operator_access_review','budget_review','stop_and_reconcile','reconcile_then_bounded_backoff','worker_and_billing_reconciliation','review_pinned_image'}:
            result.update(state='provider_reconciliation')
    except Exception:
        result.update(state='unknown',reason='handler_or_lineage_reconciliation_required')
    queue.finish(event_id,claimed['lease'],result)
    return result
