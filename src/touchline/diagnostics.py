"""Local structured diagnostics; deliberately never reads credentials or environment.

Only static stage/error codes and bounded numeric counters are persisted. Raw
messages, arguments, URLs, HTTP bodies/headers, paths and frame locals are not.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

_CURRENT = ContextVar('touchline_diagnostic_trace', default=None)
_PARENT = ContextVar('touchline_diagnostic_span', default=None)
_STAGES = frozenset({
    'cli', 'media.preflight', 'media.probe', 'media.normalize', 'pipeline', 'detector.load',
    'detector.infer', 'analytics', 'audio.normalize', 'audio.align',
    'audio.transcribe', 'data.inventory', 'data.stage', 'data.etl',
    'data.release', 'data.validate', 'training', 'evaluation', 'audit',
    'report.draft', 'report.approve', 'report.correct', 'report.compare',
    'budget.verify', 'budget.reserve', 'budget.reconcile', 'studio.request',
    'doctor', 'config', 'artifacts', 'vision.sam', 'player.profiles','player.score','run.export',
    'repair.prepare','repair.validate','repair.stage','repair.check',
    'provider.modal', 'provider.meta', 'provider.baseten', 'provider.fireworks',
})
_EVENTS = frozenset({'started', 'completed', 'failed', 'cancelled', 'progress', 'truncated'})
_COUNTERS = frozenset({'decoded_frames','selected_frames','observations','step','epoch',
    'images','duration_s','http_status','reserved_micro_usd','actual_micro_usd','loss'})
_EXCEPTION_TYPES=frozenset({'ValueError','TypeError','KeyError','RuntimeError','OSError','FileNotFoundError',
    'FileExistsError','PermissionError','TimeoutError','ImportError','ModuleNotFoundError','MemoryError',
    'KeyboardInterrupt','GeneratorExit','SystemExit','Exception'})
_GUIDANCE = {
    'INPUT_INVALID': 'Review the input/configuration for the failing stage.',
    'MEDIA_UNSUPPORTED': 'Use unrotated landscape constant 24/1 FPS video; normalize other cadences explicitly and inspect timestamps.',
    'DATA_INVALID': 'Review labels, coordinate origins, class mapping and capture-group splits.',
    'DEPENDENCY_MISSING': 'Install the pinned dependencies in the selected runtime.',
    'RESOURCE_EXHAUSTED': 'Stop and inspect memory/resource evidence; use a reviewed bounded recipe.',
    'TRAINING_NONFINITE': 'Stop the trial; inspect inputs, precision and the fixed recipe before a new bounded run.',
    'AUTH_REJECTED': 'Operator must verify access privately; never paste or log a credential.',
    'RATE_LIMITED': 'Use bounded backoff within the original reservation; do not repeat blindly.',
    'PROVIDER_UNAVAILABLE': 'Check provider status and job state before any bounded retry.',
    'TIMEOUT': 'Reconcile worker and billing state before retrying.',
    'STORAGE_UNAVAILABLE': 'Check output permissions, capacity and storage visibility.',
    'BUDGET_REJECTED': 'Verify prior usage, eligible headroom and remaining project reservations.',
    'BUDGET_OVERRUN': 'Stop dispatch and reconcile the preserved overrun before manual clearance.',
    'REVIEW_REQUIRED': 'Check source, identity, review revision and exact approval evidence.',
    'HTTP_REJECTED': 'Inspect the request contract and authorization without sharing its contents.',
    'CANCELLED': 'Reconcile any active worker and retain pending charge reservations.',
    'INTERNAL_ERROR': 'Use the trace ID and repository stack locations to investigate.',
}


class DiagnosticUnavailable(RuntimeError):
    pass


def failure_details(exc, stage='cli', *, status=None):
    """Classify without persisting exception text; never serialize SDK responses."""
    if stage not in _STAGES:
        raise ValueError('Unknown diagnostic stage')
    if status is None:
        for attr in ('status_code', 'code'):
            candidate = getattr(exc, attr, None)
            if type(candidate) is int and 100 <= candidate <= 599:
                status = candidate
                break
    if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
        code='CANCELLED'
    elif status in (401,403): code='AUTH_REJECTED'
    elif status == 429: code='RATE_LIMITED'
    elif status is not None and status >= 500: code='INTERNAL_ERROR' if stage=='studio.request' else 'PROVIDER_UNAVAILABLE'
    elif status is not None and status >= 400: code='HTTP_REJECTED'
    elif isinstance(exc, (TimeoutError,)): code='TIMEOUT'
    elif isinstance(exc, (ImportError,ModuleNotFoundError)): code='DEPENDENCY_MISSING'
    elif isinstance(exc,MemoryError): code='RESOURCE_EXHAUSTED'
    elif isinstance(exc,OSError): code='STORAGE_UNAVAILABLE'
    elif stage.startswith('budget.'): code='BUDGET_REJECTED'
    elif stage.startswith('media.'): code='MEDIA_UNSUPPORTED'
    elif stage.startswith('data.'): code='DATA_INVALID'
    elif stage.startswith('report.'): code='REVIEW_REQUIRED'
    elif isinstance(exc,(ValueError,TypeError,KeyError)): code='INPUT_INVALID'
    else: code='INTERNAL_ERROR'
    # A few runtime categories need message inspection, never message storage.
    # Limit inspection and discard it immediately; no redaction-based log promise.
    if code=='INTERNAL_ERROR' or stage=='training':
        message=str(exc)[:2048].casefold()
        if 'out of memory' in message: code='RESOURCE_EXHAUSTED'
        elif 'timed out' in message: code='TIMEOUT'
        elif stage=='training' and 'non-finite loss' in message: code='TRAINING_NONFINITE'
    stack=[]
    tb=exc.__traceback__
    source_root=Path(__file__).resolve().parent
    while tb is not None:
        filename=Path(tb.tb_frame.f_code.co_filename)
        try:
            relative=filename.resolve().relative_to(source_root).as_posix()
            # Code locations only, never local variables or source lines.
            stack.append({'module':relative,'line':tb.tb_lineno})
        except (ValueError,OSError):pass
        tb=tb.tb_next
    stack=stack[-8:]
    fingerprint=hashlib.sha256(json.dumps([code,stack],sort_keys=True).encode()).hexdigest()[:20]
    name=type(exc).__name__
    result={'code':code,'guidance':_GUIDANCE[code],'fingerprint':fingerprint,'stack':stack,
            'type':name if name in _EXCEPTION_TYPES else 'Exception'}
    if type(status) is int and 100 <= status <= 599:result['http_status']=status
    return result


def _safe_directory(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.name.casefold() in ('.env','.modal.toml') or part.name.casefold().startswith('.env.'):
            raise DiagnosticUnavailable('Credential paths are forbidden for diagnostics')
        if part.is_symlink():
            raise DiagnosticUnavailable('Symlink diagnostic paths are forbidden')
    path.mkdir(parents=True,exist_ok=True)
    return path


class DiagnosticJournal:
    def __init__(self,directory=None,*,max_events=2000,max_bytes=1048576):
        if type(max_events) is not int or max_events<4 or type(max_bytes) is not int or max_bytes<4096:
            raise ValueError('Diagnostic limits are too small')
        self.trace_id=uuid.uuid4().hex
        self.directory=_safe_directory(directory or Path.cwd()/'runs'/'diagnostics')
        self.path=self.directory/(self.trace_id+'.jsonl')
        try:
            fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            os.close(fd)
        except OSError as exc:
            raise DiagnosticUnavailable('Cannot create private diagnostic journal') from exc
        self.max_events=max_events;self.max_bytes=max_bytes
        self.sequence=0;self.bytes_written=0;self.failures=0;self.truncated=False
        self.lock=threading.Lock()

    def event(self,stage,event,*,span_id=None,parent_span_id=None,elapsed_ms=None,counters=None,failure=None):
        if stage not in _STAGES or event not in _EVENTS:
            raise ValueError('Unknown diagnostic stage/event')
        clean={}
        for name,value in (counters or {}).items():
            if name not in _COUNTERS or type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('Only bounded numeric diagnostic counters are accepted')
            clean[name]=value
        if span_id is not None and not re.fullmatch('[a-f0-9]{32}',span_id):raise ValueError('Invalid span ID')
        if parent_span_id is not None and not re.fullmatch('[a-f0-9]{32}',parent_span_id):raise ValueError('Invalid parent span ID')
        with self.lock:
            if self.truncated:return
            record={'schema':'touchline.diagnostic.v1','trace_id':self.trace_id,'sequence':self.sequence+1,
                    'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'event':event}
            if span_id:record['span_id']=span_id
            if parent_span_id:record['parent_span_id']=parent_span_id
            if elapsed_ms is not None:
                if type(elapsed_ms) not in (int,float) or not math.isfinite(elapsed_ms) or elapsed_ms<0:raise ValueError('Invalid elapsed duration')
                record['elapsed_ms']=round(elapsed_ms,3)
            if clean:record['counters']=clean
            if failure is not None:
                # This is intentionally a closed schema; arbitrary fields are rejected.
                if set(failure)-{'code','guidance','fingerprint','stack','http_status','type'} or failure.get('code') not in _GUIDANCE:
                    raise ValueError('Invalid diagnostic failure record')
                record['error']={
                    'code':failure['code'],'guidance':_GUIDANCE[failure['code']],
                    'fingerprint':failure.get('fingerprint',''),
                }
                if not re.fullmatch('[a-f0-9]{20}',record['error']['fingerprint']):raise ValueError('Invalid error fingerprint')
                stack=failure.get('stack',[])
                if len(stack)>8 or any(set(v)!={'module','line'} or not re.fullmatch(r'[a-zA-Z0-9_/]+\.py',v['module']) or type(v['line']) is not int or v['line']<1 for v in stack):
                    raise ValueError('Invalid diagnostic stack')
                record['error']['stack']=stack
                if 'type' in failure:
                    if failure['type'] not in _EXCEPTION_TYPES:raise ValueError('Unknown exception category')
                    record['error']['type']=failure['type']
                if 'http_status' in failure:
                    value=failure['http_status']
                    if type(value) is not int or not 100<=value<=599:raise ValueError('Invalid HTTP status')
                    record['error']['http_status']=value
            raw=json.dumps(record,allow_nan=False,separators=(',',':'))+'\n'
            if self.sequence>=self.max_events-1 or self.bytes_written+len(raw.encode())>self.max_bytes-512:
                record={k:v for k,v in record.items() if k in ('schema','trace_id','sequence','utc','stage')}
                record['event']='truncated';raw=json.dumps(record)+'\n';self.truncated=True
            try:
                if self.path.is_symlink():raise DiagnosticUnavailable('Diagnostic file was replaced by a symlink')
                with self.path.open('a',encoding='utf-8',newline='\n') as stream:
                    stream.write(raw);stream.flush()
            except OSError as exc:
                raise DiagnosticUnavailable('Diagnostic write failed; do not dispatch further work') from exc
            self.sequence+=1;self.bytes_written+=len(raw.encode())
            if event=='failed':self.failures+=1


def current_trace_id():
    journal=_CURRENT.get()
    return journal.trace_id if journal else None


@contextmanager
def diagnostic_session(directory=None):
    current=_CURRENT.get()
    if current is not None:
        yield current
        return
    journal=DiagnosticJournal(directory)
    token=_CURRENT.set(journal)
    try:yield journal
    finally:_CURRENT.reset(token)


@contextmanager
def diagnostic_span(stage):
    with diagnostic_session() as journal:
        span_id=uuid.uuid4().hex;parent=_PARENT.get();start=time.monotonic();before=journal.failures
        journal.event(stage,'started',span_id=span_id,parent_span_id=parent)
        token=_PARENT.set(span_id)
        try:
            yield journal
        except BaseException as exc:
            if isinstance(exc,SystemExit) and exc.code in (None,0):
                journal.event(stage,'completed',span_id=span_id,parent_span_id=parent,elapsed_ms=(time.monotonic()-start)*1000)
            else:
                event='cancelled' if isinstance(exc,(KeyboardInterrupt,GeneratorExit)) else 'failed'
                journal.event(stage,event,span_id=span_id,parent_span_id=parent,elapsed_ms=(time.monotonic()-start)*1000,failure=failure_details(exc,stage))
            raise
        else:
            journal.event(stage,'failed' if journal.failures>before else 'completed',span_id=span_id,parent_span_id=parent,elapsed_ms=(time.monotonic()-start)*1000)
        finally:_PARENT.reset(token)


def diagnosed(stage,*,failures_only=False):
    if stage not in _STAGES:raise ValueError('Unknown diagnostic stage')
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            if failures_only and _CURRENT.get() is not None:
                try:return function(*args,**kwargs)
                except Exception as exc:
                    record_failure(exc,stage)
                    raise
            with diagnostic_span(stage):return function(*args,**kwargs)
        return wrapped
    return decorate


def record_failure(exc,stage='cli',*,status=None):
    detail=failure_details(exc,stage,status=status)
    journal=_CURRENT.get()
    if journal:
        journal.event(stage,'failed',span_id=_PARENT.get(),failure=detail)
    return {'code':detail['code'],'message':detail['guidance'],'trace_id':current_trace_id()}


def diagnostic_progress(stage,**counters):
    journal=_CURRENT.get()
    if journal:journal.event(stage,'progress',span_id=_PARENT.get(),counters=counters)


def diagnostic_issue(stage,code):
    if code not in _GUIDANCE:raise ValueError('Unknown diagnostic code')
    detail={'code':code,'guidance':_GUIDANCE[code],
            'fingerprint':hashlib.sha256((stage+':'+code).encode()).hexdigest()[:20],'stack':[]}
    journal=_CURRENT.get()
    if journal:journal.event(stage,'failed',span_id=_PARENT.get(),failure=detail)


def local_doctor():
    """Non-credentialed dependency/configuration report; never imports provider SDKs."""
    from .analysis_policy import PROVIDER_CAPS_USD
    modules={name:importlib.util.find_spec(name) is not None for name in ('av','PIL','numpy','torch','torchvision','modal','sam3')}
    required=('av','PIL','numpy','torch','torchvision')
    return {'schema':'touchline.doctor.v1','trace_id':current_trace_id(),
        'local_dependencies':modules,'credentials':'not inspected','env_loader':'explicit operator-only; not invoked by doctor',
        'local_dependency_check':'passed' if all(modules[k] for k in required) else 'missing dependencies',
        'missing_required_modules':[k for k in required if not modules[k]],
        'ready_for_paid_execution':False,
        'provider_execution':'not tested; adapters/controller still pending',
        'budget_caps_usd':PROVIDER_CAPS_USD,'provider_dashboard_limits':'not inspected/configured',
        'input_policy':{'width':1920,'height':1080,'fps':'24/1','constant_rate_required':True},
        'accuracy_95_percent':'not validated','paid_calls':0}


def diagnostic_summary(path):
    path=Path(path)
    if not re.fullmatch(r'[a-f0-9]{32}\.jsonl',path.name):
        raise ValueError('Select a generated trace-ID.jsonl file, not a credential or arbitrary log')
    for part in (path,*path.parents):
        if part.is_symlink() or part.name.casefold()=='.env' or part.name.casefold().startswith('.env.'):
            raise ValueError('Unsafe diagnostic input path')
    if path.stat().st_size>1048576:raise ValueError('Diagnostic file exceeds the bounded report limit')
    counts={};errors={};trace_id=path.stem;last=0;truncated=False;active=set();failed=False
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('schema')!='touchline.diagnostic.v1' or row.get('trace_id')!=trace_id or row.get('sequence')!=last+1:
                raise ValueError('Invalid diagnostic sequence or schema')
            stage,event=row.get('stage'),row.get('event')
            if stage not in _STAGES or event not in _EVENTS:raise ValueError('Invalid diagnostic stage/event')
            last+=1;key=stage+':'+event;counts[key]=counts.get(key,0)+1
            if event=='started':active.add(row.get('span_id'))
            elif event in ('completed','failed','cancelled'):active.discard(row.get('span_id'))
            failed=failed or event in ('failed','cancelled')
            code=row.get('error',{}).get('code')
            if code is not None:
                if code not in _GUIDANCE:raise ValueError('Unknown diagnostic error code')
                errors[code]=errors.get(code,0)+1
            truncated=truncated or event=='truncated'
    # Do not forward unknown fields or free-form text from a selected input file.
    return {'schema':'touchline.diagnostic-summary.v1','trace_id':trace_id,'events':last,
        'stage_events':counts,'error_counts':errors,'truncated':truncated,
        'state':'incomplete' if truncated or not last or active else ('failed' if errors or failed else 'recorded')}
