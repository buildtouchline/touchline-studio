"""Optional isolated repair checks. Local analysis never imports/starts Modal."""
import json
import math
import re
from pathlib import Path
from .diagnostics import diagnosed,record_failure,current_trace_id
from .repair import validate_packet,validate_proposal,canonical_hash,CHECK_ID
from .run_records import new_record,write_record

MAX_RESULT_BYTES=8192


def read_worker_result(stream):
    """Consume unbuffered SDK byte chunks, never accumulate an unbounded stream."""
    result=bytearray()
    for chunk in stream:
        if not isinstance(chunk,bytes):raise ValueError('Expected binary worker output')
        if len(chunk)>MAX_RESULT_BYTES-len(result):
            raise ValueError('Worker result exceeds the output budget')
        result.extend(chunk)
    return result.decode('utf-8',errors='strict')


def validate_image_id(image_id):
    if not isinstance(image_id,str) or not re.fullmatch(r'im-[a-zA-Z0-9_-]{1,150}',image_id):
        raise ValueError('An approved immutable Modal image ID is required')
    return image_id

# Inputs go to the worker; expected answers remain in the trusted controller.
CASES={'frames':48,'rate':24,'fixed_rate':6,
       'intervals':[[0.0,1.0],[0.5,2.0]],'bad_indices':[0,2]}
RUNNER='''import contextlib, importlib.util, json
class Discard:
    def write(self,value): return len(value)
    def flush(self): pass
try:
    with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
        spec=importlib.util.spec_from_file_location("candidate","/tmp/sampling.py")
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with open("/tmp/cases.json") as stream: cases=json.load(stream)
        anchor=module.Anchor24Sampler();fixed=module.FixedRateSampler(cases["fixed_rate"])
        a=[i for i in range(cases["frames"]) if anchor.select(i,i/cases["rate"])]
        b=[i for i in range(cases["frames"]) if fixed.select(i,i/cases["rate"])]
        rejected=[]
        for gap in (True,False):
            try:
                sampler=module.Anchor24Sampler()
                sampler.select(0,0.0)
                sampler.select(2 if gap else 1,(2/24) if gap else (1/30))
                rejected.append(False)
            except Exception: rejected.append(True)
        merged=module.merge_intervals(cases["intervals"])
        if not isinstance(merged,list) or len(merged)>32: raise ValueError()
        result={"anchor":a,"fixed":b,"gap_rejected":rejected[0],"wrong_fps_rejected":rejected[1],"merged":merged}
    print(json.dumps(result,allow_nan=False))
except BaseException:
    print('{"candidate_error":true}')
'''


def evaluate_results(raw):
    if not isinstance(raw,str) or len(raw.encode())>MAX_RESULT_BYTES:raise ValueError('Invalid bounded worker result')
    result=json.loads(raw)
    if not isinstance(result,dict) or set(result)!={'anchor','fixed','gap_rejected','wrong_fps_rejected','merged'}:
        raise ValueError('Worker result does not match the fixed check schema')
    for key in ('anchor','fixed'):
        if not isinstance(result[key],list) or len(result[key])>48 or any(type(v) is not int or not 0<=v<48 for v in result[key]):
            raise ValueError('Invalid sampled-index result')
    if any(type(result[k]) is not bool for k in ('gap_rejected','wrong_fps_rejected')):
        raise ValueError('Invalid negative-check result')
    if not isinstance(result['merged'],list) or len(result['merged'])>32:raise ValueError('Invalid merged intervals')
    merged=[]
    for pair in result['merged']:
        if not isinstance(pair,list) or len(pair)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pair):
            raise ValueError('Invalid interval result')
        merged.append([float(v) for v in pair])
    normalized={**result,'merged':merged}
    expected={'anchor':[5,17,29,41],'fixed':list(range(0,48,4)),
        'gap_rejected':True,'wrong_fps_rejected':True,'merged':[[0.0,2.0]]}
    return canonical_hash(normalized)==canonical_hash(expected)


def named_modal_factory(app_name,image_id):
    """OWNER ONLY: existing app and immutable image; no build, secrets or mounts."""
    if not isinstance(app_name,str) or not re.fullmatch('[a-z][a-z0-9-]{2,62}',app_name):
        raise ValueError('Explicit approved app name required')
    validate_image_id(image_id)
    def create(*,job_name):
        import modal
        app=modal.App.lookup(app_name,create_if_missing=False)
        image=modal.Image.from_id(image_id)
        return modal.Sandbox.create(app=app,image=image,runtime='gvisor',block_network=True,
            secrets=[],volumes={},env={},cpu=(0.5,1),memory=(512,512),timeout=120,name=job_name)
    create.image_id=image_id
    return create


class ModalRepairChecks:
    def __init__(self,sandbox_factory,ledger,*,image_id):
        self.image_id=validate_image_id(image_id)
        if getattr(sandbox_factory,'image_id',image_id)!=image_id:
            raise ValueError('Factory and approved image IDs do not match')
        self.factory=sandbox_factory;self.ledger=ledger

    @diagnosed('repair.check')
    def check(self,packet,proposal,*,attempt_id,worst_case_usd,enabled=False,quote_verified=False,run_dir=None):
        if enabled is not True or quote_verified is not True:raise ValueError('Optional remote checks need explicit operator allocation/access')
        validated=validate_proposal(packet,proposal)
        if not re.fullmatch('[a-zA-Z0-9_-]{1,80}',attempt_id):raise ValueError('Invalid attempt ID')
        reservation='repair-modal-'+attempt_id
        fingerprint=canonical_hash({'candidate':validated['candidate_id'],'check_id':CHECK_ID,'cases':CASES,'runner':RUNNER,'image_id':self.image_id})
        held=self.ledger.reserve('modal',reservation,fingerprint=fingerprint,worst_case_usd=worst_case_usd)
        if not held['created']:raise ValueError('Check already dispatched; reconcile existing job instead of replaying')
        record=new_record('repair',backend='modal',run_id='repair-'+attempt_id)
        record.update(diagnostic_trace_id=current_trace_id(),case_id=packet['case_id'],candidate_id=validated['candidate_id'],reservation_id=reservation,image_id=self.image_id)
        directory=Path(run_dir) if run_dir is not None else Path('runs')/'repair-checks'/record['run_id']
        directory.mkdir(parents=True,exist_ok=False);write_record(directory,record)
        sandbox=None;result=None;terminated=False;failed=False
        try:
            sandbox=self.factory(job_name=record['run_id'])
            worker=getattr(sandbox,'object_id',None)
            if not isinstance(worker,str) or not re.fullmatch('sb-[a-zA-Z0-9_-]{1,150}',worker):raise ValueError('Invalid worker identifier')
            record.update(status='running',worker_id=worker);write_record(directory,record)
            sandbox.filesystem.write_text(proposal['replacement']['content'],'/tmp/sampling.py')
            sandbox.filesystem.write_text(json.dumps(CASES),'/tmp/cases.json')
            process=sandbox.exec('python','-I','-c',RUNNER,timeout=30,text=False,bufsize=-1)
            # Binary SDK iteration avoids read-all and newline buffering. Overflow
            # exits straight into finally, which terminates the entire worker.
            raw=read_worker_result(process.stdout)
            process.wait()
            result=evaluate_results(raw) if process.returncode==0 else False
        except Exception as exc:
            record_failure(exc,'repair.check');failed=True
        finally:
            if sandbox is not None:
                try:sandbox.terminate(wait=True);terminated=True
                except Exception as exc:record_failure(exc,'repair.check')
            if sandbox is None or not terminated:
                self.ledger.block_pool('modal')
            self.ledger.reconcile(reservation,finalized=False)
            record['status']='failed' if failed or not terminated or not result else 'completed'
            record['worker_terminated']=terminated;write_record(directory,record)
        if failed or not terminated:raise RuntimeError('Repair worker failed/uncertain; keep reservation and reconcile') from None
        return {'schema':'touchline.repair-check.v1','case_id':packet['case_id'],'candidate_id':validated['candidate_id'],
            'run_id':record['run_id'],'worker_id':record.get('worker_id'),'image_id':self.image_id,'check_id':CHECK_ID,'passed':result,'worker_terminated':terminated,'backend':'modal_optional',
            'promotion_state':'human_review_required' if result else 'rejected','provider_billing':'pending_reconciliation'}
