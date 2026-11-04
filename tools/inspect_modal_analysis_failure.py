"""Read the existing stopped analysis call only; never submits a new job."""
from contextlib import redirect_stdout,redirect_stderr
import json
import logging
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error


def inspect(work):
    record=json.loads((work/'modal-function.json').read_text(encoding='utf-8'))
    if record.get('app_stop')!='confirmed':raise ValueError('App termination must be reconciled first')
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={})
        modal,client=modal_client(settings,work)
        call=modal.FunctionCall.from_id(record['function_call_id'],client=client)
        try:
            result=call.get(timeout=0)
            detail={'state':'existing_result_available','result_is_dict':isinstance(result,dict)}
        except BaseException as exc:
            detail={'state':'existing_call_error',**safe_error(exc)}
            exception_type=type(exc).__name__
            detail['exception_type']=exception_type if re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]{0,79}',exception_type) else 'unclassified'
            text=str(exc).lower()
            # Known dependency names only; never arbitrary provider text/paths.
            names=('touchline','torch','torchvision','supervision','numpy','av','modal_analysis_function')
            detail['known_missing_modules']=[n for n in names if re.search(r"no module named ['\"]"+re.escape(n)+r"['\"]",text)]
            if exception_type=='InternalFailure':
                detail['classification']='provider_internal_failure'
                detail['action']='provider_reconciliation; no automatic code rewrite or retry'
    detail['function_call_id']=record['function_call_id']
    detail['new_job_submitted']=False
    save(work/'failure-inspection.json',detail)
    return detail


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:print(json.dumps(inspect(ROOT/'runs/modal-analysis-20261102'),indent=2))
    except Exception as exc:
        print(json.dumps({'state':'inspection_failed',**safe_error(exc)}));raise SystemExit(1) from None
