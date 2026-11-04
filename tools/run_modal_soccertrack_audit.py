"""No-key, no-training remote intake review plus PXL metadata diagnosis."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,json,logging,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.spending import SpendingLedger
from run_modal_remote_checks import make_bundle
from run_modal_training_recovery import phase
from run_modal_video_trial import Discard,modal_client
from probe_modal_gpu import safe_error
def run():
    work=ROOT/'runs/modal-soccertrack-audit-20261103';source=ROOT/'artifacts/soccertrack-v2';meta=json.loads((source/'source-metadata.json').read_text())
    code=ROOT/'runs/soccertrack-audit-source-20261103.zip';make_bundle(code)
    ledger=SpendingLedger(ROOT/'runs/soccertrack-pxl-20261103-budget.sqlite3')
    held=ledger.reserve('modal','soccertrack-audit-20261103',fingerprint=meta['revision'],worst_case_usd='1.00')
    if not held['created']:raise ValueError('Consumed intake claim')
    result=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-training-recovery',client=client,environment_name='main',create_if_missing=False)
            image=modal.Image.from_id('im-A4IdpUIDiLlRbBha7sqzYy',client=client)
        inputs={'code.zip':code,'worker.py':ROOT/'tools/modal_soccertrack_audit_worker.py',**{name:source/'probe'/name for name in ('head.bin','moov.bin','layout.json')}}
        for match in ('118575','118577','117093'):
            inputs[match+'.mp4']=source/'mot/clips'/(match+'.mp4');inputs[match+'.txt']=source/'mot'/(match+'.txt')
        result=phase(work,modal=modal,client=client,app=app,image=image,inputs=inputs,job={'source_revision':meta['revision']},is_analysis=True)
        return result
    finally:
        ledger.reconcile('soccertrack-audit-20261103',finalized=False)
        if result is None or result.get('worker_terminated') is not True:ledger.block_pool('modal')
if __name__=='__main__':
    logging.disable(logging.CRITICAL);p=argparse.ArgumentParser();p.add_argument('--approved-existing-budget',action='store_true');a=p.parse_args()
    if not a.approved_existing_budget:raise SystemExit('Owner dataset-intake scope required')
    try:r=run();print(json.dumps({'status':r['status'],'worker_terminated':r['worker_terminated']}))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
