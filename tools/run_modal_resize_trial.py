"""Exact user-approved major resize recipe; no automatic model promotion."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,json,logging,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.spending import SpendingLedger
from touchline.data.manifest import sha256_file
from touchline.training.recipes import validate_resize_recipe
from run_modal_remote_checks import make_bundle
from run_modal_training_recovery import phase
from run_modal_video_trial import Discard,modal_client,save
from probe_modal_gpu import safe_error
def run():
    parent=ROOT/'runs/modal-training-recovery-20261102';recovery=json.loads((parent/'recovery.json').read_text());checkpoint=parent/'minor/candidate/candidate.pt'
    doc_path=ROOT/'artifacts/recipes/reviewed-resize-proposal.json';doc=json.loads(doc_path.read_text());validate_resize_recipe(doc,sha256_file(checkpoint))
    work=ROOT/'runs/modal-resize-training-20261103';source=ROOT/'runs/resize-source-20261103.zip';make_bundle(source)
    ledger=SpendingLedger(ROOT/'runs/soccertrack-pxl-20261103-budget.sqlite3')
    held=ledger.reserve('modal','resize-t4-20261103',fingerprint=doc['proposal_sha256'],worst_case_usd='5.00')
    if not held['created']:raise ValueError('Consumed resize claim')
    result=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-training-recovery',client=client,environment_name='main',create_if_missing=False)
            image=modal.Image.from_id(recovery['data']['snapshot_image_id'],client=client)
        reference=json.loads((ROOT/'runs/modal-ap50-20261103/evaluation-comparison.json').read_text())
        inputs={'code.zip':source,'worker.py':ROOT/'tools/modal_resize_training_worker.py','candidate.pt':checkpoint,'recipe-review.json':doc_path,'reference.json':parent/'minor/candidate-val/evaluation.json'}
        result=phase(work,modal=modal,client=client,app=app,image=image,inputs=inputs,job={'parent_run_id':'798b794c-3ab0-4ac1-9c7f-a880a03c3fbd','reference_native_ap50':reference['native_cpu_ap50']},gpu='T4')
        return result
    finally:
        ledger.reconcile('resize-t4-20261103',finalized=False)
        if result is None or result.get('worker_terminated') is not True:ledger.block_pool('modal')
if __name__=='__main__':
    logging.disable(logging.CRITICAL);p=argparse.ArgumentParser();p.add_argument('--execute-owner-reviewed',action='store_true');a=p.parse_args()
    if not a.execute_owner_reviewed:raise SystemExit('Exact owner review required')
    try:r=run();print(json.dumps({'status':r['status'],'worker_terminated':r['worker_terminated']}))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
