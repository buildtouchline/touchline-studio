"""One bounded CPU AP50 comparison on the same reviewed validation release."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,sqlite3,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,modal_client,save
from run_modal_training_recovery import phase
from run_modal_remote_checks import make_bundle
from touchline.data.manifest import sha256_file
from probe_modal_gpu import safe_error

def run():
    parent=ROOT/'runs/modal-training-recovery-20261102';recovery=json.loads((parent/'recovery.json').read_text());minor=parent/'minor'
    checkpoint=minor/'candidate/candidate.pt';reference=minor/'candidate-val/evaluation.json';ref=json.loads(reference.read_text())
    if sha256_file(checkpoint)!=ref['model']['checkpoint']:raise ValueError('Selected model changed')
    bundle=ROOT/'runs/ap50-source-20261103.zip';make_bundle(bundle)
    inputs={'code.zip':bundle,'worker.py':ROOT/'tools/modal_ap50_worker.py','candidate.pt':checkpoint,'reference.json':reference}
    with sqlite3.connect(ROOT/'runs/modal-ap50-20261103-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY,cap_micro_usd INTEGER)')
        db.execute('INSERT INTO claims VALUES (?,?)',('requested-ap50-rerun-20261103',2000000))
    work=ROOT/'runs/modal-ap50-20261103'
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,parent)
        app=modal.App.lookup('touchline-training-recovery',client=client,environment_name='main',create_if_missing=False)
        image=modal.Image.from_id(recovery['data']['snapshot_image_id'],client=client)
    record=phase(work,modal=modal,client=client,app=app,image=image,inputs=inputs,job={'mode':'evaluation'},is_analysis=True)
    (work/'gpu-results.zip').rename(work/'ap50-results.zip');record.update(result_archive='ap50-results.zip',reserved_usd='2.00',cumulative_modal_ceiling_usd='50.00')
    save(work/'job-receipt.json',record);return record

if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--approved-existing-budget',action='store_true');args=p.parse_args()
    if not args.approved_existing_budget:raise SystemExit('Explicit requested AP50/budget scope required')
    try:r=run();print(json.dumps({'status':r['status'],'worker_terminated':r['worker_terminated']}))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
