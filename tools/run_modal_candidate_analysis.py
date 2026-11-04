"""Visual-test the provisional validation winner on CPU; keep approved weights unchanged."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,hashlib,json,logging,sqlite3,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.data.manifest import sha256_file
from run_modal_video_trial import Discard,modal_client,save
from run_modal_training_recovery import phase
from run_modal_remote_checks import make_bundle
from probe_modal_gpu import safe_error


def run(*,native=False):
    parent=ROOT/'runs/modal-training-recovery-20261102';recovery=json.loads((parent/'recovery.json').read_text())
    if recovery['status']!='complete' or not recovery['data'].get('snapshot_image_id'):raise ValueError('Reconciled training/data snapshot required')
    b=json.loads((parent/'baseline/training-receipt.json').read_text());m=json.loads((parent/'minor/minor-trial-receipt.json').read_text())
    selected='minor' if m['candidate_ap50']>=m['baseline_ap50'] else 'baseline'
    inputs={'candidate.pt':parent/selected/'candidate/candidate.pt','worker.py':ROOT/'tools/modal_candidate_analysis_worker.py',
        'video24.mp4':ROOT/'runs/modal-cpu-full-video-20261102/source/video24.mp4',
        'baseline-observations.jsonl':ROOT/'runs/modal-cpu-full-video-20261102/run/observations.jsonl'}
    work=ROOT/'runs'/('modal-native-analysis-20261103' if native else 'modal-candidate-analysis-20261102')
    if native:
        sam_root=ROOT/'runs/modal-sam-analysis-20261103';sam_record=json.loads((sam_root/'sam-analysis.json').read_text())
        if sam_record['status']!='complete' or sam_record['worker_terminated'] is not True:raise ValueError('Completed SAM review required')
        code_path=ROOT/'runs/native-analysis-code-20261103.zip';make_bundle(code_path)
        inputs.update({'code.zip':code_path,'sam.json':sam_root/'sam.json','sam_worker.py':ROOT/'tools/modal_sam_analysis_worker.py'})
        if sha256_file(inputs['video24.mp4'])!=sam_record['hashes']['video24.mp4']:raise ValueError('SAM footage changed')
    expected=m['training']['checkpoint_sha256'] if selected=='minor' else b['training']['checkpoint_sha256']
    if sha256_file(inputs['candidate.pt'])!=expected or sha256_file(inputs['video24.mp4'])!='cd15fab8a76e3c7c6ae5d3682ea96956dbedd83ebed3c4929cfb561819cc5baa':raise ValueError('Candidate/source selection changed')
    digest=hashlib.sha256(json.dumps({n:sha256_file(p) for n,p in inputs.items()},sort_keys=True).encode()).hexdigest()
    # Single bounded phase within the original retained CPU analysis/check hold.
    with sqlite3.connect(ROOT/'runs/modal-cpu-analysis-20261102-allowance.sqlite3') as db:
        if db.execute('SELECT reserved_micro_usd FROM claims WHERE authorization=?',('modal-cpu-analysis-20261102-ten-dollars',)).fetchone()!=(10000000,):raise ValueError('Original CPU hold required')
        db.execute('CREATE TABLE IF NOT EXISTS candidate_smoke (authorization TEXT PRIMARY KEY,fingerprint TEXT)')
        db.execute('INSERT INTO candidate_smoke VALUES (?,?)',('native-smoke-20261103' if native else 'candidate-smoke-20261102',digest))
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,parent)
        app=modal.App.lookup('touchline-training-recovery',client=client,environment_name='main',create_if_missing=False)
        image=modal.Image.from_id(recovery['data']['snapshot_image_id'],client=client)
    record=phase(work,modal=modal,client=client,app=app,image=image,inputs=inputs,
        job={'mode':'analysis','native':native,'selection':{'candidate':selected,'native_resolution':native,'provisional_validation_selection':True,'promotion':'not performed'}},is_analysis=True)
    (work/'gpu-results.zip').rename(work/'analysis-results.zip')
    record['result_archive']='analysis-results.zip';save(work/'job-receipt.json',record)
    return record


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--approved-existing-cpu-cap',action='store_true');p.add_argument('--native',action='store_true');args=p.parse_args()
    if not args.approved_existing_cpu_cap:raise SystemExit('Explicit retained CPU cap required')
    try:r=run(native=args.native);print(json.dumps({'status':r['status'],'worker_terminated':r['worker_terminated']}))
    except Exception as exc:print(json.dumps({'status':'failed',**safe_error(exc)}));raise SystemExit(1) from None
