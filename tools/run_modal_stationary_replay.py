"""One owner-approved $0.50 CPU repair replay. Private settings are runtime-only."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import argparse,ast,hashlib,json,logging,shutil,sqlite3,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_modal_video_trial import Discard,modal_client,save
from run_modal_remote_checks import make_bundle
from run_modal_panorama_review import extract_selected
from touchline.data.manifest import sha256_file


def run(operator, *, priority_fix_retry=False):
    operator=operator.resolve()
    if priority_fix_retry:
        previous=json.loads((operator/'runs/modal-stationary-repair-20261104/receipt.json').read_text())
        if previous.get('worker_terminated') is not True or previous.get('status')!='complete' or previous.get('summary',{}).get('improved_on_development_clip') is not False:
            raise ValueError('Reconciled rejected first trial required')
    work=operator/('runs/modal-stationary-repair-20261104-priority' if priority_fix_retry else 'runs/modal-stationary-repair-20261104');work.mkdir(exist_ok=False)
    prior=operator/'runs/modal-panorama-resnet-20261104'
    receipt=json.loads((prior/'receipt.json').read_text())
    if receipt['status']!='complete' or receipt['worker_terminated'] is not True:raise ValueError('Completed exported ResNet run required')
    digest=make_bundle(work/'code.zip')
    with sqlite3.connect(operator/'runs/tracking-repair-20261104-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (phase TEXT PRIMARY KEY,reserved_micro_usd INTEGER,retained_ceiling_micro_usd INTEGER)')
        db.execute('INSERT INTO claims VALUES (?,?,?)',('owner-approved-priority-replay' if priority_fix_retry else 'owner-approved-cpu-replay',250000 if priority_fix_retry else 500000,55750000 if priority_fix_retry else 55500000))
    # Reuse the prior provisional evaluator's exact metric/matching functions.
    # Only bind the two predeclared candidate files; no reviewed flag fabricated.
    evaluate=(ROOT/'tools/provisional_panorama_metrics.py').read_text(encoding='utf-8')
    (work/'evaluate-replay.py').write_text(evaluate,encoding='utf-8')
    inputs={'code.zip':work/'code.zip','video24.mp4':operator/'runs/modal-cpu-full-video-20261102/source/video24.mp4',
        'raw_proposals.jsonl':prior/'evidence/raw_proposals.jsonl','sam-proposals.json':prior/'sam-proposals.json',
        'frame-map.jsonl':operator/'runs/modal-cpu-full-video-20261102/source/frame-map.jsonl',
        'compare-reference.txt':operator/'artifacts/soccertrack-v2/mot/118577.txt',
        'panorama/summary.json':prior/'evidence/summary.json','panorama/raw_proposals.jsonl':prior/'evidence/raw_proposals.jsonl',
        'panorama/observations.jsonl':prior/'evidence/observations.jsonl','evaluate-replay.py':work/'evaluate-replay.py'}
    if sha256_file(inputs['video24.mp4'])!='cd15fab8a76e3c7c6ae5d3682ea96956dbedd83ebed3c4929cfb561819cc5baa':raise ValueError('Exact source video required')
    if sha256_file(inputs['compare-reference.txt'])!='425c9fa97e14f50298150f366828854639cfa52529470bd6ed95b8ac278dd2e7':raise ValueError('Exact unchanged reference required')
    job={'inputs':{k:sha256_file(v) for k,v in inputs.items()},'video_sha256':sha256_file(inputs['video24.mp4']),
        'attempt_id':work.name,'recipes':['motion','sam_motion'],'baseline_run_id':receipt['run_id'],'quality_acceptance':False}
    record={'status':'prepared','reserved_usd':'.25' if priority_fix_retry else '.50','retained_ceiling_usd':'55.75' if priority_fix_retry else '55.50','gpu':None,'cpu':4,'memory_mib':8192,
        'timeout_s':900 if priority_fix_retry else 1800,'resource_and_transfer_upper_bound_usd':'.229936' if priority_fix_retry else '.459872','worker_terminated':'not_created','billing':'unreconciled',
        'code_sha256':digest,'job':job,'detector_inference':'not run; exact cached proposals','training':'not run'}
    save(work/'receipt.json',record);sandbox=None
    try:
        from touchline.settings import load_operator_settings
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(operator/'.env',environ={});modal,client=modal_client(settings,work)
            app=modal.App.lookup('touchline-stationary-repair',client=client,environment_name='main',create_if_missing=True)
            sandbox=modal.Sandbox.create(app=app,client=client,name=work.name,image=modal.Image.from_id('im-A4IdpUIDiLlRbBha7sqzYy',client=client),
                cpu=(4,4),memory=(8192,8192),timeout=record['timeout_s'],runtime='gvisor',block_network=True,secrets=[],volumes={},workdir='/work',
                env={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        allocated=time.time();record.update(status='running',worker_id=sandbox.object_id,worker_terminated='pending');save(work/'receipt.json',record)
        print(json.dumps({'stage':'sandbox','status':'created','worker_id':sandbox.object_id,'gpu':None}),flush=True)
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            for name,path in inputs.items():sandbox.filesystem.copy_from_local(path,'/work/'+name)
            sandbox.filesystem.copy_from_local(ROOT/'tools/modal_stationary_replay_worker.py','/work/replay-worker.py')
            sandbox.filesystem.write_text(json.dumps(job),'/work/replay-job.json');process=sandbox.exec('python','-I','/work/replay-worker.py',timeout=760 if priority_fix_retry else 1500,text=False)
        last=None;misses=0;complete=False
        while time.time()-allocated<(810 if priority_fix_retry else 1620):
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):state=json.loads(sandbox.filesystem.read_text('/work/replay/worker-state.json'))
                misses=0
                if state!=last:
                    with (work/'events.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
                    print(json.dumps(state),flush=True);last=state
                if state['stage']=='failed':raise RuntimeError('Repair replay failed')
                if state['stage']=='complete':complete=True;break
            except RuntimeError:raise
            except Exception as exc:
                misses+=1
                with redirect_stdout(Discard()),redirect_stderr(Discard()):exit_code=sandbox.poll()
                if exit_code is not None or misses>12:raise RuntimeError('Replay status unavailable') from None
            time.sleep(10)
        if not complete:raise TimeoutError('Bounded CPU replay deadline')
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            for name,limit in [('overlay24.mp4',500*1024**2),('evidence.zip',200*1024**2)]:
                if not 0<sandbox.filesystem.stat('/work/replay/'+name).size<=limit:raise ValueError('Replay transfer exceeds bound')
                sandbox.filesystem.copy_to_local('/work/replay/'+name,work/name)
        extract_selected(work/'evidence.zip',work/'evidence',{'.json','.jsonl','.jpg','.log'},700*1024**2)
        summary=json.loads((work/'evidence/summary.json').read_text());record.update(status='complete',summary=summary,result_sha256=sha256_file(work/'overlay24.mp4'))
    except BaseException as exc:
        record.update(status='failed',failure_type=type(exc).__name__)
        if sandbox is not None:
            for name in ('failure.json','tests.log','artifacts.log','comparison.log','summary.json','comparison.json','worker-state.json','motion-observations.jsonl','sam_motion-observations.jsonl'):
                try:
                    with redirect_stdout(Discard()),redirect_stderr(Discard()):
                        if sandbox.filesystem.stat('/work/replay/'+name).size<100*1024**2:sandbox.filesystem.copy_to_local('/work/replay/'+name,work/('partial-'+name))
                except Exception:pass
    finally:
        if sandbox is not None:
            try:
                with redirect_stdout(Discard()),redirect_stderr(Discard()):sandbox.terminate(wait=True)
                record['worker_terminated']=True
            except Exception:record['worker_terminated']='unknown'
        save(work/'receipt.json',record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--operator-root',type=Path,required=True);parser.add_argument('--execute-reviewed-replay',action='store_true');parser.add_argument('--priority-fix-retry',action='store_true');args=parser.parse_args()
    if not args.execute_reviewed_replay or not args.operator_root.is_absolute():raise SystemExit('Explicit approved replay and absolute operator checkout required')
    logging.disable(logging.CRITICAL)
    try:
        result=run(args.operator_root,priority_fix_retry=args.priority_fix_retry);print(json.dumps({k:result.get(k) for k in ('status','worker_terminated','result_sha256')}),flush=True)
        if result['status']!='complete':raise SystemExit(1)
    except Exception as exc:print(json.dumps({'status':'failed','failure_type':type(exc).__name__}),flush=True);raise SystemExit(1) from None
