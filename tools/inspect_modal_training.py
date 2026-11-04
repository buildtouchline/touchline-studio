"""Read bounded progress counters from the owned training Sandbox; no new job."""
from contextlib import redirect_stdout,redirect_stderr
import json,logging,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_modal_video_trial import Discard,modal_client
from touchline.repair_checks import read_worker_result
from probe_modal_gpu import safe_error

COMMAND="""import json
from pathlib import Path
p=Path('/work/dataset/images');files=list(p.glob('*.png'))
sizes=[f.stat().st_size for f in files]
losses=Path('/work/candidate/losses.jsonl');step=0
if losses.is_file() and losses.stat().st_size<1024**2:
 for line in losses.read_text().splitlines()[-3:]:
  try:
   value=json.loads(line).get('step')
   if type(value) is int and 0<=value<=1200:step=max(step,value)
  except ValueError:pass
print(json.dumps({'restored_files':len(sizes),'restored_bytes':sum(sizes),'mean_image_bytes':sum(sizes)//len(sizes) if sizes else 0,'candidate_exported':Path('/work/candidate/result.json').is_file(),'successful_training_steps':step}))
"""


def main():
    work=ROOT/'runs/modal-reviewed-training-20261102'
    job=json.loads((work/'training-job.json').read_text())
    if job['status']!='running' or job['worker_terminated']!='pending':raise ValueError('No owned running training job')
    from touchline.settings import load_operator_settings
    with redirect_stdout(Discard()),redirect_stderr(Discard()):
        settings=load_operator_settings(ROOT/'.env',environ={});modal,client=modal_client(settings,work)
        sandbox=modal.Sandbox.from_id(job['worker_id'],client=client)
        process=sandbox.exec('python','-I','-c',COMMAND,timeout=30,text=False)
        data=json.loads(read_worker_result(process.stdout));process.wait()
    if process.returncode or set(data)!={'restored_files','restored_bytes','mean_image_bytes','candidate_exported','successful_training_steps'}:raise ValueError('Invalid progress receipt')
    print(json.dumps(data))


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:main()
    except Exception as exc:print(json.dumps({'status':'unavailable',**safe_error(exc)}));raise SystemExit(1) from None
