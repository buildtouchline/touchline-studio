"""Ask the pinned adviser for a rulebook-constrained minor training proposal."""
from contextlib import redirect_stdout,redirect_stderr
import argparse,json,logging,sqlite3,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from touchline.data.manifest import sha256_file
from touchline.repair_policy import CHECKS,assess_change
from run_modal_video_trial import Discard,save
from probe_modal_gpu import safe_error


def run():
    source=ROOT.parent/'Models-Trained/person-20261031T115722222540Z'
    baseline={'recipe_sha256':sha256_file(source/'candidate-train/recipe.json'),
        'checkpoint_sha256':sha256_file(source/'candidate-train/candidate.pt'),
        'learning_rate':.001,'weight_decay':.0005}
    metrics=json.loads((source/'candidate-val/evaluation.json').read_text())
    work=ROOT/'runs/training-tuning-advice-20261102';work.mkdir(exist_ok=False)
    with sqlite3.connect(ROOT/'runs/training-tuning-advice-20261102-allowance.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS claims (authorization TEXT PRIMARY KEY, reserved_micro_usd INTEGER)')
        db.execute('INSERT INTO claims VALUES (?,?)',('rulebook-tuning-half-budget-20261102',500000))
    record={'status':'prepared','baseline':baseline,'model':'meta-models/muse-spark-open-1026',
        'reserved_usd':'0.50','cumulative_baseten_ceiling_usd':'50.00','billing':'unreconciled',
        'source_changed':False,'automatic_promotion':False,'execution':'not started'}
    save(work/'proposal.json',record);stage='configuration'
    try:
        from touchline.settings import load_operator_settings
        from openai import OpenAI
        import httpx
        with redirect_stdout(Discard()),redirect_stderr(Discard()):
            settings=load_operator_settings(ROOT/'.env',environ={})
            key=settings.credentials.get('BASETEN_API_KEY')
            if not key:raise ValueError('Required adviser credential missing')
            stage='provider_request'
            with httpx.Client(trust_env=False,follow_redirects=False,timeout=90) as transport:
                with OpenAI(api_key=key,base_url='https://inference.baseten.co/v1',http_client=transport,max_retries=0,timeout=90) as client:
                    response=client.chat.completions.create(model=record['model'],max_tokens=4096,messages=[
                        {'role':'developer','content':'You advise one minor football detector fine-tuning experiment. Follow the rulebook exactly. Return JSON only with exactly kind="tuning", baseline_sha256, changes, expected_outcome, validation_checks, rollback_sha256. Copy hashes from the supplied baseline. changes may contain only learning_rate and/or weight_decay, each numeric 0.5x to 2x its baseline. Include every supplied fixed check. Do not propose code, shell, labels/splits/evaluator/budget changes or promotion. These small validation metrics do not measure tracking accuracy. Expected outcome is a testable hypothesis, not a promised gain. Choose one conservative minor adjustment and state limitations.'},
                        {'role':'user','content':json.dumps({'baseline':baseline,'validation_metrics':metrics['metrics'],
                            'validation_images':metrics['images'],'subset':metrics['subset'],'prior_training_steps':100,
                            'required_checks':sorted(CHECKS),'current_trial':'Independent frozen baseline warm-start is running; do not mutate it. Proposal execution waits for verified dataset and baseline result.'})}])
        record['provider_responded']=True
        record['usage']={'input_tokens':response.usage.prompt_tokens,'output_tokens':response.usage.completion_tokens} if response.usage else None
        stage='response_validation'
        if len(response.choices)!=1:raise ValueError('Unexpected choices')
        reason=response.choices[0].finish_reason
        record['finish_reason']=reason if reason in {'stop','length','content_filter','tool_calls'} else 'other'
        if reason!='stop':raise ValueError('Incomplete proposal')
        text=response.choices[0].message.content
        if not isinstance(text,str) or len(text)>16000:raise ValueError('Proposal length invalid')
        text=text.strip()
        if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
        stage='proposal_validation';proposal=json.loads(text)
        decision=assess_change(proposal,baseline,reviewed_data=True,frozen_evaluation=True,isolated_candidate=True,budget_approved=True)
        record.update(status='proposal_classified',proposal=proposal,decision=decision,
            execution='waiting for verified restored images, completed baseline and a separately bounded isolated candidate trial')
    except Exception as exc:record.update(status='rejected_no_execution',failure_stage=stage,**safe_error(exc))
    save(work/'proposal.json',record)
    return {k:record[k] for k in ('status','execution','source_changed','automatic_promotion')}


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approved-rulebook-cap',action='store_true');args=parser.parse_args()
    if not args.approved_rulebook_cap:raise SystemExit('Explicit rulebook/cumulative cap required')
    try:print(json.dumps(run()))
    except Exception as exc:print(json.dumps({'status':'blocked',**safe_error(exc)}));raise SystemExit(1) from None
