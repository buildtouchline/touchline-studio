"""Fixed review-only replay experiments; never train or promote a checkpoint."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import hashlib,json,os,re,subprocess,sys,time,zipfile


def main():
    start=time.monotonic();work=Path('/work');project=work/'project';project.mkdir()
    job=json.loads((work/'job.json').read_text())
    for name,digest in job['hashes'].items():
        with (work/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Input hash mismatch')
    with zipfile.ZipFile(work/'checks-code.zip') as archive:
        for entry in archive.infolist():
            target=(project/entry.filename).resolve()
            if not target.is_relative_to(project) or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe source')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(entry))
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    check_results={}
    for mode,args in (('tests',[]),('artifacts',['artifacts'])):
        with (work/(mode+'.log')).open('wb') as log:
            result=subprocess.run([sys.executable,'-B','tools/run_offline_checks.py',*args],stdout=log,stderr=subprocess.STDOUT,timeout=650)
        data=(work/(mode+'.log')).read_text(errors='replace')
        count=re.findall(r'Ran (\d+) tests?',data)
        check_results[mode]={'exit_code':result.returncode,'tests':int(count[-1]) if count else None}
        print(json.dumps({'stage':'checks','mode':mode,**check_results[mode]}),flush=True)
        if result.returncode:raise RuntimeError('Fixed remote checks failed')
    parent=work/'parent';parent.mkdir()
    for name in ('manifest.json','coverage.jsonl','raw_proposals.jsonl'):
        (parent/name).write_bytes((work/name).read_bytes())
    from touchline.analysis_trials import RECIPES,validate_recipe,summarize_run
    from touchline.tracking_replay import retrack_run
    from touchline.video.preview import render_run_preview
    reports=[]
    for recipe in RECIPES:
        recipe=validate_recipe(recipe);name=recipe['name'];output=work/'run'/name
        with (work/'trial.log').open('a') as log,redirect_stdout(log),redirect_stderr(log):
            manifest=retrack_run(parent,output,activation_threshold=recipe['activation_threshold'],max_gap_s=recipe['max_gap_s'])
            if manifest['status']!='complete':raise ValueError('Complete replay required')
            stats=summarize_run(output)
            render_run_preview(output,work/'preview'/name,source=work/'video24.mp4',max_seconds=20,hold_seconds=.17)
        reports.append({'recipe':recipe,**stats})
        print(json.dumps({'stage':'replay','name':name,'temporary_tracks':stats['temporary_tracks'],'observations':stats['observations']}),flush=True)
    profiles=work/'profiles';profiles.mkdir()
    (profiles/'comparison.json').write_text(json.dumps(reports,indent=2))
    (profiles/'review.md').write_text('# Tracking experiment review\n\nTemporary tracks are not unique players. More observations or fewer IDs do not establish accuracy. Compare all three previews and supply reviewed reference IDs before selection. No model training, code generation or promotion occurred.\n')
    receipt={'status':'complete','checks':check_results,'reports':reports,'input_hashes':job['hashes'],
        'duration_s':round(time.monotonic()-start,2),'identity_accuracy':None,'promotion':'not performed','training':'not run'}
    (work/'cpu-receipt.json').write_text(json.dumps(receipt,indent=2))
    (work/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><h1>Tracking candidates for human review</h1><p>No measured identity accuracy; no candidate promoted.</p>'+''.join('<h2>'+r['recipe']['name']+'</h2><video controls width="1100" src="preview/'+r['recipe']['name']+'/overlay24.mp4"></video>' for r in reports))
    with zipfile.ZipFile(work/'tracking-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('run','preview','profiles'):
            for p in (work/folder).rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(work).as_posix())
        for name in ('cpu-receipt.json','RESULTS.html'):archive.write(work/name,name)
    print(json.dumps({'stage':'complete','status':'complete'}),flush=True)


if __name__=='__main__':main()
