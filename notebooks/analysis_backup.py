"""Closed-file-list analysis backup; standard library only, no model or credentials."""
from pathlib import Path
import datetime
import hashlib
import json
import re


def save_complete_analysis(work, run_dir, project_root, video_path, *,
                           parent_work=None, parent_run_dir=None, snapshots=None):
    """Save generated results and exact source video; completion receipt is last.

    The current run is restored from <backup>/current/run; its video is under
    <backup>/video. Closed file lists exclude code, datasets, weights and secrets.
    Interrupted/failed copies remain for inspection without a completion receipt.
    """
    work, run_dir, project_root = Path(work), Path(run_dir), Path(project_root)
    if not project_root.is_dir():raise ValueError('Mount the Touchline project folder first')
    if video_path is None:raise ValueError('Complete backup needs the exact analyzed video; restore VIDEO_PATH first')
    video_path=Path(video_path)
    if not video_path.is_file() or video_path.suffix.lower() not in ('.mp4','.mov','.mkv'):
        raise ValueError('Select the existing analyzed video, not a model or configuration file')
    for candidate in (run_dir/'manifest.json',video_path):
        if any(p.is_symlink() for p in (candidate,*candidate.absolute().parents)):
            raise ValueError('Symlinks are not accepted in backup inputs')
    manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    run_id=manifest['run_id']
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}',run_id):
        raise ValueError('Invalid persisted run ID')
    roots=[('current',work,run_dir)]
    if parent_run_dir is not None and Path(parent_run_dir).resolve()!=run_dir.resolve():
        parent_run_dir=Path(parent_run_dir)
        roots.append(('parent',Path(parent_work) if parent_work is not None else parent_run_dir.parent,parent_run_dir))
    for _,root,run in roots:
        if not run.resolve().is_relative_to(root.resolve()):raise ValueError('Run must be inside its selected work folder')
        for name in ('manifest.json','coverage.jsonl','observations.jsonl'):
            if not (run/name).is_file():raise ValueError('Missing required saved-run artifact: '+name)
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination=project_root/'complete-results'/(run_id+'-'+timestamp)
    destination.mkdir(parents=True,exist_ok=False)
    hashes={};sizes={}

    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()

    def copy(source,relative,expected=None):
        source=Path(source)
        if any(p.is_symlink() for p in (source,*source.absolute().parents)):
            raise ValueError('Symlinks are not accepted in backup inputs')
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256();length=0
        with source.open('rb') as src,target.open('xb') as dst:
            for chunk in iter(lambda:src.read(8*1024*1024),b''):
                dst.write(chunk);h.update(chunk);length+=len(chunk)
        actual=h.hexdigest()
        if expected is not None and actual!=expected:
            raise ValueError('Selected video differs from the analyzed source; backup is incomplete')
        if digest(target)!=actual:raise ValueError('Backup copy verification failed; preserve the original')
        key=Path(relative).as_posix();hashes[key]=actual;sizes[key]=length
        print('Verified:',key)

    for label,root,run in roots:
        for name in ('runtime.json','normalization.json','frame-map.jsonl'):
            if (root/name).is_file():copy(root/name,Path(label)/name)
        for name in ('manifest.json','observations.jsonl','coverage.jsonl','raw_proposals.jsonl',
                     'analytics.json','review.html','associations.json'):
            if (run/name).is_file():copy(run/name,Path(label)/'run'/name)
        for pattern,names in (
            ('preview-*',('overlay24.mp4','focus24.mp4','contact-sheet.jpg','players.jpg','track-index.json','preview.json')),
            ('profiles-*',('player-analysis.json','review.md')),
            ('support-*',('support.json','losses.jsonl')),
            ('evaluation-*',('tracking-evaluation.json',)),
        ):
            for folder in sorted(root.glob(pattern)):
                if folder.is_dir():
                    for name in names:
                        if (folder/name).is_file():copy(folder/name,Path(label)/folder.name/name)
    video_relative=Path('video')/video_path.name
    copy(video_path,video_relative,expected=manifest['source']['id'])
    for name in ('normalization.json','frame-map.jsonl'):
        if (video_path.parent/name).is_file():copy(video_path.parent/name,Path('video')/name)
    serializable={};skipped={}
    allowed={'result','preview_result','replay_manifest','profiles','runtime','saved_manifest'}
    for name,value in (snapshots or {}).items():
        if name not in allowed:raise ValueError('Snapshot name is not on the output allowlist')
        try:serializable[name]=json.loads(json.dumps(value,allow_nan=False))
        except (TypeError,ValueError):skipped[name]='Not JSON-compatible; use the corresponding saved artifact'
    payload=json.dumps({'results':serializable,'skipped':skipped},indent=2,allow_nan=False).encode('utf-8')
    snapshot=destination/'notebook-results.json'
    with snapshot.open('xb') as stream:stream.write(payload)
    expected=hashlib.sha256(payload).hexdigest()
    if digest(snapshot)!=expected:raise ValueError('Snapshot verification failed')
    hashes[snapshot.name]=expected;sizes[snapshot.name]=len(payload)
    receipt={'schema':'touchline.complete-analysis-backup.v1','run_id':run_id,'created_utc':timestamp,
        'status':'complete','verified_files':len(hashes),'sha256':hashes,'bytes':sizes,
        'restore_run_subfolder':'current/run','analyzed_video':video_relative.as_posix(),
        'parent_included':len(roots)>1,'skipped_snapshots':skipped}
    with (destination/'TRANSFER_COMPLETE.json').open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print('BACKUP COMPLETE:',destination)
    print('Restore SAVED_RUN_DIR:',destination/'current'/'run')
    print('Restore VIDEO_OVERRIDE:',destination/video_relative)
    if skipped:print('Non-JSON snapshots skipped; see notebook-results.json:',sorted(skipped))
    return destination
