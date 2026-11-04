"""Build an unexecuted analysis notebook using the existing training bootstrap."""
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
training=json.loads((ROOT/'notebooks/01_dataset_to_model.ipynb').read_text(encoding='utf-8'))
bootstrap=next(c['source'] for c in training['cells'] if c['cell_type']=='code')
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'})
def code(s):cells.append({'cell_type':'code','metadata':{},'source':textwrap.dedent(s).strip()+'\n','execution_count':None,'outputs':[]})

md('''
# Touchline: analyze existing video in Colab

**Notebook revision 2.2 — complete backup and clear CPU-replay run order.** This repository notebook does not automatically replace a copy saved in your Drive. Download this .ipynb and matching code ZIP. TPU is unsupported. Save runtime-only results before changing runtime; a saved notebook alone does not contain its generated files.

With saved detections and no GPU: setup → CPU runtime mode → Drive/input → 3a restore → 4R replay → 4b visualize → optional reviewed evaluation/profiles → 7c complete backup. Skip 3b when the verified video already exists, and skip 4A new detector inference. The restore cell selects files; the replay cell actually runs ByteTrack. Keep its new RUN_DIR selected afterward.

Colab is the primary runtime for this workflow. Use your already-exported compatible Torchvision `candidate.pt`; this notebook does not train, download pretrained weights, load `.env`, call model APIs or launch Modal. The training notebook remains separate and unchanged.

Run in order, starting with a short capped smoke run. Analysis accepts consistent unrotated landscape dimensions, including 4096x1080 panoramas, at constant 24/1 FPS. Use step 3b to convert other cadences without resizing or changing playback speed. A SoccerNet 25 FPS training image dataset remains valid for training. Track IDs are not named players, and draft metrics/scores are not accepted performance evidence.
''')
md('''
## 1. Load the matching code

The following cell is copied from the training notebook's bootstrap so ZIP/import handling stays consistent. Keep the current checkout if already extracted, upload a current feature/modal-training code ZIP, or set its repository/ref explicitly. No provider keys are needed. Do not replace an already imported checkout in a running kernel; restart when changing code versions.
''')
code(bootstrap)
code('''
NOTEBOOK_REVISION = '2.2-complete-backup-and-delivery-handover'
required_code = ('src/touchline/byte_tracking.py','src/touchline/cuda_motion.py',
    'src/touchline/tracking_replay.py','src/touchline/tracking_evaluation.py','tools/check_cuda_tracking.py')
missing_code = [name for name in required_code if not (REPO_DIR/name).is_file()]
assert not missing_code, 'Old code checkout detected. Load the matching current code ZIP into a fresh code directory and restart the kernel when changing imported code. Missing: ' + ', '.join(missing_code)
print('Notebook:', NOTEBOOK_REVISION)
''')
md('''
## 2. Check the Colab runtime

Choose EXECUTION_MODE='cpu_replay' below when GPU quota is exhausted: use a standard CPU runtime and saved proposals; no Torch/CUDA inference check or checkpoint is required. Use 'cuda_analysis' for new detector inference on an available free GPU. TPU is unsupported. Both modes install vision/tracking helpers without requesting a new Torch build or provider SDK. GPU mode checks the existing Torch/Torchvision CUDA NMS and synthetic motion parity; it fails rather than silently falling back.
''')
code('''
EXECUTION_MODE = 'cpu_replay'  # 'cuda_analysis' only when free GPU access is available.
assert EXECUTION_MODE in ('cpu_replay','cuda_analysis'), 'TPU is unsupported.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[vision,tracking]'], check=True)
CUDA_TRACKING_CHECK_PASSED = False
runtime = {'python': sys.version, 'execution_mode': EXECUTION_MODE}
if EXECUTION_MODE == 'cuda_analysis':
    import torch, torchvision
    from torchvision.ops import nms
    assert torch.cuda.is_available(), 'GPU unavailable. Explicitly choose cpu_replay for saved results; no silent fallback.'
    nms(torch.tensor([[0., 0., 10., 10.]], device='cuda'), torch.tensor([0.9], device='cuda'), 0.5)
    DEVICE = 'cuda'
    runtime.update(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    subprocess.run([sys.executable, 'tools/check_cuda_tracking.py'], check=True, cwd=REPO_DIR)
    CUDA_TRACKING_CHECK_PASSED = True
else:
    DEVICE = 'cpu'
    print('CPU replay selected: no detector inference, model loading or CUDA check. Restore a saved run below.')
print(json.dumps(runtime, indent=2))
cli('doctor')
''')
md('''
## 3. Mount Drive and select existing files

The next cell searches only MyDrive/Touchline for the extracted candidate.pt, skipping image datasets, videos, hidden directories and analysis outputs. It lists alternatives and requires explicit selection when multiple candidates exist. last.pt is never selected automatically. Select only trusted project weights: `.pt`/`.pth` extension alone does not establish compatibility. The checkpoint hash must match its adjacent result.json when present; the loader separately validates embedded metadata/runtime. release.json is not weights. No dataset image tree is required for video inference.

Keep working files in `/content`; save the generated run bundle to Drive in step 7. Drive permissions/quota and Colab runtime availability are not guaranteed. This notebook does not start a web server or create a public tunnel.
''')
code((ROOT/'notebooks/analysis_inputs.py').read_text(encoding='utf-8'))
md('''
## 3a. Restore an existing run for CPU replay

In CPU-replay mode set SAVED_RUN_DIR to the folder containing manifest.json, coverage.jsonl and raw_proposals.jsonl, under MyDrive/Touchline or /content/touchline-analysis. A Drive export usually contains a run/ subfolder: select that subfolder. No weights or source video are needed for association replay. To watch the overlay, VIDEO_OVERRIDE must point to the exact same normalized 24 FPS video (its hash must match); if its original local copy was lost, replay still works but the visual preview needs that matching copy restored. Do not substitute the original 25 FPS video.
''')
code('''
SAVED_RUN_DIR = None  # Path('/content/drive/MyDrive/Touchline/analysis-runs/YOUR_BUNDLE/run')
VIDEO_OVERRIDE = None  # Optional exact analyzed 24 FPS video if its original path changed.
if EXECUTION_MODE == 'cpu_replay':
    RUN_ANALYSIS = False
    if SAVED_RUN_DIR is not None:
        SAVED_RUN_DIR = Path(SAVED_RUN_DIR).resolve()
        assert SAVED_RUN_DIR.is_relative_to(TOUCHLINE_ROOT.resolve()) or SAVED_RUN_DIR.is_relative_to(Path('/content/touchline-analysis')), 'Keep saved runs under Touchline or the local analysis folder.'
        for name in ('manifest.json','coverage.jsonl','raw_proposals.jsonl'):
            assert (SAVED_RUN_DIR/name).is_file(), 'Missing saved-run artifact: ' + name
        RUN_DIR, WORK = SAVED_RUN_DIR, SAVED_RUN_DIR.parent
        saved_manifest = json.loads((RUN_DIR/'manifest.json').read_text(encoding='utf-8'))
        video_value = VIDEO_OVERRIDE if VIDEO_OVERRIDE is not None else saved_manifest['source'].get('path')
        VIDEO_PATH = None
        if video_value is not None:
            candidate_video = Path(video_value)
            assert candidate_video.resolve().is_relative_to(TOUCHLINE_ROOT.resolve()) or candidate_video.resolve().is_relative_to(Path('/content/touchline-normalized')) or candidate_video.resolve().is_relative_to(Path('/content/touchline-analysis')), 'Select project footage only.'
            VIDEO_PATH = candidate_video if candidate_video.is_file() else None
        if VIDEO_PATH is not None:
            print('Checking video hash; large files can take time...')
            assert checkpoint_digest(VIDEO_PATH) == saved_manifest['source']['id'], 'Video exists but is not the analyzed source. Do not overwrite the saved hash.'
        print('Saved run selected:', RUN_DIR, '| Matching video available:', VIDEO_PATH is not None)
        print('RESTORE ONLY: next run step 4R with RUN_RETRACK=True. Skip 4A. Do not restore again after replay.')
    else:
        print('Set SAVED_RUN_DIR above, then rerun this cell. No replay has started.')
''')
md('''
## 3b. Create a 24 FPS analysis copy when needed

For a 25/30/60/23.976 FPS or variable-rate video, set RUN_NORMALIZE=True. This CPU encode can take substantial time and local disk space for a full match; MAX_SELECTED below does not cap it. The original remains untouched. The new video-only copy keeps its dimensions, drops/duplicates frames by source timestamps, and preserves duration within one output frame. Original audio stays in the original source; align separate coach recordings using source-relative time and reviewed offsets.

The exact zero-based frame correspondence and source/output hashes are saved alongside the copy. Reference labels must follow frame-map.jsonl, not the old frame number or an assumed global offset. Keep normalization.json and frame-map.jsonl for later evaluation. A completed 24 FPS file can be reused by setting VIDEO_PATH to it and leaving RUN_NORMALIZE=False. Do not repeat encoding unnecessarily.

The detector still resizes internally to its configured limits (default maximum side 1920); accepting 4096-wide input does not guarantee full-resolution inference or tracking accuracy. Larger inference sizes/tiles require measured GPU-memory and quality comparisons.
''')
code('''
RUN_NORMALIZE = False
ORIGINAL_VIDEO_PATH = None  # If recovering lost footage, explicitly select the original source here.
if RUN_NORMALIZE:
    from touchline.video.normalize import normalize_video
    normalization_source = ORIGINAL_VIDEO_PATH if ORIGINAL_VIDEO_PATH is not None else VIDEO_PATH
    assert normalization_source is not None and Path(normalization_source).is_file(), 'Select the original source video first.'
    NORMALIZATION_DIR = Path('/content/touchline-normalized') / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    VIDEO_PATH = normalize_video(normalization_source, NORMALIZATION_DIR)
    if EXECUTION_MODE == 'cpu_replay' and 'saved_manifest' in globals():
        assert checkpoint_digest(VIDEO_PATH) == saved_manifest['source']['id'], 'Recreated video hash differs. Stop before preview; preserve the original run hash.'
        VIDEO_OVERRIDE = VIDEO_PATH
        print('Recreated video matches the saved analyzed source.')
    print('Use this verified 24 FPS copy for analysis:', VIDEO_PATH)
else:
    print('No conversion started. Non-24 FPS inputs must be normalized before analysis.')
''')
md('''
## 4A. New detector analysis — skip this in CPU-replay mode

Set RUN_ANALYSIS=True only when ready. Keep MAX_SELECTED=240 for the first 12-sample/s smoke test. It limits detector-selected frames, not full-source CPU preflight or total wall time; remaining coverage is unknown. After inspection, set MAX_SELECTED=None and rerun this cell for a NEW full run directory. Full-source FPS/geometry validation occurs before detector construction. All runs use the chosen weights, with pretrained-download fallback disabled.

For the stationary panorama, the new trial uses 12 detector samples/second and ByteTrack with CUDA batched motion prediction; matching and Kalman correction remain CPU-side. This is a hybrid tracker, not full-GPU tracking or an accuracy guarantee. Source FPS remains 24. For the previous baseline choose DETECTOR_FPS=6, TRACKER_MODE='iou', MAX_SELECTED=120. Compare the same source duration and reviewed frame schedule. anchors24 still selects zero-based 5/17 at 2 updates/second. Neither mode implements adaptive sliding-window bursts or appearance ReID.
''')
code('''
RUN_ANALYSIS = False
MAX_SELECTED = 240  # ~20 seconds at 12 samples/s. None only after the smoke run is inspected.
SAMPLER_MODE = 'fixed'
DETECTOR_FPS = 12
TRACKER_MODE = 'bytetrack-cuda' if EXECUTION_MODE == 'cuda_analysis' else 'bytetrack'
DETECTOR_MIN_SIZE = 1080
DETECTOR_MAX_SIZE = 1920  # Internal long-side limit. Benchmark 4096 separately for panoramas; watch GPU memory.
if RUN_ANALYSIS:
    assert EXECUTION_MODE == 'cuda_analysis', 'CPU replay skips new detector inference; use the saved-run/replay cells.'
    assert DEVICE == 'cuda', 'This notebook requires CUDA inference. Stop and select an available free GPU runtime.'
    assert TRACKER_MODE != 'bytetrack-cuda' or CUDA_TRACKING_CHECK_PASSED, 'Run the explicit CUDA tracker parity check before using this experimental path.'
    assert VIDEO_PATH is not None and Path(VIDEO_PATH).is_file(), 'Set VIDEO_PATH to an actual supported video first.'
    assert checkpoint_digest(CHECKPOINT_PATH) == CHECKPOINT_SHA256, 'Checkpoint changed since selection.'
    assert SAMPLER_MODE in ('anchors24', 'fixed')
    assert MAX_SELECTED is None or type(MAX_SELECTED) is int and MAX_SELECTED > 0
    ATTEMPT_ID = datetime.datetime.now(datetime.timezone.utc).strftime('analysis-%Y%m%dT%H%M%S%fZ')
    WORK = Path('/content/touchline-analysis') / ATTEMPT_ID
    WORK.mkdir(parents=True, exist_ok=False)
    RUN_DIR = WORK / 'run'
    (WORK/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\\n', encoding='utf-8')
    if 'NORMALIZATION_DIR' in globals() and Path(VIDEO_PATH).resolve() == (NORMALIZATION_DIR/'video24.mp4').resolve():
        for name in ('normalization.json', 'frame-map.jsonl'):
            shutil.copyfile(NORMALIZATION_DIR/name, WORK/name)
    cli('analyze', VIDEO_PATH, '--checkpoint', CHECKPOINT_PATH, '--device', DEVICE,
        '--mode', SAMPLER_MODE, '--output', RUN_DIR, '--record-proposals',
        '--fps', DETECTOR_FPS, '--tracker', TRACKER_MODE,
        '--min-size', DETECTOR_MIN_SIZE, '--max-size', DETECTOR_MAX_SIZE,
        *(['--max-selected', MAX_SELECTED] if MAX_SELECTED is not None else []))
    manifest = json.loads((RUN_DIR/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['detector']['checkpoint'] == CHECKPOINT_SHA256, 'Run did not record the selected model hash.'
    print('Verified: this run used the selected uploaded checkpoint.')
    print('Detector device requested:', manifest['detector'].get('device'))
    print('Actual detector runtime:', json.dumps(manifest.get('detector_runtime', {'note': 'Older code did not record actual placement telemetry.'}), indent=2))
    print('Actual tracker runtime:', json.dumps(manifest.get('tracker_runtime', {}), indent=2))
    print(json.dumps({k: manifest.get(k) for k in ('run_id','status','decoded_frames','selected_frames','observation_count','diagnostic_trace_id')}, indent=2))
else:
    print('Analysis disabled. Select files, then set RUN_ANALYSIS=True.')
''')
md('''
## 4R. Replay saved proposals — run this cell in CPU-replay mode

Use this cell with an existing RUN_DIR from the older six-FPS run. Leave RUN_RETRACK=False unless you intend to replay that selected run. It does not add detections, read/decode video or use the detector GPU: it changes association only, retaining partial status and the parent hashes. RETRACK_DEVICE='cuda' runs batched motion prediction on CUDA; 'cpu' uses CPU throughout. New raw proposals, observations and run ID are generated in a fresh directory. You can then run step 4b against the replay result. Do not rerun the analysis cell first if it would replace RUN_DIR with a different run.
''')
code('''
RUN_RETRACK = False
RETRACK_DEVICE = 'cuda' if EXECUTION_MODE == 'cuda_analysis' else 'cpu'
if RUN_RETRACK:
    assert 'RUN_DIR' in globals() and (RUN_DIR/'manifest.json').is_file(), 'Select an existing saved run in step 3a first.'
    assert RETRACK_DEVICE != 'cuda' or CUDA_TRACKING_CHECK_PASSED, 'Run the CUDA parity check first, or explicitly choose CPU replay.'
    from touchline.tracking_replay import retrack_run
    PARENT_RUN_DIR, PARENT_WORK = RUN_DIR, WORK
    REPLAY_WORK = Path('/content/touchline-analysis') / datetime.datetime.now(datetime.timezone.utc).strftime('retrack-%Y%m%dT%H%M%S%fZ')
    REPLAY_WORK.mkdir(parents=True, exist_ok=False)
    replay_manifest = retrack_run(PARENT_RUN_DIR, REPLAY_WORK/'run', prediction_device=RETRACK_DEVICE)
    for name in ('runtime.json','normalization.json','frame-map.jsonl'):
        if (PARENT_WORK/name).is_file():
            shutil.copyfile(PARENT_WORK/name, REPLAY_WORK/name)
    WORK, RUN_DIR = REPLAY_WORK, REPLAY_WORK/'run'
    RUN_ANALYSIS = True  # Enables downstream profiles/exports; does not start inference.
    print(json.dumps(replay_manifest, indent=2))
else:
    print('No replay started; current saved run remains selected.')
''')
md('''
## 4b. Watch detections and inspect player boxes

Run after analysis. This renders saved results without rerunning the GPU model: a source-resolution 24 FPS overlay MP4, contact sheet, player gallery and preview manifest. The default preview is at most 20 seconds and stops at processed coverage in partial runs. Large colored T-number badges identify saved temporary tracks, not jersey numbers or player names. The players.jpg gallery maps each track to a first-seen crop; track-index.json lists coverage in this preview. Colors repeat, so use the printed ID. Set FOCUS_TRACK_ID to an integer from the index to produce a zoomed focus24.mp4. Missing observations produce an explicit unavailable frame, rather than switching to another ID. Optional amber proposals are hidden by default. Confidence is not accuracy or roster identity.

Between sampled inference frames, the preview can hold the last overlay for at most 0.17 seconds. Every such frame says HELD / no new inference; it is a display convenience, not measured tracking at 24 FPS. Set VISUAL_HOLD_SECONDS=0 to show boxes only on genuinely sampled frames. If proposals were not recorded, amber diagnostics are unavailable. CPU decode/drawing/encoding is expected here. Do not switch to a CPU runtime for detector inference; the model and input CUDA telemetry above are recorded by the inference subprocess, unlike memory queried from the separate notebook process.

Run the Drive-save cell afterward to retain preview files. For the same run you can render a new preview without running analyze again. A final profile/report is not a substitute for inspecting this footage.
''')
code('''
RENDER_PREVIEW = True
PREVIEW_SECONDS = 20
VISUAL_HOLD_SECONDS = 0.17  # 0 = exact sampled-frame overlays only.
FOCUS_TRACK_ID = None  # E.g. 7 follows T7; this is NOT the jersey number.
SHOW_PROPOSALS = False  # Optional amber diagnostics; keep hidden for clear labels.
if RENDER_PREVIEW and 'RUN_DIR' in globals() and (RUN_DIR/'manifest.json').is_file():
    assert VIDEO_PATH is not None and Path(VIDEO_PATH).is_file(), 'Replay does not require video, but visual overlays need the exact analyzed 24 FPS copy. Restore it and set VIDEO_OVERRIDE.'
    from IPython.display import display, Video, Image as DisplayImage
    from touchline.video.preview import render_run_preview
    PREVIEW_DIR = WORK / ('preview-' + datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f'))
    preview_result = render_run_preview(RUN_DIR, PREVIEW_DIR, source=VIDEO_PATH,
        max_seconds=PREVIEW_SECONDS, hold_seconds=VISUAL_HOLD_SECONDS,
        focus_track_id=FOCUS_TRACK_ID, show_proposals=SHOW_PROPOSALS)
    print(json.dumps(preview_result, indent=2))
    display(DisplayImage(filename=str(PREVIEW_DIR/'contact-sheet.jpg')))
    display(DisplayImage(filename=str(PREVIEW_DIR/'players.jpg')))
    print((PREVIEW_DIR/'track-index.json').read_text(encoding='utf-8'))
    display(Video(str(PREVIEW_DIR/'overlay24.mp4'), embed=True, width=1100))
    if FOCUS_TRACK_ID is not None:
        display(Video(str(PREVIEW_DIR/'focus24.mp4'), embed=True, width=960))
    print('Video file:', PREVIEW_DIR/'overlay24.mp4')
else:
    print('No saved analysis run selected; no preview was rendered.')
''')
md('''
## 4c. Score identity consistency when reviewed reference IDs are available

Leave TRACKING_REFERENCE_PATH=None until you have genuine independent reference IDs/boxes. See examples/tracking-reference.template.json and docs/BYTETRACK_RUNBOOK.md. This CPU evaluation does not run the detector and cannot infer accuracy from unlabelled footage. Use exactly the same reviewed sampled frames when comparing runs; original SoccerNet labels require the normalized-source frame mapping. The exported score remains limited to that reviewed scope, not full-video acceptance.
''')
code('''
TRACKING_REFERENCE_PATH = None  # Path to genuinely reviewed tracking-reference JSON.
if TRACKING_REFERENCE_PATH is not None:
    assert 'RUN_DIR' in globals() and (RUN_DIR/'manifest.json').is_file(), 'Select or replay a saved run first.'
    reference_path = Path(TRACKING_REFERENCE_PATH)
    assert reference_path.suffix.lower() == '.json' and reference_path.is_file()
    EVALUATION_DIR = WORK / ('evaluation-' + datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f'))
    cli('evaluate-tracking', RUN_DIR, reference_path, '--output', EVALUATION_DIR/'tracking-evaluation.json')
    print((EVALUATION_DIR/'tracking-evaluation.json').read_text(encoding='utf-8'))
else:
    print('No reviewed reference supplied: tracking accuracy remains unknown.')
''')
md('''
## 5. Generate player evidence profiles locally in this Colab runtime

Run after successful analysis. Every observed track is retained; named identity stays unresolved until reviewed association. Without valid reviewed calibration, physical distance/speed remains unavailable. This step is CPU-side and uses no provider key. Optional JSON files must come from genuine review; do not set review flags just to obtain scores.

The existing `associate` CLI supports reviewed identity entries. Its reviewer identity, player ID and track must reflect a real review; see the code runbook. Persist the resulting associations before regenerating profiles. The first notebook pass can leave all optional inputs unset.
''')
code('''
CALIBRATION_PATH = None  # Optional Path to reviewed Calibration JSON.
RUBRIC_PATH = None  # Optional Path to reviewed touchline.rubric.v1 JSON.
SESSION_CONTEXT_PATH = None  # Optional Path to matching reviewed drill/context JSON.
if RUN_ANALYSIS and (RUN_DIR/'manifest.json').is_file():
    PROFILE_DIR = WORK / ('profiles-' + datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f'))
    optional = []
    for flag, path in (('--calibration', CALIBRATION_PATH), ('--rubric', RUBRIC_PATH), ('--session-context', SESSION_CONTEXT_PATH)):
        if path is not None:
            assert Path(path).is_file() and Path(path).suffix.lower() == '.json'
            optional += [flag, path]
    cli('player-analysis', RUN_DIR, '--output', PROFILE_DIR, *optional)
    profiles = json.loads((PROFILE_DIR/'player-analysis.json').read_text(encoding='utf-8'))
    print('Profiles:', len(profiles['profiles']), 'Unresolved tracks:', profiles['unresolved_tracks'])
    print((PROFILE_DIR/'review.md').read_text(encoding='utf-8'))
else:
    print('Run analysis first; no profile job started.')
''')
md('''
## 6. Collect support logs, including after an analysis failure

Run this cell even if step 4 or 5 failed after writing a manifest. It exports the actual run ID, safe diagnostic summary and numeric losses if present. It is a snapshot, not remote monitoring. Raw trace journals remain in the private runtime; only the closed-schema summary is copied below. If no manifest was written, use the printed CLI trace with `diagnostic-report` before disconnecting.
''')
code('''
if RUN_ANALYSIS and 'RUN_DIR' in globals() and (RUN_DIR/'manifest.json').is_file():
    SUPPORT_DIR = WORK / ('support-' + datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f'))
    cli('run-log-export', RUN_DIR, '--output', SUPPORT_DIR, '--diagnostics-dir', REPO_DIR/'runs'/'diagnostics')
    print((SUPPORT_DIR/'support.json').read_text(encoding='utf-8'))
else:
    print('No persisted analysis manifest yet. Preserve the CLI trace/error before resetting the runtime.')
''')
md('''
## 7. Save a verified run bundle to Drive

Copies only generated analysis/profile/support/runtime artifacts, never the checkout, `.env`, original video or model weights. This bundle may contain player/session data: keep it private and share the support snapshot separately for debugging. Unique destinations prevent overwrite. A final `TRANSFER_COMPLETE.json` is written only after all copied file hashes match; a failed transfer leaves an incomplete directory to inspect/retry under a new name. Rerun this cell after collecting support, even after a failed analysis. Do not delete the local run before transfer verifies.
''')
code('''
def save_analysis_bundle(work, run_dir, drive_root):
    from touchline.data.manifest import sha256_file
    work, run_dir, drive_root = Path(work), Path(run_dir), Path(drive_root)
    manifest = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    import re
    run_id = manifest['run_id']
    assert isinstance(run_id, str) and re.fullmatch('[a-zA-Z0-9_-]{1,120}', run_id), 'Invalid persisted run ID.'
    destination = drive_root / (run_id + '-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True, exist_ok=False)
    selected = [work/name for name in ('runtime.json', 'normalization.json', 'frame-map.jsonl')]
    selected += [run_dir/name for name in ('manifest.json','observations.jsonl','coverage.jsonl','raw_proposals.jsonl','analytics.json','review.html','associations.json')]
    for pattern, names in (('profiles-*', ('player-analysis.json','review.md')), ('support-*', ('support.json','losses.jsonl')), ('preview-*', ('overlay24.mp4','contact-sheet.jpg','players.jpg','track-index.json','focus24.mp4','preview.json')), ('evaluation-*', ('tracking-evaluation.json',))):
        for folder in work.glob(pattern):
            selected += [folder/name for name in names]
    hashes = {}
    for source in selected:
        assert not any(p.is_symlink() for p in (source, *source.parents)), 'Symlink in generated artifacts.'
        assert source.resolve().is_relative_to(work.resolve()), 'Artifact outside selected run.'
        if not source.is_file():
            continue
        relative = source.relative_to(work)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256_file(source)
        shutil.copyfile(source, target)
        assert sha256_file(target) == expected, 'Transfer verification failed; preserve the local run.'
        hashes[relative.as_posix()] = expected
    assert 'run/manifest.json' in hashes, 'Manifest was not exported.'
    (destination/'TRANSFER_COMPLETE.json').write_text(json.dumps({'run_id': run_id, 'sha256': hashes}, indent=2)+'\\n', encoding='utf-8')
    return destination

if RUN_ANALYSIS and 'RUN_DIR' in globals() and (RUN_DIR/'manifest.json').is_file():
    SAVED_RUN = save_analysis_bundle(WORK, RUN_DIR, DRIVE_OUTPUT_ROOT)
    print('Verified Drive bundle:', SAVED_RUN)
else:
    print('No generated run to export.')
''')
md('''
## 7b. Optional: preserve the exact analyzed video before changing runtime

The result bundle above excludes footage. If the normalized 24 FPS video exists only under /content, switching/losing the runtime can make later overlays unavailable. Enable this optional cell while that exact file is still available. It creates a fresh folder under Touchline/analysis-videos, copies the video without changing it, verifies its hash against the run, and writes VIDEO_COPY_COMPLETE.json last. Check Drive quota first; large-video copies can take time or fail. The original stays untouched. In a later CPU session set VIDEO_OVERRIDE to this saved file. An interrupted copy has no completion marker and must not be assumed complete.
''')
code('''
SAVE_ANALYZED_VIDEO = False
if SAVE_ANALYZED_VIDEO:
    from touchline.data.manifest import sha256_file
    assert VIDEO_PATH is not None and Path(VIDEO_PATH).is_file()
    assert Path(VIDEO_PATH).suffix.lower() in ('.mp4','.mov','.mkv')
    video_manifest = json.loads((RUN_DIR/'manifest.json').read_text(encoding='utf-8'))
    expected_video_hash = video_manifest['source']['id']
    assert sha256_file(VIDEO_PATH) == expected_video_hash, 'Video differs from the analyzed source.'
    video_folder = TOUCHLINE_ROOT/'analysis-videos'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    video_folder.mkdir(parents=True, exist_ok=False)
    saved_video = video_folder/Path(VIDEO_PATH).name
    shutil.copyfile(VIDEO_PATH, saved_video)
    assert sha256_file(saved_video) == expected_video_hash, 'Video copy failed verification; preserve the original.'
    (video_folder/'VIDEO_COPY_COMPLETE.json').write_text(json.dumps({'run_id':video_manifest['run_id'],
        'filename':saved_video.name,'sha256':expected_video_hash}, indent=2)+'\\n', encoding='utf-8')
    print('Verified video copy. Use as VIDEO_OVERRIDE in a later session:', saved_video)
else:
    print('No source video copied. Preserve local footage yourself before changing runtime if needed.')
''')
md('''
## 7c. Final complete backup — recommended before disconnecting

Run after the previews/profiles and any reviewed evaluation. This saves all known generated artifacts in the current work folder, available parent-run artifacts, the exact analyzed video, normalization provenance and JSON-compatible result dictionaries to Touchline/complete-results. It excludes credentials, code, model weights and datasets. Enough Drive space is required; video copying and hash verification can take time. Keep the originals until BACKUP COMPLETE is printed. A failed/interrupted folder has no completion receipt and is not an accepted backup.

This is the all-in-one backup. Steps7/7b remain available for smaller separate exports; you do not have to duplicate those copies first. The final cell prints the exact SAVED_RUN_DIR and VIDEO_OVERRIDE paths for a later restore. In these complete backups, select current/run, not the backup root. It captures saved files and selected result dictionaries, not arbitrary console text or browser-widget state.
''')
code((ROOT/'notebooks/analysis_backup.py').read_text(encoding='utf-8')+'''

SAVE_COMPLETE_RESULTS = True
if SAVE_COMPLETE_RESULTS and 'RUN_DIR' in globals() and (Path(RUN_DIR)/'manifest.json').is_file():
    result_snapshots = {name:globals()[name] for name in (
        'result','preview_result','replay_manifest','profiles','runtime','saved_manifest') if name in globals()}
    COMPLETE_BACKUP = save_complete_analysis(WORK, RUN_DIR, TOUCHLINE_ROOT, VIDEO_PATH,
        parent_work=globals().get('PARENT_WORK'), parent_run_dir=globals().get('PARENT_RUN_DIR'),
        snapshots=result_snapshots)
else:
    print('No selected saved run to back up. Restore/replay first; no backup was started.')
''')
md('''
## Next review

### Operator update — 31 October 2026

Model source: the operator's extracted Touchline-model-last-pt bundle in MyDrive/Touchline. Prefer its unique candidate.pt; verify result.json and the exact manifest checkpoint SHA256. No automatic pretrained replacement or arbitrary last.pt selection. Cell 3 can inspect project files after Drive authorization; leave VIDEO_PATH unset until an actual supported clip is chosen.

Compute: free Colab only. Do not buy compute units, upgrade, start paid cloud jobs or use provider keys. Use a free T4 when available; stop if the interface requires payment or paid units.

The current requested tracking minimum is reviewed sampled-frame IDF1 >= 0.75, desired 0.80; earlier 0.85/0.90 ambitions remain stretch targets. Use genuine held-out reference IDs, report detection precision/recall and reviewed identity changes, and compare identical frame schedules. This is not an achieved score or roster-identification claim. Detection confidence and AP50 are separate metrics. SoccerNet labels must follow the exact normalized frame map for converted videos; dedicated native-FPS image-sequence evaluation remains separate. CPU ByteTrack and optional CUDA batched prediction are now integrated, but actual CUDA/footage/benchmark acceptance remains pending. The old IoU CLI baseline is retained. Calibrate on validation; keep test sealed. See docs/BYTETRACK_RUNBOOK.md.

Inspect `run/review.html`, `run/analytics.json` and the profile drafts. Capped runs are partial; unknown identity/geometry/coverage never becomes a zero or validated score. Retain the existing approved candidate if new training is worse. The proposed 95% precision/recall goal still requires independent own-camera evaluation.

Modal remains a separate optional executor. Baseten repair candidates remain staged and require review/isolated checks; do not execute generated patches inside a Drive-authorized Colab kernel. This notebook adds no automatic repair watcher or continuous training from uploaded videos.

See `docs/COLAB_ANALYSIS_RUNBOOK.md` in the matching checkout and [Colab's resource/Drive guidance](https://research.google.com/colaboratory/faq.html). Runtime/GPU availability varies; save verified results before disconnecting.
''')
for i,cell in enumerate(cells):cell['id']=f'touchline-analysis-{i:03d}'
target=ROOT/'notebooks/02_video_analysis.ipynb'
target.write_text(json.dumps({'cells':cells,'metadata':{'colab':{'name':target.name},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5},indent=2)+'\n',encoding='utf-8')
print(target)
