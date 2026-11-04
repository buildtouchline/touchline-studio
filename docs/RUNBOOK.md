# Manual local runbook

Start with [Colab training](COLAB_RUNBOOK.md); analysis implementation follows a working trained/evaluated/reloaded model bundle. This local runbook checks repository wiring without infrastructure.

## Set up and check

From the repository root, Python 3.12+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[data]"
python -m unittest discover -s tests -v
touchline preflight
touchline demo --output runs/demo-001
```

Open `runs/demo-001/review.html`. Every artifact is synthetic and labelled as such. If package installation is unnecessary, run directly with `$env:PYTHONPATH='src'` then `python -m touchline ...`. Temporary-directory access problems can be avoided by setting TEMP/TMP to a writable local test directory before Python starts. No CI or cloud infrastructure is needed.

## Existing datasets to model

```text
touchline inventory /absolute/data/root --output runs/inventory-001.json
touchline etl /absolute/reviewed-config.json --output data/release-001
touchline validate-dataset data/release-001/release.json --training-gate
touchline train-detector data/release-001/release.json --output runs/train-001 --device cuda --epochs 2 --max-steps 100 --max-seconds 600 --allow-weight-download
touchline evaluate-detector data/release-001/release.json --output runs/baseline-val --device cuda --allow-weight-download
touchline evaluate-detector data/release-001/release.json --output runs/candidate-val --device cuda --checkpoint runs/train-001/candidate.pt
```

Use actual Windows paths on Windows. Optional SoccerNet staging precedes ETL: `touchline stage-soccernet CONFIG.json --output data/staging-001`. A compatible Torch/Torchvision pair, Pillow, pycocotools and GPU runtime are required for training/evaluation. Inventory/import do not need Torch. See the notebook for overlays, runtime capture, tiny-train/reload, same-sample comparison and export. Do not interpret an import pass as label-quality validation.

## Later: short real-video smoke check

Install `.[vision]` plus the separately verified Torch/Torchvision pair. With a trusted local candidate bundle:

```text
touchline analyze /absolute/clip.mp4 --checkpoint runs/train-001/candidate.pt --output runs/video-001 --device cuda --mode fixed --fps 6 --max-selected 120
```

The frame cap creates a **partial** run, not a complete analysis. Review manifest status, original PTS/time base in observations, coverage/no-detection records and tentative track IDs. The HTML trail is image-space and prefix-limited. Strict `--mode anchors24` requires a complete uniform 24 FPS decode; VFR/skipped frames fail. It is not a general frame-rate converter. No roster identity or physical metric clearance is inferred.

## Trial checks to implement/perform after the model gate

1. Record a fixed-camera short drill with visible pitch references, measured course and independent timing. Keep original source file/hash and capture metadata.
2. Test detection on near/far players, blur and occlusion. Manually audit identity segments. Mark unknown periods, never inactivity from missing observations.
3. Compare image-space paths/occupancy and reviewed drill/attempt timestamps with source playback. Clip extraction/full video viewer remain implementation tickets.
4. Calibrate the actual camera, verify against unused ground references, exclude invalid regions/zoom changes, then compare distance/speed against measurements. Evaluate intensity/activity labels separately.
5. Use the Android app only after an actual device build passes record/stop/export/replay and video alignment tests. See [the recorder spec](https://github.com/buildtouchline/touchline/blob/tickets/docs/specs/TL-S13-android-coach-recorder.md); no APK is shipped here.
6. Freeze data/model/code/environment and report pass/fail/unavailable per group. Do not promote a demo-only fixture to field evidence.

## Failure and recovery

All run/release directories must be fresh; never overwrite evidence. ETL failures retain an error report but no successful release. Video runs retain failed/partial manifests; automatic resume is not supplied. Training has final checkpoint export, not periodic interruption recovery. Preserve logs and repeat a bounded run under a new name. Retain the baseline if the candidate regresses. Record actual command, environment, hashes, result and reviewer in the ticket completion record.
