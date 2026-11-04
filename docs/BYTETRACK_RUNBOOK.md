# Stationary-panorama ByteTrack and CUDA trial

The view is stationary. This trial targets fewer temporary-ID splits using motion and low-score association. It does not recognize jerseys or promise 75–80% identity accuracy. All actual CUDA/quality acceptance is pending.

## Load current code and dependencies

Use the matching feature/modal-training code ZIP/notebook. Preserve existing results and the verified 24 FPS video. Install `pip install -e '.[vision,tracking]'` without replacing Colab's working Torch/Torchvision pair. The tracking backend is pinned to supervision==0.26.0; it requires CPU numerical/image dependencies. Do not load .env or provider keys. Free Colab only. Keep the existing source and model selection/hashes.

## Explicit CUDA validation before real input

On the free GPU runtime, run `python tools/check_cuda_tracking.py`. This performs small synthetic CUDA/CPU prediction and association comparisons only, without weights/video. It must report synthetic_cuda_parity_passed before enabling the experimental CUDA path. The agent has not executed this check. A failure is a stop; do not silently substitute CPU or claim GPU tracking. The notebook runtime cell includes this explicit gate before the analysis cell.

## Reuse the current saved run first

No detector rerun is needed when raw_proposals.jsonl is present. In the updated checkout:

```python
from touchline.tracking_replay import retrack_run
import datetime
PARENT_RUN_DIR, PARENT_WORK = RUN_DIR, WORK
REPLAY_WORK = WORK.parent / datetime.datetime.now(datetime.timezone.utc).strftime('retrack-%Y%m%dT%H%M%S%fZ')
REPLAY_WORK.mkdir(parents=True, exist_ok=False)
result = retrack_run(PARENT_RUN_DIR, REPLAY_WORK/'run', prediction_device='cuda')
print(result['tracker_runtime'])
# Select the new run for the existing visualization cells:
WORK, RUN_DIR = REPLAY_WORK, REPLAY_WORK/'run'
```

CUDA replay uses CUDA motion prediction and CPU association/correction, with zero detector inference. Choose prediction_device='cpu' explicitly for the CPU comparison. The notebook's optional replay cell also copies the allowlisted runtime/normalization/frame-map files from PARENT_WORK; copy those for this manual route too before exporting if needed. Render the new run's overlay/gallery/focused video using the existing VIDEO_PATH. Names and original observations are untouched; a capped parent remains partial. Neither replay nor the preview fixes a person missing from saved proposals.

## New detector trial

Use TRACKER_MODE='bytetrack-cuda', DETECTOR_FPS=12, MAX_SELECTED=240 (~20s) and DEVICE='cuda'. Source remains24FPS. CUDA is required for the detector and batched motion prediction. CPU handles assignment, Kalman correction, decoding/preflight and display encoding: this is HYBRID ByteTrack, not a fully GPU tracker. For a CPU-tracker comparison use TRACKER_MODE='bytetrack' while leaving DEVICE='cuda'. For the original baseline use TRACKER_MODE='iou', DETECTOR_FPS=6, MAX_SELECTED=120. Never compare caps as equal duration without accounting for sample rate. No actual speedup has been measured; moving small single-stream math to CUDA can cost more than CPU.

DETECTOR_MAX_SIZE remains1920 for comparability. A separate bounded max-size4096 trial can preserve more panoramic inference detail if free GPU memory permits; log runtime/peak memory and compare held-out misses. Do not treat filling VRAM as an accuracy objective or silently retry on CPU. Exact resolution is not an accuracy guarantee.

Check manifest.detector_runtime for actual CUDA model/input placement and inference count. Check manifest.tracker_runtime for motion_prediction_device starting with cuda and positive cuda_prediction_batches; association_device and kalman_correction_device remain cpu. Prediction and total tracking wall seconds include transfers, not GPU utilization. Only real operator execution can verify those values. New ByteTrack motion state is not resumable; always use fresh run directories. Do not attempt to resume an old IoU run as ByteTrack.

## Measure identity consistency

Use examples/tracking-reference.template.json to construct a genuinely reviewed reference. Keep it unreviewed until complete; no auto-mapping from predicted IDs. Provide source_sha256 and frames=[{frame_index:0,people:[{reference_id:'stable-person-A',bbox_xyxy:[x1,y1,x2,y2]}]}]. Use zero-based indices in the normalized source and all people/empty frames for the admitted review scope. Original SoccerNet 25-FPS annotations need the recorded frame map and reviewed coordinate/clip offsets.

Run `python -m touchline evaluate-tracking RUN_DIR REFERENCE_JSON --output NEW_REPORT_JSON`. Only explicitly reviewed sampled frames are admitted; no silent omission of unsampled or missing frames. Compare identical reviewed frame IDs for 6 vs12FPS. Report sampled-frame IDF1 (minimum0.75, desired0.80), detection precision/recall, reviewed identity changes and predicted IDs per reference. A perfect detector split into two IDs can still score IDF1=0.5. Synthetic input never satisfies quality targets. Full-video accuracy remains unknown; use [TrackEval](https://github.com/JonathonLuiten/TrackEval) for an independent canonical dataset benchmark/preprocessing cross-check. Real footage, CUDA parity/performance, appearances/crossings and quality acceptance remain open. Earlier 0.85–0.90 tracking ambitions and the separate 95% detection goal are unpassed.

The [ByteTrack method](https://github.com/FoundationVision/ByteTrack) combines predicted motion and two-stage detection association; the [pinned Supervision API](https://supervision.roboflow.com/0.26.0/trackers/) supplies the CPU backend. Our optional CUDA module replaces only batched prediction on that backend instance. It adds no ReID model, long-gap identity guesses or provider execution.

## GPU-quota exhaustion: notebook revision 2.1

Notebook revision 2.1 adds an explicit cpu_replay default for exhausted GPU quota, skips Torch/CUDA and checkpoint admission in that mode, restores saved-run artifacts, retains explicit cuda_analysis for later GPU use, adds reviewed IDF1 evaluation/export and optional hash-verified source-video preservation. A visible notebook version and required-code check reject stale feature checkouts. TPU remains unsupported. 336 guarded offline tests pass; actual Colab/Drive/CUDA and real-footage quality remain unverified. CPU replay changes association only and cannot recover players absent from cached proposals. No real credentials, browser integration, paid/GPU/model job or existing PR changes.

Download/open the new notebooks/02_video_analysis.ipynb; pushing a repository notebook does not update a separate Drive notebook. Load its matching code ZIP. The notebook checks required tracking files early; an old /content/touchline-starter checkout is not upgraded merely by opening a new notebook. Use a fresh code directory and restart the kernel when replacing imported code; preserve run files and source footage first. Do not delete the runtime as an upgrade step.

For today choose a standard CPU runtime, EXECUTION_MODE='cpu_replay'. Run setup/runtime and Drive/input cells. In step3a set SAVED_RUN_DIR to the run subfolder containing manifest.json, coverage.jsonl and raw_proposals.jsonl. Set VIDEO_OVERRIDE to the exact analyzed normalized 24 FPS source if its path changed. Skip the new detector-analysis cell. Enable RUN_RETRACK=True; RETRACK_DEVICE is explicitly cpu in this mode. Then run visualization/profiles, optional genuinely reviewed IDF1 scoring, and Drive export. No checkpoint/Torch/CUDA requirement for CPU replay. If raw proposals were lost, cached reassociation is unavailable and detection must be rerun separately; an unlabelled saved model alone cannot reconstruct the prior detections.

The run bundle excludes footage. Before changing or losing the runtime, optionally set SAVE_ANALYZED_VIDEO=True in step7b to copy the exact analyzed video under Touchline/analysis-videos, verify its run-source hash, and write VIDEO_COPY_COMPLETE.json last. Check Drive quota yourself and retain the original until verification completes. A later preview requires that identical source; the original 25 FPS input is not interchangeable with its converted 24 FPS copy. Replay itself can proceed without footage, but video overlays cannot. [Colab FAQ](https://research.google.com/colaboratory/faq.html) explains that runtime files are temporary and resource limits vary; no fixed GPU-reset time is promised.

When free GPU access returns, select EXECUTION_MODE='cuda_analysis', run the CUDA/NMS/parity checks, select weights/video, and use the existing bounded GPU trial. There is no automatic hardware fallback or TPU implementation. CPU mode imports no Torch/CUDA runtime; no paid provider path is enabled. New IDF1 evaluation runs only when TRACKING_REFERENCE_PATH is explicitly supplied and genuinely reviewed. Scores cover the admitted sampled frames; no accuracy is inferred from missing labels.
