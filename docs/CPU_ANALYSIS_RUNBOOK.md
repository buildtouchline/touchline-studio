# CPU video analysis; GPU reserved for training

## Latest measured AP50 comparison

3 November full AP50 rerun: same checkpoint, immutable570-image validation list/labels and unchanged COCO metric. Default CPU AP50=41.3580%, higher-resolution(min1080/max4096)=61.1757%, change=+19.8177 percentage points. Previous GPU baseline=41.3578%. Both complete reports/predictions and resize provenance exported; no training or final-test evaluation, no automatic promotion. 431 tests/artifact checks passed inside Modal before full validation. Worker termination confirmed. This measures SoccerNet person detection, not panoramic player identity or SAM accuracy. SAM remains an independent disagreement-review pass, not ground truth. The separate$2 reservation shares the cumulative$50 ceiling; all earlier holds remain unreconciled.

Results: runs/modal-ap50-20261103/evaluation-comparison.json and ap50-results.zip. evaluate-detector now accepts paired --min-size/--max-size options; omission preserves the previous model transform. The metric code, final test and approved model are unchanged.

Controlling workflow, 2 November 2026: use CPU for video analysis. Reserve Modal
GPU compute for separately authorized training on reviewed releases. Existing
GPU analysis helpers are retained; do not dispatch them under this instruction.

## Selected execution: Modal CPU Sandbox

The user's clarification is explicit: CPU analysis runs in a **Modal CPU Sandbox**,
not on the desktop. They approved up to $10 additional for the entire four-minute
job. Entry point:

```shell
python -B tools/run_modal_cpu_analysis.py --work runs/NEW_APPROVED_CPU_SANDBOX_RUN --approved-ten-dollar
```

The current single-use claim/output folder is runs/modal-cpu-full-video-20261102.
That run completed: 5760 frames, 1440 detector samples and 11429 observations,
with an actual 240-second overlay, profiles and mapping images exported. CPU
parameter/input placement was recorded; Sandbox termination succeeded. The
installed Torch build has a +cu128 suffix, but gpu_requested is false and actual
model/input devices are cpu. Installed CUDA libraries do not imply GPU usage.

Open RESULTS.html in that folder. The cpu-results.zip contains the exported
cloud artifacts; source/video24.mp4 and its frame map/normalization record were
also restored locally for later replay. The CPU receipt and controller/worker
logs retain hashes and run ID ef942e00-d536-4d50-95e6-1ea153354b2f.
Do not create another claim using this example: the existing approval and ledger
are tied to this job. The runner uses environment main, four CPU cores, 8 GiB RAM,
no GPU, gVisor, blocked egress, no injected secrets/mounts and a 7200-second maximum
Sandbox lifetime. The fixed worker process is bounded to 6500 seconds. A cached
image is reused; no new dependency install or image build is performed.

The trusted controller alone loads Modal credentials from the selected root .env.
The Sandbox receives only frozen code, the full normalized video, checkpoint and
worker script, bound by hashes. It runs CPU NMS, full-video CPU inference/tracking,
profiles and full overlay rendering. GPU analysis is never dispatched. Results
are downloaded and checked before Sandbox termination; the local normalized source
is copied into the result folder for later replay. Raw weights and keys are not
included in the result ZIP.

Watch modal-cpu.json, controller.events.jsonl and worker.events.jsonl. The worker
streams safe stage/counter events every 30 seconds, including preflight, inference,
profiles, rendering and export. Failure exports include available diagnostic/run
JSON records before termination. These streamed exec logs differ from the earlier
GPU Function's provider-log retrieval; both forms remain preserved separately.

The $10 is a retained reservation, not a measured charge or provider-enforced
budget. Earlier GPU/provider holds are preserved. A confirmed stop is not proof
of zero charges. No model training is started under this CPU-only authorization.

Code validation is also executed remotely under the user's latest preference.
tools/run_modal_remote_checks.py uploads source only (no media/weights/keys) to
a separate no-GPU, 2-core/4-GiB, 900-second Sandbox sharing the same retained CPU
hold. It returns bounded logs and a result JSON. It does not run a local test
suite, promote a model or start training. The internal network guard still
prevents the code checks from accessing secrets or external services.

Remote validation completed: 400 tests passed; 136 Python files, 28 notebook
cells and 115 links passed in the frozen upload. The initial upload omitted
Skills.md; source packaging was corrected and the remote recheck passed. Both
validation Sandboxes were terminated. Logs/receipts are under
runs/modal-cpu-checks-20261102 and runs/modal-cpu-checks-20261102-retry.

The earlier local attempt completed normalization to 5760 frames/24 FPS but failed
during detector import; it did not complete analysis. Its normalized source is
reused, so do not delete runs/cpu-full-video-20261102 until the cloud results and
source-restoration copy are verified. The optional desktop helper below is kept
for compatibility and is not the selected execution route.

## Optional desktop helper (not selected)

The user selected the whole four-minute artifacts/videos/118577.mp4 and the
approved artifacts/models/football-person/candidate.pt. Explicit entry point:

```shell
python -B tools/run_cpu_video_analysis.py --video artifacts/videos/118577.mp4 --checkpoint artifacts/models/football-person/candidate.pt --output runs/NEW_CPU_RUN
```

Use a compatible installed CPU Torch/Torchvision pair with vision/tracking
dependencies. No API key, Modal, GPU or model download is needed. Existing output
folders and sources longer than five minutes are refused. Current output folder:
`runs/cpu-full-video-20261102`.

The runner preserves the original panorama and creates a constant 24 FPS,
video-only copy with source hashes and exact frame/PTS correspondence. It uses
six detector samples per second, CPU Torchvision and CPU ByteTrack, records raw
proposals and applies no selected-frame cap. Internal detector resize remains
min 1080/max 1920, matching the previous CPU baseline. This is not full-width
4096-pixel inference; source/overlay geometry remains 4096x1080. Runtime settings
alone do not establish better recall or tracking accuracy.

The inference child uses four CPU threads and a two-hour timeout. It receives
minimal system settings, without provider credentials, and never loads .env.
Progress is recorded every 30 seconds. Coverage rows can lag due to buffering;
zero rows during normalization/preflight are expected. Failures retain partial
outputs. A complete status requires inference, profiles and full overlay export.

## Outputs

| Path under the CPU run | Purpose |
| --- | --- |
| status.json, progress.jsonl | Current stage, timestamps, elapsed time and final counts |
| normalized/video24.mp4 | Full 24 FPS analysis copy |
| normalized/frame-map.jsonl, normalization.json | Original correspondence and hashes |
| run/manifest.json, observations.jsonl, raw_proposals.jsonl, coverage.jsonl | Inference/tracking evidence |
| profiles/player-analysis.json, review.md | Draft profiles and unknown/unsupported states |
| preview/overlay24.mp4 | Entire analyzed coverage rendered as video |
| preview/players.jpg, track-index.json, contact-sheet.jpg | Temporary-track visual mapping |
| preview/preview.json | Render provenance |
| analysis.stdout.log, analysis.stderr.log | Private local process logs |
| diagnostics/ | Safe normalization/profile journals |
| RESULTS.html | Viewer generated after successful completion |

Labels are temporary track IDs, not jersey numbers or confirmed players. Held
overlays are display holds, not new detections. Full rendering requires
`full_run=True, max_seconds=None`; bounded previews retain their 120-second limit.
No reviewed tracking reference, pitch calibration, coach rubric or context is
provided for this job. Accuracy, physical distance/speed and approved player
scores remain unverified.

## Colab

Use notebooks/02_video_analysis.ipynb (revision 2.3) in a standard CPU runtime.

1. Run code setup, runtime and input cells. Restrict Drive operations to Touchline.
2. Keep EXECUTION_MODE='cpu_analysis'. It loads Torch/Torchvision and checks CPU
   NMS; no CUDA/provider request runs.
3. Select the checkpoint and source. Explicitly normalize a 25 FPS original first.
4. Enable RUN_ANALYSIS=True for the initial 120 samples at 6 Hz. After inspecting
   that smoke run, set MAX_SELECTED=None and use a new run for the entire video.
5. Run profiles, visual review and the hash-verified Drive export. Notebook
   previews remain bounded; the local tool generates a full overlay.

For saved proposals choose cpu_replay, restore SAVED_RUN_DIR and the exact
VIDEO_OVERRIDE, then retrack. Replay needs no model and cannot recover people
missing from cached detections. The legacy cuda_analysis branch remains for
compatibility but is not selected by the current workflow.

## Training boundary

### Current separated recovery sequence

Use `tools/run_modal_training_recovery.py --approved-half-budget-recovery` for
the recorded recovery sequence: evaluation-dependency preflight with pinned
pycocotools 2.0.11, parallel CPU data restoration, seven-day immutable snapshot,
then isolated T4 default/minor training and full validation. Its dated $9 claim
is consumed. Do not rerun by deleting output or resetting a ledger.

Actual results: all 1710 images restored in 380.76 seconds; both T4 trials ran
1200 warm-start steps. Original full-validation AP50 was 32.39%; default recipe
41.27%; Baseten LR 0.0008 candidate 41.36%, all on the same 570 images. The tiny
difference between the two candidates is not reliable evidence from one run.
Final test and promotion remain untouched. The data snapshot is recorded in
runs/modal-training-recovery-20261102/recovery.json and expires after seven days;
durable Volume/Drive retention remains follow-up work.

The first-generation trial below completed steps but lost its output when
evaluation failed on a missing dependency. The separated controller downloads
each completed checkpoint on its saved event, before evaluation, and preserves
failure evidence. Read the current handover for usable candidate locations.

`tools/run_modal_candidate_analysis.py --approved-existing-cpu-cap` tested the
provisional candidate on 120 CPU detector samples with a 20-second overlay.
That separate dated claim is also consumed. Its partial analysis status is the
intentional sample cap; 1743 observations are not a measured accuracy result.

### Historical first-generation entry point

The latest owner permits cumulative provider ceilings of Modal $50, Meta $75,
Baseten $50 and Fireworks $50, including previous charges/reservations. Retain
all ledgers. One additional recovery/training trial reserves $5 with a T4,
4 CPU cores/8 GiB and a 3600-second hard Sandbox lifetime. This is a ceiling,
not a measured bill. No unbounded retry or automatic promotion follows.

`tools/run_modal_reviewed_training.py --approved-half-budget` was the first
single-use entry point for the inspected Models-Trained export. The worker verifies
the selected manifest/checkpoint, runs code checks remotely and tests CUDA NMS.
It restores missing images from the pinned SoccerNet archive using both original
JPEG and normalized PNG hashes. Any mismatch blocks training; do not rewrite the
manifest hashes to get past the gate. Then it warm-starts saved weights with a
fresh optimizer (not exact resume), at most 1200 steps/1800 training seconds.
It compares all validation images and leaves the final test sealed. The prior
candidate stays unchanged. Downloaded training-results.zip contains the restored
dataset/images; the controller extracts only small reports/weights. Preserve the
archive separately from the public code ZIP. Actual acceptance is receipt-based;
running this command is not evidence that every stage succeeded.

For ordinary CLI use, `train-detector ... --initial-checkpoint candidate.pt`
selects existing weights without an implicit download. `--allow-weight-download`
retains the original pretrained initialization route. Neither mode implements
optimizer/RNG/sampler resume. Keep each attempt under a fresh run ID.

candidate.pt contains weights; release.json is a dataset manifest. Training also
requires the referenced images, reviewed complete labels, capture-group splits
and frozen independent validation. Four SAM proposal frames are still unreviewed
and must not automatically become training data. No new training is started by
the CPU analysis job. GPU training needs separate bounded approval and verified
CUDA/NMS, checkpoint/resume and export/reload gates.

Earlier Modal analysis failures/logs and retained budget records remain in
[MODAL_ANALYSIS_RUNBOOK.md](MODAL_ANALYSIS_RUNBOOK.md).

## Tracking comparison and bounded fixer

`tools/run_modal_tracking_trial.py --approved-existing-cap` runs the registered
baseline and two review-only threshold/recovery recipes in Modal on saved
proposals. It keeps all frames/timestamps and never reruns a detector or trains.
The delivered comparison has 434 baseline tracks, 1190 lower-activation tracks
and 815 recovery-variant tracks. Lowering thresholds increased short fragments;
no candidate was promoted. All three have 20-second overlay/gallery exports.
Use labelled identities before claiming better tracking. Retracking cannot
recover a missed proposal or turn a temporary ID into a jersey number.

The CLI accepts explicit `retrack-run ... --activation-threshold 0.5
--max-gap-s 1.5`; existing defaults remain unchanged. Do not silently substitute
an experimental run for the reviewed baseline.

`tools/request_tracking_advice.py` sends aggregate metrics only to the pinned
Baseten model. Advice is untrusted data, never executable code. The first two
bounded attempts did not yield an accepted complete response; no advice was
applied. General source fixing, independent quality gates, exact checkpoint
resume and automatic experiment selection remain unfinished. Infrastructure
failure, incomplete advice or more temporary tracks must not trigger a model
rewrite. See [AUTOFIX_RULEBOOK.md](AUTOFIX_RULEBOOK.md).
