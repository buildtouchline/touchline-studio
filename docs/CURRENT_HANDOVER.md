# Touchline operational handover — 1 November 2026

Use feature/modal-training for code and tickets for authoritative scope/status. Existing PRs and main/develop were not changed. No CI or cloud deployment was added. Follow the repository's existing CODEOWNERS/review rules; shadow reviewers do not gain owner authority. No credentials are in the handover.

## Current result

The recorded-video baseline can load exported Torchvision weights, normalize landscape/panoramic footage to24FPS with correspondence metadata, persist observations, replay proposals using ByteTrack and display overlays, track labels, a player-crop gallery and optional focused video. CPU replay is the current default because the operator exhausted free Colab GPU quota. Physical metrics need reviewed calibration; track IDs are not jersey numbers or roster names. 75–80% IDF1 is a target, not an achieved result. Fine-tuning is deferred and ETL/training remain partial.

## Code map

- [Training notebook](../notebooks/01_dataset_to_model.ipynb): existing dataset-to-candidate baseline; unchanged by this delivery.
- [Analysis notebook2.2](../notebooks/02_video_analysis.ipynb): code/runtime setup, restore, normalization, GPU trial or CPU replay, visual review, optional reference scoring and final backup.
- [Input helper](../notebooks/analysis_inputs.py): bounded Touchline model discovery/hash validation for inference; bypassed for CPU replay.
- [Backup helper](../notebooks/analysis_backup.py): standard-library-only, embedded in final cell; closed output lists, source/hash verification and completion receipt.
- src/touchline/video/normalize.py and preview.py: explicit24FPS copy/provenance and saved-result visual rendering.
- src/touchline/byte_tracking.py, cuda_motion.py, tracking_replay.py and tracking_evaluation.py: pinned CPU ByteTrack, optional CUDA batched prediction, cached replay, reviewed sampled-frame IDF1.
- src/touchline/adapters/torchvision.py, pipeline.py and diagnostics.py: checkpoint loading/CUDA detector, persistence/run identity and safe telemetry.
- tools/check_cuda_tracking.py: operator-only synthetic CUDA parity test, not an accuracy benchmark. No real CUDA execution was performed by the agent.

## Start / continue now

1. Open the matching .ipynb plus current code ZIP in Colab. A Drive notebook copy does not update automatically from GitHub. Preserve /content files before changing runtimes. Do not use TPU; it is unsupported.
2. For saved results choose EXECUTION_MODE='cpu_replay' and a standard CPU runtime. Run setup/runtime/Drive-input cells. No model checkpoint or Torch/CUDA check is required for this path.
3. Step3a: set SAVED_RUN_DIR to the folder with manifest.json, coverage.jsonl and raw_proposals.jsonl. Set VIDEO_OVERRIDE to the exact analyzed24FPS video when its path changed. Existing-file checks now also verify its hash. If the video was lost, explicitly recreate from the original source and compare its hash; do not edit the saved source hash to bypass a mismatch.
4. Skip3b when that matching video exists. Skip4A new inference. In4R set RUN_RETRACK=True; CPU replay keeps the original cadence and partial status, gets a new run ID and leaves the parent untouched. Do not rerun restore afterward, since it selects the parent again.
5. Step4b: start FOCUS_TRACK_ID=None and SHOW_PROPOSALS=False for clarity. Choose an ID from the new gallery/index before focused playback. T8 is a track, not jersey8. Held boxes are labelled; no observations are invented during gaps. CPU replay cannot recover players never proposed by the detector.
6. Step4c is optional. Leave TRACKING_REFERENCE_PATH=None until independent labels exist; no accuracy score is inferred. Label all people/empty frames in the reviewed sampled scope, with stable reference IDs independent of predicted T-IDs. Original25FPS SoccerNet labels require the normalization frame map.
7. Generate profiles if desired. CALIBRATION_PATH/RUBRIC_PATH/SESSION_CONTEXT_PATH may remain None; corresponding physical/drill/score fields stay unavailable.
8. Run final cell7c. It copies current and available parent outputs, all known preview videos/images/JSON, profiles/evaluations/support snapshots, exact analyzed footage and result dictionaries into Touchline/complete-results. Wait for BACKUP COMPLETE and TRANSFER_COMPLETE.json. Enough Drive quota is required. Older separate exports7/7b remain available but are not prerequisites.
9. Later restore from the printed <backup>/current/run and <backup>/video/<filename>, not the backup root. The original dataset and model weights must be shared separately; the complete analysis backup deliberately excludes them, source code and credential files.

## GPU later / validation

Select cuda_analysis only when a free GPU is available. Preserve the working Torch/Torchvision pair, run CUDA NMS and tools/check_cuda_tracking.py, then use the bounded240-selected-frame/12-samples-per-second trial. The source remains24FPS. Detection and batched motion prediction use CUDA; matching, correction, decoding and rendering remain CPU-side. This is hybrid, not full-GPU ByteTrack, and is not guaranteed faster for one stream. Missing CUDA fails rather than silently falling back. Inspect detector_runtime/tracker_runtime and actual output before a full run. Larger panorama inference/tiling and appearance ReID remain measured follow-ups.

## Acceptance and transfer

339 guarded offline tests pass, including real CPU ByteTrack and complete-backup success/failure fixtures. 107 Python files and28 notebook cells parse. Tests use local synthetic files/mocked providers, deny real .env/credential reads and external networking; no real GPU/model/Drive job was executed by the agent. Operator-reported initial capped inference and matching-video recovery are evidence of progress, not full-match or quality acceptance. Demo progress is now explicitly recorded as Partially done for demo for19 parents/38 subtasks;28 parents/68 subtasks remain Planned. See [the pinned ticket reference](TICKET_REFERENCE.md) for delivered scope and remaining work.

Share this handover, the matching analysis .ipynb, code ZIP and ticket delivery snapshot. Grant the incoming team appropriate access to private video/run backups, weights and dataset releases separately; do not put them into Git or share credentials. Keep secrets operator-managed; the agent is forbidden to inspect actual .env/credential files. Existing PRs remain untouched. [Detailed Colab guide](COLAB_ANALYSIS_RUNBOOK.md) and [ByteTrack guide](BYTETRACK_RUNBOOK.md).

## Demo progress status — revision 2.3

At the user's request, `Partially done for demo` records a ticket with delivered baseline code, fixtures, preparation or scaffolding while the broader long-term task remains unfinished. This is not a percentage-complete estimate, full ticket closure, production readiness, measured accuracy or live-provider acceptance. The demo scope note on each changed record states what is available and what remains unverified. Fine-tuning and all remaining acceptance criteria stay open. This status does not satisfy dependencies that require accepted/completed work.

All 19 previously In progress parents and 38 previously In progress subtasks now use this demo progress status. The 28 Planned parents and 68 Planned subtasks remain Planned. No unstarted feature was marked delivered; no scope, dependency, acceptance criterion or evidence note was removed.
