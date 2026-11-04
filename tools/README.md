# Operator tools

Read [the current handover](../docs/HANDOVER.md) before running provider tools.
Imports must not dispatch jobs. Provider credentials are configured privately;
the trusted controller may load the selected root.env at explicit execution,
never print keys or pass them to analysis workers.

## Reusable components

- `run_handover_job.py`: existing bounded baseline takeover launcher.
- `fetch_rcnn_comparison_weights.py`: public official-weight download and hash
  pinning; no model execution.
- `fetch_soccertrack_intake.py`: allowlisted pinned source intake, not training
  admission. Human label completeness/group review remains required.
- `check_ticket_pack.py`: static ticket/CSV/dependency/link validation, no cloud.
- `run_offline_checks.py` and `check_artifacts.py`: manual checks, executed in
  Modal for the current workflow; no CI was requested.

## Consumed experiments and recovery controllers

The dated recording, resize, audit, R-CNN preview/full/demo and adviser scripts
freeze selected artifacts and single-use claims. They preserve how the existing
results were produced. They are not generic frontend endpoints or reusable
permissions to spend. Existing run directories/claims deliberately block replay.
Do not rename a directory or reset a ledger to bypass that guard.

`modal_rcnn_gpu_worker.py` is the experimental R-CNN worker. `worker_status`
snapshots support future reconnectable controllers; the experimental ResNet
observation format is still separate from canonical coach-review input.
`render_cached_resnet_demo.py` produced the verified three-minute clip from
preserved real GPU observations. It did not run new inference or training.

`run_modal_recording.py` and `modal_recording_worker.py` preserve the earlier
MobileNet workflow. `adopt_modal_recording.py`/`adopt_rcnn_gpu_amp.py` preserve
specific same-worker recovery attempts. Read their final receipts before any
handover. A failed long run is not completed by a successful short demo.

The four-minute ResNet panorama experiment is TL-030.6. Its new up-to$2 owner
authorization and fresh lineage must be recorded separately after current
publication; older PXL claims cannot authorize it.

## Stationary identity repair (handover stopped)

`run_modal_stationary_replay.py` snapshots this source and uses an explicit
private operator checkout for existing videos, proposals and runtime-only keys.
It runs CPU-only; `modal_stationary_replay_worker.py` does not load a detector.
`provisional_panorama_metrics.py` is development-only upstream-reference scoring,
separate from the genuine reviewed-reference evaluator. Dated claims are consumed;
do not blindly rerun or delete their budget databases. Read [the handover](../HANDOVER.md).
