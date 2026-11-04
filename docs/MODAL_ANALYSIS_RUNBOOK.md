# Modal Function analysis and logs

Updated 2 November 2026. Current branch: feature/modal-provider-pipeline.
This is an explicit optional backend; existing Colab/local analysis is unchanged.

Current allocation supersedes GPU analysis: use the Modal CPU Sandbox described
in [CPU_ANALYSIS_RUNBOOK.md](CPU_ANALYSIS_RUNBOOK.md). GPU compute is reserved for
reviewed training. This document preserves the earlier GPU failure/log evidence
and does not authorize further GPU analysis attempts.

## Current result

The earlier post-payment plain-image T4 Function passed nvidia-smi. The two
subsequent 20-second analysis submissions returned Modal `InternalFailure`,
without a worker receipt or CUDA-stage marker. Both temporary Apps were explicitly
stopped. Provider log retrieval succeeded for both. The retry's best-effort call
graph reports FAILURE without an assigned task; this is supporting evidence,
not a guarantee of real-time lifecycle state or a zero-charge assertion.

No GPU detector analysis, CUDA/NMS/parity result, new training or endpoint
deployment is accepted from these attempts. The original local CPU results and
four unreviewed hosted SAM proposal images remain available.

| Attempt | Private folder under runs/ | App |
| --- | --- | --- |
| Initial analysis | modal-analysis-20261102 | ap-XxcOxydTNeA77GDsPOHNjd |
| Explicit logged retry | modal-analysis-20261102-retry | ap-ZIimZG5i38nc3ASjWFKZQg |

See each folder's modal-function.json, modal-function.events.jsonl,
provider-log-summary.json and provider-events.jsonl. The retry also has a live
snapshot in provider-log-summary-live.json. The initial stopped-call inspection
is in failure-inspection.json. No raw credential, HTTP response/error or provider
log text is retained; fixed stage events and diagnostic counts are retained.

## Implemented contract

- `src/touchline/function_jobs.py`: exact five-file allowlist, byte/hash limits,
  path checks and durable single-use SQLite reservation. Inputs cannot be
  redirected to an arbitrary local file through a modified job specification.
- `tools/run_modal_analysis_function.py`: separate offline preparation and
  explicitly authorized execution; records hashes, App/call IDs, resource bounds,
  cancellation and stop state. An unknown worker/stop state blocks acceptance.
- `tools/modal_analysis_function.py`: one ephemeral T4 Function, two CPU cores,
  8 GiB RAM, 540-second execution and 45-second startup limits, one container,
  no warm minimum, no automatic retries, no secrets/mounts/egress or checkout
  upload. Function result wait is bounded to 600 seconds. Scheduling and transfer
  are distinct from the execution timeout; these are not provider dollar caps.
- `tools/modal_video_worker.py`: verifies frozen inputs, then CUDA availability,
  compiled NMS and synthetic tracker parity before invoking existing analysis.
  The selected excerpt is 480 frames at 24 FPS; detector sampling is 6 Hz,
  resize min 1080/max 4096, with hybrid CUDA prediction and CPU association.
- The worker would export manifest, proposals, coverage, GPU receipt and safe
  diagnostic journals in a <=64 MiB ZIP. The controller verifies checkpoint,
  frame counts, CUDA tensor/device telemetry and hashes before accepting output.
  This output path is implemented but was not exercised successfully remotely.

The frozen source bundle used in each attempt is preserved separately from later
controller/logging fixes. Use its job-specification.json hashes to reproduce that
attempt; do not label the entire latest checkout as identical to the submitted
snapshot. No uploads become reviewed labels or training data automatically.

## Logs and read-only inspection

Log fetching uses Modal's supported `modal app logs` CLI against an existing App
ID in main. A trusted child runtime receives only explicitly selected Modal
credentials and minimal system settings, with an absent private-profile path.
Output is bounded to 1 MiB and retrieval to 45 seconds. Raw text is hashed and
discarded. Fixed TOUCHLINE_EVENT stages, static error categories and known missing
dependency names are saved. Nonempty stdout alone is not treated as inference
evidence. Function call graphs are best-effort and may lag.

Owner-only read-only retrieval, after installing provider dependencies:

```shell
python -B tools/fetch_modal_analysis_logs.py --work runs/modal-analysis-20261102-retry
python -B tools/fetch_modal_analysis_logs.py --work runs/modal-analysis-20261102-retry --live-snapshot
```

These commands load the selected root .env privately at runtime. Never paste keys,
run an environment dump, read actual credentials as an agent, or place provider
keys in generated-code workers. The separate local receipt/event logs remain
available even when the platform does not start the worker.

## Spending and retry boundary

The owner approved $1 additional Modal spend for this analysis trial and then
explicitly allowed a logged retry. Both attempts share the same retained dollar
reservation in runs/modal-analysis-20261102-allowance.sqlite3. Earlier $1 smoke
and $3 video-trial reservations are preserved. No new Meta/Baseten request ran.
Actual charges/credit balance remain unreconciled.

One original claim and one separately recorded retry are permitted by this
implementation. The retry requires a confirmed stopped parent, a recorded
InternalFailure and successful log fetch. A third claim fails. Do not delete the
databases/claims, change authorization IDs, create another output folder to bypass
them or silently upgrade the GPU. Resource/time/transfer bounds were selected
within the approved total; local accounting does not configure Modal budgets.

The preparation/execution entry points are documented for a future separately
approved run, not instructions to consume another authorization now:

```shell
python -B tools/run_modal_analysis_function.py prepare --work runs/NEW_APPROVED_RUN
python -B tools/run_modal_analysis_function.py execute --work runs/NEW_APPROVED_RUN --approved-one-dollar
```

The current authorization is consumed and the existing ledger will reject another
normal dispatch. A new authorization needs explicit operator accounting and a
reviewed controller change; it is not granted by the flag above.

## Next actions and repair behavior

1. Share the private SUPPORT.md and sanitized App/call/log receipts with Modal
   support. Investigate image/scheduling/input initialization using backend logs;
   no exact root cause is established by the current metadata.
2. Reconcile actual charges and the credit ledger. Agree on one new bounded
   diagnostic only after reviewing that evidence. Do not claim the detector is
   faulty because the platform returned InternalFailure.
3. Once the worker runs, require CUDA/NMS/parity and actual detector telemetry,
   then render and visually review the exported observations. Measure tracking
   against genuine reference IDs before reporting accuracy.
4. Review/correct SAM labels before building a training release. Training,
   checkpoint resume and deployment remain separate gated tickets.

InternalFailure maps to provider reconciliation, not automatic code edits.
Unknown/oversized worker output and real code failures have distinct review
paths. Baseten advice/patch execution is not triggered or paid for by this
Modal-only authorization. Read [AUTOFIX_RULEBOOK.md](AUTOFIX_RULEBOOK.md).

Sources: [Modal app logs](https://modal.com/docs/cli/latest/app),
[Function timeout semantics](https://modal.com/docs/guide/timeouts),
[current unit pricing](https://modal.com/pricing).
