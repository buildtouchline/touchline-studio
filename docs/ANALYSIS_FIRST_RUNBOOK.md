# Analysis-first setup and budget guard

## Controlling media revision — 1.7

Operator revision 1.7 supersedes exact 1920x1080 admission: accept consistent unrotated landscape dimensions, including 4096x1080 panoramas; analysis remains exact constant 24/1 FPS. Explicit CPU normalization creates a video-only analysis copy without resizing or changing playback speed, with source/output hashes and an exact zero-based source-frame/PTS correspondence. Original footage and audio remain untouched. CLI/notebook and offline conversion checks are implemented; actual long-video Colab conversion, GPU inference and player/tracking quality acceptance remain open. Free compute only; no browser integration, credential access, paid services or existing PR changes.

See [24 FPS conversion instructions](COLAB_ANALYSIS_RUNBOOK.md#panoramic-video-conversion).

## Run analysis in Colab — version 1.5

Use [COLAB_ANALYSIS_RUNBOOK.md](COLAB_ANALYSIS_RUNBOOK.md) and notebooks/02_video_analysis.ipynb to run the existing video/profile pipeline in Colab with exported weights. No local Python installation or Modal is required. The training notebook remains unchanged. Drive export verifies generated-file hashes; actual runtime/quality acceptance remains open. 297 guarded offline tests pass.

## Local-first implementation update — version 1.4

Local Python analysis and the existing Colab notebook remain primary. Modal is secondary and explicit; local analysis neither imports provider SDKs nor falls back to cloud. Additive player evidence profiles, deterministic provisional rubric drafts, narrow repair proposals/checks, training run IDs and local support exports are implemented and offline-tested. Real provider/GPU/Drive execution, coach correction acceptance and the 95% own-camera quality gate remain open. ETL/training optimization remains partial; no paid jobs or actual credential reads, and no existing PR changes.

[LOCAL_PRIMARY_RUNBOOK.md](LOCAL_PRIMARY_RUNBOOK.md) owns current operation. Narrow optional Baseten/Modal repair adapters exist, but general remote training/analysis services remain open; older blanket no-client statements below are historical.

31 October 2026. Current branch feature/modal-training. Reviewed backend fixes and this local guard implementation are published here; existing PRs are not updated/merged and main/develop do not receive these fixes automatically.

## What is ready versus pending

Existing: local Torchvision detector, recorded-video/tracking/metric primitives, review/report gates, local Studio/audio helpers and Colab ETL/training (still partial). New: CPU full-source 1080p/24 CFR preflight wired before detector construction in CLI analyze and Studio upload; durable local SpendingLedger; legacy budget helper additionally capped at the new ceilings. Remote Modal runner, hosted SAM adapter, Baseten repair service, complete scoring/review loop and dual-upload session workflow are still pending. Explicit operator-only loading is now available; no paid call has been made.

Work order: TL-025 spending controls + TL-031 configuration → TL-032 executor and TL-033 visual adapter → TL-039 supported automatic player measurements → TL-040 deterministic rubric drafts → TL-041 coach corrections → TL-042 grounded analysis/improvement plans → TL-043 actual acceptance. TL-030 missed-player diagnostics run alongside the automatic measurements. TL-044 uploads and TL-045 voice alignment can proceed against the frozen contracts. Dataset admission/retraining, native Android and autonomous repair remain follow-up; do not block the first analysis slice on new training.

## Input gate

Only unrotated landscape 1920x1080 sources with constant 24/1 FPS are accepted by the current analysis entry points. 23.976, 25, 30, 60, variable rates, missing timing, frame gaps and smaller/upscaled-at-upload conversions are not silently accepted. The check cannot prove whether a file was upscaled or converted before upload. Existing datasets and low-level research decoders are not converted or deleted.

Operator command, after installing the documented video dependencies:

```text
python -m touchline analysis-preflight /absolute/path/to/video.mp4
```

The full stream is CPU-decoded to verify PTS/geometry without invoking a model. Analysis currently decodes again; this adds preflight latency but avoids discovering an unsupported late segment after model allocation. Future caching must bind source hash/policy/version and preserve complete verification. CLI/Studio detector resize defaults are 1080/1920 to preserve source scale; measure VRAM/latency before a paid run. Explicit research resize overrides do not demonstrate the declared accuracy target.

Input FPS is distinct from expensive detector invocation rate. The existing sampler may inspect fewer frames, including 6th/18th anchors; evaluate missed short events and tracking gaps before claiming temporal accuracy. 24 FPS acceptance alone does not validate speed, identity or sprint coverage. Existing Studio is a local prototype; it is not the new durable upload service or a production server.

## Fifty-percent project ceilings

| Provider | Reported credit | Cumulative ceiling |
| --- | ---: | ---: |
| Modal | $100 | $50 |
| Meta SAM | $150 | $75 |
| Baseten | $100 | $50 |
| Fireworks AI | $100 | $50 |

Maximum nominal project spend is $225, not a requirement to use each pool. Prior attributable usage counts inside each ceiling; availability also cannot exceed verified eligible credit headroom. No automatic reset on billing-cycle rollover, restart, failed call or retry. The remaining half stays untouched. Unknown prior spend, credit eligibility or rates blocks execution. Modal includes CPU/RAM/GPU/build/storage/transfer, not just GPU seconds. Passive storage charges must also be reserved and reconciled. Fireworks stays disabled unless a specific model task is selected.

`touchline.spending.SpendingLedger` uses integer micro-USD, SQLite BEGIN IMMEDIATE, fixed ceilings, one-time operator-reviewed baseline, immutable request fingerprints and held pending charges. Reserve the complete worst case (all attempts, startup, maximum input/output/frames/runtime and ancillary costs) before dispatch. Return an existing job/result when created=false; it is not retry permission. Only final provider-confirmed billing releases unused reservation. Observed overrun persists and blocks further requests; manual audit is needed. This ledger is not a rate estimator or remote job executor and is not wired to nonexistent provider adapters.

Use one trusted controller with its local durable SQLite file. Do not place live multi-writer SQLite on a shared Modal Volume. For multiple controllers, implement a transactional authoritative store first. All paid calls and background charges must flow through the same reservation authority; direct API calls, notebooks and other teammates' jobs can bypass an application ledger. Never reset/delete the ledger to regain allowance. Scope and reconcile account usage outside this controller before starting it.

## Provider settings the operator configures privately

- Modal Usage & Billing: set the gross workspace usage budget to $50 for an otherwise unused dedicated billing cycle/workspace; set net out-of-pocket spend limit to $0 where supported. A net spend limit alone does not stop credit consumption. For shared/already-used accounts calculate the account ceiling from recorded non-project usage plus remaining project allowance; never raise an existing tighter cap. Monthly controls supplement the cumulative project ledger. [Modal budgets](https://modal.com/docs/guide/budgets).
- Baseten: set workspace monthly budget $50 and enable enforcement for Model APIs. Enforcement rejects Model API requests after the budget is reached; it does not govern Dedicated Inference or Training. Include existing workspace charges/other users when choosing the account limit. Verify exact model meta-models/muse-spark-open-1026 and price through the operator's catalog, without silent fallback. [Baseten budget documentation](https://docs.baseten.co/inference/model-apis/pricing-and-limits).
- Meta: Billing → Spend Limits can set an alert (suggest $60 warning within the $75 ceiling), but it is not a hard cap. Use bounded SAM requests and the application ledger to stop new dispatches. SAM video requests have a minimum billed-frame count; reserve that floor and every retry. [Meta alerts](https://dev.meta.ai/help/billing/spend-limit-alert), [SAM pricing](https://dev.meta.ai/docs/pricing-rate-limits).
- Fireworks: self-serve monthly usage limit $50 can pause API requests; Enterprise alerts do not suspend service. On the operator's own machine, the documented command is `firectl quota update monthly-spend-usd --value 50`. Disable Auto Reload if unwanted. Account settings and billing-cycle semantics need verification before use. [Fireworks limits](https://docs.fireworks.ai/guides/quotas_usage/account-quotas).

The agent has not changed any account settings, queried credentials/catalogs or run paid jobs. These steps are for the operator; no actual .env, secret file or private profile is accessed by the agent.

## Secret names to enter privately

In the selected checkout's root .env, if the operator chooses to load it explicitly:

```dotenv
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
BASETEN_API_KEY=
META_API_KEY=
```

The two Modal tokens are unnecessary when the operator already uses a working private Modal CLI profile. Optional later: FIREWORKS_API_KEY for a selected Fireworks task; HF_TOKEN only for approved local checkpoint downloads, not hosted SAM. Do not add Sarvam or a second Meta model automatically. This release loads .env only when the operator explicitly selects it with --env-file; the agent will not inspect it. Importing the modules above reads no environment variables or credentials.

In Modal, create touchline-baseten with BASETEN_API_KEY and touchline-meta-sam with META_API_KEY, attached only to trusted broker services. Keep Modal controller tokens out of workers; repaired code receives none of these. Existing [provider contract](MODAL_PROVIDER_CONTRACT.md) lists non-secret URLs/model names. Endpoint availability is verified publicly; account credit eligibility, exact dated Baseten access and remaining balance are not verified.

## Accuracy acceptance

95% is a target, not a current result. Proposed initial operational gate: player detection precision >=0.95 and recall >=0.95 at one frozen operating threshold and IoU>=0.5, on independently reviewed held-out footage from the intended 1080p/24 camera. Freeze confidence/NMS/preprocessing on validation before using final test. Count misses and false positives, include empty controls, and report numerator/denominator, confidence intervals and per-small/medium/large, near/far, occlusion/motion/lighting strata. Choose representative support and acceptance policy before inspecting final results; insufficient evidence is inconclusive.

Precision/recall do not establish 95% tracking/identity, speed, distance, event classification or coach-score correctness. Report IDF1/ID switches and coverage separately; validate physical outputs against independent references/calibration, and agree numeric tolerances before acceptance. Abstentions and unsampled intervals cannot disappear from coverage. SAM agreement and model confidence are not ground truth. Real performance acceptance remains open even when all offline tests pass. The 20-hour demonstration and a validated 95% gate are separate deliverables.

[Pinned tickets and authoritative acceptance spec](TICKET_REFERENCE.md).

## Manual verification — 31 October 2026

230 offline tests passed, including encoded 1080p/24 acceptance, 30 FPS rejection, late-frame timing failure, rejection before detector construction, concurrent budget reservations, restart/idempotency, pending charges, prior spend, four separate ceilings and persisted overrun blocking. Windows temporary-file sandbox restrictions required the guarded test runner outside the filesystem sandbox; its credential-file and external-network denial remained enabled. No real footage, model/provider/GPU accuracy or account configuration was verified.

## Diagnostics before keys

[Diagnostics runbook](DIAGNOSTICS_RUNBOOK.md) describes implemented local traces, safe failure categories, coverage and doctor/summary commands. 248 offline tests pass. Remote adapters, credential loading, provider settings and live telemetry remain pending; adding keys alone does not enable remote execution. The agent will not access actual .env/credentials.

## Reuse the existing export

[USING_COLAB_EXPORT.md](USING_COLAB_EXPORT.md) maps the dataset/weights files and provides the settings template plus export-check/analyze commands. TL-031.2 local configuration is implemented; remote compute/provider integration remains open.
