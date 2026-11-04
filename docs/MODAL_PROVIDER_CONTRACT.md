# Modal, Baseten, Meta SAM and optional Fireworks contract

## Current operation — 2 November 2026

The owner authorized selected runtime-only private key loading and cumulative
ceilings Modal $50, Meta $75, Baseten $50 and Fireworks $50. The agent must not
inspect/print credential values. All model/tests run in Modal; workers receive
no provider keys. CPU video analysis, CPU reviewed-data recovery and two T4
training trials completed. Baseten supplied the typed minor recipe; SAM's four
proposal frames remain unreviewed. Endpoint deployment, global repair listener,
production billing/recovery and own-camera identity quality remain open.
Read [HANDOVER.md](HANDOVER.md) for actual evidence and selected routes. Earlier
dated local-first/no-execution scope below is historical where superseded.

Current provider-fix branch: feature/modal-provider-pipeline, based on
feature/modal-training. See [repair setup and recovery](PROVIDER_REPAIR_RUNBOOK.md)
for private configuration placement, immutable image IDs, bounded output and
operator-only reconciliation. No actual credential access or live execution.

## Local-first implementation update — version 1.4

Local Python analysis and the existing Colab notebook remain primary. Modal is secondary and explicit; local analysis neither imports provider SDKs nor falls back to cloud. Additive player evidence profiles, deterministic provisional rubric drafts, narrow repair proposals/checks, training run IDs and local support exports are implemented and offline-tested. Real provider/GPU/Drive execution, coach correction acceptance and the 95% own-camera quality gate remain open. ETL/training optimization remains partial; no paid jobs or actual credential reads, and no existing PR changes.

[LOCAL_PRIMARY_RUNBOOK.md](LOCAL_PRIMARY_RUNBOOK.md) owns current operation. Narrow optional Baseten/Modal repair adapters exist, but general remote training/analysis services remain open; older blanket no-client statements below are historical.

Version 1.0 — 31 October 2026. Explicit operator-only settings loading is implemented; remote clients/controller remain pending. Active code branch feature/modal-training is in touchline-integration-fixes-20261031 beside pyproject.toml. Keep the older Colab checkout separate.

The agent may never open, read, source, parse, copy, upload or print actual .env/credential files or values, including through helpers, tests or provider CLIs. Do not provide keys in chat. Only the operator configures authentication and named Modal Secrets privately. An optional local .env belongs at the selected checkout root or explicitly configured private path; presence does not load it automatically. Never serve a checkout or credential directory.

| Operator-owned location | Secret variable | Trusted consumer |
| --- | --- | --- |
| Private local Modal authentication | MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or CLI profile | Operator terminal/controller only |
| Modal Secret touchline-baseten | BASETEN_API_KEY | Adviser client only |
| Modal Secret touchline-meta-sam | META_API_KEY | Hosted SAM client only |
| Optional Modal Secret touchline-sam-download | HF_TOKEN | Approved local-checkpoint downloader only |
| Optional Modal Secret touchline-fireworks | FIREWORKS_API_KEY | Approved Fireworks backend client only; not required for the core demo |

No secret is attached to generated-code repair Sandboxes or sent to the browser. A worker needing a hosted call asks a trusted broker; the whole environment is never injected into repaired code. Meta examples call their variable MODEL_API_KEY; our adapter would explicitly pass META_API_KEY to the SDK. Secret variable names are our application convention, not provider-mandated magic.

```text
META_BASE_URL=https://api.meta.ai/v1
META_SAM_MODEL=sam-3.1
TOUCHLINE_META_SAM_SECRET_NAME=touchline-meta-sam
BASETEN_BASE_URL=https://inference.baseten.co/v1
BASETEN_MODEL=meta-models/muse-spark-open-1026
TOUCHLINE_BASETEN_SECRET_NAME=touchline-baseten
TOUCHLINE_MODAL_APP_NAME=touchline-training
TOUCHLINE_TRAINING_ENABLED=false
TOUCHLINE_AUTOFIX_ENABLED=false
```

Meta hosted SAM uses POST /responses, not /chat/completions. Baseten uses POST /chat/completions; verify the exact dated model before runtime, without silent fallback. Optional local checkpoint configuration remains SAM_MODEL_ID=facebook/sam3.1 plus pinned SAM_MODEL_REVISION/SAM_CODE_REVISION and TOUCHLINE_SAM_SECRET_NAME=touchline-sam-download. HF_TOKEN is unnecessary for hosted SAM requests. Fireworks model/base/timestamp contract remains a separate decision; no dedicated transcription endpoint is invented.

Leave spending/concurrency/trial limits disabled until a named authority records provider-specific allocations. User reports $100 Modal, $150 Meta, $100 Baseten and $100 Fireworks; these are four separate, unverified remaining balances. CPU/RAM/Volume/storage/transfer and retries count against Modal too. Prefer a measured compatible L4/T4 configuration; no automatic expensive GPU upgrade. Access, terms, budget and runtime validation remain open.

[Meta SAM API](https://dev.meta.ai/docs/sam/overview), [Baseten API](https://docs.baseten.co/inference/model-apis/overview), [Modal Secrets](https://modal.com/docs/guide/secrets), [Modal HTTP](https://modal.com/docs/guide/webhooks), [Fireworks scope](https://docs.fireworks.ai/getting-started/introduction).

## Code status

Remote interfaces remain planned. Local settings/export diagnostics have 262 passing offline tests; this does not accept remote services. No PR is updated/merged; work stays on feature/modal-training based on PR #13 with #11/#12 and without PR #14–#16. Read [the rulebook](AUTOFIX_RULEBOOK.md) and [pinned tickets](TICKET_REFERENCE.md).

## Analysis-first update

Read [ANALYSIS_FIRST_RUNBOOK.md](ANALYSIS_FIRST_RUNBOOK.md) for strict 1080p24 preflight, proposed 95% acceptance, fixed $50/$75/$50/$50USD project ceilings, provider dashboard steps and operator-only ENV names. Local guards are implemented; paid provider wiring and player-scoring runtime remain open. Earlier training-first/zero-allocation-only scheduling is superseded; no unverified job is enabled.

## Diagnostics before keys

[Diagnostics runbook](DIAGNOSTICS_RUNBOOK.md) describes implemented local traces, safe failure categories, coverage and doctor/summary commands. 248 offline tests pass. Remote adapters, provider settings and live telemetry remain pending; credential loading is explicit and operator-controlled. The agent will not access actual .env/credentials.

## Dataset and checkpoint settings

[USING_COLAB_EXPORT.md](USING_COLAB_EXPORT.md) describes the actual notebook output format. Configure TOUCHLINE_CHECKPOINT_PATH for candidate-train/candidate.pt and TOUCHLINE_DATASET_MANIFEST for the separate dataset-releases/RUN_ID/release.json; TOUCHLINE_EXPORT_DIR selects the model-runs/RUN_ID integrity bundle. Do not use release.json as trained weights. The agent never reads the actual .env, including through the new loader.
