# Ticket reference for this implementation

Current revision 2.15: work stopped for owner handover. [Root handover](../HANDOVER.md) and [tracking quality report](TRACKING_QUALITY_REPORT.md) distinguish the 31.27% provisional identity baseline, rejected coverage-regressing trials and the unfinished flow-recovery quality replay.492 remote regression tests passed on the final source snapshot; this is not tracking acceptance. The CPU Sandbox terminated and partial results were exported.

[TL-030.7 tracking repair](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/subtasks/TL-030.7.md) is partially done for demo. Current pack:49 parents/117 subtasks; no new full acceptance. All previous holds remain through exceptional Modal ceiling$55.75; no further execution is authorized by this handover.

Earlier revision 2.14: selective component integration on `feature/player-mapping-modal-integration`, retaining Modal Sandboxes, Meta SAM and guarded Baseten repairs. 484 tests and code/notebook checks passed in the existing Modal Sandbox. Ticket checks passed 49 parents/116 subtasks, 1,589 local links, matching CSV/JSON and acyclic dependencies. [Component comparison](SELECTIVE_INTEGRATION.md).

[TL-038.3 selected integration](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/subtasks/TL-038.3.md) is partially done for demo. Live speech/headset, deployed endpoints, canonical ResNet coach ingestion and independent quality acceptance remain open. Both source branches remain intact.

Earlier revision 2.13 — 4 November 2026: reviewed resize training, ResNet video comparison/demo, SAM proposal receipts, guarded repair changes and reconnectable worker status are delivered as partial demo work. Publication checks passed in Modal: 461 tests, notebook/source checks and synchronized ticket links. A verified three-minute PXL demo was exported; the full-recording quality gate remains open.

The MobileNet candidate reached 84.85% AP50 on the unchanged 570-image SoccerNet validation split. This is separate from ResNet video output and does not measure tracking IDF1. The four-minute stationary panoramic ResNet comparison is planned in TL-030.6, within the owner's new $2 allowance and $55 retained ceiling. Canonical ResNet coach integration, durable frontend job API/storage, independent accuracy acceptance and the always-on fixer remain open. Status counts: 25/49 parents and 51/115 subtasks are partially done for demo; the remainder are planned.

The sections below preserve dated implementation history; the latest handover and ticket pack control current status.

Earlier revision 2.10: Modal CPU video analysis, CPU data snapshot, two reviewed T4 trials and a real Baseten minor proposal are delivered. Failure-event routing is implemented; global listener/production acceptance remains open. All current checks run remotely. Existing PRs/main/develop stay unchanged. Earlier dated scope sections below are historical.

Authoritative branch: [`tickets`](https://github.com/buildtouchline/touchline/tree/tickets).

Pinned revision: `761ffd48e5d46419d3fa94a3a211847c74d875d2`.

- [Backlog](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/BACKLOG.md)
- [TL-029 Colab dataset-to-model](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/TL-029.md)
- [TL-023 Dataset release](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/TL-023.md)
- [TL-024 Training/evaluation](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/TL-024.md)
- [TL-030 Missed-player pipeline optimization](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/TL-030.md): P0; delivered CPU analysis, reviewed GPU trials and bounded tracking comparisons; own-camera identity quality and higher-resolution/tiling work remain open.
- [TL-028 Android recorder](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/TL-028.md): specified only; no mobile code in this branch.
- [Plan](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/PLAN.md) · [Source audit](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/SOURCE_AUDIT.md) · [Implementation map](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/IMPLEMENTATION_MAP.md)

Progress updated 28 October 2026: TL-023, TL-024 and TL-029 remain In progress with CPU-tested ETL recovery/scale diagnostics recorded on tickets; target-runtime verification, GPU training-resume, full-validation selection and independent dataset review remain open. TL-030 is In progress with scale diagnostics in TL-030.2, while TL-030.1 and TL-030.3 remain unassigned. Repository code ownership updated to @jv-013. When ticket requirements change, commit them on `tickets` first, update the exact revision/links here in the associated code change, and record affected ticket IDs in the code commit. Do not copy the full ticket/spec pack into this branch.


## Current backend and provider scope — 31 October 2026

Implementation branch: feature/modal-training, based on PR #13 including #11/#12 and excluding later frontend #14–#16. Existing PR branches are not updated or merged. Backend gates have 212 passing offline tests; this does not establish real model/player accuracy or service authorization.

[Terminal provider plan](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/training-platform/HLD.md) and [today’s player scoring delivery](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/PLAYER_SCORING_TODAY.md). P0 scoring tickets are TL-039 automatic measurements/profiles, TL-040 rubric drafts, TL-041 coach review/corrections, TL-042 analysis/improvement plans and TL-043 real end-to-end acceptance. They remain Planned, targeting the conditional twenty-hour demo by 1 November, with actual data/rubric/coach gates.

Read [MODAL_PROVIDER_CONTRACT.md](MODAL_PROVIDER_CONTRACT.md), [AUTOFIX_RULEBOOK.md](AUTOFIX_RULEBOOK.md) and [REVIEW_GATES.md](REVIEW_GATES.md). Provider integration and scoring runtime are not implemented by these documents. Agent access to the real .env and credential files/values is forbidden.

## Current dual-upload and twenty-hour scope — version 1.0

Authoritative planning update: TL-044 private independent video/audio uploads, TL-045 timestamp/anchor/manual alignment and voice evidence, TL-046 staged reviewed training/repair (follow-up, outside the deadline). All new scope is Planned. Existing TL-039–TL-043 automatic drafts → coach correction → recompute/report order remains.

- [Twenty-hour three-stream schedule](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/DEMO_20_HOUR_PLAN.md)
- [Complete HLD](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/training-platform/HLD.md)
- [Dual-upload/time/voice specification](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S20-dual-upload-voice-alignment.md)
- [Current ticket index](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/BACKLOG.md)
- [Official provider audit](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/training-platform/PROVIDER_AUDIT.md)

Meta-hosted sam-3.1 is now the primary proposed Meta-credit route. Modal CPU is proposed web hosting; Fireworks AI is optional model inference, not documented general frontend hosting. Four reported pools total $450 nominal; balances/access unverified and execution disabled. No new runtime or real model quality is claimed. Read [the operational dual-upload guide](DUAL_UPLOAD_DEMO.md). Existing PRs unchanged; actual .env/credentials remain inaccessible to the agent.

## Analysis-first version 1.1

[TL-S21: analysis order, media/quality gate, provider budgets and ENV names](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S21-analysis-first-quality-and-spending.md). Local 1080p/24 CFR preflight and durable fifty-percent budget guards implemented under TL-002/TL-025 with 230 offline tests passing; real provider/dashboard and 95% own-camera acceptance remain open. Player analysis precedes retraining. [Operator runbook](ANALYSIS_FIRST_RUNBOOK.md). No actual .env/credentials or paid provider calls accessed. Existing PRs unchanged.

## Diagnostic prerequisite — version 1.2

[TL-047/TL-S22: diagnostics before credentials](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S22-diagnostics-and-readiness.md). Shared local operation traces, safe error codes/artifacts, dependency doctor and bounded summary commands implemented; 248 offline tests pass. Actual remote job/provider telemetry, retention and owner/target-runtime acceptance remain open. No actual credential/profile/.env access or paid job. Existing PRs unchanged. Read [DIAGNOSTICS_RUNBOOK.md](DIAGNOSTICS_RUNBOOK.md).

## Existing export and configuration — version 1.3

[TL-S23: dataset/weights output and operator settings](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S23-export-artifacts-and-operator-config.md). TL-031/TL-031.2 are In progress with explicit owner-only loading, selected-checkpoint CLI/Studio use and export-check; 262 offline tests pass. release.json is the dataset manifest; candidate-train/candidate.pt contains trained weights. [USING_COLAB_EXPORT.md](USING_COLAB_EXPORT.md) and [settings template](../examples/operator-settings.template) explain the actual notebook output. Actual private artifacts, runtime/quality and remote provider integration remain unverified; the agent never accessed real .env/credentials. Existing PRs unchanged.

## Local-primary revision 1.4

[TL-S24 — local profiles and optional repair](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S24-local-primary-analysis-and-repair.md) · [TL-S25 — run IDs and storage handoffs](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S25-run-identity-and-storage-handoff.md).

293 guarded offline tests pass. Existing local/Colab remains primary; Modal is explicitly optional. Player profiles/rubric drafts, bounded repair proposals/checks, training run IDs and local support exports are partial implementation. Real provider/GPU/Drive execution, coach corrections and independently measured accuracy remain open. [Operational runbook](LOCAL_PRIMARY_RUNBOOK.md). No actual credentials, paid jobs or existing PR changes.

## Colab analysis revision 1.5

[TL-S26 — Colab video/player analysis](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S26-colab-video-analysis.md). Colab is primary for training and analysis, with the existing local CLI supported and Modal optional. Separate 02_video_analysis notebook reuses exported weights and the backend, preserves run IDs and verifies generated-file Drive exports. 297 offline tests pass; actual Colab/GPU/checkpoint/Drive and quality gates remain open. [Runbook](COLAB_ANALYSIS_RUNBOOK.md). Training notebook unchanged; no actual credentials, paid calls or existing PR changes.

## Checkpoint/tracking revision 1.6

[TL-S27 — uploaded checkpoint and free-Colab tracking benchmark](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S27-uploaded-checkpoint-and-tracking-benchmark.md). Scoped checkpoint discovery/hash verification, fixed-six-FPS comparison and raw-proposal diagnostics are implemented with 301 guarded offline tests. The uploaded artifact hash/recipe/result were inspected in Colab; inference, ByteTrack integration and the proposed IDF1 0.85–0.90 target remain unverified. Free compute only, Drive access restricted to Touchline, no .env/Secrets access. Browser integration stopped on operator request. Existing PRs unchanged.

## Controlling media revision 1.7

[TL-S28 — panoramic geometry and mandatory 24 FPS analysis copies](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S28-panorama-and-24fps-normalization.md). This supersedes earlier exact-1080p/no-conversion requirements. TL-002 now supports source landscape dimensions and an explicit video-only 24 FPS copy with exact source-index/PTS correspondence. 307 guarded offline tests pass; actual long-video Colab conversion, model inference and quality acceptance remain open. No browser integration, credentials, paid compute or existing PR changes.

## Visual-review and GPU evidence revision 1.8

[TL-S29 — visual review and verifiable GPU inference](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S29-visual-review-and-gpu-inference.md). Saved-result 24 FPS overlay video/contact sheet, proposal diagnostics, preview Drive export and actual detector/input placement telemetry are implemented. 314 guarded offline tests pass; real free-Colab GPU telemetry and all player-quality gates remain open. CUDA is required by the notebook for inference; CPU preparation/rendering remains expected. Existing PRs and credential restrictions unchanged.

## Readable player mapping revision 1.9

[TL-S29](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S29-visual-review-and-gpu-inference.md). Readability revision 1.9 adds resolution-scaled black-backed T-ID badges, deterministic colors and leader lines, a first-observed player-crop gallery and track index, and an optional 960x540 24 FPS focused-track video. Proposal clutter is hidden by default. Focused view never switches identity to fill missing observations; held overlays remain explicit. 316 guarded offline tests pass, with a synthetic panorama readability check. This changes display only; jersey recognition, roster identity, actual Colab/footage acceptance and tracking quality remain unverified. No new inference, GPU/provider job, credential access or existing PR change.

## Stationary-panorama tracking revision 2.0

[TL-S30](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S30-stationary-panorama-byte-tracking.md). Optional CPU ByteTrack, saved-proposal replay, reviewed sampled-frame IDF1 and gated CUDA batched prediction are implemented with 333 guarded offline tests. CUDA detector plus CUDA motion prediction is HYBRID: assignment/correction remain CPU. Actual CUDA parity, speed and 75% minimum/80% desired IDF1 on genuine footage remain unverified. TL-030.3 is In progress, not complete. See [the runbook](BYTETRACK_RUNBOOK.md). No model/GPU/provider job, real credentials, browser integration or existing PR changes.

## Quota-aware notebook revision 2.1

[TL-S30](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/specs/TL-S30-stationary-panorama-byte-tracking.md). Notebook defaults to explicit CPU replay with no Torch/CUDA/checkpoint admission, adds saved-run restoration, IDF1 scoring/export and optional verified video preservation. GPU analysis remains explicit; TPU unsupported. Version banner and minimum-code check expose outdated notebook/code copies. 336 guarded offline tests pass; actual Colab/Drive/CUDA and quality acceptance remain open. No browser integration, credentials, paid jobs or existing PR changes.

## Delivery and handover revision 2.2 — 1 November 2026

[Delivery/acceptance snapshot](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/DELIVERY_STATUS_2026-11-01.md). Baseline code delivered is distinct from ticket closure:0 closed,19 In progress/28 Planned parents;38 In progress/68 Planned subtasks. Corrected stale Planned statuses for TL-001.1/TL-004.1/TL-039.3 without changing acceptance criteria. Notebook2.2 adds final current/parent/video/JSON/image backup with SHA256 receipt, explicit run order and video recovery hash checks. 339 guarded offline tests pass. ETL/training remain partial; fine-tuning deferred. Actual quality, CUDA prediction and complete Drive restore acceptance remain open. [Operational handover](CURRENT_HANDOVER.md). Existing PRs, credentials and compute restrictions unchanged.

## Demo progress classification — revision2.3

[Current demo status](https://github.com/buildtouchline/touchline/blob/761ffd48e5d46419d3fa94a3a211847c74d875d2/docs/tickets/DELIVERY_STATUS_2026-11-01.md). At the user request,19 parents and38 subtasks are Partially done for demo;28 parents and68 subtasks remain Planned. This credits delivered baseline/preparation/scaffolding while preserving every unfinished acceptance criterion, dependency, evidence note and fine-tuning follow-up. Live providers, CUDA validation and quality targets are not thereby accepted. Notebook remains revision2.2; this change is ticket/documentation only.


## Provider/audio revision 2.4 - 1 November 2026

Code continues on feature/modal-provider-pipeline. Runtime-only key loading is explicitly authorized for a $1 total smoke test; keys/headers/raw bodies/errors remain private. Meta catalog and Modal authentication passed; no Modal jobs launched. After the owner update, Baseten catalog and exact-model Hello passed (147 input / 23 output tokens), with the original test hold retained. SARVAM_API_KEY is recognized but was absent. Sarvam-generated speech to Meta transcription is the new voice route, separate from coach evidence. SAM proposals still require review and separate detector training/evaluation. 355 offline tests pass; full voice/video/training and quality acceptance remain open.

## Current video trial and detector service - revision 2.5

TL-032/033 now record actual CPU sandbox/SAM evidence and rejected T4/L4 creation; no GPU training or quality acceptance. TL-036/046 record deterministic minor/major policy, with the general automatic fine-tuning executor still pending. TL-048/TL-S31 cover the prepared authenticated custom detector API, not a deployed service. Read VIDEO_TRIAL_RUNBOOK.md. 373 guarded offline tests pass.

Latest main/T4 plain-image probe: environment/image preparation passed; GPU creation rejected with billing-related wording, no running worker confirmed. Three new offline probe tests passed. Private event/receipt logs are documented in VIDEO_TRIAL_RUNBOOK.md. Exact billing restriction remains unresolved.

Separate ephemeral T4 Function setup in main was also rejected with GPU/billing-related wording before invocation. Temporary App stop confirmed. Safe Function event/receipt logs retained; three new offline tests passed. Account/billing investigation remains open.

Current post-payment result: separate T4 Function in main passed nvidia-smi (15360 MiB, driver 610.57.04); explicit temporary App stop confirmed. Four Function-probe offline tests passed. Earlier billing rejection is historical; credit ledger, Torchvision/video/training/deployment acceptance remain open.

## Current analysis runner and logs - 2 November 2026

TL-032.2 has frozen-input Function dispatch and durable one-dollar/explicit-retry gates. TL-032.3 has actual cancel/App-stop evidence but no checkpoint recovery. TL-047 includes live/final sanitized provider-log fetch. Two submissions returned InternalFailure without worker output; GPU inference remains unverified. TL-046 correctly routes this to provider reconciliation without code edits. 388 guarded offline tests pass. See MODAL_ANALYSIS_RUNBOOK.md.
