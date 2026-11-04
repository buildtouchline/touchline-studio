# Touchline scalability strategy

**Status: 5 November 2026.** Touchline currently runs as a local file-backed Studio with background processing and explicit provider boundaries. This document describes a **migration strategy**, not a deployed multi-tenant platform. Current behavior and evidence are in the [system architecture](TOUCHLINE_SYSTEM_ARCHITECTURE.md) and [README](../README.md).

## Scaling principle

Separate short coach interactions from expensive and failure-prone media processing. Preserve one traceable chain for every result: authorized source recording → exact normalization and frame map → selected model/code/checkpoint → validated output → coach review. Scale each stage only after its security, persistence and quality gates exist.

| Dimension | Current implementation | Next production boundary |
| --- | --- | --- |
| Application | One local HTTP Studio serving UI and APIs; loopback by default | Stateless, authenticated API instances behind a gateway, with shared metadata storage |
| Jobs | File-backed status and daemon background thread; uncertain in-flight status after restart | Durable queue, distinct job/attempt IDs, lease/recovery and bounded concurrency |
| GPU | Prepared bounded Modal Sandbox adapter, offline mocked execution | Verified deployed image/checkpoint and one approved live pilot before controlled autoscaling |
| AI calls | Server-side Meta activity with per-run local records and a bounded call allowance | Tenant-aware quotas, provider adapter policy, durable audit and model-version registry |
| Storage | Local run directories with manifests, hashes and child corrections | Authorized object storage plus transactional metadata, retention and revision controls |
| Tenancy | No production roster or tenant service | Authenticated organization/team/player roles with server-enforced isolation |

## 1. Horizontal application scaling

The browser already talks through an API client rather than directly to Modal or Meta. The present [`run_studio_server`](../src/touchline/studio.py) also owns local file paths and in-process work, so simply starting replicas would split their view of runs and jobs. A production transition would move metadata, sessions and policy decisions into shared managed storage, keep API handlers short, and have them enqueue work rather than own a long-lived thread. App instances could then scale with coach traffic independently of GPU workers. This **has not been implemented or load-tested**.

Keep identity and authorization at the API boundary: organization, team, player, recording and artifact reads must all resolve against an authenticated actor. Signed short-lived media links or an authenticated range proxy can permit playback without exposing worker filesystem paths. Optional local bearer tokens and injected callbacks in Studio are pilot controls, not production role-based access.

## 2. Asynchronous video jobs

The current [job record](../src/touchline/job_state.py) persists status on local disk and the [adapter](../src/touchline/adapters/panorama_worker.py) runs in a background thread. This keeps a normal upload response from waiting for the entire analysis, but the thread is not a durable distributed queue. A restarted process reports an unconfirmed in-flight attempt as unavailable. Capacity is effectively one local service's resources and configuration.

A durable job service should atomically admit an authorized upload, assign a stable job ID plus a distinct attempt ID, and retain an idempotency key. Workers lease bounded attempts; heartbeats and stage events are persisted. The API reports queued, processing, partial, completed, failed and cancelled states from durable records. Retry means a **new attempt** against the same immutable input and approved model revision, never an untracked replay. Prioritization can account for club SLA, video duration, available GPU and budget. Progress should report actual stage and verified coverage; do not invent percentages or ETAs. Success requires all required artifacts to be exported, hash-checked and durably stored before worker teardown.

## 3. Independent GPU capacity

The repository contains a bounded [Modal Sandbox dispatch path](../src/touchline/adapters/panorama_worker.py) for the [Panorama/ResNet worker](../tools/modal_panorama_worker.py). It verifies source and normalized input lineage, checkpoint, code bundle and worker-script hashes. Live mode requires an operator-approved image, checkpoint, deployment probe, stationary-camera attestation, authentication and spending reservation. The exact image and checkpoint are absent from this public repository, and **no fresh Studio-to-live-Modal run has been verified**.

Once a pilot validates that boundary, a scheduler could assign GPU jobs independently of Studio instances. It should select a GPU type and timeout from an approved resource profile, cap per-tenant concurrency, reserve a spending ceiling before dispatch, and reconcile actual usage after completion or failure. Multiple Sandboxes could process independent recordings; a single Sandbox must not be treated as a general multi-tenant shared process. Model or image revisions are deployed deliberately and each run records the exact code/checkpoint/image identity. Do not infer a cost or throughput benchmark from the current mock tests.

## 4. Modular model services

Panorama/ResNet detection and tracking, Meta selected-frame segmentation/reasoning, and Meta file transcription have different input, cost and validity rules. They already live behind separate server-side modules rather than browser credentials. A production model registry should record identifier, version, supported media type, request budget, parsing schema, calibration/quality evidence and rollout state. New model versions must not silently rewrite old observations. An optional provider can be replaced only through a reviewed adapter and revalidated outputs, while original evidence and prior model versions remain inspectable.

The four Meta workflows have genuine local smoke responses, with saved activity and coach-review UI, but not a reliability or availability study. SAM proposals are visual regions; Spark and Wonder generate evidence-constrained text; Voice transcribes a file. None verifies roster identity, physical metrics or longitudinal improvement. Future vision and voice modules should use the same provenance and approval rules. Provider flexibility is an architectural option, not a claim of currently interchangeable providers.

## 5. Storage, versioning and traceability

The local run directory is useful for reproducible review but cannot serve multiple application replicas or protect data against same-process modification. A staged migration would put original/normalized recordings, frame maps, annotated video and evidence in durable object storage; put jobs, attempts, sessions, review events and authorization metadata in a transactional store; and keep immutable content hashes and versioned manifests linking them. The backend, not a worker or browser, should mint authorized artifact retrieval. Retention, encryption, backups, deletion requests and regional data policy need explicit design before production use with youth footage.

Child corrections currently preserve parent evidence as separate local runs, but local hashes alone are not tamper-proof. A durable revision design should enforce append-only corrections with parent run/version, reviewer identity from authentication, exact evidence references and conflict handling. Cache or reuse a processed artifact only when source hash, normalization, frame map, model/code/checkpoint and requested analysis scope match; otherwise run a new attempt. This saves compute without hiding a changed model or recording.

## 6. Multi-club and academy expansion

The proposed hierarchy is organization → academy/club → team/cohort → athlete → session/run. Coaches and authorized staff need scoped roles; parents should see only explicit child-specific, coach-approved material. A trusted roster mapping must connect a temporary local track to a player; tracker labels alone are not identity. Longitudinal comparison requires matching player, objective, review state and session evidence, with explicit missing-data states.

Production authentication, tenant isolation, roster authorization, auditability and parent access **are not implemented or verified here**. Before inviting multiple clubs, test cross-tenant denial on every route and artifact, enforce review/revocation rules, and ensure worker inputs and logs cannot cross tenant boundaries. Do not present existing browser-local development drafts or generated model text as approved school or parent records.

## 7. Cost and operations

On-demand GPU execution, bounded video length, approved resource profiles and per-attempt budgets limit exposure. A queue can schedule work when capacity is available instead of tying up Studio request threads. Persist structured stages, timestamps, input/output hashes, failure class and spend reservation; alert on stuck attempts, export failures and budget overruns. Observe app latency separately from GPU duration and provider latency. Cache verified artifacts where the exact input/model lineage allows it. There is no measured platform-wide cost, throughput, availability or latency target in this repository.

## 8. Product expansion and delivery order

Roadmap capabilities include longitudinal player development, tactical analysis and minimaps, validated physical metrics, richer coach voice interaction, multi-academy collaboration and additional sports modules. They need their own evidence and access gates; the present demo must not display them as computed results. A practical order is:

1. Validate one approved fresh Panorama run end-to-end, including checkpoint/image identity, export and saved-run reopening.
2. Add durable upload/object storage, job/attempt records, idempotency and operator recovery with tests.
3. Add authenticated club/roster access and authorized artifact delivery, then child-run review transactions.
4. Pilot bounded concurrency and cost monitoring; measure actual behavior before claiming scale.
5. Expand development and parent reporting only after review, privacy and quality gates are verified.

See [review gates](REVIEW_GATES.md), [Panorama remediation](PANORAMA_STUDIO_REMEDIATION.md), [historical integration proposal](TOUCHLINE_PANORAMA_INTEGRATION.md), and [migration provenance](../MIGRATION_PROVENANCE.md). Proposed routes in historical documents are not automatically live production APIs.
