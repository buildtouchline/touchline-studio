# Diagnostics before provider setup

31 October 2026 | feature/modal-training | TL-047

## Run the checks

From this checkout root, after installing the documented local dependencies:

```text
python -m touchline doctor
python -m touchline analysis-preflight /absolute/path/to/video.mp4
python -m touchline diagnostic-report runs/diagnostics/REPLACE_WITH_TRACE_ID.jsonl
```

Replace REPLACE_WITH_TRACE_ID with the 32-character trace ID printed in an error or doctor output. Each boundary creates a trace when called directly; nested calls share that trace. Journals are local ignored files under runs/diagnostics by default. Studio requests have their own trace and X-Touchline-Trace-ID response header. Pipeline manifests reference diagnostic_trace_id. A manually invoked Python workflow can choose an explicit private log directory with diagnostic_session(directory); future Modal workers must provide writable private storage and carry job/trace correlation through the controller.

doctor checks dependency availability and reports input policy/budget ceilings. It reads no .env, credential profile or provider account, performs no authenticated network request, imports no provider SDK and launches no model/GPU job. Exit zero means the report was produced; inspect missing_required_modules, local_dependency_check and ready_for_paid_execution. That last field remains false while remote adapters/controller and account/budget gates are pending. It is not an authenticated readiness test or a model quality certificate.

## Coverage implemented now

| Boundary | Evidence recorded |
| --- | --- |
| CLI commands and argument errors | Trace, command boundary timing, safe failure code; no argv dump |
| Video probe and full-source analysis preflight | Geometry/timing failure category and repository stack locations; original media not logged |
| Pipeline | Trace reference in manifest, elapsed time, decoded/selected/observation counters every 24 decoded frames, safe failure artifact |
| Torchvision loading/inference | Load span and inference failures; successful per-frame inference is aggregated through the pipeline to avoid a log flood |
| Analytics | Boundary and caught physical-projection failures; unsupported metrics stay unavailable |
| Audio normalization/alignment and existing Sarvam adapter | Boundary/failure categories; no audio, transcript, key, request body or provider response in journal |
| Inventory, SoccerNet staging, ETL v1 and resumable release v2 | Boundary timings/errors; input paths/labels not dumped; ETL v1 failure artifact uses safe structured details |
| Manifest loading, scale audit and missed-player audit | Validation/audit boundaries; metric/audit artifacts remain separate from diagnostic journal |
| Training/evaluation | Boundaries, safe failure artifact and numeric training progress at step 1/every 25 steps; no new training implementation |
| Report draft/approval/correction/comparison | Boundaries with review errors; no player name, report text or reviewer identity in journal |
| Durable spending ledger | Verify/reserve/reconcile boundaries; overrun event after the block is durably committed, even if logging subsequently fails |
| Studio GET/HEAD/POST | Independent request traces, safe error responses and caught unexpected exceptions; URLs/headers/body are excluded |
| Optional local SAM proposal helper | Local vision.sam boundary; this is not evidence that the hosted Meta client is implemented |

Low-level pure calculations inherit their caller's operation trace rather than logging every invocation. Generator-based decoding is covered by probe/preflight/pipeline boundaries, not by persisting individual frames. This is operation-level coverage, not instrumentation of every source line or every external process.

## Record and failure contract

JSONL records have schema, generated trace/span/parent IDs, monotonic sequence number, UTC timestamp, static stage/event, elapsed duration, allowlisted finite numeric counters and optional structured error. Error details contain a fixed code/guidance, bounded repository-relative code locations and a fingerprint. Raw exception messages and local variable dumps are not retained. Error.type is limited to known categories; pipeline error.message is now static guidance for compatibility. Existing consumers must use code/trace_id rather than matching old raw exception strings.

Categories distinguish unsupported media, invalid data, missing dependencies, resource exhaustion/non-finite training, authorization, rate limits, provider failures, timeout, storage, budget rejection/overrun, required review and internal faults. Models receive only an explicitly reviewed redacted diagnostic excerpt; no automatic code correction or paid retry follows from a failure code.

Each journal is bounded to 2,000 events and approximately 1 MiB by default. At the limit, a truncation marker makes diagnostic-report return incomplete. Empty, unfinished or corrupted traces are not declared successful. Report summaries emit only known stage/error aggregates and never forward unknown free text from a selected file. A cancelled operation remains failed/cancelled in its summary; a recorded trace alone does not mean analytics are accurate.

Journal creation/write failure stops the guarded operation. Studio returns a safe 503 when it can; it closes the connection if a response has already started. A budget overrun commits its blocking state before its diagnostic event, so logging failure cannot erase the overrun. A future paid dispatcher must verify the journal is healthy/not truncated, persist the reservation/job ID and reconcile uncertain execution before retries. No remote dispatcher is implemented by this change.

The local log directory requires trusted ownership and appropriate filesystem permissions. The code rejects symlink diagnostic paths and creates exclusive trace files; this is not a filesystem sandbox against a hostile local administrator. Per-trace bounds do not impose a total retention quota. Operator retention/collection is manual; no files are automatically deleted or uploaded, and no paid logging service is configured. Production collection, retention quotas, remote health/latency, GPU telemetry and cross-process job correlation remain TL-047.3.

## What keys are sufficient for

Keys alone do not make the complete remote workflow ready. The chosen path needs:

1. Operator Modal authentication: MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or an existing private CLI profile.
2. BASETEN_API_KEY and META_API_KEY, with the selected model access and credit eligibility confirmed privately.
3. The non-secret endpoint/model settings in MODAL_PROVIDER_CONTRACT.md; they have documented defaults, but no adapter may silently substitute a model.
4. The existing detector bundle/reviewed data, verified budget baseline/headroom, provider-side limits and the still-pending remote clients/controller.

Optional later: FIREWORKS_API_KEY for an approved Fireworks task; HF_TOKEN only for the local gated-checkpoint route. Hosted SAM does not need HF_TOKEN. In Modal, touchline-baseten contains BASETEN_API_KEY and touchline-meta-sam contains META_API_KEY; attach them only to trusted brokers. Never put keys in the browser or repair Sandbox.

If the operator uses a local .env, place it at the selected checkout root beside pyproject.toml and load it explicitly in the future operator-controlled runtime. Merely adding the file does not load it: explicit loading is now available only through the operator-selected analyze/studio --env-file path; doctor never invokes it. The agent is forbidden to access actual .env/credential files or values, including through helpers, tests or provider CLIs, and will not inspect the file after keys are added. Do not send keys in chat.

Diagnostic journals exclude raw content by schema. Product artifacts and normal successful command output may legitimately contain reviewed player/session data, and third-party libraries can emit their own output outside this journal. Review those separately before sharing. No claim is made that arbitrary external logs or older files were retroactively sanitized.

[Analysis/budget setup](ANALYSIS_FIRST_RUNBOOK.md) · [Provider contract](MODAL_PROVIDER_CONTRACT.md) · [Rulebook](AUTOFIX_RULEBOOK.md) · [Pinned diagnostic tickets](TICKET_REFERENCE.md).

## Export/configuration update

[Using Colab exports](USING_COLAB_EXPORT.md): explicit configuration and export-check are now implemented, with 262 offline tests. Older provider-readiness claims remain unverified. Actual .env/credentials and Drive weights/data were not inspected during implementation.
