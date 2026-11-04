# Provider repair setup and recovery

## Current video trial and service preparation

Read [VIDEO_TRIAL_RUNBOOK.md](VIDEO_TRIAL_RUNBOOK.md) for the newer separately
authorized $3 video trial. CPU Sandbox creation and four hosted SAM requests
succeeded; T4/L4 Sandbox creation was rejected before a GPU worker was returned.
Local CPU video analysis/preview is separate evidence, not Modal GPU execution.
The detector ASGI endpoint and minor/major repair policy are prepared and tested;
deployment, general automatic fine-tuning and reviewed-label training remain open.
This update supersedes older authentication-only status below.

## Current runtime authorization and access results

On 1 November the user explicitly authorized trusted runtime-only loading of the
selected checkout `.env` and provider smoke calls, at most $1 total. This narrowly
supersedes older blanket runtime restrictions below; direct agent inspection,
secret/header/raw-response/error disclosure and credential-profile discovery
remain forbidden. Offline checks still use only fake clients/keys.

`tools/check_provider_access.py` performs fixed-host catalog checks, an explicit
Modal authentication-only mode, or one explicitly selected capped Baseten Hello.
It does not automatically dispatch inference, sandboxes, GPUs, training or repairs.
Redirects/environment proxy discovery are disabled for HTTP checks; no raw provider
output is printed. The Hello uses the OpenAI SDK, the exact pinned model,
32 maximum output tokens, no retries and a persistent single-use $1 authorization
reservation under ignored runs. It never releases that reservation automatically;
unknown billing requires independent reconciliation. One explicit user-requested
retry after the recorded HTTP 401 may reuse the existing hold; a unique SQLite
retry record rejects additional retries. Do not delete the allowance
database or change identifiers to bypass it. This is a narrowly scoped smoke
checker, not the complete shared provider-budget controller.

Live results: Meta's catalog returned HTTP 200 and listed `sam-3.1` and
`muse-voice-transcribe-1.0`. Modal's explicit-token client passed its authentication
handshake; zero jobs were launched and an absent runtime profile path prevented
discovery of the user's existing Modal profile. Baseten initially returned HTTP 401
for both catalog lookup and the exact-model Hello request. After the owner update
and explicit retry request, catalog lookup returned HTTP 200 with the pinned model,
and the SDK Hello retry returned a completion (147 input / 23 output tokens).
Its original $1 hold remains reserved; billing is independently unreconciled.
This validates only a tiny inference smoke, not the code-repair workflow.
Sarvam's key was absent at the configuration check.
Catalog listing is not successful model inference, credit balance verification,
trained-model quality or end-to-end provider acceptance. Receipts stay in runs.

## Video inputs and the requested voice route

Put original footage under `artifacts/videos/` in the active checkout. Put separate
audio under `artifacts/audio/`. Both are ignored by Git; preserve originals.
Analysis uses a checked 24 FPS copy when conversion is necessary, with the
source-frame/PTS map retained. Neither media folder nor `.env` belongs in a share ZIP.

The current voice requirement is **Sarvam-generated speech → Meta transcription**:

| Step | Endpoint/model | Runtime credential |
| --- | --- | --- |
| Synthetic speech generation | `POST https://api.sarvam.ai/text-to-speech`, `bulbul:v3` | `SARVAM_API_KEY`, sent as `api-subscription-key` |
| File transcription | `POST https://api.meta.ai/v1/asr/transcribe`, `muse-voice-transcribe-1.0` | `META_API_KEY`, sent as Bearer authorization |
| Visual proposals | `POST https://api.meta.ai/v1/responses`, `sam-3.1` | `META_API_KEY` |

Add `SARVAM_API_KEY` privately to the same `.env`; it is now a recognized secret
setting. No additional keys should be pasted into chat. Sarvam TTS returns base64
audio; Meta's file route needs mono 16-bit PCM WAV at 16 or 24 kHz. Meta returns
turn timestamps, not word-level timestamps. Preserve text/audio/transcript hashes
and separate generation, transcription and alignment revisions. Synthesized audio
is a test/demo source; its time origin does not prove when a real coach spoke.
Keep it outside reviewed coach evidence and automatic training/scoring inputs.
The existing `audio.py` Sarvam-STT route is legacy functionality, not this new
round-trip integration. That round trip is not implemented or live-accepted here.

`candidate.pt` is the Torchvision person detector; it is not SAM weights. Use the
detector as the baseline and SAM as an independent visual-proposal teacher. New
video → proposed labels → human review/correction → immutable dataset release →
separate bounded detector training → held-out evaluation → explicit promotion.
Sending video to hosted SAM does not retrain the local checkpoint automatically.

Sources: [Meta voice](https://dev.meta.ai/docs/speech-to-text),
[Meta SAM](https://dev.meta.ai/docs/sam/overview),
[Sarvam TTS](https://docs.sarvam.ai/api-reference/text-to-speech/convert),
[Baseten authentication/errors](https://github.com/basetenlabs/Hack-the-North-2026#error-codes).

1 November 2026. Code branch: feature/modal-provider-pipeline, based on
feature/modal-training at e92db36. Covers fixes under TL-034/TL-036/TL-037/TL-047;
these tickets remain partially done for demo. No live provider acceptance,
training/analysis GPU execution or automated promotion is claimed.

## Private configuration location

Create `.env` at the active checkout root, beside `pyproject.toml`, not inside
`src`, `notebooks`, `Handovers` or the tickets checkout. This workspace uses:

```text
./.env
```

For another machine use that machine's checkout root. The agent must never read,
source, validate, upload or copy the actual file, including through tests/CLIs.
The operator may copy [the public settings template](../examples/operator-settings.template)
privately and fill these supported names:

```dotenv
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
BASETEN_API_KEY=
META_API_KEY=
# Optional later:
# FIREWORKS_API_KEY=
# HF_TOKEN=

TOUCHLINE_CHECKPOINT_PATH=artifacts/models/football-person/candidate.pt
TOUCHLINE_DATASET_MANIFEST=artifacts/dataset-releases/YOUR_RUN_ID/release.json
TOUCHLINE_RUNS_DIR=runs
TOUCHLINE_DEVICE=cpu
TOUCHLINE_MODAL_VOLUME_NAME=touchline-artifacts
BASETEN_BASE_URL=https://inference.baseten.co/v1
BASETEN_MODEL=meta-models/muse-spark-open-1026
META_BASE_URL=https://api.meta.ai/v1
META_SAM_MODEL=sam-3.1
TOUCHLINE_TRAINING_ENABLED=false
TOUCHLINE_AUTOFIX_ENABLED=false
```

Use real runtime filesystem paths; a Drive sharing URL is not a checkpoint path.
Only configure the dataset manifest when its images are available alongside it.
Keep the existing checkpoint recipe/metadata beside the weights. `artifacts/`,
models, runs and `.env` are ignored by Git. `.env` placement does not automatically
load it or enable cloud execution. The existing explicit owner-only loader takes
an absolute file path; relative artifact paths resolve against its parent.

Modal authentication uses both token fields or a privately configured profile.
Do not assume adding Modal token fields to `.env` configures the SDK: the owner
must configure its trusted controller authentication explicitly. Hosted Baseten
and Meta keys belong only in their named trusted Modal Secrets when deployed;
never attach them to repair Sandboxes. See [the provider contract](MODAL_PROVIDER_CONTRACT.md).
The broader Modal analysis/training controller and hosted SAM adapter are still
open work. Existing local/Colab checkpoint analysis needs none of these API keys.

## Immutable repair runtime

The owner prepares and approves an existing image independently, records its
`im-...` object ID and exact dependency recipe, then supplies that ID explicitly:

```python
factory = named_modal_factory('touchline-repair', approved_image_id)
checker = ModalRepairChecks(factory, ledger, image_id=approved_image_id)
```

This is a library configuration example, not a launch command. No image is built
or named tag resolved by this factory. Mutable names and mismatched image IDs
are rejected. Image ID is bound into the dispatch fingerprint, run record and
check receipt. Reusing an attempt with another image cannot dispatch again.
An image ID must still be checked by the operator for approved content/access.

Worker output uses binary, unbuffered SDK chunks. The controller accumulates at
most 8192 bytes, rejects overflow/invalid UTF-8 and terminates the worker without
draining further output. Transport chunk allocation is controlled by the SDK;
this bounds application accumulation, not total transport/runtime memory.
Failed termination blocks the Modal pool; no guessed billing savings are released.

## Reconcile an uncertain attempt

After a timeout, do not delete the attempt database or change IDs to bypass it.
An operator must independently verify no work remains active, finalize provider
billing and retain a restricted evidence receipt binding the case/attempt/job and
termination/billing facts. Pass only that receipt's SHA256 and a non-personal
reviewer ID to the trusted controller:

```python
attempts.reconcile_unknown(
    attempt_id,
    reviewer_id='operator-a',
    evidence_sha256=reviewed_receipt_sha256,
    worker_terminated=True,
    billing_finalized=True,
)
```

This atomically records an immutable audit entry and retires only an `unknown`
attempt as `rejected`, never `checked`. Repeating the exact attestation is
idempotent; changing evidence is rejected. The original attempt still counts
toward the two-attempt cap. Ordinary `finish()` cannot resolve unknown states.

These arguments are operator attestations, not authenticated review or automatic
provider verification. Do not expose this method as an unauthenticated endpoint
or model tool. It does not reconcile/release spending reservations, remove any
provider block, retry work or promote a patch. Billing/blocked-pool reconciliation
is a separate trusted-controller responsibility; automatic unblock remains disabled.

## Verification

Use the guarded manual checks described in [the local runbook](LOCAL_PRIMARY_RUNBOOK.md).
Regression tests cover stream overflow/UTF-8, termination and retained reservations,
image provenance/replay rejection, audited recovery, immutable evidence and caps.
All provider clients are fake; no credential file or real provider account is used.
Verification for this delivery: 42 targeted tests and the full 345-test guarded
offline suite passed; artifact checks found no broken local documentation links.
Live model availability, sandbox behavior, billing and checkpoint recovery remain
operator acceptance gates. Read [the rulebook](AUTOFIX_RULEBOOK.md) before execution.

## Verified local CPU environment — 1 November 2026

An isolated `.venv` at the active checkout root now has the project installed
editable with `vision,tracking,providers`. Use its Python explicitly; the desktop
agent's base Python is a different environment. Setup did not load `.env`, import
an authenticated provider client or consume provider credits.

The tested Windows/Python 3.12.14 versions are Torch 2.11.0+cpu,
Torchvision 0.26.0+cpu, PyAV 17.1.0, Pillow 12.3.0, Supervision 0.26.0,
SciPy 1.18.1, OpenCV 5.0.0.93, Modal 1.6.3 and OpenAI 2.54.0.
`pip check` reports no broken requirements. The full 345-test guarded suite
passed in this environment, including the real optional CPU tracking backend.

The supplied candidate's SHA256 matches its exported result:
`92d2c29b21ff562b9199f03116c1e8ceff88afd39ab3d5e1cf59f84ea2768044`.
It loaded with the matching CPU Torch pair and passed compiled CPU NMS and
forward inference. A separate synthetic video smoke used eight 640x360 frames
at constant 24 FPS, selected two samples at 6 Hz, ran this actual detector plus
CPU ByteTrack, persisted a complete run and rendered its preview/gallery files.
The detector resize was bounded to 320/640 for this smoke only. No people or
reference identities were present, so this proves execution, not player accuracy,
tracking quality, field acceptance or production throughput. Real footage,
independent references and GPU/provider acceptance remain open. SAM's optional
local code/checkpoint package is not installed; SDK installation does not implement
the hosted SAM route or the general Modal controller.

From the checkout root, the operator can check local readiness without loading keys:

```powershell
& .\.venv\Scripts\python.exe -m pip --isolated check
& .\.venv\Scripts\python.exe -m touchline doctor
```

For an explicitly selected real 24 FPS landscape video, begin with a bounded CPU
smoke and a new output directory (replace the video path before running):

```powershell
& .\.venv\Scripts\python.exe -m touchline analyze C:\path\to\video24.mp4 --checkpoint artifacts/models/football-person/candidate.pt --device cpu --mode fixed --fps 6 --tracker bytetrack --record-proposals --max-selected 12 --output runs/real-video-smoke-001
```

This intentionally omits `--env-file`: checkpoint inference requires no provider
key. The real-video command preserves default detector resize settings, unlike
the reduced synthetic smoke, and its capped run will be partial. Full-source
preflight still decodes the selected video to verify cadence before inference;
use a short reviewed source for initial testing. A local CPU environment does not
provide a GPU or launch Modal implicitly. Keep generated runs and `.venv` out of Git.
