# Successor handover: local terminal → Modal compute

Handover verification: 433 tests plus syntax/notebook/local-link checks passed in a Modal CPU Sandbox; termination confirmed. Fresh launcher contract checks passed; no second-operator live takeover or new model run was performed.

Updated 3 November 2026, Asia/Kolkata. Read this first for an LLM-assisted takeover.
The desktop runs the controller; models, video processing and tests run in Modal.
The fresh-run launcher supports CPU analysis and full AP50 validation. It does not
activate a global repair daemon, launch training or promote a model.

## 1. Current state

| Component | Delivered | Remaining |
| --- | --- | --- |
| Person detector | Selected trained weights; 61.18% AP50 with higher-resolution inference, versus 41.36% default, same 570 validation images | 70% is a target; own-camera detection/identity acceptance remains open |
| Modal | CPU analysis, dataset recovery/snapshot, T4 training and independent validation executed | Durable storage/expiry, billing reconciliation and generalized deployment |
| Baseten | Exact model meta-models/muse-spark-open-1026 returned a typed LR0.0008 proposal; isolated T4 trial/validation executed | General code-fixing and an always-on event consumer are not deployed |
| SAM3.1 | Eight real analysis requests plus four earlier labelling requests; visual disagreements saved | Proposals require human review; they are not accuracy truth or training labels |
| Failure handling | Deduplicated events, atomic claims, policy routing, staged repair adapter and rollback gates | Automatic source target currently limited to sampling.py plus an independent reproducer |

The validation resize experiment changed model input resolution, not the COCO
metric, checkpoint, labels or sample list. Final test is sealed. IDs in videos
are temporary tracks, not jersey numbers. Calibration/rubric/context are needed
before physical metrics or approved player scores.

## 2. Get the two branches

Install Git and Python **3.12**. Authenticate GitHub privately with your account
if the repository is private. Never put a token in a clone URL or paste it to an LLM.

```powershell
git clone --branch feature/modal-provider-pipeline --single-branch https://github.com/buildtouchline/touchline.git touchline
git clone --branch tickets --single-branch https://github.com/buildtouchline/touchline.git touchline-tickets
Set-Location touchline
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[providers]" "modal==1.6.3" "openai==2.54.0"
```

On macOS/Linux use `python3.12 -m venv .venv` and `.venv/bin/python` instead.
Local Torch, CUDA and model weights installation into the Python environment
are unnecessary for this controller route. The pinned remote image supplies
Torch/Torchvision/COCO/video dependencies. These are the SDK versions used here;
do not upgrade unrelated components during takeover.

## 3. Obtain private artifacts separately

Git contains no weights, footage, reviewed images, runtime receipts or keys.
Request these files from the outgoing team and place them as follows:

| Destination inside checkout | Source from the completed run |
| --- | --- |
| artifacts/handover/candidate.pt | modal-training-recovery-20261102/minor/candidate/candidate.pt |
| artifacts/handover/video24.mp4 | modal-cpu-full-video-20261102/source/video24.mp4 |
| artifacts/handover/baseline-observations.jsonl | modal-cpu-full-video-20261102/run/observations.jsonl |
| artifacts/handover/reference-evaluation.json | modal-training-recovery-20261102/minor/candidate-val/evaluation.json |
| artifacts/handover/recovery.json | modal-training-recovery-20261102/recovery.json |

The selected checkpoint SHA256 is
`b9200825a2bb3e39589fb705ced502531136b2cc70496c0a0d9a8519e2e99063`.

The receiving Modal credentials must access **the same workspace/environment main**,
the existing touchline-training-recovery App and the private immutable dataset
snapshot in recovery.json → data.snapshot_image_id. The snapshot has **seven-day
retention**, not permanent storage. It contains /work/dataset/release.json with
all 1710 verified images. A model export's release.json alone is insufficient.

If changing Modal workspaces or the snapshot expired, stop this quickstart:
rebuild/import the pinned environment and complete reviewed dataset first using
[CPU_ANALYSIS_RUNBOOK.md](CPU_ANALYSIS_RUNBOOK.md). Do not substitute an empty
directory or mark missing/unreviewed images eligible. Request fresh owner-approved
recovery allocation. Private image IDs do not transfer access across accounts.

For other source videos, normalize to exact constant24FPS inside Modal with
source/frame-map preservation before using this launcher. The supplied video24
copy already meets that contract. Merely renaming a25FPS file is insufficient.

## 4. Configure keys privately

```powershell
New-Item -ItemType Directory -Force artifacts/handover
if (-not (Test-Path .env)) { Copy-Item examples/operator-settings.template .env }
Copy-Item examples/handover-session.template.json artifacts/handover/session.json
```

The human edits root **.env beside pyproject.toml**, not a file under tools/src/docs.
Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET. Set BASETEN_API_KEY and META_API_KEY
for the optional adviser/SAM broker paths; these are not needed by the CPU worker.
Keep TOUCHLINE_TRAINING_ENABLED=false and TOUCHLINE_AUTOFIX_ENABLED=false:
those toggles do not authorize paid jobs. Keep the exact pinned provider URLs/models
in the template. No new key names are needed for the handover launcher.

Only the trusted runtime loads this exact .env. The LLM must not read, print,
copy, inspect through helpers, upload or include secret values in prompts.
Do not call SDK debug logging or print raw HTTP responses/headers. A key's
presence does not grant permission to launch jobs.

## 5. Fill one fresh session config

Edit artifacts/handover/session.json. It contains **no keys**:

- Replace run_id with a fresh `takeover-YYYYMMDD-analysis-001` identifier.
- Set operation to `analysis` for a20-second/120-sample preview; native=true
  selects higher-resolution input. Set `evaluate` for both default and higher-
  resolution full570-image validation against the supplied reference.
- Copy the real snapshot_image_id from the shared recovery receipt.
- Keep paths under artifacts/handover; verify checkpoint SHA against its receipt.
- The owner fills prior_upper_bound_usd and available_headroom_usd from prior
  charges **and retained reservations**, and sets owner_budget_confirmed=true.
  Zero values in the template are placeholders, not the project baseline.
- job_cap_usd is $0.50–$2.00; cumulative Modal ceiling remains $50 for the current
  approved project. Another account/team must obtain its own owner authorization.

Fill all values before preparation. Editing a prepared config requires a new
run ID; the launcher freezes its config, code and input hashes. Keep one trusted
budget controller and its runs/handover-spending.sqlite3; do not use multiple
writers or synchronize a live SQLite ledger through Drive. Never reset ledgers
to recreate credit headroom. Local reservations are not provider billing controls.

## 6. Prepare, then execute from the local terminal

```powershell
.\.venv\Scripts\python.exe tools/run_handover_job.py --config artifacts/handover/session.json
```

This prepares code/hash records only: **no credentials loaded, provider called
or model run**. Inspect the prepared result and owner budget settings. Then:

```powershell
.\.venv\Scripts\python.exe tools/run_handover_job.py --config artifacts/handover/session.json --execute-approved
```

The controller privately loads the selected settings, reserves spending once,
creates a bounded CPU Sandbox, uploads only allowlisted inputs, streams fixed
events and downloads results before confirming termination. The worker has
no secrets, outbound network or GPU. A duplicate dispatch is rejected.

Results are under runs/<run_id>/<run_id>-worker/:

- Analysis: RESULTS.html, preview/overlay24.mp4, contact-sheet.jpg, players.jpg,
  profile drafts, raw proposals, coverage and manifest.
- Evaluation: evaluation-comparison.json, baseline-val/evaluation.json,
  candidate-val/evaluation.json and predictions. Same checkpoint and570 sample IDs.
- job-receipt.json records worker ID and termination; result.json is the controller
  summary. gpu-results.zip is a historical shared-helper filename; the receipt's
  gpu=null and CPU model provenance identify actual execution.

Partial analysis status is expected for the120-sample smoke cap. Do not interpret
it as a complete four-minute run. Physical/identity accuracy remains unknown.
The fresh analysis launcher does not automatically call SAM; existing SAM review
artifacts and its broker are documented in [SAM_ANALYSIS_RUNBOOK.md](SAM_ANALYSIS_RUNBOOK.md).

If startup/termination is uncertain, stop and reconcile the exact worker before
any retry. The spending pool remains blocked and the hold retained. Preserve
receipts and ask the owner; do not blindly rerun or delete state.

## 7. Training, SAM and AI-fixer continuation

The tools/run_modal_*2026 experiment workflow is recorded through dated paths
in its scripts. **Do not run dated trial controllers as fresh setup commands**:
their original authorizations are consumed and they reference private artifacts.
Read [HANDOVER.md](HANDOVER.md), [CPU_ANALYSIS_RUNBOOK.md](CPU_ANALYSIS_RUNBOOK.md)
and [AUTOFIX_RULEBOOK.md](AUTOFIX_RULEBOOK.md) before preparing a new bounded job.

New training requires reviewed release.json plus images, fixed game/recording
splits, a fresh run ID and a declared recipe. Higher-resolution/backbone changes
are major recipes requiring exact review; the current approved minor scope only
changes LR/weight decay within0.5x–2x baseline. GPU training uses T4; CPU analysis
remains separate. Check evaluation dependencies before training and download
completed checkpoints before validation. Keep final test sealed until selection.

Baseten's connection and minor-trial path work. The optional code adviser accepts
typed proposals; model text never becomes an arbitrary shell command. The event
dispatcher requires configured budgeted handlers, fresh verified quotes, reviewed
source access and worker reconciliation. It is **not an installed background
watcher**. Missing data, provider/auth/budget issues route to recovery, not code
rewrites. Source fixes currently require sampling-contract-v1 on the registered
sampling.py target and an independent reproducer; other targets need registration
and major-change approval. Source/model promotion stays human-reviewed.

Next integration ticket: activate an authenticated consumer, attach the trusted
Baseten/checker handlers, prove duplicate/restart/timeout recovery and expand the
independent target registry. See TL-034, TL-036, TL-046 and TL-047 on tickets.

## 8. Paste this prompt into the successor's coding assistant

> Work in the cloned feature/modal-provider-pipeline checkout. Read AGENTS.md,
> docs/LOCAL_TERMINAL_HANDOVER.md, docs/HANDOVER.md, docs/AUTOFIX_RULEBOOK.md and
> docs/TICKET_REFERENCE.md first. The sibling tickets checkout is canonical.
> Keep model execution/tests in Modal; local work is editing/orchestration.
> Never inspect .env or credentials; only the selected trusted runtime may load
> them. Verify the supplied private artifacts, source/model hashes, workspace
> access and owner-confirmed budget config. Prepare a fresh takeover run first;
> execute only after its exact paid operation is authorized. Do not rerun consumed
> dated controllers, reset ledgers, alter evaluation truth or promote models.
> Baseten's minor path works, but no global auto-fixer service is deployed.
> Report run IDs, output paths, termination and measured results honestly. Update
> ticket Markdown/JSON/CSV together, pin the ticket commit in the feature branch,
> and leave existing PRs/main/develop unchanged.
