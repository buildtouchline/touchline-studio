# Touchline handover — 4 November 2026

Start here. Current branch: `feature/player-mapping-modal-integration`.
Prepared for project handover. Work stopped at the owner's request. The CPU Sandbox was terminated
and partial evidence exported; no remote worker is intentionally left running.

## Current result

**Tracking quality is not fixed or accepted yet.** The last baseline panorama
measurement is **31.27% provisional identity F1**. Do not present detector AP50 as
player identity accuracy, or the experimental repair as an accepted improvement.

The four-minute ResNet video was fully analyzed and exported: 5,760 frames,
4096×1080, 24 FPS, six-Hz detector anchors and explicitly labelled predictions.
The GPU worker terminated. Six additional Meta SAM3.1 review calls completed.

On the same 1,440 sampled frames, player-reference detection AP50 improved from
19.64% for the old saved MobileNet run to84.29% for native ResNet. Detection recall
was93.27%, but tracked-box recall was60.26%, with224 identity changes. The reference
covers22 players/goalkeepers and omits other people; local completeness/identity
review remains outstanding. These are provisional development metrics.

Separately, the trained MobileNet candidate reached84.85% AP50 on570 unchanged
SoccerNet validation images. That is a different model, dataset and metric scope.

## What this branch contains

- The complete provider baseline at `143b8f9`, retaining primary Modal Sandbox
  execution, SAM brokers, Baseten rulebook controls, safe diagnostics, training
  and retained-budget handling.
- Seven selected player-mapping/voice components with additional guards:
  calibration loading, coach voice cues and clip appending, approved report
  export, optional speech adapters, headset controls, protected detector client
  and bounded SAM polygon display.
- Dynamic panorama canvas geometry and bounded observation paging, replacing
  the Studio's silent1,200-row preview truncation.
- Experimental stationary tracking repair, source-level regression tests,
  CPU-only replay controller/worker and a separate provisional metric runner.

The original `feature/modal-provider-pipeline` and
`feature/player-mapping-voiceover` branches remain intact. No main/develop or
existing PR was merged. No YOLO, alternate Function deployment, hard-coded field
homography or file-size-based annotation shortcut was imported.

Read [the exact component comparison](docs/SELECTIVE_INTEGRATION.md).

## Tracking repair status

1. The first motion/SAM replay failed the coverage gate. Best identity F1 was
   31.62%, with23.68% tracked-box recall. It was rejected; the worker terminated.
2. A confirmed-before-tentative association bug was fixed and reproduced in a
   regression test. The corrected trial reached33.18% identity F1 but only44.95%
   recall, so it also failed the gate and is not a successful replacement.
3. A further source change adds measured foreground pixel motion to recovery,
   with explicit prediction labels. 492 remote regression tests and artifact checks passed before this replay
   began. The owner requested handover before a completed quality comparison;
   this version remains **quality-unverified**.

Baseten reviewed the source. The supervising agent rejected its incorrect syntax
claim and invalid edit, and used its Kalman state/covariance concern to improve
the source. No generated patch ran automatically. No new training was performed.

Details and metric definitions: [tracking quality report](docs/TRACKING_QUALITY_REPORT.md).

## Where the private artifacts are

The existing operator checkout is `touchline-integration-fixes-20261031`, beside
this checkout in the All Chat workspace. Git does not contain its `.env`, media,
weights, datasets, run databases or execution receipts.

| Operator-relative path | Contents |
| --- | --- |
| `runs/modal-panorama-resnet-20261104/` | Completed ResNet video, evidence ZIP, exact metadata and six SAM receipts/proposals |
| `runs/panorama-reference-comparison-20261104-cpu/` | Full-recording provisional old/new reference comparison |
| `runs/modal-stationary-repair-20261104/` | Rejected first repair,490-test source snapshot and terminated-worker receipt |
| `runs/modal-stationary-repair-20261104-priority/` | Rejected priority-only candidate and source/run lineage |
| `runs/stationary-motion-recovery-20261104/` | Further flow-recovery source snapshot; incomplete quality result |
| `runs/tracking-stop-handover-20261104/` | Confirmed termination receipt,492-test log and preserved partial logs/results |
| `runs/baseten-stationary-review-20261104/` | Baseten review, safe usage receipt and unexecuted advice |
| `runs/modal-cpu-full-video-20261102/source/` | Exact normalized video, normalization receipt and25-to24-FPS frame map |
| `artifacts/soccertrack-v2/mot/118577.txt` | Unchanged upstream player-only reference |

The outer `Handovers/Video-Results/` folder contains
`Touchline-ResNet-Panorama-4min-review.mp4`, player gallery, contact sheet, metadata
and README. These are private review artifacts, not accepted tracking output.
Transfer the operator run folders privately; never transfer credentials in a ZIP.

## Local terminal → Modal setup

Use Python3.12 and Git. Clone this branch and the `tickets` branch separately.
Authenticate privately; never put a GitHub token in a URL or an LLM message.

```powershell
git clone --branch feature/player-mapping-modal-integration --single-branch https://github.com/buildtouchline/touchline.git touchline
Set-Location touchline
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[providers]" "modal==1.6.3" "openai==2.54.0"
```

Local Torch/GPU installation is unnecessary for the controller route. Model/media
execution and regression tests run in a Modal Sandbox. Supply a private root
`.env` using [the settings template](examples/operator-settings.template), or use
the existing operator checkout with the replay controller's `--operator-root`.
The agent may not inspect or print credentials. Workers have no provider secrets
or network access. Environment is `main`; remote dependency image IDs may expire
and must be checked before another launch.

The dated replay launcher refuses consumed attempts and existing output folders.
Do not rerun it blindly or remove its spending records. For another experiment,
review a fresh run ID, allowance and timeout, then update the explicit controller
scope. Do not mark upstream labels reviewed just to unlock an evaluator.

## Next person: do this in order

1. Read the stop receipt and confirm no worker remains; preserve failed/partial
   results and every previous budget hold.
2. Review the flow-recovery patch and its regression evidence. Reproduce a fresh
   bounded CPU replay on the same frozen detections, without detector retraining.
3. Require BOTH identity F1 improvement and coverage at least as good as the
   60.26% baseline. Inspect crossings and beginning/middle/end; do not choose a
   tracker simply because it displays24 IDs or fewer switches.
4. Keep24 non-recycled session IDs distinct from tentative internal tracklets.
   Investigate capacity starvation, dormant recovery and false-positive births.
5. Obtain complete reviewed tracking references and separate held-out acceptance
   before claiming75/80% IDF1. Production coach ingestion still needs the canonical
   ResNet schema adapter and a real recording-to-correction acceptance run.

Tickets are authoritative on `tickets`; see [the exact pin](docs/TICKET_REFERENCE.md).
Spending reservations are retained through exceptional Modal ceiling **$55.75**;
this is not a remaining balance or a new authorization. Baseten/Meta retain their
existing separate ceilings. There is no automatic model promotion or deployed
general-purpose repair daemon.
