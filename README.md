Copyright (c) 2026 Team Vikzar. All rights reserved. This repository is made public solely for hackathon submission and judging purposes. No permission is granted to copy, modify, distribute, or use this software for any other purpose, commercial or non-commercial.

# Touchline Studio

Touchline Studio is a football video-review workspace for coaches and academies. It connects a recording to timestamped observations, temporary player tracks, coach review, and later corrections. The goal is to let a coach inspect evidence before associating a track with a player or drawing a development conclusion.

This public repository contains source code, tests and technical documentation. It contains **no match recording, model checkpoint, trained weight, credential or completed analysis export**. Public visibility permits Meta Global AI Developer Hackathon evaluation; it is not an open-source licence.

## What is implemented

The browser workspace is plain HTML, CSS and JavaScript under [`src/touchline/web/`](src/touchline/web/). Its API client separates explicit synthetic demo, local Studio, and configured authenticated production modes; production requests do not silently fall back to fixtures. The Python [`Studio server`](src/touchline/studio.py) serves the workspace, accepts bounded local MP4 uploads when readiness checks pass, stores file-backed jobs and runs, and offers run, observation, artifact and correction routes. It is **not** a complete multi-tenant production backend.

For the selected Panorama/ResNet path, the [`worker adapter`](src/touchline/adapters/panorama_worker.py) probes a fresh source MP4, checks its SHA-256, runs [timestamp-based normalization](src/touchline/video/normalize.py), records a source-to-normalized [frame map](src/touchline/panorama_import.py), verifies separate input and code hashes, and prepares a bounded [Modal Sandbox worker](tools/modal_panorama_worker.py) job. The worker uses a pinned Faster R-CNN ResNet50-FPN-v2 checkpoint contract. Completed exports can be checked and converted to canonical observations and a local run bundle. Coach review can create a separate child run while leaving the parent evidence unchanged; file-backed child runs are **not tamper-proof storage**.

```text
Coach workspace → local Studio API and job state → validated source and 24 fps copy
  → bounded Panorama/ResNet Sandbox adapter → verified exported artifacts
  → file-backed run and observations → coach evidence review and child correction
```

The full sequence above has passed an **offline mocked-Sandbox test with a fresh MP4 fixture**. A fresh browser-to-live-Modal-to-coach run has **not** been verified. The historical four-minute export and tracking results are review evidence, not production acceptance or a validated player-identity result.

## Meta model demonstration

Coach Studio now has a **Meta model activity** panel inside a real recording's Analysis workspace. It extracts one selected frame from the server's source video using its presentation timestamp; when the coach watches annotated output, the selected time is mapped back to source time through the supplied frame correspondence map. It sends a bounded request from the Python server to Meta Model API, then stores the request status, returned model ID, provider request ID, timestamps, latency, source-frame hash and safe output preview alongside that run. The browser receives no provider credential. A coach can confirm or reject a visible model observation; that local review does **not** confirm a roster identity or publish a parent report.

```text
Coach selects a real run and source-video moment
  → Studio decodes that frame from original video PTS
  → server-side Meta Model API request (when enabled)
  → local per-run AI activity record and source-frame hash
  → returned boxes/text shown beside the actual frame
  → coach reviews the proposal; player identity remains unconfirmed
```

| Intended Meta model | Role | Current verification |
| --- | --- | --- |
| **SAM 3.1** (`sam-3.1`) | Selected-frame person segmentation; boxes and encoded masks | One real Model API response received in local Studio from a frame of authorized football footage. Bounding-box proposals are displayed over the source frame; masks are retained as technical output, not rendered or independently validated. |
| **Muse Spark** (`muse-spark-1.3`) | Cautious selected-frame explanation | The account listed the model, but one real request failed without reviewable output. No successful integration claim. |
| **Muse Wonder** (`muse-wonder-1.0`) | Evidence-grounded reasoning assistance | The account listed the model, but one real request failed without reviewable output. No successful integration claim. |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Coach-audio transcription | Listed by the account; the current Studio audio route is not verified against the current Meta file-transcription contract. No live transcription claim. |

To enable **at most one to four** selected-frame calls in a trusted local server process, set `MODEL_API_KEY` and `TOUCHLINE_META_MAX_CALLS` (an integer from `1` to `4`) in that process's environment, then start Studio with an authorized run directory. The default allowance is **zero**. Open a run with a real `source_video.mp4`, seek to a frame and use **Analyse selected frame**. Each model can be requested only once per run; failed requests remain visible and count against the allowance. Credentials must be configured server-side and must never be pasted into the browser or committed. A fresh clone contains no recording, checkpoint or Model API key, so a judge cannot reproduce a live call from source alone. [Meta Model API](https://dev.meta.ai/docs/overview), [segmentation](https://dev.meta.ai/docs/sam/segmenting), and [voice transcription](https://dev.meta.ai/docs/speech-to-text) document the provider interfaces.

## Local setup

Use Python **3.12 or newer**. From the repository root on macOS or Linux:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[vision]'
python -m touchline studio --runs-dir ./runs --port 8765
```

Open `http://127.0.0.1:8765/`. The local server and workspace can open without a Modal account. **Real Panorama upload readiness remains closed** until its checkpoint, approved code bundle, compatible runtime, deployment probe, stationary-camera approval, access controls and spending ledger are supplied. Do not place secrets in browser assets or commit them. The default Studio server is for loopback development, not public hosting.

No frontend build is needed; the current client is served directly. To inspect the **explicit synthetic QA fixture**, start its separate local server and open `http://127.0.0.1:8766/`:

```sh
python -B src/touchline/web/qa/serve.py --port 8766
```

The fixture is labelled as synthetic. Its test video and observations are not footage or model results from a football match.

## Offline verification

After installing the vision extra, run the focused fresh-MP4 mock, importer and normalizer tests:

```sh
PYTHONPATH=src:tests python -B -m unittest test_panorama_backend_orchestration test_panorama_import test_video_normalize -q
PYTHONPATH=src:tests python -B -m unittest test_meta_activity test_meta_studio_api -q
node --test src/touchline/web/*.test.cjs
node --check src/touchline/web/app.js
python -B tools/run_offline_checks.py artifacts
```

The broader suite is `python -B tools/run_offline_checks.py`. Some tests need additional optional packages, including Torch/Torchvision, SciPy and OpenCV; report missing-dependency errors separately from code failures. These commands do not launch Modal or process a private recording.

## Status and limits

| Capability | Current status |
| --- | --- |
| Coach workspace, local run browsing, video timeline and evidence presentation | Implemented; local and synthetic paths tested |
| Fresh MP4 preparation, frame map, distinct original/normalized hashes, export conversion and restart reopening | Tested with a small fixture and mocked Sandbox |
| Panorama/ResNet worker and pinned checkpoint contract | Source present; checkpoint bytes and deployed image are not in this repository |
| Live Modal GPU execution from Studio | Gated and **not verified** for a fresh recording |
| Meta selected-frame AI activity | SAM 3.1 returned one real response in local Studio; other intended models are blocked or unverified as listed above |
| Production authentication, roster authorization and durable cloud storage | Integration boundaries exist; a complete production service is not verified |
| Temporary track review and separate correction child runs | Local file-backed workflow; roster identity requires trusted review; storage is not tamper-proof |
| Tactical minimap, MOTChallenge export, jersey recognition, validated speed/distance, automatic retraining, coach voice commentary and advanced scoring | Planned or separately experimental; not presented as verified recording outputs |

Detections and temporary track IDs do not prove a player's identity. Pixel movement is not validated distance or speed. Coach assessments and synthetic examples must not be presented as measured improvement. A historical export's detection quality does not establish tracking or identity accuracy. Camera motion, incorrect geometry or timing, corrupt video, missing lineage, and incomplete worker outputs can invalidate a run.

Live execution requires the exact approved checkpoint and source bundle, a compatible Modal SDK/image and access, a qualifying stationary panorama, trusted server-side authorization, and an explicit bounded spending approval. See [remediation status](docs/PANORAMA_STUDIO_REMEDIATION.md), [integration architecture](docs/TOUCHLINE_PANORAMA_INTEGRATION.md), [review gates](docs/REVIEW_GATES.md), and [snapshot provenance](MIGRATION_PROVENANCE.md). Other dated documents in `docs/` preserve engineering context; their historical plans and test counts are not current product claims.

## Acknowledgements and provenance

Touchline is collaborative Team Vikzar work. The backend, Coach Studio, Panorama/ResNet worker, and model components were integrated. **Jai Vinush (`jv-013`)** prepared the reviewed integration snapshot and remediation. The initial snapshot commit records migration into this repository; it does not claim that its committer individually authored every component. The [archived Touchline repository](https://github.com/buildtouchline/touchline) retains historical commits and attribution. Dataset access conditions and third-party notices cited in the technical documents remain applicable.
