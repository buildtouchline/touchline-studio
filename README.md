Copyright (c) 2026 Team Vikzar. All rights reserved. This repository is made public solely for hackathon submission and judging purposes. No permission is granted to copy, modify, distribute, or use this software for any other purpose, commercial or non-commercial.

# Touchline Studio

Touchline is building an AI-powered football video intelligence platform that transforms match and training footage into structured, evidence-backed insights for coaches and player development.

This repository contains **Touchline Studio**, the current coach-facing implementation. A coach can browse a saved run, inspect source or available annotated video and timestamped evidence, review provisional tracks and Meta-generated proposals, and record local corrections. The product is designed to connect those moments to future player development; it is not yet a deployed multi-club or parent platform.

This public repository contains source code, tests and technical documentation. It contains **no match recording, model checkpoint, trained weight, credential or completed analysis export**. Public visibility permits Meta Global AI Developer Hackathon evaluation; it is not an open-source licence.

**Architecture guide:** [system architecture and diagram](docs/TOUCHLINE_SYSTEM_ARCHITECTURE.md) · [scalability strategy](docs/TOUCHLINE_SCALABILITY.md) · [Panorama integration boundary](docs/TOUCHLINE_PANORAMA_INTEGRATION.md) · [review gates](docs/REVIEW_GATES.md) · [remediation status](docs/PANORAMA_STUDIO_REMEDIATION.md) · [migration provenance](MIGRATION_PROVENANCE.md). The Panorama integration document is a dated proposal; its proposed production routes are not claims of deployed endpoints.

## What is implemented

The browser workspace is plain HTML, CSS and JavaScript under [`src/touchline/web/`](src/touchline/web/). Its API client separates explicit synthetic demo, local Studio, and configured authenticated production modes; production requests do not silently fall back to fixtures. The Python [`Studio server`](src/touchline/studio.py) serves the workspace, accepts bounded local MP4 uploads when readiness checks pass, stores file-backed jobs and runs, and offers run, observation, artifact and correction routes. It is **not** a complete multi-tenant production backend.

For the selected Panorama/ResNet path, the [`worker adapter`](src/touchline/adapters/panorama_worker.py) probes a fresh source MP4, checks its SHA-256, runs [timestamp-based normalization](src/touchline/video/normalize.py), records a source-to-normalized [frame map](src/touchline/panorama_import.py), verifies separate input and code hashes, and prepares a bounded [Modal Sandbox worker](tools/modal_panorama_worker.py) job. The worker uses a pinned Faster R-CNN ResNet50-FPN-v2 checkpoint contract. Completed exports can be checked and converted to canonical observations and a local run bundle. Coach review can create a separate child run while leaving the parent evidence unchanged; file-backed child runs are **not tamper-proof storage**.

```text
Coach workspace → local Studio API and job state → validated source and 24 fps copy
  → bounded Panorama/ResNet Sandbox adapter → verified exported artifacts
  → file-backed run and observations → coach evidence review and child correction
```

The full sequence above has passed an **offline mocked-Sandbox test with a fresh MP4 fixture**. A fresh browser-to-live-Modal-to-coach run has **not** been verified. The historical four-minute export and tracking results are review evidence, not production acceptance or a validated player-identity result.

The architecture separates coach interaction from bounded processing. The current local Studio stores jobs and run artifacts on disk and performs Panorama work in a background thread through an adapter; a durable queue, cloud object store, authenticated tenant service and horizontally scaled deployment are **future production boundaries**. The [system diagram](docs/TOUCHLINE_SYSTEM_ARCHITECTURE.md#system-at-a-glance) marks those boundaries, and the [scalability strategy](docs/TOUCHLINE_SCALABILITY.md) explains how to reach them without treating local mocks as live GPU capacity.

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
| **SAM 3.1** (`sam-3.1`) | Selected-frame person segmentation; boxes and encoded masks | One real Model API response received in local Studio from a frame of authorized football footage. Bounding-box proposals can be overlaid when the exact saved source frame passes hash verification; masks are retained as technical output, not rendered or independently validated. |
| **Muse Spark** (`muse-spark-1.3`) | Selected-frame visual observation for coach review | A genuine completed `/v1/responses` call returned a structured observation from the real football frame. Its saved text is displayed for coach review; an earlier failed attempt remains in the activity history. |
| **Muse Wonder** (`muse-wonder-1.0`) | Assess what one frame supports about a coach objective, and identify missing evidence | A genuine completed `/v1/responses` call returned a cautious response about the shoulder-check objective. It correctly stated that the wide still cannot establish a shoulder check or improvement. An earlier failed attempt remains visible. |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | File transcription of coach audio | A genuine `/v1/asr/transcribe` call returned text and one timed turn from a **locally generated spoken-audio test file**, not a real coach recording. Coach approval can attach an edited transcript to the local session; speaker identity is unverified. |

To enable bounded calls in a trusted local server process, set `MODEL_API_KEY` and `TOUCHLINE_META_MAX_CALLS` (an integer from `1` to `8`) in that process's environment, then start Studio with an authorized run directory. The default allowance is **zero**; the recorded smoke-test run exhausted its configured allowance. Open a real source recording, seek to a frame, and choose SAM, Spark or Wonder. Wonder also requires a specific coach objective. For Voice, record or select an audio file, review it, then explicitly submit it. The server converts audio to mono 16-bit PCM WAV at 16 kHz, accepts at most 8 MB and 120 seconds, and sends one multipart request to Meta's file endpoint. Each model permits a second attempt only after a recorded failure; every attempt counts against the server allowance. No provider call is made when the allowance is closed. Credentials stay server-side and must never be committed or supplied through the browser. A fresh clone contains no recording, checkpoint or Model API key. [Meta Model API](https://dev.meta.ai/docs/overview), [image understanding](https://dev.meta.ai/docs/image-understanding), [segmentation](https://dev.meta.ai/docs/sam/segmenting), and [voice transcription](https://dev.meta.ai/docs/speech-to-text) document the interfaces.

Activity JSON persists per run with timestamps, model ID, request or provider session reference, latency, input hash, safe output preview, failure metadata and local coach review. It does not preserve the complete raw provider envelope or uploaded audio. Approving Voice adds a local transcript clip to the session; edited wording uses the full clip interval because the provider's original turn boundaries no longer locate each edited word. These local records are not production authentication, approved parent reports or verified player identities. The successful model smoke tests establish access and a working local UI path, not production reliability or tracking quality.

New selected-frame requests also save the exact submitted JPEG beside their activity record, with its SHA-256 checked on retrieval. Older saved requests without that JPEG may require the original decoding runtime to reproduce the bytes exactly. If verification fails, Studio shows an unavailable frame-preview state while keeping the genuine saved model text and review status; it never substitutes a different frame. This limitation applies to the recovered historical run used in the local demonstration.

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

## Current capability status

| Capability | Status and evidence |
| --- | --- |
| Coach workspace and local run browsing | **Implemented locally.** Studio lists file-backed runs; the browser can reopen a saved run. No production club directory is present. |
| Video timeline and evidence presentation | **Implemented locally.** Source/annotated media and timestamped observations display when corresponding verified artifacts exist; synthetic QA is explicitly labelled. |
| Video preparation and normalization | **Mock-tested.** Fresh MP4 probing and bounded PyAV normalization are exercised by `test_video_normalize.py` and `test_panorama_backend_orchestration.py`. |
| Frame mapping and provenance | **Mock-tested and importer-tested.** Separate original/normalized hashes and the source correspondence map are validated; historical saved-export import was reviewed. |
| Panorama/ResNet worker | **Source and contract present.** Worker, adapter and checkpoint hash requirement exist; checkpoint bytes and deployed image are absent from this repository. |
| Modal GPU execution | **Integration boundary present; fresh live run unverified.** Sandbox creation/export code exists, but the fresh Studio-to-Modal path has only mocked execution evidence. |
| Meta AI activity | **Four genuine local smoke responses recorded.** SAM 3.1, Muse Spark, Muse Wonder and Muse Voice Transcribe returned usable responses; Voice used generated test speech. Saved outputs and coach review are local, not production reliability evidence. |
| Coach review and temporary-track correction | **Implemented locally.** Model proposals can be reviewed; separate correction child runs are file-backed. A track ID is not a verified roster identity. |
| Production authentication and roster authorization | **Not implemented as a production service.** Loopback development and optional injected bearer/access callbacks are boundaries, not tenant or roster management. |
| Durable cloud storage | **Not implemented.** Jobs, media, activity and revisions are local files; local hashes do not make them tamper-proof. |
| Multi-tenant production deployment | **Not implemented or verified.** No demonstrated tenant isolation, managed queue or horizontally scaled application deployment. |

Planned analysis capabilities include a tactical minimap, MOTChallenge export, jersey recognition, validated speed and distance, automatic retraining, coach voice commentary and advanced scoring. The interface labels unavailable outputs as planned; no fabricated values should appear as recording results.

Detections and temporary track IDs do not prove a player's identity. Pixel movement is not validated distance or speed. Coach assessments and synthetic examples must not be presented as measured improvement. A historical export's detection quality does not establish tracking or identity accuracy. Camera motion, incorrect geometry or timing, corrupt video, missing lineage, and incomplete worker outputs can invalidate a run.

Live execution requires the exact approved checkpoint and source bundle, a compatible Modal SDK/image and access, a qualifying stationary panorama, trusted server-side authorization, and an explicit bounded spending approval. See the [architecture](docs/TOUCHLINE_SYSTEM_ARCHITECTURE.md) and [scalability strategy](docs/TOUCHLINE_SCALABILITY.md) for the implementation-to-production boundary. Other dated documents in `docs/` preserve engineering context; their historical plans and test counts are not current product claims.
