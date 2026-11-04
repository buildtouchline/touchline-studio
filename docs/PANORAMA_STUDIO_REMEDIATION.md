# Panorama Studio remediation status

This branch extends the submitted Panorama consolidation commit `bd89c68` on the
current published `feature/player-mapping-modal-integration` lineage (`b0b74a0`).
The historical `e943f5d` line diverges after `b0b74a0`; its three later commits
are not on the published integration ref and were not silently imported.

## Supported offline path

The Studio upload route accepts bounded raw MP4 bytes into a newly created run
directory and retains the original SHA-256. Before mock or live dispatch, the
adapter probes the source, runs the repository's PyAV normalization, records a
frame correspondence map and metadata, checks exact normalized geometry, rate
and frame count, and verifies original, normalized, frame-map, checkpoint,
source-bundle, and worker-script hashes. The reviewed worker scope is a
stationary 4096 × 1080 panorama normalized to 5,760 frames at 24 fps, sampled
every fourth frame. The source duration must fit the four-minute scope. A
fresh recording that does not meet these limits fails before dispatch.

Mocked Sandbox tests exercise upload, preparation, export verification, canonical
run packaging, annotated video range delivery, coach association review, a
separate child run, and reopening after Studio restart. The parent remains
unchanged. This is **mocked execution only**; no real Modal run has been made.
Detector anchors and motion/camera predictions remain separate; local track IDs
are provisional until a trusted roster reviewer confirms an association.

## Live execution gates

Live dispatch is off by default. It requires an explicit server-side adapter
configuration, verified checkpoint and source bundle, exact operator-approved
source-bundle and worker-script hashes, PyAV/libx264 and Modal SDK, a positive
deployment probe, operator attestation that the camera is stationary, and a
spending ledger with a reservable USD 1.80 cap. The server also requires a
Bearer token and trusted run-access, roster and reviewer callbacks. A client
header alone cannot enable live dispatch. The backend reserves the spend before
Sandbox creation. The actual account image, app, checkpoint bytes, authentication
and worker availability have **not** been independently verified here.

## Remaining limits before a live pilot

- The reported model checkpoint and pinned Modal image are not present in this
  checkout. A trusted operator must provide and independently verify their bytes
  and deployed identity. No filename proves model identity.
- Stationary-camera suitability requires human review. The current probe checks
  geometry, rotation, duration and decodability, not camera motion quality.
- The shared Bearer token and injected access callbacks are a boundary for a
  local/pilot deployment. They are not a complete authenticated, multi-tenant
  roster service. Do not expose this Studio as a public production API.
- Child runs are separate, but file-backed artifacts remain writable by the
  server process. This is logical revision history, not tamper-proof storage.
- A process restart during an active Sandbox attempt does not reconnect to that
  attempt automatically. Its persisted job is not marked successful merely
  because it was running; operator reconciliation is required.
- Artifact reads use directory-relative no-follow descriptors on supporting
  POSIX platforms. Other path-based Studio reads/writes and a hostile same-user
  process replacing run paths still need a broader filesystem isolation design.
- The source normalizer uses decoded presentation timestamps and frame mapping;
  its output must still be compared to a real worker export in one explicitly
  approved bounded pilot. No physical metrics or stable player identity are
  validated by these tests.

## Validation commands

Use an isolated environment with the project `vision` extra (PyAV, Pillow,
NumPy) and optional offline-suite dependencies. No provider calls are needed.

```sh
PYTHONPATH=src:tests python -B -m unittest test_panorama_backend_orchestration -q
python -B tools/run_offline_checks.py
python -B tools/run_offline_checks.py artifacts
node --test src/touchline/web/*.test.cjs
node --check src/touchline/web/app.js
git diff --check
```

The full offline suite may require optional Torch and other dependencies.
Report skips and environment errors separately from test failures. Never run a
real Sandbox as part of routine validation.
