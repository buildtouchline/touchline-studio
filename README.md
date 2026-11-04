Copyright (c) 2026 Team Vikzar. All rights reserved. This is a private Team Vikzar repository. No permission is granted to copy, modify, distribute, or use this software outside Team Vikzar's authorisation.

# Touchline Studio

Touchline Studio is Team Vikzar's football video review and coach workspace. This repository is a **clean source snapshot** of the reviewed local remediation commit, with a fresh Git history. It is not evidence that a live Panorama/ResNet Modal run or production player-identity workflow has passed.

## Start locally

Python 3.12 or newer is required. Create an isolated environment and install only the extras needed for your task:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[vision]'
.venv/bin/python -m touchline studio --port 8765
```

Open `http://127.0.0.1:8765/`. Studio is a local review server. Without a verified checkpoint, source bundle, spending authorisation and Modal runtime, real Panorama upload readiness remains closed. For an explicit **synthetic QA** preview, use:

```sh
python3 -B src/touchline/web/qa/serve.py --port 8766
```

Synthetic observations and temporary track IDs are not verified player identities. Physical measurements require independently validated calibration. See [Panorama remediation status](docs/PANORAMA_STUDIO_REMEDIATION.md), [integration architecture](docs/TOUCHLINE_PANORAMA_INTEGRATION.md), and [migration provenance](MIGRATION_PROVENANCE.md).

## Offline checks

```sh
.venv/bin/python -B tools/run_offline_checks.py
.venv/bin/python -B tools/run_offline_checks.py artifacts
node --test src/touchline/web/*.test.cjs
```

Some offline tests require additional optional dependencies, including Torch. Test failures and skips must be reported separately. Tests with a mock Sandbox do not establish a live Modal result.

## Acknowledgements

Touchline is collaborative Team Vikzar work. The backend, Coach Studio, Panorama/ResNet worker, and model components were integrated. **Jai Vinush (`jv-013`)** prepared this reviewed integration snapshot and remediation. The single initial commit records the migration; it does not claim that its committer individually authored every component. Component provenance remains available in the archived source repository.
