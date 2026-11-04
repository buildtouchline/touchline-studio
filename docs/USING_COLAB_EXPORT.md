# Use the dataset and trained weights exported by Colab

31 October 2026. This describes the format produced by this repository's notebook/export code. The actual private Drive files have not been downloaded or validated by the agent.

## What each output contains

```text
Touchline/
  model-runs/<RUN_ID>/
    release.json                 # Dataset manifest copy; images are not in this export
    SHA256.json                  # Export artifact integrity index
    candidate-train/
      candidate.pt               # Trained Torchvision weights + embedded metadata
      recipe.json                # Architecture, classes, runtime and training lineage
      result.json                # Checkpoint hash, completed steps and candidate status
      losses.jsonl
    baseline-val/                # Evaluation artifacts when executed
    candidate-val/
    selected-test/               # Optional, only when explicitly executed
    runtime.json
    environment.txt
    source-hashes.json
    code-revision.txt
  dataset-releases/<RUN_ID>/
    release.json                 # Full dataset manifest beside the actual image tree
    images/...
    etl-report.json
```

release.json is an ETL dataset manifest, not trained parameters. It describes samples, train/val/test capture groups, source/image hashes, dimensions, review/eligibility and person boxes in source-pixel XYXY. Classes are background=0/person=1. The actual weights are candidate-train/candidate.pt. There is no separate service/model named Touchline to obtain or configure.

The checkpoint bundle contains model_state, metadata and training_state. The current adapter expects the repository's Torchvision architecture and class mapping, checks runtime compatibility and uses torch.load(weights_only=True). A random .pt/.pth file, YOLO checkpoint or raw state_dict is not automatically interchangeable. recipe.json is the exported metadata companion; a separately named metadata.json is not required by this export format.

## Configure paths separately from provider keys

Use [operator-settings.template](../examples/operator-settings.template) as a reference. The operator can copy/edit it privately as .env beside pyproject.toml. The agent must never open or modify that actual file. Key artifact settings are:

```dotenv
TOUCHLINE_EXPORT_DIR=artifacts/model-runs/YOUR_RUN_ID
TOUCHLINE_CHECKPOINT_PATH=artifacts/model-runs/YOUR_RUN_ID/candidate-train/candidate.pt
TOUCHLINE_DATASET_MANIFEST=artifacts/dataset-releases/YOUR_RUN_ID/release.json
TOUCHLINE_RUNS_DIR=runs
TOUCHLINE_DEVICE=cpu
```

Paths above are relative to the explicitly selected settings file's parent, independent of notebook/terminal CWD. Replace YOUR_RUN_ID and put the exported folders there, or use absolute paths. On Colab, existing paths may start /content/drive/MyDrive/Touchline/. On Modal, first import/copy the files to the mounted storage and use that worker's filesystem paths. A Drive sharing URL is not a mounted path, and a local Windows path is not visible in Modal automatically. /artifacts/ under a Modal mount is also distinct from the relative artifacts/ example above.

Video inference needs trained weights; it does not need dataset images on every request. Evaluation/retraining need the full dataset-releases manifest and images. Do not point dataset validation at the small model-export copy and expect its missing images to exist. New video input remains 1080p/24 CFR; original training images and source video annotations keep their original timing/geometry.

Keep BASETEN_API_KEY/META_API_KEY and optional Modal tokens in separate settings lines. Those keys are for external services; they do not replace checkpoint/data files. Local inference with the existing weights needs no model API key. Remote Modal/Meta/Baseten orchestration remains a separate implementation gate; adding keys does not start paid work or authorize spending.

## Check the exported format and integrity first

After installing the project, the operator can run this without any .env or provider key:

```text
python -m touchline export-check /absolute/path/to/model-runs/RUN_ID
```

This checks the known exported files, SHA256.json entries, checkpoint/result hashes, recipe architecture/classes and dataset lineage. It never unpickles/loads the weights, imports Torch for inference or claims 95% accuracy. It prints weights_loaded=false and runtime_compatibility=not checked deliberately. The export's self-consistent hashes are an integrity check, not proof of an authenticated publisher or correct labels.

To also validate the actual image release, select its separate manifest explicitly:

```text
python -m touchline export-check /absolute/path/to/model-runs/RUN_ID --dataset-manifest /absolute/path/to/dataset-releases/RUN_ID/release.json --verify-images
```

This requires the same dataset hash used for training and the existing training eligibility/image integrity gates. It may take time because it reads/hashes images; it runs no training or hosted model. Synthetic fixtures do not pass the real training/image gate.

## Run the existing weights on a video

Without reading any env file:

```text
python -m touchline analyze /absolute/path/to/video.mp4 --checkpoint /absolute/path/to/candidate-train/candidate.pt --output runs/review-001 --device cpu
```

Or, the operator alone can explicitly load their settings file:

```text
python -m touchline analyze /absolute/path/to/video.mp4 --env-file /absolute/path/to/project/.env --output runs/review-001
python -m touchline studio --env-file /absolute/path/to/project/.env
```

The loader never searches parents/CWD, performs shell/variable expansion or prints keys. Known process variables override the selected file; explicit checkpoint/device CLI flags override those settings. Quoted and unquoted single-line NAME=value settings are supported; comments occupy their own lines. Unknown/duplicate names, shell expansion and unapproved endpoint/model changes fail closed. Training/autofix remain disabled. doctor never invokes the loader or checks credential contents.

Configured Studio uses these trained weights/device and disables implicit pretrained downloads; a missing/invalid selected checkpoint fails instead of falling back. Studio without settings retains its legacy local prototype behavior; it is not the durable two-upload service. Analysis uses fresh run output directories, preserves PTS and produces provisional tracks/analytics. Calibrated physical metrics, confirmed player identity, full rubric scoring and improvement plans still require their independent implementation/review gates.

CPU/GPU memory and quality must be tested with the actual bundle. Choose cuda only on a configured GPU runtime; no automatic expensive GPU upgrade or provider deployment occurs. Do not commit .env, actual artifacts or images; the example artifacts/ directory is ignored.
