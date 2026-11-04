# Reproduce the Level 3 validation evidence

The published JSON and verifier can be checked with **Python 3 and its standard library**. No Torch, footage, network access, or model checkpoint is needed for this arithmetic and consistency check:

```bash
python3 -B docs/validation/level3-2026-10-29/verify_results.py
python3 tools/check_artifacts.py
```

`verify_results.py` checks the 24 frozen frame identities; 12-frame MOT detection TP/FP/FN denominators and player/GK recall; 30-frame tracking counts and adjacent/gap ID-change counts; the still-blocked human annotation markers; and the saved mean-timing ratio. This **does not** independently verify the gated annotation contents, image alignment, detector inference, video decode, or timing measurement.

## Raw-count recomputation with authorized local inputs

An authorized reviewer who already has the paired curated `mot/118577.txt`, the two saved prediction JSON files, and the saved `full-118577/observations.jsonl` can run:

```bash
python3 -B docs/validation/level3-2026-10-29/verify_results.py \
  --mot /absolute/path/to/mot/118577.txt \
  --predictions-default /absolute/path/to/predictions_default.json \
  --predictions-higher /absolute/path/to/predictions_higher_resolution.json \
  --observations /absolute/path/to/full-118577/observations.jsonl
```

The verifier checks the local MOT and observation SHA-256 against the published results, matches prediction image hashes to the sanitized manifest, validates 22 unique player/GK rows on every selected source frame, then recomputes one-to-one IoU≥0.50 detection counts and bounded tracking ID changes. Omit `--observations` to recompute only detection. It writes nothing and will not fetch gated data. The equivalent command using the original machine's absolute paths was run and passed: default TP/FP/FN **16/30/248**, higher **112/75/152**, tracking **65/660 matches**, six adjacent ID changes and five after gaps. In this player/GK-only scope, unmatched predictions may be correct detections of referees or other humans; their count is not all-person false positives.

The reference file was obtained from the official [SoccerTrack v2 MOT directory](https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/main/mot). Local SHA-256: `425c9fa97e14f50298150f366828854639cfa52529470bd6ed95b8ac278dd2e7` for 6,250,160 bytes. The source clip hash is in `frame_manifest.json`. The Hub listing gave the same byte sizes for the MOT file and paired video; an immutable remote commit/hash comparison was not completed. The local visual spot check covered source frames 0, 3000 and 5500 only. The published metrics deliberately retain `preliminary_external_gt_alignment_review_pending`.

## Reproducing the original inference and timing experiments

The original frozen experiment and its source inputs are **not** in this PR. The source directory was `/private/tmp/touchline-level3-groundtruth-2026-10-29/`; the Level 2 runs were under `/private/tmp/touchline-level2-footage-eval-2026-10-29/`. Those paths describe the original machine and may disappear. The original 24 source frames, annotation UI, full prediction boxes, raw proposals, checkpoint and decoder environment must be retrieved from an approved persistent copy before re-running any source-level experiment. No such copy was confirmed when this bundle was prepared.

The original detector configuration was Torchvision `fasterrcnn_mobilenet_v3_large_320_fpn` with official `COCO_V1` weights, checkpoint SHA-256 `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`, CPU, score threshold 0.50, and `min_size=640` with `max_size=1280` versus `2560`. The isolated original environment recorded Python 3.12, Torch 2.15.0, Torchvision 0.29.2, PyAV 17.1.0 and Pillow 12.3.0; its full lock file is local and excluded. The timing summary measured adapter-equivalent frame calls, excluding disk load, model construction, tracking and artifact creation. **This PR has not rerun inference, decoding, timing, GPU, 4K throughput, live processing, or calibration.**

For the next all-person benchmark, the original annotation UI and frozen PNGs are required. `annotation_batch.json` selects five frames from both videos. A human must review every visible person at zoom and mark all five explicitly reviewed before the original `score.py --batch` can produce a valid preliminary all-person score. Both published blocked markers intentionally contain no such metric.
