# Preliminary external MOT player detection check — 29 October 2026

> **Publication note:** The local paths below are provenance from the original experiment, not files included in this PR. [REPRODUCE.md](REPRODUCE.md) provides the checked repository-only and authorized-input commands. The gated MOT rows, paired video, saved prediction boxes and tracking observations are excluded.

This is a **12-frame, player/goalkeeper-only** comparison against the SoccerTrack v2 curated MOT subset. It is separate from the 24-frame all-visible-person manual benchmark, whose `annotations.json` still has zero reviewed frames. No detector inference was rerun and no human annotations were created.

## Reference and alignment evidence

- Official source: [`mot/118577.txt`](https://huggingface.co/datasets/atomscott/soccertrack-v2/blob/main/mot/118577.txt) and its [MOT README](https://huggingface.co/datasets/atomscott/soccertrack-v2/blob/main/mot/README.md). The local file is `/Users/jaivinush.c/Downloads/118577.txt`, 6,250,160 bytes, SHA-256 `425c9fa97e14f50298150f366828854639cfa52529470bd6ed95b8ac278dd2e7`. The official Hub listing independently gave this file size; an immutable remote revision or LFS hash was not obtained. The dataset is gated and its [data licence](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA) is CC BY 4.0; preserve its access conditions.
- Paired official clip `mot/clips/118577.mp4` was listed as 270,151,489 bytes, matching the existing local `/Users/jaivinush.c/Downloads/118577.mp4`. The frozen manifest records its SHA-256. Matching byte size is supporting evidence, not proof of identical remote bytes.
- `inspect_mot_gt.py` parsed the local MOT file and verified **22 distinct ID/box rows for each of the 12 selected zero-based frame indices**; every rectangle has positive area and lies within its 4096×1080 frozen image. It saved `external_mot_118577_subset.json` without modifying source files, predictions or manual labels.
- I visually inspected the cyan-box overlays at source frames 0, 3000 and 5500 in `external_mot_preview/`. Their boxes align with on-field players at those sampled moments. This is a limited visual spot check, not a full 12-frame independent annotation audit or remote content-hash verification. The JSON metric status retains `preliminary_external_gt_alignment_review_pending` for that reason.

## IoU 0.50 results on identical frozen frames

The existing `score.py` maximum-cardinality one-to-one matching was applied to each configuration's saved person boxes. Both configurations use the same 12 frozen `118577` frames and the same 264 MOT player/goalkeeper boxes. All figures below refer only to this external target scope.

| Configuration | Predicted boxes | TP | Unmatched predictions* | FN | Player/GK recall | Scope precision* | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 46 | 16 | 30 | 248 | 6.06% | 34.78% | 0.676 (16 matches) |
| Higher resolution | 187 | 112 | 75 | 152 | 42.42% | 59.89% | 0.713 (112 matches) |

\* The MOT reference omits referees, sideline people and other non-player humans. A legitimate person detection outside the 22 player/GK labels is counted as an unmatched prediction in this comparison. Therefore these counts are **not** all-person false positives or all-person precision. The player/GK recall denominator is exactly 264 reference boxes, so recall is the clearest finding. The matched IoU is conditional on detections that already met 0.50, not average localization quality over all players.

Higher resolution matched 96 more labelled players across the 12 frames, but still missed 152/264 at the saved confidence threshold. Every sampled frame improved in TP count (per-frame values are in `external_mot_detection_metrics.json`). This supports prioritising detection coverage investigation before identity or physical-metric claims. It does **not** establish performance over the full video, different camera conditions, or Touchline's own capture setup; the 12 frame sample was frozen before this reference arrived and the official labels still need broader alignment review.

An **exploratory, post-hoc** split by reference-box pixel height points to a small-player coverage problem: below 50 px, default matched 0/138 and higher resolution 28/138 (20.3%); from 50 to under 100 px, 8/115 (7.0%) versus 73/115 (63.5%); at 100 px or more, 8/11 versus 11/11. These are recall counts on the same sampled frames, not a pre-registered size benchmark. The smallest group remains weak even at higher resolution, and the largest group has only 11 examples. The exact counts are saved under `exploratory_gt_box_height_bins` in the machine-readable result.

The same MOT file also permits a **bounded identity diagnostic** on the preserved 30 sampled frames from 0–4.84 s. `score_mot_tracking.py` matched 65/660 player/GK reference boxes to 191 saved observations at IoU≥0.50, found six adjacent-sample local-ID changes for matched reference identities, and five further ID changes after detection gaps. See `external_mot_tracking_0_5s.json` and `TRACKING_REVIEW.md` for the definitions and event pairs. Low detection coverage and limited visual review prevent a full tracking-accuracy claim.

## Reproduction

```bash
ROOT=/private/tmp/touchline-level3-groundtruth-2026-10-29
python3 -B "$ROOT/verify.py"
python3 -B "$ROOT/score_mot_players.py"
python3 -B "$ROOT/score_mot_tracking.py"
```

The one-time extraction command used was `python3 -B "$ROOT/inspect_mot_gt.py" --mot-file /Users/jaivinush.c/Downloads/118577.txt`; it refuses to overwrite the preserved subset, so do not repeat it here. Reproduce the scores by verifying the recorded source hash, reviewing the subset and previews, and rerunning the two scorers if needed. `external_mot_detection_metrics.json` contains per-frame counts, configuration, source hashes, and the caveat. No input file was overwritten.

## Next checks

1. Have a human review all 12 overlays, especially distant, occluded and non-player people, and pin the exact dataset revision and paired clip hashes. Treat current counts as preliminary until that review is recorded.
2. Complete the independent five-frame **all-visible-person** manual batch in `annotation_batch.json`; then run `score.py --batch`. The MOT result cannot fill that schema or answer all-person precision.
3. Investigate the high player miss count through saved proposals and filtering before changing model settings. Tracking and physical metrics remain separate gated stages.
