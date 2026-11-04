# Tracking validation: bounded next review

> **Publication note:** The review JPEGs and source observations referenced here are local gated experiment artifacts and are not included in this PR. The aggregate diagnostic is [external_mot_tracking_0_5s.json](external_mot_tracking_0_5s.json); see [README.md](README.md) for scope and next actions.

`tracking_review_0_5s/tracking_review.json` and 30 numbered JPEGs are a **model-output review package**. They were made from the existing `full-118577` Level 2 `observations.jsonl` and exact video frames at source PTS 0.00–4.84 s. No model inference ran for this package. Cyan rectangles show tentative local track IDs; orange marks an ambiguous assignment. A separate official SoccerTrack MOT reference with player IDs has now been checked against this same sequence; see below. It covers 22 on-field players/GKs, not every person, and has not been independently reviewed frame by frame by the team.

The reviewed package has 30 frames, 191 saved observations and 36 distinct temporary local IDs. These counts do not imply 36 people, identity switches or fragmentation. The Level 2 run uses `IoUTracker`: detections from the Torchvision adapter enter a greedy box-IoU association at threshold 0.30, with a 0.75 s maximum gap. It creates new IDs for unmatched detections, keeps candidate ambiguity state, and has an in-memory started/matched/lost/terminated event ledger. The pipeline saves frame `coverage.jsonl` and per-detection `observations.jsonl`, but does not persist the tracker event ledger in this run; the saved `analytics.json` shows `tracking_events: []`. Its full-run summary reports 1,182 model track IDs and zero confirmed players, which are algorithmic outputs and not identity accuracy.

In the 30-frame sample, 11 model IDs appear once and 42 of the 191 observations carry `association_state="ambiguous"`. These are useful frames to inspect first, but a one-observation ID is not automatically a fragment, and an ambiguous flag is not a verified switch.

## Preliminary external MOT identity diagnostic

`score_mot_tracking.py` compared the preserved 30 sampled frames against the official curated `mot/118577.txt` player/GK boxes and stable IDs. It verified source video and annotation hashes, run ID, 22 unique GT identities per frame, 191 saved observations, and applied one-to-one IoU≥0.50 matching separately per frame. `external_mot_tracking_0_5s.json` holds the 30-frame ledger and event details. Of 660 labelled player/GK boxes, **65** matched saved observations; 10 distinct reference identities were matched at least once. The diagnostic found **six adjacent-sample ID changes** for matched reference identities and five further ID changes after one or more unmatched sampled frames. One adjacent change for MOT ID 9 at source frames 30→34 corresponds to visibly changing Touchline labels T7→T17 on the same nearby red player in the saved overlays. These are bounded algorithmic findings, not full-video IDF1/HOTA or a complete reviewed switch rate; 595 labelled boxes had no match, and those detection gaps can hide identity errors. A human should inspect all six event pairs and the five gap cases before promoting a tracking quality claim.

## Human review procedure

Open the 30 JPEGs in source-frame order, preferably at native resolution, alongside `tracking_review.json` and `tracking_contact_sheet.png`. For each physical person visible through the sequence, the reviewer should record a stable human-assigned temporary identity and every model track ID used for that person. Explicitly mark periods with no detection, occlusion, ambiguous overlap/crossing, a new ID after reappearance, and any ID assigned to a different person. A second reviewer should check at least the crossings and smallest distant players.

Use this convention before counting errors: an **ID switch** is a model ID reassigned to a different reviewed person or a reviewed person changing model ID across consecutive observable frames; **fragmentation** is a new model ID for the same reviewed person after a detection gap. Record gap duration separately, rather than inferring either event from the total ID count. Mark an interval unresolved when the source pixels do not support a reliable identity decision. Do not calculate IDF1/HOTA until every relevant person/frame has independent identity labels and a tested metric implementation.

The repository's `PlayerAssociationLedger` can bind a run-scoped local track to a confirmed player and records corrections, but the current Level 2 run has no confirmed association. In `src/touchline/analytics.py:119`, `identity_review_ref` is constructed from `track_id` (`track_rev_{track_id}`) without a real review record. This is a **separate source-code gate** before any calibrated distance, speed, activity or intensity metric should be described as identity-reviewed. The present uncalibrated Level 2 outputs suppress those physical values.

## Acceptance criteria for the next tracking stage

1. A human-reviewed chronological frame ledger identifies each in-scope person or marks the identity unresolved, with frame/PTS provenance.
2. For each person, each frame records the observed model track ID or explicit no-detection state; ambiguity and occlusion are explicit.
3. A second review resolves or marks disagreements, especially crossings.
4. ID switches, fragments, misses and recoveries are counted from that ledger with definitions and denominators; no claim relies on raw track-ID totals.
5. Any link to physical metrics uses a real reviewed association reference, not `track_rev_{id}` generated from the tracker output.
