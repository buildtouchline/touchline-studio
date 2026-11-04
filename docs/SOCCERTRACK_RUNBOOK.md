# SoccerTrack intake and current model trial

Current delivery update4 November: the verified PXL demo is180 seconds,
1080p24FPS, rendered from preserved ResNet GPU observations. Full-video exports
did not complete; failed receipts remain failed. Six PXL SAM requests and a
reviewed Baseten source proposal are additional to the intake calls below.
Read [the current handover](HANDOVER.md) for the current model/cost boundaries.

Updated 3 November 2026. Model/data execution stays in Modal; the local terminal
brokers allowlisted files and privately loads the selected root .env.

## Verified model result

The owner-approved recipe f6a6052b5fd659046a1902fb55d4b0dbce4a71ded774be8632b0a97ec70752e3
completed 2,000 steps on T4 and full validation on the unchanged570 SoccerNet
images. AP50 increased61.1757%→84.8523% (+23.6766 percentage points), with
mAP50:95=38.9358% and AP75=30.1845%. This combined trial changed training scale,
LR and training duration; the gain cannot be attributed to one factor alone.
Final test is sealed. No production model was promoted and tracking accuracy
has not been measured by this detection evaluation.

Training used the existing600 reviewed training images, a frozen backbone,
720–1280 training resize, LR0.0005, weight decay0.0005, seed42 and a fresh
optimizer. Inference evaluation remained min1080/max4096. Original weights,
dataset and evaluation metric were preserved. 438 remote tests and artifact
checks passed before training; checkpoint export preceded validation, and the
T4 Sandbox terminated. The training step cap is expected, not a failed job.

Private results:

- runs/modal-resize-training-20261103/training-receipt.json
- runs/modal-resize-training-20261103/candidate/candidate.pt
- runs/modal-resize-training-20261103/candidate/recipe.json
- runs/modal-resize-training-20261103/candidate-val/evaluation.json
- runs/modal-resize-training-20261103/gpu-results.zip

Checkpoint SHA256:6340f81877408e693dcd4734d84214c29095011a184e4c8187a98139ef3b7d80.
Use --checkpoint with this file explicitly for a provisional analysis trial;
only the human changes root .env settings. The MobileNet architecture/class map
is unchanged. The new reviewed-resize metadata is accepted without changing
old v1 checkpoint loading or default training recipes.

## Source selection and custody

[SoccerTrack-v2](https://github.com/AtomScott/SoccerTrack-v2) distributes a
[gated Hugging Face dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2).
The owner accepted access and provided HF_TOKEN privately. Code is MIT; data
uses [CC BY4.0](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA).
Attribute Atom Scott and coauthors, cite the linked paper, retain the licence
and disclose extracted-frame/overlay transformations when sharing.

Pinned HF revision:325bd4c57ee8e135ab652f3921d274526ada496b (releasev1.2).
artifacts/soccertrack-v2/source-metadata.json pins sizes/file hashes; the trusted
broker strips Authorization on cross-host redirects. No HF/Meta/Modal key enters
a worker or public source archive. Tokens, signed download URLs and raw HTTP
errors are never logged. Original data is not executed as code.

The20 full-match half videos total82.68GB, excluding their large GSR JSON files.
The first intake downloaded four curated4-minute MOT clips, about1.45GB total:
117092,117093,118575,118577. The already present118577 file was reused only after
matching the pinned LFS hash. Three clips were audited remotely:118575,118577
and117093;24 exact source frames were exported for review. Their annotations
contain132000,132000 and131472 rows respectively,22 IDs each, with no
out-of-image boxes. The un-audited117092 clip is not admitted merely by download.

## Why training admission remains blocked

The released MOT ground truth covers players and goalkeepers, omitting referees
and other people. The current detector class is person=1. Missing people would
become background during full-image training. Thus intake-review.json retains
person_labels_complete=false, training_eligible=false and pending human review.
SAM disagreements remain proposals, never ground truth or automatic eligibility.

MOT files have zero-based source frame IDs and score=-1 as the documented
annotation sentinel. Match them to the curated source clips at their native
25FPS. Do not feed these files to the SoccerNet parser, change the score to1 or
pair them with full-match halves. Analysis copies use24FPS separately; any
tracking reference conversion must follow the exact source-frame map.

117093 is withheld for validation; it is not training data. Future full-match
intake preserves both halves of each match in one split. Source train matches:
117092,118575,118576,118577,118578,128058; validation117093/132877; test128057/132831.
Do not pull source validation/test clips into training simply because labels are
public. Keep the original SoccerNet570-image validation and sealed test frozen.

Full-match GSR is a separate importer. The upstream
[format warning](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md)
documents structural and declared-dimension mismatches. Inspect real files,
decode actual dimensions, map bbox_image dictionaries and roles, and use
streaming JSON plus selected training-match frame windows. MOT row indices
cannot be reused for these long videos. Audit repeated scenes/camera overlap.

The sampled currentv1.2 GSR header for118575_1st.json declares4096x1080 correctly.
Its annotation version is1.3, image IDs are strings, filenames are one-based and
bbox_image is a dictionary. Only bounded head/tail fragments were inspected;
this is not a completed streaming importer or proof that all GSR assets match.
Apply upstream warnings conditionally after inspecting each actual asset.

## Operator steps

1. Human accepts HF conditions and puts HF_TOKEN in root .env. Never paste it
   into chat or pass it to workers. The trusted runtime alone loads it.
2. Run tools/fetch_soccertrack_intake.py for pinned labels/metadata, with --clips
   for the four allowlisted curated clips. Existing verified files are reused;
   mismatched files are rejected, not overwritten.
3. Current completed review: runs/modal-soccertrack-audit-20261103/review/index.html
   and intake-review.json. Inspect omitted referees, sideline people, occlusions,
   box alignment, completeness and exact frame pairing. Correct annotations;
   record reviewer identity and immutable correction provenance.
4. Freeze a new grouped release only after completeness/eligibility review.
   Mix approved source training images with existing data under a declared
   source-balanced recipe. Evaluate source-held-out and own-camera sets separately.
5. Rerun under fresh IDs/budgeted controllers. The dated Modal audit/training
   scripts are consumed trials, not reusable permissions or fresh-run templates.

## New PXL analysis

PXL_20261101_045045814~2.mp4 is analysis-only, never training data. Its source
hash is6210e20cd0cbabc1dc5c4f932be0db8c7fc3b7f9db3fb492f19d06ba4c2c0cdf,
3840×2160,742.8856 seconds, average29.8700FPS. It is normalized explicitly to
24FPS with original/source hashes and per-frame correspondence. Full detection
uses CPU,6Hz sampling, ByteTrack and the new6340f818 candidate, then exports
the complete overlay, gallery, profiles, JSON/JSONL and exact normalized source.
Calibration/reviewed identity/rubric remain required for physical metrics/scores.

The first attempt failed at input stage with ValueError; its exact root cause
was not established. Source metadata and local input hashes were independently
verified. The retry uses128MiB chunks, remote per-chunk hashes and full-input
hash checks. More precise static failure locations are recorded without raw
error disclosure. Original receipts and reservations remain retained.

The retry is runs/modal-pxl-analysis-20261103-retry. A same-worker controller
handoff to tools/adopt_modal_recording.py expands export bounds to24GiB, with
disk-headroom, archive-hash and traversal checks. It creates no new Sandbox,
changes no worker model/input, and preserves the10800-second lifetime and$4 cap.
Status is authoritative only in its job-receipt.json; do not call an in-progress
run complete or share its partial archive.

Budget: conservative previous Modal upper bound$35 plus first PXL$5,
reviewed T4$5, intake$1 and retry$4 reaches the existing$50 ceiling in retained
reservations, not known billed charges. No additional Modal jobs until billing
is reconciled or the owner revises the ceiling. SAM intake review separately
reserves$2 within the existing$75 Meta ceiling; zero returned text tokens do
not establish zero image charges. Baseten/Fireworks were not used in this trial.

Read [the handover](HANDOVER.md) and [repair rulebook](AUTOFIX_RULEBOOK.md).
