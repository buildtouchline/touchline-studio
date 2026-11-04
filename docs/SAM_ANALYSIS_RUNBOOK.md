# SAM 3.1 second-pass video analysis

SAM is now used for analysis, not only labelling: a bounded Meta broker reviews
eight exact frames from the trained-candidate panorama. Torchvision/ByteTrack
remain the detector/tracker. The independent SAM proposals flag disagreements
for review; they do not automatically become detections, identities or training
labels. No approved model was changed.

## Actual 3 November evidence

- Eight new SAM 3.1 requests returned HTTP 200, with 21–24 proposals per frame.
  The earlier four labelling calls are separate historical evidence.
- Frames are 0, 60, 120, 180, 240, 300, 360 and 420 of the exact 24 FPS source.
  Full source geometry is 4096×1080. Same checkpoint and frames in both comparisons.
- At detector max-size 1920: 1743 observations over the capped 120 samples,
  86 SAM-only disagreement candidates across the eight snapshots.
- At max-size 4096 (native input geometry): 2070 observations and 37 SAM-only
  disagreements. Detector CPU time rose from 20.64 to 63.74 seconds.
- These are coverage/disagreement diagnostics, not precision, recall, AP50 or
  identity accuracy. SAM also produces false positives; inspect the gallery.
- 429 tests and artifact checks passed remotely before dispatch. The native
  inference/render path then completed in another Modal CPU Sandbox. Both stopped.

## View and reproduce

The broker entry point is tools/run_modal_sam_analysis.py with explicit
--approved-meta-analysis. It freezes source/geometry/proposals, runs checks in a
no-secret/no-egress CPU Sandbox, exports JPEG frames, calls Meta from the trusted
controller, and returns sanitized geometry for comparison/rendering in Modal.
Its single-use claims reserve Meta $2 and Modal $1 within cumulative $75/$50 caps.
Those dated claims are consumed; retain previous holds and receipts.

Private output: runs/modal-sam-analysis-20261103/review/index.html and sam-review.zip.
sam-analysis.json contains request status, response IDs and the API's usage fields.
All eight successful responses report zero input/output/total tokens. This is not
evidence of zero dollar cost; credit/billing reconciliation remains open. No raw
headers, response masks or credentials are stored in review artifacts.

The native comparison uses tools/run_modal_candidate_analysis.py with
--approved-existing-cpu-cap --native. It reuses the already returned SAM geometry;
it does not bill Meta again for identical frames. Its results are under
runs/modal-native-analysis-20261103/: analysis-results.zip, preview/overlay24.mp4,
review/index.html and analysis-receipt.json. Green boxes are detector proposals
with score≥0.5; orange boxes are unmatched SAM proposals at one-to-one IoU≥0.5.

Only the trusted broker loads META_API_KEY from the selected private root .env.
Workers receive no provider keys and use CPU; GPUs remain reserved for training.
API route: POST https://api.meta.ai/v1/responses, model sam-3.1. No new .env key
names are required. Authentication failure, malformed output or timeout stops the
bounded pass without rewriting the detector or automatically retrying charges.

## Next acceptance

Have a reviewer label missed/false boxes and reference identities on the exact
frames, then evaluate a declared inference recipe against unchanged held-out
truth. Test the native setting across the full recording before changing a
default. SAM-only disagreements cannot certify 70% AP50 or tracking quality.
Training on accepted hard examples requires a new reviewed grouped data release.
