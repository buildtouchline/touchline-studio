# R-CNN video quality review

Current delivery4 November2026: a complete180-second PXL demo was exported at
1920x1080/24FPS from preserved GPU ResNet observations. Video SHA256
d461153ef4fd6d307be62a0d3c46a4e1e9b616e9a96af8b6d5e55b1fcc083587.
Cached inference used min720/max1280 and adaptive3-6Hz; all4320 frames rendered.
IDs still fragment/switch; no own-camera AP50/IDF1 acceptance. The render worker
terminated and the unused L4 allocation was cancelled.

The full native-resolution attempts below were experiments, not accepted full
recordings. The T4 replacement failed; exit137 did not establish a root cause.
Atomic worker status snapshots are prepared for reconnectable control; their
end-to-end provider acceptance remains open. Next is TL-030.6: one approved$2
four-minute panorama trial after publication, retained Modal ceiling$55.

Updated 3 November 2026. All inference and checks run in Modal. The local terminal
only downloads public weights, stages allowlisted inputs and retrieves artifacts.

## Baseline and comparison

The baseline remains the reviewed-resize Faster R-CNN MobileNetV3-FPN checkpoint
with SHA256 `6340f81877408e693dcd4734d84214c29095011a184e4c8187a98139ef3b7d80`.
Its 84.85% AP50 result is specific to 570 SoccerNet validation images. PXL player
identity accuracy remains unmeasured. The experimental motion tracker has not
passed visual acceptance and must not replace baseline outputs.

The separate comparison uses Torchvision 0.26 Faster R-CNN ResNet50-FPN-v2,
official COCO_V1 weights, SHA256
`dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`.
These weights have not been trained on our football dataset. No YOLO inference,
training or package installation is part of this selected approach.

This is our R-CNN Sandbox template, based on the official Torchvision builder
inside the existing compatible Modal image. A ready-made Modal R-CNN example was
not verified. No external repository code is executed.

## Paired comparison procedure

1. Preserve the original video, normalized 24 FPS copy, frame map and completed
   baseline raw proposals.
2. Retrieve official weights with `tools/fetch_rcnn_comparison_weights.py`.
   This performs no local model execution and pins the full file hash.
3. With the existing recording worker still live, explicitly invoke
   `tools/run_rcnn_comparison_existing.py --existing-worker-only`.
   It creates no new worker and does not extend its original lifetime or budget.
4. `tools/modal_rcnn_comparison_worker.py` checks hashes, loads weights with
   `weights_only=True`, and compares eight exact frames in the first twenty
   seconds. Both detectors use min1080/max4096; displayed boxes use the same
   score threshold 0.5 and source-pixel coordinates.
5. Inspect `review.html`, the paired JPEGs, `proposals.jsonl` and `receipt.json`
   under the local comparison run. Left is the trained MobileNet baseline;
   right is the generic COCO ResNet50 candidate.
6. Check missed near/far players, false boxes on equipment/netting and duplicate
   boxes. More detections do not establish higher accuracy. Review actual people
   before changing thresholds or selecting a model.

The comparison receipt records actual device, weights/source hashes, input
transform, runtime versions and frame count. AP50/IDF1 remain null because these
unlabelled frames are not an independent labelled evaluation.

## Tracking acceptance

Use 24 FPS point/motion updates with explicit detector anchors. Predicted boxes
must remain visibly distinguishable from detections. Do not freeze a border on
screen when the player moves, force an ID across an ambiguous gap, or treat a
24-ID capacity limit as evidence of accuracy. The requested limit is at most
24 allocated session IDs; unmatched excess detections must remain unassigned.

Before selecting a tracking change, inspect crossings, angle changes, overlap,
re-entry and people outside the field on the same clip. Keep the previous output
and record the model/tracker configuration for every attempt. Measure AP50 and
IDF1 separately on reviewed held-out truth; neither is inferred from a gallery.

## Current operational limits

CPU analysis remains selected; GPUs are for reviewed training. The active worker
has 4 CPU cores, 8 GiB RAM, no network/provider keys and an original 10800-second
lifetime. The $4 recording reservation includes these auxiliary checks and
exports. Do not create another worker or extend the lifetime without reconciling
the retained cumulative Modal budget. A separate controller retrieves results
before terminating the recording Sandbox.

## Owner selected ResNet and one GPU inference exception

The owner subsequently selected ResNet50 for analysis and approved one T4
full-video inference trial, up to $3 additional Modal spend. Retained reservations
rise from $50 to $53 for this single claim; original holds remain intact and the
global default spending policy is unchanged. The separate authorization claim
is stored in `runs/rcnn-gpu-owner-exception-20261103.sqlite3`.

`tools/run_modal_rcnn_full.py --owner-approved-t4-exception` is the consumed
single-use launcher. It snapshots the existing prepared filesystem with a
one-day TTL, creates one T4 Sandbox in main, and stages source-only changes.
No second multi-GB video upload, provider credential injection, network access,
training or serving deployment is part of the trial. Bounds:1800 seconds,
4 CPU cores,8 GiB RAM,4 GiB result archive. No automatic retry/fallback.

`tools/modal_rcnn_gpu_worker.py` verifies normalized-video, model and code hashes,
runs fixed checks, verifies CUDA NMS/model placement, and analyzes all17830 frames
with6Hz R-CNN anchors. Generic nonnegative ROI descriptors and color assist
association; these features are not a trained person-ReID model. Automatic grass
scope is experimental; rejected proposals are retained for review. IDs are
capped at24 and extras remain unassigned. Selected SAM anchors inform births;
they do not become reviewed training labels.

The full output renders at1920x1080 for review, with source-pixel coordinates
kept at3840x2160 in JSON/CSV.24FPS display includes explicitly tagged held
estimates between6Hz detections. It is not24FPS detector inference. AP50/IDF1
remain unknown for this unlabelled recording.

Before accepting the output, inspect the gallery, beginning/middle/end,
unassigned counts, camera movement and any pitch-scope exclusions. The owner
selected the architecture; that does not establish measured tracking accuracy.

References: [Torchvision R-CNN source](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/detection/faster_rcnn.py)
and [Modal Sandboxes](https://modal.com/docs/guide/sandboxes).
