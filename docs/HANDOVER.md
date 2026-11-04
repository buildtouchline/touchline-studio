# Touchline current handover

**Latest stop/handover:** read [the root handover](../HANDOVER.md).
The latest tracking experiment was stopped on owner request;492 remote tests
passed, but identity quality is not accepted. Both tested repairs failed coverage.
The final flow-recovery change has no completed quality result. Worker termination
is confirmed in the private handover-stop receipt.

Selective integration branch: `feature/player-mapping-modal-integration`.
Read [the component comparison and operating boundaries](SELECTIVE_INTEGRATION.md).
Modal Sandboxes, existing SAM brokers and guarded Baseten repairs remain primary.
The provider baseline and dated evidence below remain intact.

Updated 4 November 2026. Code belongs on feature/modal-provider-pipeline;
the tickets branch owns requirements and status. Existing PRs, main and develop
are unchanged. All model/media execution and checks run in Modal. Local terminals
author source, stage approved inputs and retrieve results.

## Delivered evidence

- Real PXL ResNet demo exported:180 seconds,4320 frames,1920x1080 at24FPS.
  Video SHA256 d461153ef4fd6d307be62a0d3c46a4e1e9b616e9a96af8b6d5e55b1fcc083587.
  It renders preserved T4 results with adaptive3-6Hz detection and
  min720/max1280 inference. IDs may fragment/switch; no PXL AP50/IDF1.
- Separate reviewed MobileNet training completed2000 steps on600 reviewed
  images and unchanged570-image SoccerNet validation. AP50 improved61.1757%
  to84.8523%; mAP50:95=38.9358%, AP75=30.1845%. Final test is sealed. This result
  does not describe ResNet, PXL tracking or generalization to every camera.
- Official ResNet50 FPN v2 weights were compared on matched frames, including
  beginning/middle/end. The owner selected ResNet for analysis.
- Four pinned SoccerTrack clips downloaded; three audited with24 review frames.
  Missing referees/other people block admission to all-person training.
- Eight SoccerTrack and six PXL Meta SAM3.1 calls completed. Proposals remain
  unreviewed training candidates, not truth or confirmed identities.
- Baseten returned one valid source proposal after two truncated outputs. The
  agent adjusted proposed border clamping into explicit invalid-flow rejection.
  Relevant remote checks passed; no unrestricted unattended repair service.

Publication verification in Modal passed461 tests, syntax checks for209 Python
files and28 notebook code cells, plus132 local code-document links. Ticket
validation passed49 parents/115 subtasks,1572 local links, matching CSV/JSON
and an acyclic dependency graph. The check worker terminated. A result-counter
regex typo was corrected from the saved test log; the actual test suite passed.
Unrun notebook cells are not execution evidence.

## Artifact custody and failed runs

Weights, footage, generated media, run databases and real credentials are not
committed. The operator share folder contains the self-contained demo MP4,
gallery, contact sheet and model/run metadata. Its render worker terminated.

The full PXL CPU detector pass traversed17830 normalized frames, but its large
full-video export did not reach a verified local archive before worker expiry.
Long ResNet attempts also did not yield an accepted full recording. Keep those
receipts failed; the bounded demo is a separate successful artifact. Exit137
alone does not establish GPU OOM, corrupt input or the root cause. The unused
L4 allocation was cancelled and termination confirmed.

The prepared filesystem image had a one-day TTL; it is not permanent shared
storage. Transfer required private artifacts before expiry. Never transfer
the real.env, tokens or credential profiles.

## Models and source contracts

MobileNet candidate SHA256:
6340f81877408e693dcd4734d84214c29095011a184e4c8187a98139ef3b7d80.
It needs the reviewed-resize metadata support in this branch.

ResNet official COCO weights SHA256:
dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf.
Use the ResNet adapter; do not load these into MobileNet. YOLO is excluded.

Preserve geometry, exact24FPS source timing, original/normalized hashes and
frame correspondence. Detector cadence is separately declared. Held/predicted
boxes are not detector evidence. Track IDs are not jersey numbers; pixels are
not calibrated physical distance.

## Before frontend integration

Read [the integration agreement](RECORDING_TO_COACH_INTEGRATION.md).
There is no verified track_video_gpu.remote interface. The dated full-recording
scripts are operator tools, not a deployed asynchronous web API. The prepared
image ASGI service is separate and undeployed. MOT text and calibrated minimap
outputs are not verified contracts of the selected worker.

Experimental ResNet rows need a canonical adapter before associate/player-analysis
can consume them. Verify one real recording, exported evidence, genuine coach
identity correction, recomputed review and durable backend retrieval together.
Calibration/rubric/context gates remain required for metrics and approved scores.

## Next work

1. Publish synchronized code and tickets before new improvements.
2. Execute TL-030.6 on the four-minute stationary panorama118577. Compare with
   the saved5760-frame baseline, which had434 unresolved temporary tracks.
3. Inspect misses, false boxes, duplicates, crossings and unassigned people in
   beginning/middle/end. Use appropriately scoped truth only after inference;
   source player-only labels omit some valid person detections.
4. Complete canonical coach integration, persistent artifacts and reconnectable
   supervision. Lost stdout must not by itself discard a live job.
5. Admit corrected SoccerTrack releases and measure separate held-out AP50 and
   IDF1 before promoting another trained model.

The owner approved a new up-to$2 panoramic trial, exceptional retained Modal
ceiling$55 including old holds. Earlier$3 PXL attempts are consumed; retain
their receipts/holds. Provider bills remain unreconciled. Default global caps
are unchanged; dated trial scripts do not grant reusable spend.

## Operational references

- [R-CNN review and boundaries](RCNN_VIDEO_REVIEW.md).
- [SoccerTrack admission](SOCCERTRACK_RUNBOOK.md).
- [Meta usage and demo script](PRODUCT_DEMO_AND_META_MODELS.md).
- [Local terminal setup](LOCAL_TERMINAL_HANDOVER.md): installation and older
  baseline launcher remain useful; older result sections are historical.
- [Repair rulebook](AUTOFIX_RULEBOOK.md).
- [Authoritative ticket pin](TICKET_REFERENCE.md).
