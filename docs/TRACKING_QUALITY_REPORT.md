# Tracking quality: four-minute stationary panorama

Updated 4 November 2026. This is development evidence, not an accepted accuracy claim.

## What was measured

The earlier MobileNet run and the native-resolution ResNet run were compared on
the same 1,440 frames, sampled at 6 Hz across all 240 seconds of recording 118577.
The 24-FPS normalized video and the exact 25-to-24-FPS frame correspondence are
hash-pinned. Both model architecture and inference resolution differ between
these pipelines; the comparison does not isolate the effect of architecture alone.

Reference: upstream SoccerTrack-v2 MOT annotations, 22 players/goalkeepers,
zero-based source frames. Referees and other people are omitted, so some valid
person detections may count as false positives. Local human completeness and
identity review remain outstanding. Labels were not fed into inference.

| Metric | Earlier saved MobileNet run | ResNet before tracking repair |
| --- | ---: | ---: |
| Player-reference detection AP50 | 19.64% | 84.29% |
| Detection precision, score >= 0.5 / IoU >= 0.5 | 44.18% | 72.43% |
| Detection recall, same thresholds | 22.95% | 93.27% |
| Detection spatial F1 | 30.21% | 81.54% |
| Tracked-box precision | 45.15% | 79.87% |
| Tracked-box recall | 16.29% | 60.26% |
| Provisional global identity F1 | 6.99% | 31.27% |
| Identity changes among spatial matches | 484 | 224 |
| Allocated temporary IDs | 434 | 24 |

The identity calculation uses global one-to-one identity-count matching at IoU
0.5. It is not TrackEval dataset preprocessing or the project's accepted,
human-reviewed-reference evaluation. No jersey or roster identity is verified.

## Why a video can still look worse

The detector finds most reference players, but the tracking stage loses much of
that coverage and switches identities. ResNet tracked-box recall by minute was
73.52%, 52.59%, 56.83%, then 58.08%. Better AP50 is not stable player identity.

The 24-ID lifetime counter exhausted early. The experimental worker also expired
old tracks after 30 seconds without recovering their session slots. This creates
capacity starvation. Generic ROI appearance is not a trained person ReID model;
same-kit players and crossings remain ambiguous. Smoothing measured boxes can
also move them away from the actual detector anchor. These are tracking and
association issues; additional detector training alone will not resolve them.

The separate trained MobileNet candidate's 84.85% AP50 on 570 SoccerNet validation
images is a different result. It must not be attached to this ResNet video or
treated as tracking accuracy.

## Repair under evaluation

- Tentative tracklets do not consume session IDs until repeated detections
  support them. The 24 non-recycled session IDs remain a separate constraint.
- Allocated session records survive short-lived track loss; conservative motion
  and appearance gates can reacquire them without deleting the identity record.
- Kalman footpoint/size velocity predicts motion. Measured anchor boxes remain
  exact; filtered state and covariance remain consistent for future prediction.
- SAM supports candidate births in one predeclared variant. Its proposals do not
  establish identity or become reviewed training labels.
- Compare two fixed capped recipes and a diagnostic ByteTrack control on cached
  detections. Score only after tracking completes. Preserve every result.
- No new detector inference, training, truth changes or automatic model promotion.

Baseten reviewed the proposed source. The agent rejected its incorrect syntax
claim and an invalid suggested edit. Its Kalman state/covariance concern informed
the reviewed fix. Suggestions about thresholds remain hypotheses until measured.

The owner approved one CPU Sandbox replay up to $0.50, exceptional retained Modal
ceiling $55.50 with all previous holds retained. Provider bills remain unreconciled.
Remote regression checks, replay outputs and the provisional comparison must be
reviewed before publishing the repair as an improvement. Neither passing tests
nor a fixed number of displayed IDs satisfies the 75/80% identity target.

## Reproduction boundary

Run `tools/run_modal_stationary_replay.py` from this branch with an explicit
absolute `--operator-root` containing the private settings and preserved artifacts,
plus `--execute-reviewed-replay` after approving a fresh bounded allowance.
The dated launcher is single-use and refuses reusing its claim/output directory.
It is not an unlimited service or permission to repeat a consumed trial.

The operator checkout needs the completed `runs/modal-panorama-resnet-20261104`
export, exact normalized video/frame map, SAM proposals and the unchanged upstream
reference. The code checkout is independently snapshotted and hashed. Its real
`.env` is not copied into Git or a worker. No local model execution is performed.

The provisional evaluator is separate from
`src/touchline/tracking_evaluation.py`; the latter's genuine review gate is unchanged.
See [the selected integration](SELECTIVE_INTEGRATION.md) and
[recording contract](RECORDING_TO_COACH_INTEGRATION.md).

## Rejected first replay and corrective trial

The first two fixed recipes were rejected: global identity F1 was 30.95% and
31.62%, with tracked-box recall only 23.16% and 23.68%. Lower switch counts alone
did not meet the coverage gate. The diagnostic ByteTrack control retained83.21%
coverage but had25.84% identity F1 and435 internal IDs; it was not selected.

A concrete bug let tentative duplicates compete with established session tracks
in the same Hungarian assignment stage. They could repeatedly take detections
away from the established ID. The correction prioritizes confirmed session tracks
for high/low confidence matching and conservative recovery before tentative
tracklets. A reproducing test covers this failure.

The first trial is preserved and its worker terminated. The owner approved a
separate corrected CPU trial up to$0.25,900 seconds, exceptional retained ceiling
$55.75. Old holds are retained; no new GPU inference or training is authorized.
Do not promote the rejected first candidate or label its preview an improvement.

## Final handover state

The priority-only candidate also failed:33.18% identity F1,44.95% tracked-box
recall. It is not promoted. A subsequent foreground-flow correction passed492
remote regression tests and artifact checks, but its full quality replay was
stopped at the owner's handover request. No completed quality score exists for
that source revision. The Sandbox terminated; partial logs/results were exported.
See [the root handover](../HANDOVER.md) for custody and continuation steps.
