# Touchline Level 3 validation handover · 29 October 2026

**Start with [ROADMAP_HANDOVER.md](ROADMAP_HANDOVER.md).** Then read the three supporting reports: [detection against the curated MOT reference](EXTERNAL_MOT_REPORT.md), [tracking continuity](TRACKING_REVIEW.md), and [reproduction instructions](REPRODUCE.md). [The GSR audit](EXTERNAL_GSR_AUDIT.md) and [physical-metrics prerequisite plan](PHYSICAL_METRICS_PLAN.md) provide additional context.

This is a documentation and small-evidence bundle based on frozen local experiments. It is independent of the open implementation PR #11 and does not approve or change its detector, Studio, tracking, or metric behavior. The branch is based on `develop` at `0d7aeed4682df8b09ef549557e60a9ddd6f06d18`. The experiments used a local checkout at `bd761591b5ace13f3d8ccfac8f261b7569de8c48` derived from PR #11, which has not merged into `develop`.

## What the evidence says

| Question | Current result | Boundary |
|---|---|---|
| Player/GK detection at IoU 0.50 | Default **16/264 (6.06% recall)**; higher resolution **112/264 (42.42% recall)** | Same 12 frozen frames from the curated SoccerTrack MOT reference; 22 on-field player/GK labels per frame. Referees and other humans are outside its scope. This does not establish all-person precision or full-video accuracy. |
| Small players | Exploratory under-50-pixel reference boxes: 0/138 default, 28/138 higher | Post-hoc size split on the same 12 frames; higher resolution still missed 110/138. |
| Tracking | 65/660 reference boxes matched saved observations over 30 sampled frames; six adjacent sampled-frame ID changes | Preliminary external diagnostic with low detection coverage, limited visual review, and no IDF1/HOTA claim. |
| Physical workload metrics | **Unvalidated** | Footage lacks accepted calibration and independently measured movement reference; real-run outputs suppress physical values. |
| Higher-resolution setting | Candidate for further investigation | The measured adapter-equivalent CPU time was 2.77× default; no production setting change is approved. |

The [full 24-frame all-person score marker](metrics.json) and [five-frame batch marker](metrics_batch.json) both remain `blocked_unreviewed_ground_truth`: **zero human-reviewed frames**. The external MOT result does not fill that gap. See the [official MOT README](https://huggingface.co/datasets/atomscott/soccertrack-v2/blob/main/mot/README.md) for the curated label scope and the [official GSR format guide](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md) for why GSR projected boxes are not image-space detection ground truth.

## Included evidence

- [Detection result](external_mot_detection_metrics.json) and [tracking diagnostic](external_mot_tracking_0_5s.json): aggregate and per-frame counts, matching rules, source hashes, and preliminary status. These contain no source images or bounding-box coordinates.
- [Timing summary](timings.json): original 48 timed calls per configuration and CPU environment; it is not a full-video or live throughput test.
- [Sanitized frame manifest](frame_manifest.json): 24 frozen frame identities, source indices/PTS, dimensions and hashes without local filesystem paths or images.
- [Annotation batch](annotation_batch.json) and blocked score markers: define the next human-review task without presenting unreviewed boxes as truth.
- [verify_results.py](verify_results.py): standard-library verifier for published arithmetic, plus optional raw recomputation when an authorized reviewer has the gated MOT file and saved local prediction/observation JSON.

The raw match footage, frozen frame images, GSR JSON, curated MOT rows/boxes, model checkpoint, full prediction boxes, run observations, tracking overlays, credentials, and logs are **excluded**. They remain with authorized local custodians. The source experiment was under temporary `/private/tmp` storage when this bundle was prepared; this PR is not a backup of those gated assets. The source dataset uses a [CC BY 4.0 data licence](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA) and gated access; publication or demo use of its footage needs separate review of those conditions.

## Backend review sequence

1. Read the roadmap and inspect the machine-readable counts with `python3 -B docs/validation/level3-2026-10-29/verify_results.py`.
2. Have an authorized reviewer check all 12 MOT overlays against the paired clip and pin its exact dataset revision/remote hashes. Three overlays were visually spot-checked locally; the remaining alignment review is open.
3. Complete the five-frame **all-visible-person** human annotation batch, then extend to 24 frames. Measure TP/FP/FN, precision and recall on those reviewed labels before choosing a detector setting.
4. Review tracking changes against continuous reference identities. Resolve the separate identity-review reference issue before describing any player-specific metric as independently reviewed.
5. Gather surveyed calibration, held-out pitch checks and an independent movement reference before validating distance, speed, intensity or sprint claims.

No inference, training, deployment, or physical-metric field validation is performed by this PR.
