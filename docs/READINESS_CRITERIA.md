# TL-010.1 — Proposed field-readiness criteria

Status: **proposed for A/B/coach review; no group is cleared**. Source requirements:
[TL-010.1](https://github.com/buildtouchline/touchline/blob/067b50f0422254f7a574fede9be6168c531e00ac/docs/tickets/subtasks/TL-010.1.md),
[TL-S06](https://github.com/buildtouchline/touchline/blob/067b50f0422254f7a574fede9be6168c531e00ac/docs/specs/TL-S06-validation-and-rollout.md),
and [TL-S07](https://github.com/buildtouchline/touchline/blob/067b50f0422254f7a574fede9be6168c531e00ac/docs/specs/TL-S07-field-pilot.md).

These numbers are review proposals, not measured results or claims about current performance.
The paired feature PR provides a seven-group JSON template and a fail-closed reader. Its
template remains `proposed` with no outcomes. A/B and the coach must check reference
feasibility, hardware limits, and the definitions below before marking a criterion frozen.
The reviewer records an identity and UTC freeze time before examining held-out results.
The frozen criterion also records the evaluation-set ID, lowercase SHA-256 source
and label hashes, capture-group IDs and `held_out` split role. The outcome must repeat the same set ID and
hashes; a later outcome cannot define or swap the held-out set.

| Group | Proposed metric and threshold | Independent reference | Current outcome |
| --- | --- | --- | --- |
| Level 1 identity and visibility | Usable target coverage ≥ 0.90 | Human-labelled visible intervals, identity breaks, off-camera and uncertain intervals | Pending |
| Level 1 image-space review | 90th percentile bottom-centre error ≤ 0.05 of source-frame diagonal | Independent source-pixel marks on a held-out fixed-camera drill | Pending |
| Level 1 attempt review | Evidence-link rate = 1.00 | Coach-marked attempts checked against source intervals and reviewer records | Pending |
| Level 2 distance | Relative absolute error ≤ 0.10 | Surveyed course length, valid calibration, confirmed identity and synchronized source | Pending |
| Level 2 average and maximum speed | Each relative absolute error ≤ 0.15 | Independent synchronized course/timing reference; maximum uses the declared 1-second window | Pending |
| Level 2 intensity zones | Maximum absolute per-zone time error ≤ 2 seconds in a 30-second drill | Independent synchronized timing reference and declared speed-band boundaries | Pending |
| Level 2 activity | Macro F1 ≥ 0.80 | Independent, blinded activity interval labels with a frozen category map | Pending |

## Measurement rules to freeze with the thresholds

- Freeze the source/session IDs, capture groups, camera setup, labelled intervals, metric definitions, match tolerances, and evaluation set before looking at held-out outcomes. Keep training, validation, and untouched final evaluation separate. Record source, code, model, calibration, criteria and result hashes.
- For usable coverage, divide correctly associated, source-linked target time by independently labelled *visible* target time. Report labelled visible seconds, usable seconds, unknown/off-camera seconds, lost intervals and identity switches separately. Unknown is never inactivity or zero movement.
- For image-space error, match observations to independent bottom-centre marks at the same source PTS. Divide Euclidean pixel error by the diagonal of that source frame; report the support count and unmatched marks. This is image-space review, not a claim of pitch-distance accuracy.
- For attempt evidence, the numerator is reviewed attempt records with a source interval, reviewer, outcome and objective reference; the denominator is all coach-marked attempts in the declared drill. Pending/inconclusive attempts remain distinct from error. No conclusive attempts yields an unavailable success rate.
- For distance and both speed measures, suppress a result when calibration, identity, coverage, synchronized reference or a complete 1-second peak window is missing. Compute relative absolute error only when the reference value is positive; report absolute error and unavailable cases separately. The average and maximum speed must both pass to clear their one group.
- For intensity zones, freeze band edges and inclusion rules before evaluation. Compare eligible seconds per band with synchronized reference intervals; report each band, coverage and excluded seconds. For activity, freeze categories and report per-class support and confusion as well as macro F1; missing intervals cannot be treated as standing.
- Record the actual device, runtime, total and stage time, peak RAM/VRAM, disk use, frame count, false positives, missed detections, and failed or partial runs. The hardware and turnaround budgets are **unset** until the field device and owner-approved limits are recorded. Individual metric groups can have derived results, but the overall field-readiness status cannot clear while the resource budget is unset or failed.
- Store each group's implementation and test references, independent reference evidence, frozen criterion, evaluation outcome and limitations. A reviewer must verify that references are genuinely independent and that provenance links resolve. A syntactically complete record alone is not scientific validation.

## Review decision

The accountable C stream prepares the scorecard. Stream B reviews measurement and
hardware feasibility; stream A reviews model/data provenance; the coach reviews the
Level 1 journey and reference practicality. Record their decisions and any threshold
changes before running the held-out field evaluation. Until then, the paired template
and every group remain pending.
