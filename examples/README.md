# Dataset configuration

Copy templates outside versioned source data, fill actual paths/metadata, and change attestation flags only after review. Relative paths resolve against the config file's folder. Never check in raw footage, private data or tokens.

`etl-config.template.json`: choose inputs `coco_boxes`, `yolo_boxes`, `xyxy_boxes` or `csv_boxes`. The template shows YOLO; COCO uses `annotations` instead of labels/classes. Corner TXT needs explicit `coordinate_units`. CSV requires an `annotations` file and `columns` mapping for image,class,x1,y1,x2,y2 plus units. Only CSV-represented images are imported; use a separate reviewed format for negative frames. Unknown JSON, masks, polygons and pose need different adapters.

The ordered source class list is not an example to copy blindly. Map all annotated humans to person=1, explicitly ignore ball if applicable, and fail on unknown classes. `capture_groups` maps every `input-id/relative-image-path` to its true session/match. At least three independent groups are required; never create a fake group per image. Optional `split_assignments` maps every group to train/val/test, with all three represented. Otherwise the current recipe allocates groups deterministically.

`soccernet-config.template.json`: describes one extracted sequence per entry. Inspect actual metadata and fill every observed track's source class; `track_classes` is keyed by that sequence's track ID. The template is deliberately incomplete and cannot train. Add independent groups across all three splits. Use official training matches for internal training/validation; protect test/challenge. A `frame_stride` of 12 reduces redundant frames for the initial training subset; it does not imply a 24 FPS source. No frame rate or video timestamp is invented.

Set `bbox_origin` to integer 0 or 1 after verifying the source pixel coordinate convention; null intentionally fails. A one-based origin is shifted to zero-based image pixels before COCO staging. Do not confuse one-based frame filenames with the pixel coordinate origin.

`failure-cases.template.json`: template for recording human-labelled player bounding boxes to diagnose pipeline missed detections via `touchline audit-run` (TL-030.1).
- Core schema: a JSON list of annotation dictionaries. Requires `source_id` matching the run manifest source identity, at least one of `pts` (presentation timestamp integer) or `timestamp_s` (seconds relative to source origin), and `bbox_xyxy` (`[x1, y1, x2, y2]`). Optional `frame_index` records presentation order; optional `label_id` assigns a readable label name.
- Pixel coordinates: `bbox_xyxy` must be in zero-origin source image pixels. Pixel box size is NOT physical distance.
- Annotation metadata: optional `annotation_metadata` separates review context (`target_category`, `near_far`, `occlusion`, `motion_blur`, `edge_truncation`, `crossing`, `visibility_state`, `label_uncertainty`) from the core audit fields.
- Controls: include positive controls (`successful_control`) to verify detector baseline. Empty background frames with zero people are explicitly recorded with `bbox_xyxy: null` and `target_category: "empty_background_control"` to verify absence of false-positive observations.
- All entries in the template are placeholders (`REPLACE-...`, `PLACEHOLDER-...`) and do not represent real validation data.

Example invocations:

```text
touchline stage-soccernet /data/my-soccernet-config.json --output /data/staging-v1
touchline etl /data/staging-v1/etl-config.json --output /data/release-v1
touchline validate-dataset /data/release-v1/release.json --training-gate
touchline audit-run /runs/run-01 /labels/failure-cases.json --output /reports/audit-ledger.json
```
