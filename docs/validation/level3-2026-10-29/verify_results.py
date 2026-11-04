"""Verify published aggregates; optionally recompute them from authorized local inputs.

Standard-library only. This never downloads data, runs inference, or writes artifacts.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    if not intersection:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def match(gt, predicted):
    """Maximum-cardinality one-to-one matching with descending-IoU candidate order."""
    edges = []
    for box in predicted:
        candidates = [(overlap(box, truth), index) for index, truth in enumerate(gt)]
        edges.append([index for score, index in sorted(candidates, reverse=True) if score >= 0.5])
    assigned = {}

    def augment(prediction_index, seen):
        for truth_index in edges[prediction_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            if truth_index not in assigned or augment(assigned[truth_index], seen):
                assigned[truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predicted)):
        augment(prediction_index, set())
    return sorted(assigned.items())


def verify_bundle():
    manifest = read("frame_manifest.json")
    detection = read("external_mot_detection_metrics.json")
    tracking = read("external_mot_tracking_0_5s.json")
    timings = read("timings.json")
    batch = read("metrics_batch.json")
    full = read("metrics.json")
    frames = manifest["frames"]
    assert len(frames) == 24 and len({frame["id"] for frame in frames}) == 24
    assert all("video_path" not in frame and "image_path" not in frame for frame in frames)
    assert len(detection["frame_ids"]) == 12
    assert detection["frame_ids"] == [frame["id"] for frame in frames if frame["video"] == "118577.mp4"]
    assert detection["source_video_sha256"] == tracking["source_video_sha256"]
    assert detection["mot_file_sha256"] == tracking["mot_file_sha256"]
    assert detection["status"].startswith("preliminary_") and tracking["status"].startswith("preliminary_")
    for setting, expected_tp, expected_fn in (("default", 16, 248), ("higher_resolution", 112, 152)):
        entry = detection[setting]
        rows = entry["per_frame"]
        summary = entry["summary"]
        assert [row["frame_id"] for row in rows] == detection["frame_ids"]
        assert all(row["gt_count"] == 22 for row in rows)
        for key in ("tp", "fp", "fn", "gt_count", "prediction_count", "ignored_predictions"):
            if key in summary:
                assert summary[key] == sum(row[key] for row in rows)
        assert summary["tp"] == expected_tp and summary["fn"] == expected_fn
        assert summary["tp"] + summary["fn"] == 264
        assert abs(summary["recall"] - expected_tp / 264) < 1e-12
        assert summary["mean_iou_denominator_matches"] == expected_tp
        bins = entry["exploratory_gt_box_height_bins"]
        assert sum(item["gt"] for item in bins.values()) == 264
        assert sum(item["matched"] for item in bins.values()) == expected_tp
    assert tracking["samples"] == len(tracking["per_frame"]) == 30
    assert tracking["gt_boxes"] == sum(row["gt_player_gk_count"] for row in tracking["per_frame"]) == 660
    assert tracking["saved_observations"] == sum(row["observation_count"] for row in tracking["per_frame"]) == 191
    assert tracking["matched_boxes"] == sum(row["matched_count"] for row in tracking["per_frame"]) == 65
    assert tracking["adjacent_switch_count"] == len(tracking["adjacent_switches"]) == 6
    assert tracking["gap_changed_id_count"] == len(tracking["gap_changed_ids"]) == 5
    assert full["status"] == batch["status"] == "blocked_unreviewed_ground_truth"
    assert full["reviewed_frames"] == batch["reviewed_frames"] == 0
    assert full["total_frames"] == 24 and batch["total_frames"] == 5
    ratio = timings["higher_resolution"]["summary"]["mean_s"] / timings["default"]["summary"]["mean_s"]
    assert abs(ratio - timings["comparison"]["mean_latency_ratio_higher_over_default"]) < 1e-10
    print("PASS bundle: 24 frame identities; detection 16/264 and 112/264; tracking 65/660, 6 adjacent ID changes; manual labels blocked; timing ratio consistent")
    return manifest, detection, tracking


def read_mot(path, indices, manifest_by_index):
    rows = defaultdict(list)
    for number, line in enumerate(path.open(), 1):
        columns = line.strip().split(",")
        if len(columns) != 10:
            raise ValueError(f"Invalid MOT row {number}")
        index = int(columns[0])
        if index not in indices:
            continue
        identity = int(columns[1])
        x, y, width, height = map(float, columns[2:6])
        frame = manifest_by_index.get(index, next(iter(manifest_by_index.values())))
        if not (width > 0 and height > 0 and 0 <= x < x + width <= frame["width"] and
                0 <= y < y + height <= frame["height"]):
            raise ValueError(f"Invalid MOT box on row {number}")
        rows[index].append((identity, [x, y, x + width, y + height]))
    assert set(rows) == set(indices)
    assert all(len(rows[index]) == 22 and len({identity for identity, _ in rows[index]}) == 22 for index in indices)
    return rows


def recompute(mot_path, default_path, higher_path, observations_path, manifest, detection, tracking):
    assert sha256(mot_path) == detection["mot_file_sha256"], "MOT file hash mismatch"
    match_frames = [frame for frame in manifest["frames"] if frame["video"] == "118577.mp4"]
    assert len({(frame["width"], frame["height"]) for frame in match_frames}) == 1
    by_index = {frame["source_frame_index"]: frame for frame in match_frames}
    track_indices = [row["source_frame_index"] for row in tracking["per_frame"]]
    mot_rows = read_mot(mot_path, set(by_index) | set(track_indices), by_index)
    for setting, path in (("default", default_path), ("higher_resolution", higher_path)):
        predictions = {row["frame_id"]: row for row in json.loads(path.read_text())["frames"]}
        totals = defaultdict(int)
        for frame in match_frames:
            row = predictions[frame["id"]]
            assert row["image_sha256"] == frame["image_sha256"]
            assert row["source_frame_index"] == frame["source_frame_index"]
            gt = [box for _, box in mot_rows[frame["source_frame_index"]]]
            pred = [item["bbox_xyxy"] for item in row["boxes"]]
            pairs = match(gt, pred)
            totals["tp"] += len(pairs)
            totals["fp"] += len(pred) - len(pairs)
            totals["fn"] += len(gt) - len(pairs)
        published = detection[setting]["summary"]
        assert all(totals[key] == published[key] for key in ("tp", "fp", "fn"))
        print(f"PASS raw detection {setting}: TP={totals['tp']} FP={totals['fp']} FN={totals['fn']}")
    if observations_path is None:
        print("SKIP raw tracking: --observations not supplied")
        return
    assert sha256(observations_path) == tracking["observations_sha256"]
    obs_by_index = defaultdict(list)
    for line in observations_path.open():
        row = json.loads(line)
        index = row["frame"]["index"]
        if index in track_indices:
            assert row["frame"]["source_id"] == tracking["source_video_sha256"]
            obs_by_index[index].append(row)
    sequence = defaultdict(list)
    matched = 0
    for position, index in enumerate(track_indices):
        gt = mot_rows[index]
        obs = obs_by_index[index]
        assert len(obs) == tracking["per_frame"][position]["observation_count"]
        pairs = match([box for _, box in gt], [row["bbox_xyxy"] for row in obs])
        assert len(pairs) == tracking["per_frame"][position]["matched_count"]
        matched += len(pairs)
        for gi, pi in pairs:
            sequence[gt[gi][0]].append((position, obs[pi]["local_track_id"]))
    adjacent = gap = 0
    for rows in sequence.values():
        for previous, current in zip(rows, rows[1:]):
            if previous[1] != current[1]:
                if current[0] == previous[0] + 1:
                    adjacent += 1
                else:
                    gap += 1
    assert (matched, len(sequence), adjacent, gap) == (
        tracking["matched_boxes"], tracking["matched_gt_identity_count"],
        tracking["adjacent_switch_count"], tracking["gap_changed_id_count"])
    print(f"PASS raw tracking: {matched}/660 matched, {adjacent} adjacent ID changes, {gap} gap changes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot", type=Path)
    parser.add_argument("--predictions-default", type=Path)
    parser.add_argument("--predictions-higher", type=Path)
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    manifest, detection, tracking = verify_bundle()
    supplied = [args.mot, args.predictions_default, args.predictions_higher]
    if any(supplied):
        if not all(supplied):
            parser.error("Raw recomputation needs --mot and both prediction files")
        recompute(args.mot, args.predictions_default, args.predictions_higher,
                  args.observations, manifest, detection, tracking)
    elif args.observations:
        parser.error("--observations requires all three detection inputs")


if __name__ == "__main__":
    main()
