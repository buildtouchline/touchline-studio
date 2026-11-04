"""Operator-only import of a completed panorama Sandbox export into local Studio.

This performs no inference or provider call. It does not make Studio an
authenticated upload API. Predicted track boxes remain separate from measured
detector-anchor observations; neither establishes roster identity.
"""

from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import shutil

from .contracts import deserialize_observation
from .data.manifest import sha256_file


SHA256 = re.compile(r"[a-f0-9]{64}\Z")
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")


def _regular(path: Path, max_bytes: int) -> Path:
    path = Path(path)
    # The operator selects the parent workspace; reject linked input files and
    # linked immediate directories without rejecting macOS's /var system alias.
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("A regular unlinked input file is required")
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("Input size is outside the reviewed bound")
    return path


def _json(path: Path, max_bytes: int = 4 * 1024**2) -> dict:
    value = json.loads(_regular(path, max_bytes).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Object JSON expected")
    return value


def _hash(path: Path, expected: str, max_bytes: int) -> str:
    if not isinstance(expected, str) or not SHA256.fullmatch(expected.lower()):
        raise ValueError("Expected SHA-256 is missing")
    actual = sha256_file(_regular(path, max_bytes))
    if actual != expected.lower():
        raise ValueError("Artifact SHA-256 mismatch")
    return actual


def _frame_map(path: Path, expected_frames: int) -> list[dict]:
    rows = []
    last_source_index = -1
    last_source_time = Fraction(-1)
    with _regular(path, 64 * 1024**2).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index = len(rows)
            fraction = row.get("source_relative_time")
            if (row.get("output_frame_index") != index or
                    not isinstance(fraction, list) or len(fraction) != 2 or
                    any(type(v) is not int for v in fraction) or fraction[1] <= 0 or
                    Fraction(*fraction) < 0 or
                    type(row.get("source_frame_index")) is not int or
                    type(row.get("source_pts")) is not int or
                    row.get("output_pts") not in (None, index) or
                    row.get("output_time_base") not in (None, [1, 24]) or
                    row["source_frame_index"] < last_source_index or
                    Fraction(*fraction) < last_source_time):
                raise ValueError("Invalid original-to-normalized frame correspondence")
            rows.append(row)
            last_source_index = row["source_frame_index"]
            last_source_time = Fraction(*fraction)
            if len(rows) > 200000:
                raise ValueError("Frame map exceeds bound")
    if len(rows) != expected_frames:
        raise ValueError("Incomplete frame correspondence")
    return rows


def _proposal_scores(path: Path, summary: dict) -> dict[tuple[int, tuple[float, ...]], float]:
    """Join recorded detector confidence by exact normalized frame and box."""
    scores = {}
    count = 0
    with _regular(path, 64 * 1024**2).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index, box, score = row.get("frame_index"), row.get("bbox_xyxy"), row.get("score")
            if (type(index) is not int or not 0 <= index < summary["frames"] or
                    not isinstance(box, list) or len(box) != 4 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
                    type(score) not in (int, float) or not math.isfinite(score) or
                    not 0 <= score <= 1):
                raise ValueError("Invalid recorded detector proposal")
            key = (index, tuple(box))
            if key in scores and scores[key] != score:
                raise ValueError("Conflicting recorded detector proposals")
            scores[key] = float(score)
            count += 1
            if count > 200000:
                raise ValueError("Detector proposals exceed bound")
    if count != summary.get("counters", {}).get("raw_proposals"):
        raise ValueError("Detector proposal count does not match summary")
    return scores


def _observations(path: Path, summary: dict, proposal_scores: dict) -> tuple[list[dict], list[dict], Counter]:
    anchors, predictions, counts = [], [], Counter()
    width, height = summary["source_geometry"]
    with _regular(path, 256 * 1024**2).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index = row.get("frame_index")
            if (row.get("run_id") != summary["run_id"] or type(index) is not int or
                    not 0 <= index < summary["frames"] or
                    type(row.get("track_id")) is not int or row["track_id"] <= 0 or
                    type(row.get("timestamp_s")) not in (int, float) or
                    abs(row["timestamp_s"] - index / 24) > 1e-4):
                raise ValueError("Experimental observation has invalid lineage or time")
            if row.get("evidence") != "detector_anchor":
                if row.get("evidence") not in {"optical_flow_prediction", "kalman_prediction", "camera_prediction"}:
                    raise ValueError("Unknown tracking evidence class")
                predictions.append(row)
                if len(anchors) + len(predictions) > 200000:
                    raise ValueError("Observation export exceeds bound")
                continue
            box = row.get("detector_bbox_xyxy")
            key = (index, tuple(box)) if isinstance(box, list) else None
            if key not in proposal_scores:
                raise ValueError("Detector anchor lacks an exact recorded proposal")
            canonical = {
                "schema_version": "touchline.observation.v1",
                "run_id": summary["run_id"],
                "frame": {"source_id": summary["source_sha256"], "index": index,
                          "pts": index, "time_base_num": 1, "time_base_den": 24,
                          "timestamp_s": row["timestamp_s"], "width": width, "height": height},
                "local_track_id": row["track_id"], "bbox_xyxy": box,
                "detector_score": proposal_scores[key],
                "association_state": "unreviewed", "player_id": None, "synthetic": False,
            }
            deserialize_observation(canonical)
            anchors.append(canonical)
            counts[row["track_id"]] += 1
            if len(anchors) + len(predictions) > 200000:
                raise ValueError("Observation export exceeds bound")
    return anchors, predictions, counts


def import_completed_panorama(*, export_dir: Path, source_video: Path,
                              normalized_video: Path, frame_map: Path,
                              normalization_metadata: Path,
                              expected_original_sha256: str, runs_root: Path) -> Path:
    """Verify saved inputs/results and write one fresh, reopenable local run.

    The operator supplies paths and the independently recorded original hash.
    No HTTP request can call this function or choose these paths.
    """
    export_dir, runs_root = Path(export_dir), Path(runs_root)
    if export_dir.is_symlink() or runs_root.is_symlink() or not runs_root.is_dir():
        raise ValueError("Unlinked existing export and runs directories required")
    receipt = _json(export_dir / "receipt.json")
    summary = _json(export_dir / "evidence" / "summary.json")
    run_id = summary.get("run_id")
    if (receipt.get("status") != "complete" or summary.get("status") != "complete" or
            summary.get("schema") != "touchline.panorama-review.v1" or
            not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or
            receipt.get("run_id") != run_id or receipt.get("summary") != summary or
            type(summary.get("frames")) is not int or not 0 < summary["frames"] <= 200000 or
            summary.get("source_fps") != 24 or summary.get("identity_confirmed") is not False or
            not isinstance(summary.get("source_geometry"), list) or
            len(summary["source_geometry"]) != 2 or
            any(type(n) is not int or n <= 0 for n in summary["source_geometry"])):
        raise ValueError("Completed panorama receipt and summary do not match")
    original_hash = _hash(source_video, expected_original_sha256, 5 * 1024**3)
    normalized_hash = _hash(normalized_video, summary.get("source_sha256"), 5 * 1024**3)
    map_hash = _hash(frame_map, summary.get("frame_map_sha256"), 64 * 1024**2)
    normalization = _json(normalization_metadata)
    if (normalization.get("schema") != "touchline.video-normalization.v1" or
            normalization.get("status") != "complete" or
            normalization.get("source_sha256") != original_hash or
            normalization.get("output_sha256") != normalized_hash or
            normalization.get("frame_map_sha256") != map_hash or
            normalization.get("output_frames") != summary["frames"] or
            normalization.get("geometry") != summary["source_geometry"]):
        raise ValueError("Normalization does not bind the original, output and frame map")
    overlay = export_dir / "overlay24.mp4"
    overlay_hash = _hash(overlay, receipt.get("result_sha256"), 2 * 1024**3)
    _frame_map(frame_map, summary["frames"])
    proposal_scores = _proposal_scores(export_dir / "evidence" / "raw_proposals.jsonl", summary)
    anchors, predictions, counts = _observations(export_dir / "evidence" / "observations.jsonl", summary,
                                                  proposal_scores)
    counters = summary.get("counters", {})
    if (len(anchors) != counters.get("detector_anchors") or
            len(predictions) != counters.get("motion_predictions")):
        raise ValueError("Observation counts do not match completed summary")
    coverage = _regular(export_dir / "evidence" / "coverage.jsonl", 64 * 1024**2)
    with coverage.open(encoding="utf-8") as stream:
        coverage_rows = sum(1 for _ in stream)
    if coverage_rows != summary["frames"]:
        raise ValueError("Incomplete coverage export")
    destination = runs_root / run_id
    destination.mkdir(exist_ok=False)
    try:
        (destination / "normalized").mkdir()
        for source, target in ((source_video, destination / "source_video.mp4"),
                               (normalized_video, destination / "normalized" / "video24.mp4"),
                               (frame_map, destination / "normalized" / "frame-map.jsonl"),
                               (overlay, destination / "overlay24.mp4"),
                               (coverage, destination / "coverage.jsonl")):
            shutil.copyfile(source, target)
        for target, expected, bound in ((destination / "source_video.mp4", original_hash, 5 * 1024**3),
                                        (destination / "normalized" / "video24.mp4", normalized_hash, 5 * 1024**3),
                                        (destination / "normalized" / "frame-map.jsonl", map_hash, 64 * 1024**2),
                                        (destination / "overlay24.mp4", overlay_hash, 2 * 1024**3)):
            _hash(target, expected, bound)
        with (destination / "observations.jsonl").open("x", encoding="utf-8") as out:
            for row in anchors:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        with (destination / "motion_predictions.jsonl").open("x", encoding="utf-8") as out:
            for row in predictions:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        shutil.copyfile(export_dir / "evidence" / "summary.json", destination / "panorama-summary.json")
        # Preserve the verified chain without historical private filesystem paths.
        public_normalization = {key: normalization[key] for key in (
            "schema", "status", "source_sha256", "output_sha256", "frame_map_sha256",
            "output_frames", "geometry")}
        (destination / "normalized" / "normalization.json").write_text(
            json.dumps(public_normalization, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "touchline.run.v1", "run_id": run_id, "status": "completed",
            "source": {"id": original_hash, "name": source_video.name, "sha256": original_hash,
                       "analysis_policy": {"width": summary["source_geometry"][0],
                                           "height": summary["source_geometry"][1],
                                           "fps_num": 24, "fps_den": 1}},
            "normalized_video_sha256": normalized_hash, "frame_map_sha256": map_hash,
            "normalization_verified": True,
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "code_sha256": summary.get("code_sha256"),
            "duration_s": summary.get("duration_s"), "selected_frames": summary.get("counters", {}).get("detector_frames"),
            "observation_count": len(anchors), "motion_prediction_count": len(predictions),
            "identity_confirmed": False, "calibration": {"status": "uncalibrated", "calibrated": False},
            "artifacts": {"annotated_video": {"filename": "overlay24.mp4", "sha256": overlay_hash,
                                               "verified_at_import": True}},
        }
        analytics = {"session": {"observation_count": len(anchors), "duration_s": summary.get("duration_s")},
                     "player_roster": [{"local_track_id": track_id, "observation_count": counts[track_id],
                                        "confirmed_player_id": None} for track_id in sorted(counts)],
                     "calibration": {"status": "uncalibrated", "calibrated": False},
                     "quality_indicators": {"total_tracks": len(counts)},
                     "experimental_tracking": True}
        (destination / "analytics.json").write_text(json.dumps(analytics, indent=2) + "\n", encoding="utf-8")
        # Publish the manifest last. Studio lists only runs with a manifest.
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination
