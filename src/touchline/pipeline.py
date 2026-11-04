from .diagnostics import diagnosed, diagnostic_progress, failure_details, current_trace_id
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import uuid
from .analytics import generate_match_analytics
from .contracts import SCHEMA_VERSION
from .tracking import IoUTracker
from .viewer import render_inspection_dashboard


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_preview(manifest, preview, calibration=None):
    analytics = generate_match_analytics(manifest, preview, calibration=calibration)
    return render_inspection_dashboard(manifest, analytics, preview)


def validate_run_for_resume(output, source, detector_provenance, sampler_name, sampler_config, synthetic=False):
    """Validate that an existing run directory contains valid, compatible artifacts for resume.

    Checks:
        1. Required artifacts exist (manifest.json, coverage.jsonl, observations.jsonl).
        2. Manifest status is not already complete.
        3. Source, detector provenance, sampler name, sampler config, and synthetic mode match.
        4. Artifact files contain valid, uncorrupted records.
    """
    output = Path(output)
    if not output.is_dir():
        raise FileNotFoundError(f"cannot resume run: output directory does not exist: {output}")

    manifest_file = output / "manifest.json"
    coverage_file = output / "coverage.jsonl"
    observations_file = output / "observations.jsonl"

    for artifact in (manifest_file, coverage_file, observations_file):
        if not artifact.is_file():
            raise FileNotFoundError(f"cannot resume run: required artifact missing: {artifact.name}")

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot resume run: corrupted manifest.json: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("cannot resume run: manifest.json must contain a JSON object")

    if manifest.get("status") == "complete":
        raise ValueError("cannot resume run: run is already marked complete")
    if manifest.get('tracker',{}).get('type') != 'greedy-iou-baseline':
        raise ValueError('Cannot resume without the recorded tracker motion state; start a fresh run')

    if manifest.get("source") != source:
        raise ValueError(f"cannot resume run: source mismatch: manifest has {manifest.get('source')}, expected {source}")

    if manifest.get("detector") != detector_provenance:
        raise ValueError("cannot resume run: detector provenance mismatch")

    if manifest.get("sampler") != sampler_name:
        raise ValueError(f"cannot resume run: sampler mismatch: manifest has {manifest.get('sampler')}, expected {sampler_name}")

    if manifest.get("sampler_config") != sampler_config:
        raise ValueError("cannot resume run: sampler_config mismatch")

    if bool(manifest.get("synthetic")) != bool(synthetic):
        raise ValueError("cannot resume run: synthetic mode mismatch")

    # Validate coverage.jsonl records
    coverage_records = []
    with coverage_file.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise ValueError(f"cannot resume run: corrupted JSON on line {line_num} in coverage.jsonl: {exc}") from exc
            if not isinstance(rec, dict) or "frame" not in rec or not isinstance(rec["frame"], dict):
                raise ValueError(f"cannot resume run: invalid coverage record on line {line_num}")
            coverage_records.append(rec)

    # Validate observations.jsonl records
    max_track_id = 0
    preview = []
    with observations_file.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
            except Exception as exc:
                raise ValueError(f"cannot resume run: corrupted JSON on line {line_num} in observations.jsonl: {exc}") from exc
            if not isinstance(obs, dict) or "local_track_id" not in obs:
                raise ValueError(f"cannot resume run: invalid observation record on line {line_num}")
            tid = obs["local_track_id"]
            if isinstance(tid, int) and tid > max_track_id:
                max_track_id = tid
            if len(preview) < 500:
                preview.append(obs)

    last_frame_info = None
    if coverage_records:
        last_rec = coverage_records[-1]
        last_frame_info = {
            "index": last_rec["frame"].get("index"),
            "pts": last_rec["frame"].get("pts"),
            "timestamp_s": last_rec["frame"].get("timestamp_s"),
        }

    return {
        "manifest": manifest,
        "last_frame_info": last_frame_info,
        "max_track_id": max_track_id,
        "preview": preview,
        "decoded_count": len(coverage_records),
    }


@diagnosed("pipeline")
def run_pipeline(
    frames,
    detector,
    sampler,
    output,
    source,
    synthetic=False,
    max_selected=None,
    resume=False,
    calibration=None,
    record_proposals=False,
    tracker=None,
):
    """Run sequential video analysis pipeline with support for verified atomic resume.

    Args:
        frames: Iterable of decoded Frame objects.
        detector: Detector implementation.
        sampler: Sampler implementation (e.g. Anchor24Sampler, FixedRateSampler).
        output: Destination directory for artifacts.
        source: Source identity dictionary or string.
        synthetic: Whether run is synthetic.
        max_selected: Optional limit on selected frames before setting status='partial'.
        resume: If True, resumes an existing incomplete/partial run after verifying source and configuration.
            If False (default), creates a new directory and raises FileExistsError if output already exists.
        calibration: Optional pitch Calibration instance for computing physical distance and speed.
        record_proposals: Optional flag to persist raw detector candidate proposals to raw_proposals.jsonl.
    """
    if max_selected is not None and (type(max_selected) is not int or max_selected <= 0):
        raise ValueError("max_selected must be a positive integer")
    output = Path(output)
    if resume and tracker is not None:
        raise ValueError('Resume with a custom tracker is unsupported; start a fresh run to preserve motion state')

    sampler_config = {k: v for k, v in vars(sampler).items() if k in ("fps", "tolerance_s")}
    detector_prov = detector.provenance()

    if resume:
        resume_data = validate_run_for_resume(
            output,
            source=source,
            detector_provenance=detector_prov,
            sampler_name=type(sampler).__name__,
            sampler_config=sampler_config,
            synthetic=synthetic,
        )
        manifest = resume_data["manifest"]
        manifest["status"] = "running"
        manifest["resume_count"] = manifest.get("resume_count", 0) + 1
        manifest["resume_supported"] = True
        manifest.pop("error", None)
        manifest.pop("stop_reason", None)
        preview = resume_data["preview"]
        last_frame_info = resume_data["last_frame_info"]
        tracker = tracker or IoUTracker()
        tracker.next_id = resume_data["max_track_id"] + 1
        file_mode = "a"
    else:
        output.mkdir(parents=True, exist_ok=False)
        tracker = tracker or IoUTracker()
        manifest = {
            "schema_version": "touchline.run.v1",
            "run_id": str(uuid.uuid4()),
            "synthetic": bool(synthetic),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "status": "running",
            "decoded_frames": 0,
            "selected_frames": 0,
            "observation_count": 0,
            "preview_limit": 500,
            "detector": detector_prov,
            "sampler": type(sampler).__name__,
            "sampler_config": sampler_config,
            "tracker": tracker.provenance() if hasattr(tracker,'provenance') else {"type": "greedy-iou-baseline", "threshold": tracker.threshold, "max_gap_s": tracker.max_gap_s},
            "resume_supported": isinstance(tracker,IoUTracker),
            "physical_metrics": "unavailable: no calibration/confirmed identity supplied",
            "limitations": [
                "No roster identification or technique recognition",
                "No exact visibility-duration/activity labels",
                "No model/metric validation",
                "Failed runs preserve partial output and can be resumed with verification",
                "Baseline tracker can switch identities; inspect crossings/occlusions",
                "Source rotation/display transforms need runtime verification",
            ],
        }
        preview = []
        last_frame_info = None
        file_mode = "w"

    manifest["diagnostic_trace_id"] = current_trace_id()
    write_json(output / "manifest.json", manifest)

    try:
        with ExitStack() as stack:
            observations = stack.enter_context((output / "observations.jsonl").open(file_mode, encoding="utf-8"))
            coverage = stack.enter_context((output / "coverage.jsonl").open(file_mode, encoding="utf-8"))
            proposals_handle = None
            if record_proposals:
                proposals_handle = stack.enter_context((output / "raw_proposals.jsonl").open(file_mode, encoding="utf-8"))

            for frame in frames:
                # If resuming, fast-forward frames already recorded in coverage ledger
                if last_frame_info is not None and frame.index <= last_frame_info["index"]:
                    if frame.index == last_frame_info["index"]:
                        # Verify boundary frame matches recorded coverage
                        if frame.pts != last_frame_info["pts"] or not math.isclose(frame.timestamp_s, last_frame_info["timestamp_s"], abs_tol=1e-3):
                            raise ValueError(
                                f"cannot resume run: boundary frame index {frame.index} has conflicting timing: "
                                f"pts {frame.pts} vs recorded {last_frame_info['pts']}"
                            )
                    # Advance sampler state so stateful samplers (e.g. Anchor24Sampler) stay synchronized
                    sampler.select(frame.index, frame.timestamp_s)
                    continue

                manifest["decoded_frames"] += 1
                if manifest["decoded_frames"] % 24 == 0:
                    diagnostic_progress("pipeline", decoded_frames=manifest["decoded_frames"], selected_frames=manifest["selected_frames"], observations=manifest["observation_count"])
                if not sampler.select(frame.index, frame.timestamp_s):
                    coverage.write(
                        json.dumps({"frame": frame.metadata(), "detections": None, "state": "unsampled_by_sampler"}) + "\n"
                    )
                    continue

                if record_proposals and hasattr(detector, "detect_with_proposals"):
                    detections, raw_props = detector.detect_with_proposals(frame)
                else:
                    detections = detector.detect(frame)
                    raw_props = [{"bbox_xyxy": d.box.values(), "score": d.score, "label": 1} for d in detections]

                if any(not item.box.fits(frame.width, frame.height) for item in detections):
                    raise ValueError("detector returned box outside source geometry")

                if proposals_handle is not None:
                    frame_meta = frame.metadata()
                    for prop in raw_props:
                        proposals_handle.write(
                            json.dumps({"frame": frame_meta, **prop}, allow_nan=False) + "\n"
                        )

                tracks = tracker.update(frame.timestamp_s, detections)
                manifest["selected_frames"] += 1
                coverage.write(
                    json.dumps({
                        "frame": frame.metadata(),
                        "detections": len(detections),
                        "state": "detections_present" if detections else "no_detection_not_inactivity",
                    }) + "\n"
                )
                for track in tracks:
                    item = {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": manifest["run_id"],
                        "frame": frame.metadata(),
                        "synthetic": bool(synthetic),
                        **track,
                    }
                    observations.write(json.dumps(item, allow_nan=False) + "\n")
                    manifest["observation_count"] += 1
                    if len(preview) < 500:
                        preview.append(item)

                if max_selected is not None and manifest["selected_frames"] >= max_selected:
                    manifest["status"] = "partial"
                    manifest["stop_reason"] = "explicit selected-frame limit; remaining source coverage unknown"
                    break
            else:
                manifest["status"] = "complete"

            if manifest["decoded_frames"] == 0:
                raise ValueError("source yielded no decoded frames")

        runtime_stats = getattr(detector, 'runtime_stats', None)
        if callable(runtime_stats):
            manifest['detector_runtime'] = runtime_stats()
        tracker_stats=getattr(tracker,'runtime_stats',None)
        if callable(tracker_stats):manifest['tracker_runtime']=tracker_stats()
        write_json(output / "manifest.json", manifest)
        obs_file = output / "observations.jsonl"

        def stream_observations():
            with obs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        analytics = generate_match_analytics(manifest, stream_observations(), calibration=calibration)
        write_json(output / "analytics.json", analytics)
        (output / "review.html").write_text(render_inspection_dashboard(manifest, analytics, preview), encoding="utf-8")
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        detail=failure_details(exc,"pipeline")
        manifest["error"] = {**detail, "message":detail['guidance'], "trace_id": current_trace_id()}
        write_json(output / "manifest.json", manifest)
        raise
