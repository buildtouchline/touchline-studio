"""Modular execution adapter for the Modal Panorama/ResNet worker.

Provides parameterized Modal Sandbox dispatch reusing tools/modal_panorama_worker.py,
pinned SHA-256 input/weights/code/frame-map validation, durable status progression via
job_state.py, and conversion of worker exports into canonical touchline.run.v1 and
touchline.observation.v1 artifacts for Coach Studio.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable, Optional
import zipfile

from ..analytics import generate_match_analytics
from ..contracts import deserialize_observation
from ..data.manifest import sha256_file
from ..job_state import JOB_ID, write_job_status
from ..panorama_import import (
    _frame_map,
    _hash,
    _json,
    _observations,
    _proposal_scores,
    _regular,
)
from ..run_records import new_record, write_record
from ..worker_status import parse_worker_status
from ..video.decode import probe_video
from ..video.normalize import normalize_video

LOGGER = logging.getLogger(__name__)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PINNED_RESNET50_WEIGHTS_SHA256 = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
PINNED_MODAL_IMAGE_ID = "im-A4IdpUIDiLlRbBha7sqzYy"
PINNED_MODAL_APP_NAME = "touchline-panorama-analysis"


def _verified_code_bundle(path: Path) -> str:
    """Validate the source archive the offline-checking worker will extract."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024**2:
        raise ValueError("Bounded unlinked source bundle is required")
    required = {"pyproject.toml", "tools/run_offline_checks.py",
                "tools/modal_panorama_worker.py", "src/touchline/adapters/rcnn_comparison.py"}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        if not required <= names or len(names) != len(entries) or len(entries) > 3000:
            raise ValueError("Source bundle is missing reviewed worker dependencies")
        if sum(entry.file_size for entry in entries) > 20 * 1024**2:
            raise ValueError("Source bundle exceeds reviewed expansion bound")
        for entry in entries:
            member = Path(entry.filename)
            if (member.is_absolute() or ".." in member.parts
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe source bundle member")
        if archive.testzip() is not None:
            raise ValueError("Source bundle contains a corrupt member")
    return sha256_file(path)


@dataclass(frozen=True)
class PanoramaJobConfig:
    expected_frames: int = 5760
    geometry: tuple[int, int] = (4096, 1080)
    detector_stride: int = 4
    session_id_cap: int = 24
    stationary_camera: bool = True
    weights_sha256: str = PINNED_RESNET50_WEIGHTS_SHA256
    image_id: str = PINNED_MODAL_IMAGE_ID
    app_name: str = PINNED_MODAL_APP_NAME
    environment_name: str = "main"
    gpu: str = "A10"
    cpu: tuple[int, int] = (4, 4)
    memory_mib: tuple[int, int] = (16384, 16384)
    timeout_s: int = 2700
    exec_timeout_s: int = 2450
    poll_interval_s: float = 2.0


def validate_panorama_job_spec(job: dict[str, Any], config: Optional[PanoramaJobConfig] = None) -> dict[str, Any]:
    """Validate job payload using the pinned Panorama/ResNet worker contract."""
    cfg = config or PanoramaJobConfig()
    if not isinstance(job, dict):
        raise ValueError("Panorama job payload must be a JSON object")
    for key in ("video_sha256", "original_video_sha256", "weights_sha256", "code_sha256", "frame_map_sha256", "worker_script_sha256"):
        value = job.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"Pinned 64-character hex SHA-256 digest required for {key}")
    if job.get("weights_sha256") != cfg.weights_sha256:
        raise ValueError("Pinned ResNet50 weights_sha256 does not match configured checkpoint digest")
    if job.get("expected_frames") != cfg.expected_frames or job.get("geometry") != list(cfg.geometry):
        raise ValueError("Panorama job geometry or frame count does not match the configured scope")
    if (
        job.get("detector_stride") != cfg.detector_stride
        or job.get("session_id_cap") != cfg.session_id_cap
        or job.get("stationary_camera") is not True
    ):
        raise ValueError("Reviewed six-Hz/24-ID stationary panorama scope required")
    return job


def _extract_selected_archive(archive_path: Path, destination: Path, suffixes: set[str], limit: int) -> None:
    """Safely extract bounded worker archive entries without path traversal or symlinks."""
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) > 100 or sum(e.file_size for e in entries) > limit:
            raise ValueError("Export expansion exceeds bound")
        targets = []
        dest_resolved = destination.resolve()
        for entry in entries:
            target = (destination / entry.filename).resolve()
            if (
                not target.is_relative_to(dest_resolved)
                or target.exists()
                or target.suffix not in suffixes
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
                or target in targets
            ):
                raise ValueError("Unsafe exported file in worker archive")
            targets.append(target)
        for entry, target in zip(entries, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024**2)


class PanoramaWorkerAdapter:
    """Orchestrates job lifecycle, Modal Sandbox execution, and canonical run packaging."""

    def __init__(
        self,
        runs_root: Path,
        runner: Optional[Callable] = None,
        *,
        sandbox_controller: Optional[Callable] = None,
        config: Optional[PanoramaJobConfig] = None,
        weights_path: Optional[Path] = None,
        code_bundle_path: Optional[Path] = None,
        worker_script_path: Optional[Path] = None,
        allow_live_dispatch: bool = False,
        stationary_camera_approved: bool = False,
        spending_ledger: Any = None,
        deployment_probe: Optional[Callable[[], bool]] = None,
        approved_code_sha256: Optional[str] = None,
        approved_worker_sha256: Optional[str] = None,
    ):
        self.runs_root = Path(runs_root).resolve()
        self.runner = runner
        self.sandbox_controller = sandbox_controller
        self.config = config or PanoramaJobConfig()
        self.weights_path = Path(weights_path) if weights_path else None
        self.code_bundle_path = Path(code_bundle_path) if code_bundle_path else None
        default_worker = Path(__file__).resolve().parents[3] / "tools" / "modal_panorama_worker.py"
        self.worker_script_path = Path(worker_script_path) if worker_script_path else default_worker
        self.allow_live_dispatch = bool(allow_live_dispatch)
        self.stationary_camera_approved = stationary_camera_approved is True
        self.spending_ledger = spending_ledger
        self.deployment_probe = deployment_probe
        self.approved_code_sha256 = approved_code_sha256
        self.approved_worker_sha256 = approved_worker_sha256

    def is_ready(self) -> bool:
        """Return True only when a mock runner, sandbox controller, or authorized live setup is configured."""
        if importlib.util.find_spec("av") is None:
            return False
        try:
            import av
            av.codec.Codec("libx264", "w")
            assets_ready = bool(
                self.weights_path is not None and self.weights_path.is_file()
                and self.code_bundle_path is not None and self.code_bundle_path.is_file()
                and self.worker_script_path.is_file()
                and sha256_file(self.weights_path) == self.config.weights_sha256
                and _verified_code_bundle(self.code_bundle_path)
            )
            if self.runner is not None or self.sandbox_controller is not None:
                return assets_ready
            return bool(
                assets_ready and self.allow_live_dispatch and self.stationary_camera_approved
                and self.spending_ledger is not None
                and self.spending_ledger.can_reserve("modal", "1.80") is True
                and importlib.util.find_spec("modal") is not None
                and self.deployment_probe is not None and self.deployment_probe() is True
                and SHA256_RE.fullmatch(self.approved_code_sha256 or "")
                and SHA256_RE.fullmatch(self.approved_worker_sha256 or "")
                and sha256_file(self.weights_path) == self.config.weights_sha256
                and _verified_code_bundle(self.code_bundle_path) == self.approved_code_sha256
                and sha256_file(self.worker_script_path) == self.approved_worker_sha256
            )
        except Exception:
            # A missing codec, malformed archive, or failed deployment probe
            # must never advertise upload readiness.
            return False

    def prepare_job_spec(self, job_dir: Path, job_id: str, original_sha256: str) -> dict[str, Any]:
        """Prepare a fresh upload with the repository's reviewed CPU normalization procedure.

        No Sandbox is created until all bytes and the exact frame correspondence
        have been produced and checked. Failed normalization remains in the run
        directory for operator diagnosis; it is never treated as a complete input.
        """
        if not JOB_ID.fullmatch(job_id) or job_dir.name != job_id or job_dir.is_symlink():
            raise ValueError("Invalid or linked job directory")
        source = job_dir / "source_video.mp4"
        if source.is_symlink() or not source.is_file() or sha256_file(source) != original_sha256:
            raise ValueError("Original upload is missing or changed")
        if self.weights_path is None or self.code_bundle_path is None:
            raise ValueError("Checkpoint and code bundle are required before preparation")
        for path in (self.weights_path, self.code_bundle_path, self.worker_script_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Verified unlinked worker input is required")
        checkpoint_sha = sha256_file(self.weights_path)
        if checkpoint_sha != self.config.weights_sha256:
            raise ValueError("Pinned checkpoint bytes do not match the selected ResNet model")
        code_sha = _verified_code_bundle(self.code_bundle_path)
        if self.allow_live_dispatch and (
            code_sha != self.approved_code_sha256
            or sha256_file(self.worker_script_path) != self.approved_worker_sha256
        ):
            raise ValueError("Panorama code or worker script is not the operator-approved revision")
        info = probe_video(source)
        if (info["is_rotated"] or (info["width"], info["height"]) != self.config.geometry
                or info.get("duration_s") is None
                or not 0 < info["duration_s"] <= self.config.expected_frames / 24 + 0.25):
            raise ValueError("Recording does not meet the reviewed Panorama geometry and duration scope")
        normalized = normalize_video(source, job_dir / "normalized",
                                     max_output_frames=self.config.expected_frames)
        norm_info = probe_video(normalized)
        if (norm_info["average_fps"] != 24
                or norm_info["frame_count"] != self.config.expected_frames
                or (norm_info["width"], norm_info["height"]) != self.config.geometry):
            raise ValueError("Normalized recording does not meet the worker's exact frame contract")
        metadata = json.loads((job_dir / "normalized" / "normalization.json").read_text())
        if (metadata.get("source_sha256") != original_sha256
                or metadata.get("output_sha256") != sha256_file(normalized)
                or metadata.get("output_frames") != self.config.expected_frames):
            raise ValueError("Normalization lineage does not match the uploaded bytes")
        frame_map = job_dir / "normalized" / "frame-map.jsonl"
        _frame_map(frame_map, self.config.expected_frames)
        return self.create_job_payload(
            job_id, sha256_file(normalized),
            weights_sha256=checkpoint_sha,
            code_sha256=code_sha,
            frame_map_sha256=sha256_file(frame_map),
            worker_script_sha256=sha256_file(self.worker_script_path),
            original_video_sha256=original_sha256,
        )

    def validate_video(self, video_path: Path, max_bytes: int = 1 * 1024**3) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file() or video_path.is_symlink():
            raise ValueError("A regular unlinked video file is required")
        size = video_path.stat().st_size
        if not 0 < size <= max_bytes:
            raise ValueError(f"Video size {size} exceeds allowed bound of {max_bytes} bytes")
        digest = sha256_file(video_path)
        return {"video_sha256": digest, "size": size}

    def create_job_payload(
        self,
        job_id: str,
        video_sha256: str,
        weights_sha256: Optional[str] = None,
        code_sha256: Optional[str] = None,
        frame_map_sha256: Optional[str] = None,
        worker_script_sha256: Optional[str] = None,
        original_video_sha256: Optional[str] = None,
        config: Optional[PanoramaJobConfig] = None,
    ) -> dict[str, Any]:
        cfg = config or self.config
        payload = {
            "schema_version": "touchline.panorama-job.v1",
            "job_id": job_id,
            "run_id": job_id,
            "video_sha256": video_sha256,
            "weights_sha256": weights_sha256 or cfg.weights_sha256,
            "code_sha256": code_sha256,
            "frame_map_sha256": frame_map_sha256,
            "worker_script_sha256": worker_script_sha256,
            "original_video_sha256": original_video_sha256,
            "expected_frames": cfg.expected_frames,
            "geometry": list(cfg.geometry),
            "detector_stride": cfg.detector_stride,
            "session_id_cap": cfg.session_id_cap,
            "stationary_camera": cfg.stationary_camera,
            "created_at": time.time(),
        }
        return validate_panorama_job_spec(payload, cfg)

    def convert_worker_export_to_canonical_run(
        self,
        job_dir: Path,
        job_id: str,
        export_dir: Path,
        *,
        source_video: Path,
        normalized_video: Path,
        frame_map_path: Path,
        normalization_metadata_path: Path,
        expected_original_sha256: str,
    ) -> dict[str, Any]:
        """Convert the Panorama worker export into canonical Coach Studio run artifacts."""
        if not JOB_ID.fullmatch(job_id) or job_dir.name != job_id:
            raise ValueError(f"Invalid Studio job ID: {job_id}")
        export_dir = Path(export_dir)
        if export_dir.is_symlink() or job_dir.is_symlink():
            raise ValueError("Unlinked export and job directories are required")

        receipt = _json(export_dir / "receipt.json")
        summary = _json(export_dir / "evidence" / "summary.json")
        submitted_job = receipt.get("job")
        if (not isinstance(submitted_job, dict)
                or submitted_job.get("job_id") != job_id
                or submitted_job.get("original_video_sha256") != expected_original_sha256
                or submitted_job.get("video_sha256") != summary.get("source_sha256")
                or submitted_job.get("weights_sha256") != summary.get("checkpoint_sha256")
                or submitted_job.get("code_sha256") != summary.get("code_sha256")
                or submitted_job.get("frame_map_sha256") != summary.get("frame_map_sha256")):
            raise ValueError("Worker receipt is not bound to submitted job inputs")
        worker_run_id = summary.get("run_id")
        if (
            receipt.get("status") != "complete"
            or summary.get("status") != "complete"
            or summary.get("schema") != "touchline.panorama-review.v1"
            or not isinstance(worker_run_id, str)
            or not worker_run_id
            or receipt.get("run_id") != worker_run_id
            or receipt.get("summary") != summary
            or type(summary.get("frames")) is not int
            or not 0 < summary["frames"] <= 200000
            or summary.get("source_fps") != 24
            or summary.get("identity_confirmed") is not False
            or not isinstance(summary.get("source_geometry"), list)
            or len(summary["source_geometry"]) != 2
            or any(type(n) is not int or n <= 0 for n in summary["source_geometry"])
        ):
            raise ValueError("Completed panorama receipt and summary do not match")

        original_hash = _hash(source_video, expected_original_sha256, 5 * 1024**3)
        normalized_hash = _hash(normalized_video, summary.get("source_sha256"), 5 * 1024**3)
        map_hash = _hash(frame_map_path, summary.get("frame_map_sha256"), 64 * 1024**2)
        normalization = _json(normalization_metadata_path)
        if (
            normalization.get("schema") != "touchline.video-normalization.v1"
            or normalization.get("status") != "complete"
            or normalization.get("source_sha256") != original_hash
            or normalization.get("output_sha256") != normalized_hash
            or normalization.get("frame_map_sha256") != map_hash
            or normalization.get("output_frames") != summary["frames"]
            or normalization.get("geometry") != summary["source_geometry"]
        ):
            raise ValueError("Normalization does not bind original video, normalized output, and frame map")

        overlay = export_dir / "overlay24.mp4"
        overlay_hash = _hash(overlay, receipt.get("result_sha256"), 2 * 1024**3)
        _frame_map(frame_map_path, summary["frames"])
        proposal_scores = _proposal_scores(export_dir / "evidence" / "raw_proposals.jsonl", summary)
        raw_anchors, predictions, counts = _observations(
            export_dir / "evidence" / "observations.jsonl", summary, proposal_scores
        )
        counters = summary.get("counters", {})
        if (
            len(raw_anchors) != counters.get("detector_anchors")
            or len(predictions) != counters.get("motion_predictions")
        ):
            raise ValueError("Observation counts do not match completed worker summary")

        coverage = _regular(export_dir / "evidence" / "coverage.jsonl", 64 * 1024**2)
        with coverage.open(encoding="utf-8") as stream:
            coverage_rows = 0
            for line in stream:
                row = json.loads(line)
                if (not isinstance(row, dict)
                        or row.get("frame_index") != coverage_rows
                        or type(row.get("timestamp_s")) not in (int, float)
                        or abs(row["timestamp_s"] - coverage_rows / 24) > 1e-4
                        or row.get("detector_sample") is not (coverage_rows % self.config.detector_stride == 0)):
                    raise ValueError("Invalid or unordered worker coverage row")
                coverage_rows += 1
        if coverage_rows != summary["frames"]:
            raise ValueError("Incomplete coverage export")

        # Rebind canonical observations to the Studio job_id while keeping source_id = original_hash
        # so Studio's canonicalObservation(row, job_id) and generate_match_analytics pass seamlessly.
        anchors = []
        for row in raw_anchors:
            canonical_row = {
                **row,
                "run_id": job_id,
                "frame": {
                    **row["frame"],
                    "source_id": original_hash,
                },
                "association_state": "unreviewed",
                "player_id": None,
                "synthetic": False,
            }
            deserialize_observation(canonical_row)
            anchors.append(canonical_row)

        norm_dir = job_dir / "normalized"
        norm_dir.mkdir(exist_ok=True)
        dest_source = job_dir / "source_video.mp4"
        if source_video.resolve() != dest_source.resolve():
            shutil.copyfile(source_video, dest_source)
        for src, dst in (
            (normalized_video, norm_dir / "video24.mp4"),
            (frame_map_path, norm_dir / "frame-map.jsonl"),
            (overlay, job_dir / "overlay24.mp4"),
            (coverage, job_dir / "coverage.jsonl"),
        ):
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)

        for target, expected, bound in (
            (dest_source, original_hash, 5 * 1024**3),
            (norm_dir / "video24.mp4", normalized_hash, 5 * 1024**3),
            (norm_dir / "frame-map.jsonl", map_hash, 64 * 1024**2),
            (job_dir / "overlay24.mp4", overlay_hash, 2 * 1024**3),
        ):
            _hash(target, expected, bound)

        with (job_dir / "observations.jsonl").open("w", encoding="utf-8") as out:
            for row in anchors:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        with (job_dir / "motion_predictions.jsonl").open("w", encoding="utf-8") as out:
            for row in predictions:
                out.write(json.dumps({**row, "run_id": job_id}, separators=(",", ":")) + "\n")

        shutil.copyfile(export_dir / "evidence" / "summary.json", job_dir / "panorama-summary.json")
        public_normalization = {
            key: normalization[key]
            for key in (
                "schema",
                "status",
                "source_sha256",
                "output_sha256",
                "frame_map_sha256",
                "output_frames",
                "geometry",
            )
        }
        (norm_dir / "normalization.json").write_text(
            json.dumps(public_normalization, indent=2) + "\n", encoding="utf-8"
        )

        manifest = {
            "schema_version": "touchline.run.v1",
            "run_id": job_id,
            "status": "completed",
            "synthetic": False,
            "source": {
                "id": original_hash,
                "name": source_video.name,
                "sha256": original_hash,
                "analysis_policy": {
                    "width": summary["source_geometry"][0],
                    "height": summary["source_geometry"][1],
                    "fps_num": 24,
                    "fps_den": 1,
                },
            },
            "normalized_video_sha256": normalized_hash,
            "frame_map_sha256": map_hash,
            "normalization_verified": True,
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "code_sha256": summary.get("code_sha256"),
            "worker_script_sha256": submitted_job["worker_script_sha256"],
            "duration_s": summary.get("duration_s"),
            "decoded_frames": summary["frames"],
            "selected_frames": counters.get("detector_frames", len(anchors)),
            "observation_count": len(anchors),
            "motion_prediction_count": len(predictions),
            "identity_confirmed": False,
            "calibration": {"status": "uncalibrated", "calibrated": False},
            "provenance": {
                "worker_schema": summary["schema"],
                "worker_run_id": worker_run_id,
            },
            "artifacts": {
                "annotated_video": {
                    "filename": "overlay24.mp4",
                    "sha256": overlay_hash,
                    "verified_at_import": True,
                }
            },
        }

        try:
            analytics = generate_match_analytics(manifest, anchors, calibration=None, associations=None)
        except Exception:
            analytics = {
                "run_id": job_id,
                "synthetic": False,
                "session": {"observation_count": len(anchors), "duration_s": summary.get("duration_s")},
                "player_roster": [
                    {
                        "local_track_id": track_id,
                        "observation_count": counts[track_id],
                        "confirmed_player_id": None,
                        "identity_confirmed": False,
                    }
                    for track_id in sorted(counts)
                ],
                "calibration": {"status": "uncalibrated", "calibrated": False},
                "quality_indicators": {"total_tracks": len(counts)},
            }
        analytics["calibration"] = {"status": "uncalibrated", "calibrated": False}
        analytics["experimental_tracking"] = True
        for track_entry in analytics.get("player_roster", []):
            track_entry.setdefault("confirmed_player_id", None)
            track_entry["identity_confirmed"] = False

        (job_dir / "analytics.json").write_text(json.dumps(analytics, indent=2) + "\n", encoding="utf-8")
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run_rec = new_record("analysis", backend="modal", run_id=job_id)
        run_rec["status"] = "completed"
        write_record(job_dir, run_rec)
        return manifest

    def run_job_sync(self, job_dir: Path, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the job synchronously and write durable status and canonical artifacts."""
        if not JOB_ID.fullmatch(job_id) or job_dir.name != job_id:
            raise ValueError(f"Invalid job ID or directory mismatch: {job_id}")

        validate_panorama_job_spec(payload, self.config)
        events: list[dict[str, Any]] = [{"stage": "queued", "timestamp": time.time()}]
        write_job_status(
            job_dir,
            job_id,
            "processing",
            stage="queued",
            progress=0.0,
            events=list(events),
        )

        run_rec = new_record("analysis", backend="modal", run_id=job_id)
        run_rec["status"] = "running"
        write_record(job_dir, run_rec)

        try:
            events.append({"stage": "executing_worker", "timestamp": time.time()})
            write_job_status(
                job_dir,
                job_id,
                "processing",
                stage="executing_worker",
                progress=0.25,
                events=list(events),
            )

            if self.runner is not None:
                result = self.runner(job_dir, job_id, payload)
            else:
                result = self._dispatch_modal_worker(job_dir, job_id, payload, events=events)

            if isinstance(result, dict) and "export_dir" in result:
                events.append({"stage": "packaging", "timestamp": time.time()})
                write_job_status(
                    job_dir,
                    job_id,
                    "processing",
                    stage="packaging",
                    progress=0.85,
                    events=list(events),
                )
                self.convert_worker_export_to_canonical_run(
                    job_dir,
                    job_id,
                    Path(result["export_dir"]),
                    source_video=Path(result["source_video"]),
                    normalized_video=Path(result["normalized_video"]),
                    frame_map_path=Path(result["frame_map_path"]),
                    normalization_metadata_path=Path(result["normalization_metadata_path"]),
                    expected_original_sha256=result["expected_original_sha256"],
                )

            manifest_file = job_dir / "manifest.json"
            obs_file = job_dir / "observations.jsonl"
            analytics_file = job_dir / "analytics.json"

            if not manifest_file.is_file() or not obs_file.is_file() or not analytics_file.is_file():
                raise RuntimeError("Worker completed without creating required canonical artifacts")

            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            obs_count = manifest.get("observation_count", 0)

            events.append({"stage": "completed", "timestamp": time.time(), "observation_count": obs_count})
            write_job_status(
                job_dir,
                job_id,
                "completed",
                stage="completed",
                progress=1.0,
                observation_count=obs_count,
                events=list(events),
            )
            run_rec["status"] = "completed"
            write_record(job_dir, run_rec)
            return {"job_id": job_id, "status": "completed", "manifest": manifest}

        except Exception as exc:
            LOGGER.exception("Job %s execution failed: %s", job_id, exc)
            events.append({"stage": "failed", "timestamp": time.time(), "error": type(exc).__name__})
            write_job_status(
                job_dir,
                job_id,
                "failed",
                stage="failed",
                message="Panorama worker execution failed; inspect job logs before retrying.",
                error_code=type(exc).__name__,
                events=list(events),
            )
            run_rec["status"] = "failed"
            write_record(job_dir, run_rec)
            raise

    def run_job_async(self, job_dir: Path, job_id: str, payload: dict[str, Any]) -> threading.Thread:
        """Dispatch job execution in a background thread."""
        thread = threading.Thread(
            target=self.run_job_sync,
            args=(job_dir, job_id, payload),
            name=f"worker-{job_id}",
            daemon=True,
        )
        thread.start()
        return thread

    def _create_live_modal_sandbox(self, job_id: str):
        """Create an isolated gVisor Modal Sandbox using the pinned Panorama image and GPU spec."""
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError("Modal client is not installed in this environment") from exc

        if not self.allow_live_dispatch:
            raise RuntimeError("Live Modal Sandbox dispatch is disabled unless explicitly authorized")

        cfg = self.config
        app = modal.App.lookup(cfg.app_name, environment_name=cfg.environment_name, create_if_missing=False)
        image = modal.Image.from_id(cfg.image_id)
        return modal.Sandbox.create(
            app=app,
            name=job_id,
            image=image,
            gpu=cfg.gpu,
            cpu=cfg.cpu,
            memory=cfg.memory_mib,
            timeout=cfg.timeout_s,
            runtime="gvisor",
            block_network=True,
            secrets=[],
            volumes={},
            workdir="/work",
            env={
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "MPLCONFIGDIR": "/work/cache/matplotlib",
            },
        )

    def _dispatch_modal_worker(
        self,
        job_dir: Path,
        job_id: str,
        payload: dict[str, Any],
        *,
        events: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Execute tools/modal_panorama_worker.py inside a Modal Sandbox and package exports."""
        cfg = self.config
        validate_panorama_job_spec(payload, cfg)
        if events is None:
            events = []

        source_video = job_dir / "source_video.mp4"
        norm_dir = job_dir / "normalized"
        normalized_video = norm_dir / "video24.mp4"
        frame_map_path = norm_dir / "frame-map.jsonl"
        normalization_metadata_path = norm_dir / "normalization.json"

        if not normalized_video.is_file():
            normalized_video = source_video
        if not source_video.is_file():
            raise ValueError("Source video is required in job_dir before worker dispatch")
        if not frame_map_path.is_file() or not normalization_metadata_path.is_file():
            raise ValueError("Normalized frame-map.jsonl and normalization.json are required for Panorama dispatch")
        if self.weights_path is None or not self.weights_path.is_file():
            raise ValueError("Pinned ResNet50 R-CNN weights file is required")
        if self.code_bundle_path is None or not self.code_bundle_path.is_file():
            raise ValueError("Pinned code bundle ZIP is required")
        if not self.worker_script_path.is_file():
            raise ValueError("Panorama worker script (tools/modal_panorama_worker.py) is missing")

        # Verify all pinned SHA-256 digests before creating or uploading to the Sandbox
        for path, key in (
            (normalized_video, "video_sha256"),
            (self.weights_path, "weights_sha256"),
            (frame_map_path, "frame_map_sha256"),
            (self.worker_script_path, "worker_script_sha256"),
        ):
            if sha256_file(path) != payload[key]:
                raise ValueError(f"Pinned panorama input digest mismatch for {key}")
        if _verified_code_bundle(self.code_bundle_path) != payload["code_sha256"]:
            raise ValueError("Pinned panorama source bundle changed")

        if self.sandbox_controller is None:
            if not self.is_ready():
                raise RuntimeError("Live Panorama dispatch prerequisites are not verified")
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            claim = self.spending_ledger.reserve(
                "modal", job_id, fingerprint=fingerprint, worst_case_usd="1.80"
            )
            if claim.get("created") is not True:
                raise RuntimeError("A prior Panorama attempt already holds this spending reservation")

        export_dir = job_dir / "worker_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        receipt: dict[str, Any] = {
            "status": "prepared",
            "gpu": cfg.gpu,
            "image_id": cfg.image_id,
            "worker_terminated": "not_created",
            "job": payload,
        }
        (export_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

        sandbox = None
        try:
            if self.sandbox_controller is not None:
                sandbox = self.sandbox_controller(job_dir, job_id, payload, cfg)
            else:
                sandbox = self._create_live_modal_sandbox(job_id)

            allocated = time.time()
            receipt.update(
                status="running",
                worker_id=getattr(sandbox, "object_id", "sandbox"),
                allocated_epoch=allocated,
                worker_terminated="pending",
            )
            (export_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

            inputs = {
                "video24.mp4": normalized_video,
                "weights.pth": self.weights_path,
                "code.zip": self.code_bundle_path,
                "frame-map.jsonl": frame_map_path,
            }
            for remote_name, local_path in inputs.items():
                sandbox.filesystem.copy_from_local(local_path, "/work/" + remote_name)
            sandbox.filesystem.copy_from_local(self.worker_script_path, "/work/panorama-worker.py")
            sandbox.filesystem.write_text(json.dumps(payload), "/work/panorama-job.json")
            sandbox.exec("python", "-I", "/work/panorama-worker.py", timeout=cfg.exec_timeout_s, text=False)

            last_seq = 0
            failures = 0
            complete = False
            deadline = allocated + cfg.exec_timeout_s
            while time.time() <= deadline:
                try:
                    raw_state = sandbox.filesystem.read_text("/work/panorama/worker-state.json")
                    state = parse_worker_status(raw_state)
                    failures = 0
                    if state["event_seq"] > last_seq:
                        last_seq = state["event_seq"]
                        stage_name = str(state.get("stage", "executing_worker"))
                        progress = 0.6 if stage_name == "analysis" else (0.45 if stage_name == "precision_parity" else 0.35)
                        events.append(
                            {
                                "stage": stage_name,
                                "event": state.get("status", "running"),
                                "event_seq": last_seq,
                                "timestamp": time.time(),
                            }
                        )
                        write_job_status(
                            job_dir,
                            job_id,
                            "processing",
                            stage=stage_name,
                            progress=progress,
                            events=list(events),
                        )
                    if state["status"] == "failed":
                        raise RuntimeError("Panorama worker reported failure")
                    if state["stage"] == "complete" and state["status"] == "complete":
                        complete = True
                        break
                except RuntimeError:
                    raise
                except Exception:
                    failures += 1
                    exited = sandbox.poll()
                    if exited is not None or failures > 6:
                        raise RuntimeError("Cannot verify live panorama worker status") from None
                if cfg.poll_interval_s > 0:
                    time.sleep(cfg.poll_interval_s)

            if not complete:
                raise TimeoutError("Bounded panorama worker deadline reached")

            summary = json.loads(sandbox.filesystem.read_text("/work/panorama/summary.json"))
            if (
                summary.get("frames") != payload["expected_frames"]
                or summary.get("checkpoint_sha256") != payload["weights_sha256"]
                or summary.get("source_sha256") != payload["video_sha256"]
                or summary.get("code_sha256") != payload["code_sha256"]
                or summary.get("frame_map_sha256") != payload["frame_map_sha256"]
            ):
                raise ValueError("Panorama worker result lineage mismatch")

            for name, limit in (("overlay24.mp4", 700 * 1024**2), ("evidence.zip", 128 * 1024**2)):
                stat_info = sandbox.filesystem.stat("/work/panorama/" + name)
                if not 0 < stat_info.size <= limit:
                    raise ValueError(f"Panorama artifact {name} exceeds reviewed transfer bound")
                sandbox.filesystem.copy_to_local("/work/panorama/" + name, export_dir / name)

            _extract_selected_archive(
                export_dir / "evidence.zip",
                export_dir / "evidence",
                {".json", ".jsonl", ".jpg", ".html", ".log"},
                512 * 1024**2,
            )
            overlay_sha256 = sha256_file(export_dir / "overlay24.mp4")
            receipt.update(
                status="complete",
                run_id=summary["run_id"],
                frames=summary["frames"],
                result_sha256=overlay_sha256,
                summary=summary,
            )
            (export_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        except BaseException as exc:
            receipt.update(status="failed", failure_type=type(exc).__name__)
            (export_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise
        finally:
            if sandbox is not None:
                try:
                    sandbox.terminate(wait=True)
                    receipt["worker_terminated"] = True
                except Exception:
                    receipt["worker_terminated"] = "unknown"
                (export_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

        original_sha256 = sha256_file(source_video)
        if original_sha256 != payload["original_video_sha256"]:
            raise ValueError("Original recording changed during worker execution")
        return {
            "export_dir": export_dir,
            "source_video": source_video,
            "normalized_video": normalized_video,
            "frame_map_path": frame_map_path,
            "normalization_metadata_path": normalization_metadata_path,
            "expected_original_sha256": original_sha256,
        }
