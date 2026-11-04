from __future__ import annotations

import hashlib
from fractions import Fraction
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

from touchline.adapters.panorama_worker import (
    PINNED_MODAL_IMAGE_ID,
    PINNED_RESNET50_WEIGHTS_SHA256,
    PanoramaJobConfig,
    PanoramaWorkerAdapter,
    validate_panorama_job_spec,
)
from touchline.studio import run_studio_server


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _test_code_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("pyproject.toml", "tools/run_offline_checks.py",
                     "tools/modal_panorama_worker.py", "src/touchline/adapters/rcnn_comparison.py"):
            archive.writestr(name, b"# offline fixture\n")


def _fresh_mp4(*, width: int = 1600, height: int = 500, frames: int = 8) -> bytes:
    """Encode an actual short 25-FPS MP4, independent of the worker fixture."""
    import av
    import io
    from PIL import Image

    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = Image.new("RGB", (width, height), (20 + index, 90, 30))
            frame = av.VideoFrame.from_image(image)
            frame.pts = index
            frame.time_base = Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _build_synthetic_worker_bundle(
    job_dir: Path,
    *,
    video_bytes: bytes = b"synthetic-raw-panorama-mp4-stream-0001",
    norm_video_bytes: bytes = b"normalized-24fps-4096x1080-video",
    overlay_bytes: bytes = b"annotated-panorama-overlay24-video-stream",
    worker_run_id: str = "worker-internal-uuid-0001",
    weights_sha256: str = PINNED_RESNET50_WEIGHTS_SHA256,
    code_sha256: str = "c" * 64,
    frame_count: int = 8,
    use_existing_inputs: bool = False,
    geometry: tuple[int, int] = (4096, 1080),
) -> dict:
    """Create normalized inputs and a synthetic worker export matching tools/modal_panorama_worker.py."""
    norm_dir = job_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    source_video = job_dir / "source_video.mp4"
    if not use_existing_inputs:
        source_video.write_bytes(video_bytes)
    video24 = norm_dir / "video24.mp4"
    if not use_existing_inputs:
        video24.write_bytes(norm_video_bytes)

    frame_map_rows = []
    for idx in range(frame_count):
        frame_map_rows.append({
            "output_frame_index": idx,
            "source_frame_index": idx,
            "source_pts": 300 + idx * 1000,
            "source_time_base": [1, 24000],
            "source_relative_time": [idx, 24],
        })
    frame_map_path = norm_dir / "frame-map.jsonl"
    if not use_existing_inputs:
        frame_map_path.write_text(
            "\n".join(json.dumps(r) for r in frame_map_rows) + "\n",
            encoding="utf-8",
        )

    source_sha = _sha256_bytes(source_video.read_bytes())
    norm_sha = _sha256_bytes(video24.read_bytes())
    map_sha = _sha256_bytes(frame_map_path.read_bytes())
    overlay_sha = _sha256_bytes(overlay_bytes)

    normalization = {
        "schema": "touchline.video-normalization.v1",
        "status": "complete",
        "source_sha256": source_sha,
        "output_sha256": norm_sha,
        "frame_map_sha256": map_sha,
        "output_frames": frame_count,
        "geometry": list(geometry),
    }
    normalization_path = norm_dir / "normalization.json"
    if use_existing_inputs:
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    else:
        normalization_path.write_text(
            json.dumps(normalization, indent=2) + "\n",
            encoding="utf-8",
        )

    raw_proposals = [
        {
            "frame_index": 0,
            "bbox_xyxy": [420.0, 310.0, 468.0, 430.0],
            "score": 0.91,
        },
        {
            "frame_index": 0,
            "bbox_xyxy": [1220.0, 340.0, 1270.0, 465.0],
            "score": 0.84,
        },
        {
            "frame_index": 4,
            "bbox_xyxy": [432.0, 312.0, 480.0, 432.0],
            "score": 0.89,
        },
    ]

    observations = [
        {
            "run_id": worker_run_id,
            "frame_index": 0,
            "timestamp_s": 0.0,
            "track_id": 1,
            "bbox_xyxy": [420.0, 310.0, 468.0, 430.0],
            "detector_bbox_xyxy": [420.0, 310.0, 468.0, 430.0],
            "evidence": "detector_anchor",
        },
        {
            "run_id": worker_run_id,
            "frame_index": 0,
            "timestamp_s": 0.0,
            "track_id": 2,
            "bbox_xyxy": [1220.0, 340.0, 1270.0, 465.0],
            "detector_bbox_xyxy": [1220.0, 340.0, 1270.0, 465.0],
            "evidence": "detector_anchor",
        },
        {
            "run_id": worker_run_id,
            "frame_index": 1,
            "timestamp_s": round(1 / 24.0, 6),
            "track_id": 1,
            "bbox_xyxy": [423.0, 310.5, 471.0, 430.5],
            "detector_bbox_xyxy": None,
            "detector_score": None,
            "evidence": "camera_prediction",
        },
        {
            "run_id": worker_run_id,
            "frame_index": 4,
            "timestamp_s": round(4 / 24.0, 6),
            "track_id": 1,
            "bbox_xyxy": [432.0, 312.0, 480.0, 432.0],
            "detector_bbox_xyxy": [432.0, 312.0, 480.0, 432.0],
            "evidence": "detector_anchor",
        },
    ]

    coverage_rows = [{"frame_index": idx, "timestamp_s": idx / 24,
                      "detector_sample": idx % 4 == 0} for idx in range(frame_count)]

    summary = {
        "schema": "touchline.panorama-review.v1",
        "status": "complete",
        "run_id": worker_run_id,
        "frames": frame_count,
        "source_fps": 24,
        "source_geometry": list(geometry),
        "source_sha256": norm_sha,
        "checkpoint_sha256": weights_sha256,
        "code_sha256": code_sha256,
        "frame_map_sha256": map_sha,
        "identity_confirmed": False,
        "duration_s": frame_count / 24.0,
        "counters": {
            "detector_frames": 2,
            "raw_proposals": len(raw_proposals),
            "detector_anchors": 3,
            "motion_predictions": 1,
        },
    }

    receipt = {
        "status": "complete",
        "run_id": worker_run_id,
        "summary": summary,
        "result_sha256": overlay_sha,
    }

    return {
        "source_video": source_video,
        "normalized_video": video24,
        "frame_map_path": frame_map_path,
        "normalization_metadata_path": normalization_path,
        "expected_original_sha256": source_sha,
        "normalization": normalization,
        "frame_map_rows": frame_map_rows,
        "raw_proposals": raw_proposals,
        "observations": observations,
        "coverage_rows": coverage_rows,
        "summary": summary,
        "receipt": receipt,
        "overlay_bytes": overlay_bytes,
    }


def _write_worker_export_to_dir(export_dir: Path, bundle: dict) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "overlay24.mp4").write_bytes(bundle["overlay_bytes"])
    (export_dir / "receipt.json").write_text(json.dumps(bundle["receipt"], indent=2) + "\n", encoding="utf-8")
    ev = export_dir / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "summary.json").write_text(json.dumps(bundle["summary"], indent=2) + "\n", encoding="utf-8")
    (ev / "raw_proposals.jsonl").write_text(
        "\n".join(json.dumps(r) for r in bundle["raw_proposals"]) + "\n",
        encoding="utf-8",
    )
    (ev / "observations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in bundle["observations"]) + "\n",
        encoding="utf-8",
    )
    (ev / "coverage.jsonl").write_text(
        "\n".join(json.dumps(r) for r in bundle["coverage_rows"]) + "\n",
        encoding="utf-8",
    )
    return export_dir


class FakeModalFilesystem:
    def __init__(self, bundle: dict, *, fail_worker: bool = False) -> None:
        self.bundle = bundle
        self.fail_worker = fail_worker
        self.uploaded_files: dict[str, bytes] = {}
        self.written_texts: dict[str, str] = {}

    def copy_from_local(self, local_path: Path, remote_path: str) -> None:
        self.uploaded_files[remote_path] = Path(local_path).read_bytes()

    def write_text(self, text: str, remote_path: str) -> None:
        self.written_texts[remote_path] = text

    def read_text(self, remote_path: str) -> str:
        if remote_path == "/work/panorama/worker-state.json":
            if self.fail_worker:
                return json.dumps({
                    "status": "failed",
                    "stage": "analysis",
                    "event_seq": 1,
                    "worker_epoch": 1700000000.0,
                    "error_category": "InferenceFailure",
                })
            return json.dumps({
                "status": "complete",
                "stage": "complete",
                "event_seq": 2,
                "worker_epoch": 1700000000.0,
                "frames": self.bundle["summary"]["frames"],
                "ids": 2,
            })
        if remote_path == "/work/panorama/summary.json":
            return json.dumps(self.bundle["summary"])
        raise FileNotFoundError(remote_path)

    def stat(self, remote_path: str):
        if remote_path.endswith("overlay24.mp4"):
            return SimpleNamespace(size=len(self.bundle["overlay_bytes"]))
        if remote_path.endswith("evidence.zip"):
            return SimpleNamespace(size=1024)
        raise FileNotFoundError(remote_path)

    def copy_to_local(self, remote_path: str, local_path: Path) -> None:
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if remote_path.endswith("overlay24.mp4"):
            dst.write_bytes(self.bundle["overlay_bytes"])
        elif remote_path.endswith("evidence.zip"):
            with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("summary.json", json.dumps(self.bundle["summary"], indent=2))
                zf.writestr(
                    "raw_proposals.jsonl",
                    "\n".join(json.dumps(r) for r in self.bundle["raw_proposals"]) + "\n",
                )
                zf.writestr(
                    "observations.jsonl",
                    "\n".join(json.dumps(r) for r in self.bundle["observations"]) + "\n",
                )
                zf.writestr(
                    "coverage.jsonl",
                    "\n".join(json.dumps(r) for r in self.bundle["coverage_rows"]) + "\n",
                )


class FakeModalSandbox:
    def __init__(self, bundle: dict, *, fail_worker: bool = False) -> None:
        self.object_id = "sb-mock-panorama-001"
        self.filesystem = FakeModalFilesystem(bundle, fail_worker=fail_worker)
        self.exec_calls: list[tuple[tuple[str, ...], dict]] = []
        self.terminated = False

    def exec(self, *args: str, **kwargs):
        self.exec_calls.append((args, kwargs))

    def poll(self):
        return None

    def terminate(self, *, wait: bool = True) -> None:
        self.terminated = True


class PanoramaBackendOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.tmp.name) / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _ready_mock_adapter(self, controller):
        weights = Path(self.tmp.name) / "mock-weights.pth"
        weights.write_bytes(b"mock-weights")
        code = Path(self.tmp.name) / "mock-code.zip"
        _test_code_bundle(code)
        return PanoramaWorkerAdapter(
            self.runs_root, sandbox_controller=controller,
            config=PanoramaJobConfig(expected_frames=8, geometry=(1600, 500),
                                     weights_sha256=_sha256_bytes(weights.read_bytes())),
            weights_path=weights, code_bundle_path=code,
        )

    def _start_server(
        self,
        adapter=None,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        auth_token: str | None = None,
        max_upload_bytes: int = 1024 * 1024 * 1024,
        roster_authorizer=None,
    ):
        target_port = port or self.port
        t = threading.Thread(
            target=run_studio_server,
            kwargs={
                "runs_root": self.runs_root,
                "host": host,
                "port": target_port,
                "worker_adapter": adapter,
                "auth_token": auth_token,
                "max_upload_bytes": max_upload_bytes,
                "roster_authorizer": roster_authorizer,
            },
            daemon=True,
        )
        t.start()
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        for _ in range(50):
            try:
                req = Request(f"http://127.0.0.1:{target_port}/api/runs", headers=headers)
                with urlopen(req, timeout=0.2) as r:
                    if r.status in (200, 401):
                        break
            except Exception:
                time.sleep(0.05)
        return target_port

    def test_job_spec_validation_enforces_pinned_panorama_resnet_contract(self):
        """Priority 1: Validate pinned ResNet50 Faster R-CNN v2 weights, image ID, geometry, and SHA-256 digests."""
        adapter = PanoramaWorkerAdapter(self.runs_root)
        self.assertEqual(adapter.config.image_id, PINNED_MODAL_IMAGE_ID)
        self.assertEqual(adapter.config.weights_sha256, PINNED_RESNET50_WEIGHTS_SHA256)

        valid_spec = adapter.create_job_payload(
            job_id="upload-1700000000-abc123",
            video_sha256="a" * 64,
            code_sha256="b" * 64,
            frame_map_sha256="c" * 64,
            worker_script_sha256="d" * 64,
            original_video_sha256="e" * 64,
        )
        validated = validate_panorama_job_spec(valid_spec)
        self.assertEqual(validated["weights_sha256"], PINNED_RESNET50_WEIGHTS_SHA256)
        self.assertEqual(validated["geometry"], [4096, 1080])
        self.assertEqual(validated["expected_frames"], 5760)
        self.assertEqual(validated["detector_stride"], 4)
        self.assertEqual(validated["session_id_cap"], 24)
        self.assertTrue(validated["stationary_camera"])

        with self.assertRaisesRegex(ValueError, "weights_sha256"):
            bad_weights = dict(valid_spec, weights_sha256="f" * 64)
            validate_panorama_job_spec(bad_weights)

        with self.assertRaisesRegex(ValueError, "geometry"):
            bad_geom = dict(valid_spec, geometry=[1920, 1080])
            validate_panorama_job_spec(bad_geom)

    def test_mocked_modal_sandbox_controller_dispatches_and_converts_to_canonical_run(self):
        """Priority 1 & 3: Parameterized Modal Sandbox dispatch + conversion to canonical touchline.run.v1/observation.v1."""
        job_id = "upload-1700000001-aa11bb"
        job_dir = self.runs_root / job_id

        weights_file = Path(self.tmp.name) / "resnet50_weights.pth"
        weights_bytes = b"synthetic-resnet50-fasterrcnn-weights"
        weights_file.write_bytes(weights_bytes)
        weights_sha = _sha256_bytes(weights_bytes)

        code_file = Path(self.tmp.name) / "code.zip"
        _test_code_bundle(code_file)
        code_sha = _sha256_bytes(code_file.read_bytes())

        bundle = _build_synthetic_worker_bundle(
            job_dir,
            weights_sha256=weights_sha,
            code_sha256=code_sha,
            frame_count=8,
        )

        fake_sb = FakeModalSandbox(bundle)
        controller_calls = []

        def fake_sandbox_controller(j_dir, j_id, payload, cfg):
            controller_calls.append((j_dir, j_id, payload, cfg))
            return fake_sb

        cfg = PanoramaJobConfig(
            expected_frames=8,
            weights_sha256=weights_sha,
            poll_interval_s=0.0,
        )
        adapter = PanoramaWorkerAdapter(
            self.runs_root,
            config=cfg,
            sandbox_controller=fake_sandbox_controller,
            weights_path=weights_file,
            code_bundle_path=code_file,
        )
        self.assertTrue(adapter.is_ready())

        spec = adapter.create_job_payload(
            job_id=job_id,
            video_sha256=bundle["normalization"]["output_sha256"],
            code_sha256=code_sha,
            frame_map_sha256=bundle["normalization"]["frame_map_sha256"],
            worker_script_sha256=_sha256_bytes(adapter.worker_script_path.read_bytes()),
            original_video_sha256=bundle["expected_original_sha256"],
        )
        res = adapter.run_job_sync(job_dir, job_id, spec)

        self.assertEqual(res["status"], "completed")
        self.assertTrue(fake_sb.terminated)
        self.assertEqual(len(controller_calls), 1)
        self.assertEqual(controller_calls[0][3].image_id, PINNED_MODAL_IMAGE_ID)
        self.assertIn("/work/panorama-worker.py", fake_sb.filesystem.uploaded_files)
        self.assertIn("/work/panorama-job.json", fake_sb.filesystem.written_texts)

        # Verify canonical artifacts on disk
        manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "touchline.run.v1")
        self.assertEqual(manifest["run_id"], job_id)
        self.assertFalse(manifest["identity_confirmed"])
        self.assertFalse(manifest["synthetic"])
        self.assertEqual(manifest["observation_count"], 3)
        self.assertEqual(manifest["motion_prediction_count"], 1)
        self.assertEqual(manifest["artifacts"]["annotated_video"]["filename"], "overlay24.mp4")

        observations = [
            json.loads(line)
            for line in (job_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(observations), 3)
        for obs in observations:
            self.assertEqual(obs["schema_version"], "touchline.observation.v1")
            self.assertEqual(obs["run_id"], job_id)
            self.assertIsNone(obs["player_id"])
            self.assertEqual(obs["association_state"], "unreviewed")

        predictions = [
            json.loads(line)
            for line in (job_dir / "motion_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(predictions), 1)

        # Also verify Sandbox is terminated even when worker fails
        fail_job_id = "upload-1700000002-cc22dd"
        fail_job_dir = self.runs_root / fail_job_id
        fail_bundle = _build_synthetic_worker_bundle(
            fail_job_dir,
            weights_sha256=weights_sha,
            code_sha256=code_sha,
            frame_count=8,
        )
        fail_sb = FakeModalSandbox(fail_bundle, fail_worker=True)
        fail_adapter = PanoramaWorkerAdapter(
            self.runs_root,
            config=cfg,
            sandbox_controller=lambda *a: fail_sb,
            weights_path=weights_file,
            code_bundle_path=code_file,
        )
        fail_spec = fail_adapter.create_job_payload(
            job_id=fail_job_id,
            video_sha256=fail_bundle["normalization"]["output_sha256"],
            code_sha256=code_sha,
            frame_map_sha256=fail_bundle["normalization"]["frame_map_sha256"],
            worker_script_sha256=_sha256_bytes(fail_adapter.worker_script_path.read_bytes()),
            original_video_sha256=fail_bundle["expected_original_sha256"],
        )
        with self.assertRaisesRegex(RuntimeError, "Panorama worker reported failure"):
            fail_adapter.run_job_sync(fail_job_dir, fail_job_id, fail_spec)
        self.assertTrue(fail_sb.terminated)

        malformed_id = "upload-1700000003-dd33ee"
        malformed_dir = self.runs_root / malformed_id
        malformed = _build_synthetic_worker_bundle(
            malformed_dir, weights_sha256=weights_sha, code_sha256=code_sha, frame_count=8
        )
        malformed["coverage_rows"][0]["frame_index"] = 7
        malformed_sb = FakeModalSandbox(malformed)
        malformed_adapter = PanoramaWorkerAdapter(
            self.runs_root, config=cfg, sandbox_controller=lambda *args: malformed_sb,
            weights_path=weights_file, code_bundle_path=code_file,
        )
        malformed_spec = malformed_adapter.create_job_payload(
            job_id=malformed_id, video_sha256=malformed["normalization"]["output_sha256"],
            code_sha256=code_sha, frame_map_sha256=malformed["normalization"]["frame_map_sha256"],
            worker_script_sha256=_sha256_bytes(malformed_adapter.worker_script_path.read_bytes()),
            original_video_sha256=malformed["expected_original_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            malformed_adapter.run_job_sync(malformed_dir, malformed_id, malformed_spec)
        self.assertTrue(malformed_sb.terminated)
        self.assertFalse((malformed_dir / "manifest.json").exists())

    def test_upload_lifecycle_coach_review_corrections_and_restart_persistence(self):
        """Priority 2, 3, 4: End-to-end Studio upload, canonical review, immutable child corrections, and server restart persistence."""
        video_payload = _fresh_mp4()
        checkpoint = Path(self.tmp.name) / "checkpoint.pth"
        checkpoint.write_bytes(b"test-checkpoint")
        code = Path(self.tmp.name) / "code.zip"
        _test_code_bundle(code)
        checkpoint_sha = _sha256_bytes(checkpoint.read_bytes())
        code_sha = _sha256_bytes(code.read_bytes())
        sandboxes = []

        def mocked_controller(job_dir: Path, job_id: str, job_spec: dict, cfg):
            bundle = _build_synthetic_worker_bundle(
                job_dir,
                use_existing_inputs=True,
                geometry=cfg.geometry,
                weights_sha256=checkpoint_sha,
                code_sha256=code_sha,
                frame_count=8,
            )
            sandbox = FakeModalSandbox(bundle)
            sandboxes.append(sandbox)
            return sandbox

        adapter = PanoramaWorkerAdapter(
            self.runs_root,
            sandbox_controller=mocked_controller,
            config=PanoramaJobConfig(expected_frames=8, geometry=(1600, 500),
                                     weights_sha256=checkpoint_sha, poll_interval_s=0),
            weights_path=checkpoint,
            code_bundle_path=code,
        )
        self._start_server(adapter=adapter, roster_authorizer=lambda run_id, player_id: player_id == "P07")

        # 1. Readiness check reports ready + upload_endpoint
        with urlopen(f"http://127.0.0.1:{self.port}/api/analyze/readiness") as r:
            readiness = json.loads(r.read().decode("utf-8"))
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["upload_endpoint"], "/api/runs")

        # 2. Upload MP4 to POST /api/runs
        req = Request(
            f"http://127.0.0.1:{self.port}/api/runs",
            data=video_payload,
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(len(video_payload)),
            },
            method="POST",
        )
        with urlopen(req) as resp:
            self.assertEqual(resp.status, 202)
            upload_data = json.loads(resp.read().decode("utf-8"))

        job_id = upload_data["job_id"]
        self.assertTrue(job_id.startswith("upload-"))
        self.assertEqual(upload_data["source_sha256"], _sha256_bytes(video_payload))

        # 3. Poll job status
        with urlopen(f"http://127.0.0.1:{self.port}/api/job/{job_id}") as r:
            job_state = json.loads(r.read().decode("utf-8"))
        for _ in range(100):
            if job_state["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
            with urlopen(f"http://127.0.0.1:{self.port}/api/job/{job_id}") as response:
                job_state = json.loads(response.read())
        self.assertEqual(job_state["status"], "completed")
        self.assertEqual(len(sandboxes), 1)
        self.assertTrue(sandboxes[0].terminated)
        payload = json.loads(sandboxes[0].filesystem.written_texts["/work/panorama-job.json"])
        self.assertEqual(payload["original_video_sha256"], _sha256_bytes(video_payload))
        self.assertEqual(payload["video_sha256"], _sha256_bytes(
            (self.runs_root / job_id / "normalized" / "video24.mp4").read_bytes()))
        self.assertNotEqual(payload["video_sha256"], payload["original_video_sha256"])
        self.assertEqual(payload["frame_map_sha256"], _sha256_bytes(
            (self.runs_root / job_id / "normalized" / "frame-map.jsonl").read_bytes()))
        self.assertEqual(payload["worker_script_sha256"], _sha256_bytes(adapter.worker_script_path.read_bytes()))
        self.assertEqual(job_state["stage"], "completed")
        self.assertEqual(job_state["observation_count"], 3)

        # 4. Inspect canonical coach-review output via GET /api/runs/{job_id}
        with urlopen(f"http://127.0.0.1:{self.port}/api/runs/{job_id}") as r:
            run_data = json.loads(r.read().decode("utf-8"))
        self.assertEqual(run_data["run_id"], job_id)
        self.assertTrue(run_data["annotated_video_verified"])
        self.assertEqual(run_data["annotated_video_url"], f"/runs/{job_id}/overlay24.mp4")
        self.assertEqual(len(run_data["frame_map"]), 8)
        self.assertEqual(len(run_data["observations"]), 3)
        self.assertFalse(run_data["manifest"]["identity_confirmed"])
        for obs in run_data["observations"]:
            self.assertEqual(obs["schema_version"], "touchline.observation.v1")
            self.assertEqual(obs["run_id"], job_id)
            self.assertIsNone(obs["player_id"])
            self.assertEqual(obs["association_state"], "unreviewed")

        # 5. Stream annotated video with HTTP 206 Range request
        range_req = Request(
            f"http://127.0.0.1:{self.port}/runs/{job_id}/overlay24.mp4",
            headers={"Range": "bytes=0-8"},
        )
        with urlopen(range_req) as r:
            self.assertEqual(r.status, 206)
            self.assertEqual(r.read(), b"annotated")

        # 6. Reject invalid correction that passes a raw track ID as player_id
        bad_corr_body = json.dumps({
            "local_track_id": 1,
            "player_id": "track_1",
            "decision": "confirm",
            "reviewer": "coach_alex",
        }).encode("utf-8")
        bad_corr_req = Request(
            f"http://127.0.0.1:{self.port}/api/runs/{job_id}/corrections",
            data=bad_corr_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(bad_corr_body))},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(bad_corr_req)
        self.assertEqual(ctx.exception.code, 400)

        # A correction cannot force an unbounded read or create a child run.
        before_children = set(self.runs_root.iterdir())
        oversized_correction = Request(
            f"http://127.0.0.1:{self.port}/api/runs/{job_id}/corrections",
            data=b" " * (64 * 1024 + 1),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(oversized_correction)
        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(set(self.runs_root.iterdir()), before_children)

        # 7. Submit valid coach correction confirming track 1 -> P07
        corr_body = json.dumps({
            "local_track_id": 1,
            "player_id": "P07",
            "decision": "confirm",
            "reviewer": "coach_alex",
            "note": "Verified #7 winger on right touchline",
        }).encode("utf-8")
        corr_req = Request(
            f"http://127.0.0.1:{self.port}/api/runs/{job_id}/corrections",
            data=corr_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(corr_body))},
            method="POST",
        )
        with urlopen(corr_req) as r:
            self.assertEqual(r.status, 201)
            corr_res = json.loads(r.read().decode("utf-8"))

        child_id = corr_res["child_run_id"]
        self.assertEqual(corr_res["parent_run_id"], job_id)
        self.assertEqual(corr_res["corrections_count"], 1)

        # 8. Verify child run has track 1 confirmed as P07, track 2 still unreviewed/None, and identity_confirmed=False
        with urlopen(f"http://127.0.0.1:{self.port}/api/runs/{child_id}") as r:
            child_data = json.loads(r.read().decode("utf-8"))
        self.assertEqual(child_data["manifest"]["parent_run_id"], job_id)
        self.assertEqual(child_data["manifest"]["revision"], 2)
        self.assertFalse(child_data["manifest"]["identity_confirmed"])
        for obs in child_data["observations"]:
            self.assertEqual(obs["run_id"], child_id)
            if obs["local_track_id"] == 1:
                self.assertEqual(obs["player_id"], "P07")
                self.assertEqual(obs["association_state"], "confirmed")
            else:
                self.assertIsNone(obs["player_id"])
                self.assertEqual(obs["association_state"], "unreviewed")

        # 9. Verify parent run remains completely unchanged
        with urlopen(f"http://127.0.0.1:{self.port}/api/runs/{job_id}") as r:
            parent_data = json.loads(r.read().decode("utf-8"))
        self.assertEqual(parent_data["manifest"].get("revision", 1), 1)
        self.assertEqual(parent_data["manifest"].get("corrections", []), [])
        for obs in parent_data["observations"]:
            self.assertIsNone(obs["player_id"])
            self.assertEqual(obs["association_state"], "unreviewed")

        # 10. Simulate backend restart on a new port over the same runs_root
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock2.bind(("127.0.0.1", 0))
        port2 = sock2.getsockname()[1]
        sock2.close()
        self._start_server(adapter=PanoramaWorkerAdapter(self.runs_root), port=port2)

        with urlopen(f"http://127.0.0.1:{port2}/api/runs") as r:
            restarted_sessions = json.loads(r.read().decode("utf-8"))
        restarted_ids = {s["id"] for s in restarted_sessions}
        self.assertIn(job_id, restarted_ids)
        self.assertIn(child_id, restarted_ids)

    def test_security_guards_reject_json_path_upload_non_loopback_and_unauthenticated_requests(self):
        """Priority 5: Reject JSON local path uploads, refuse non-loopback host without auth_token, and enforce Bearer auth."""
        # 1. Refuses non-loopback binding without auth_token
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            run_studio_server(self.runs_root, host="0.0.0.0", port=self.port)

        # 2. Start server with auth_token
        token = "studio-secret-token-2026"
        self._start_server(
            adapter=PanoramaWorkerAdapter(self.runs_root),
            auth_token=token,
            max_upload_bytes=128,
        )

        # Unauthenticated API call returns 401
        with self.assertRaises(HTTPError) as ctx:
            urlopen(f"http://127.0.0.1:{self.port}/api/runs")
        self.assertEqual(ctx.exception.code, 401)

        # Authenticated API call returns 200
        authed_req = Request(
            f"http://127.0.0.1:{self.port}/api/runs",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urlopen(authed_req) as r:
            self.assertEqual(r.status, 200)

        # 3. JSON body with source_path on POST /api/runs is rejected with 400
        secret_file = Path(self.tmp.name) / "sensitive_host_file.txt"
        secret_file.write_text("do-not-copy-me", encoding="utf-8")
        json_upload = json.dumps({"source_path": str(secret_file)}).encode("utf-8")
        json_req = Request(
            f"http://127.0.0.1:{self.port}/api/runs",
            data=json_upload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(json_upload)),
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(json_req)
        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(list(self.runs_root.iterdir()), [])

        # 4. Upload exceeding max_upload_bytes (128 bytes) is rejected with 400
        oversized = b"x" * 256
        over_req = Request(
            f"http://127.0.0.1:{self.port}/api/runs",
            data=oversized,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "video/mp4",
                "Content-Length": str(len(oversized)),
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(over_req)
        self.assertEqual(ctx.exception.code, 400)

    def test_corrupt_fresh_upload_fails_before_mock_sandbox(self):
        calls = []
        adapter = self._ready_mock_adapter(lambda *args: calls.append(args))
        self._start_server(adapter=adapter)
        request = Request(f"http://127.0.0.1:{self.port}/api/runs", data=b"not an MP4",
                          headers={"Content-Type": "video/mp4"}, method="POST")
        with urlopen(request) as response:
            job = json.loads(response.read())
        job_dir = self.runs_root / job["job_id"]
        for _ in range(100):
            with urlopen(f"http://127.0.0.1:{self.port}/api/job/{job['job_id']}") as response:
                status = json.loads(response.read())
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(calls, [])
        self.assertEqual((job_dir / "source_video.mp4").read_bytes(), b"not an MP4")
        self.assertFalse((job_dir / "manifest.json").exists())

    def test_wrong_checkpoint_is_rejected_before_upload(self):
        calls = []
        adapter = self._ready_mock_adapter(lambda *args: calls.append(args))
        adapter.weights_path.write_bytes(b"changed-checkpoint")
        self._start_server(adapter=adapter)
        with urlopen(f"http://127.0.0.1:{self.port}/api/analyze/readiness") as response:
            self.assertFalse(json.loads(response.read())["ready"])
        request = Request(f"http://127.0.0.1:{self.port}/api/runs", data=_fresh_mp4(),
                          headers={"Content-Type": "video/mp4"}, method="POST")
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 503)
        self.assertEqual(calls, [])
        self.assertEqual(list(self.runs_root.iterdir()), [])

    def test_run_id_collision_keeps_existing_directory_byte_identical(self):
        self._start_server(adapter=self._ready_mock_adapter(lambda *args: None))
        fixed = "abcdef" * 5 + "ab"
        run_id = "upload-1700000000-" + fixed
        existing = self.runs_root / run_id
        existing.mkdir()
        (existing / "source_video.mp4").write_bytes(b"existing source")
        (existing / "manifest.json").write_bytes(b"existing manifest")
        before = {item.name: item.read_bytes() for item in existing.iterdir()}
        with patch("touchline.studio._new_panorama_job_id", return_value=run_id):
            request = Request(f"http://127.0.0.1:{self.port}/api/runs", data=b"new video",
                              headers={"Content-Type": "video/mp4"}, method="POST")
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 409)
        self.assertEqual({item.name: item.read_bytes() for item in existing.iterdir()}, before)

    def test_rejected_upload_closes_http11_connection_with_unread_body(self):
        self._start_server(adapter=PanoramaWorkerAdapter(self.runs_root), max_upload_bytes=4)
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
            connection.settimeout(2)
            connection.sendall(
                b"POST /api/runs HTTP/1.1\r\nHost: localhost\r\nContent-Type: video/mp4\r\n"
                b"Content-Length: 9\r\nConnection: keep-alive\r\n\r\n"
                b"123456789GET /api/runs HTTP/1.1\r\nHost: localhost\r\n\r\n"
            )
            chunks = []
            while True:
                part = connection.recv(4096)
                if not part:
                    break
                chunks.append(part)
        reply = b"".join(chunks)
        self.assertIn(b"HTTP/1.1 400", reply)
        self.assertNotIn(b"HTTP/1.1 200", reply)
        self.assertEqual(list(self.runs_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
