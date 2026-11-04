"""Bounded, server-side Meta selected-frame calls and review telemetry.

These are model proposals linked to a source-video frame, not confirmed player
identities or Panorama tracking output. No credential enters a run artifact.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib import error, request
import uuid


MODELS = {"sam": "sam-3.1", "spark": "muse-spark-1.3", "wonder": "muse-wonder-1.0"}
API_URL = "https://api.meta.ai/v1/responses"


class MetaCallError(RuntimeError):
    def __init__(self, code: str, http_status: int | None = None):
        super().__init__(code)
        self.code = code
        self.http_status = http_status


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def activity_dir(run_dir: Path) -> Path:
    folder = run_dir / "meta_activity"
    if folder.is_symlink():
        raise ValueError("Unsafe activity directory")
    folder.mkdir(mode=0o700, exist_ok=True)
    return folder


def save_activity(run_dir: Path, record: dict):
    folder = activity_dir(run_dir)
    target = folder / (record["activity_id"] + ".json")
    if target.is_symlink():
        raise ValueError("Unsafe activity record")
    fd, name = tempfile.mkstemp(prefix=".activity-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def list_activities(run_dir: Path) -> list[dict]:
    folder = run_dir / "meta_activity"
    if not folder.is_dir() or folder.is_symlink():
        return []
    records = []
    for path in folder.glob("*.json"):
        if path.is_symlink() or path.stat().st_size > 200_000:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("activity_id") == path.stem:
                records.append(row)
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: item.get("submitted_at", ""), reverse=True)


def new_activity(run_id: str, model_kind: str, requested_time_s: float) -> dict:
    if model_kind not in MODELS:
        raise ValueError("Unsupported Meta model")
    if type(requested_time_s) not in (int, float) or not math.isfinite(requested_time_s) or requested_time_s < 0:
        raise ValueError("A finite source-video time is required")
    return {
        "activity_id": "meta-" + uuid.uuid4().hex,
        "provider": "Meta Model API",
        "model": MODELS[model_kind],
        "model_kind": model_kind,
        "run_id": run_id,
        "status": "submitted",
        "runtime": "Touchline Studio server",
        "input_type": "source_video_frame",
        "requested_time_s": requested_time_s,
        "submitted_at": utc_now(),
        "review": None,
    }


def source_frame(video: Path, requested_time_s: float) -> tuple[bytes, float]:
    """Decode the selected source frame using its actual presentation timestamp."""
    from .video.decode import decode_video, probe_video
    from PIL import Image

    info = probe_video(video)
    duration = info.get("duration_s")
    if duration is not None and requested_time_s > duration:
        raise ValueError("Selected time exceeds source-video duration")
    frames = decode_video(video, source_id=video.name,
                          start_time_s=max(0.0, requested_time_s - 0.2),
                          end_time_s=requested_time_s + 0.5,
                          decoder_threads=2)
    selected = min(frames, key=lambda frame: abs(frame.timestamp_s - requested_time_s), default=None)
    if selected is None:
        raise ValueError("No decodable frame at selected time")
    image = selected.image.to_image().convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    payload = buffer.getvalue()
    if len(payload) > 2_000_000:
        raise ValueError("Selected frame exceeds image bound")
    return payload, selected.timestamp_s


def meta_payload(kind: str, jpeg: bytes) -> dict:
    image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    if kind == "sam":
        prompt = "person"
    else:
        prompt = (
            "Describe only directly visible football actions in this single frame. "
            "Do not infer player identity, passes, speed, improvement, or events outside the frame. "
            "State uncertainty and keep the answer under 100 words."
        )
    payload = {
        "model": MODELS[kind],
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_url},
        ]}],
        "stream": False,
    }
    if kind == "sam":
        payload["metadata"] = {"mask_encoding": "one_bit", "include_confidence": "true"}
    else:
        payload["max_output_tokens"] = 220
    return payload


def call_meta(payload: dict, key: str, *, transport=None) -> tuple[dict, str | None]:
    """One request, no retries; transport is injected by offline tests."""
    if not key:
        raise ValueError("Meta Model API credential unavailable")
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(API_URL, encoded, {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }, method="POST")
    sender = transport or request.urlopen
    try:
        with sender(req, timeout=35) as response:
            if response.status != 200:
                raise RuntimeError("Meta request failed")
            body = response.read(3_000_001)
            if len(body) > 3_000_000:
                raise RuntimeError("Meta response exceeds safe bound")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise RuntimeError("Meta response is not an object")
            return result, response.headers.get("x-request-id")
    except error.HTTPError as exc:
        raise MetaCallError("provider_http_error", exc.code) from exc
    except (error.URLError, TimeoutError) as exc:
        raise MetaCallError("provider_transport_error") from exc


def output_text(result: dict) -> str:
    if isinstance(result.get("output_text"), str):
        return result["output_text"]
    chunks = []
    for row in result.get("output", []) if isinstance(result.get("output"), list) else []:
        if not isinstance(row, dict):
            continue
        for part in row.get("content", []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks)


def execute_activity(run_dir: Path, record: dict, key: str, *, transport=None):
    """Update a durable local activity record from real source bytes and response."""
    started = time.monotonic()
    record["status"] = "processing"
    record["processing_at"] = utc_now()
    save_activity(run_dir, record)
    try:
        jpeg, actual_time = source_frame(run_dir / "source_video.mp4", record["requested_time_s"])
        record["frame_time_s"] = actual_time
        record["frame_sha256"] = hashlib.sha256(jpeg).hexdigest()
        result, header_id = call_meta(meta_payload(record["model_kind"], jpeg), key, transport=transport)
        record["request_id"] = result.get("id") if isinstance(result.get("id"), str) else header_id
        record["response_model"] = result.get("model") if isinstance(result.get("model"), str) else None
        record["response_status"] = result.get("status") if isinstance(result.get("status"), str) else None
        details = result.get("incomplete_details")
        if isinstance(details, dict) and isinstance(details.get("reason"), str):
            record["incomplete_reason"] = details["reason"][:80]
        output = output_text(result)
        if not output and record["model_kind"] != "sam":
            raise MetaCallError("no_reviewable_output")
        record["output_preview"] = output[:12_000]
        record["output_truncated"] = len(output) > 12_000
        record["status"] = "response_received"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = "Frame decoding or Meta inference failed; inspect the server configuration."
        record["error_code"] = exc.code if isinstance(exc, MetaCallError) else "local_or_response_error"
        if isinstance(exc, MetaCallError) and exc.http_status is not None:
            record["provider_http_status"] = exc.http_status
    record["responded_at"] = utc_now()
    record["latency_ms"] = round((time.monotonic() - started) * 1000)
    save_activity(run_dir, record)
    return record
