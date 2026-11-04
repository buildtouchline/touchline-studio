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
import re
from pathlib import Path
import tempfile
import time
from urllib import error, request
import uuid
import wave


MODELS = {"sam": "sam-3.1", "spark": "muse-spark-1.3", "wonder": "muse-wonder-1.0", "voice": "muse-voice-transcribe-1.0"}
API_URL = "https://api.meta.ai/v1/responses"
VOICE_URL = "https://api.meta.ai/v1/asr/transcribe"


class MetaCallError(RuntimeError):
    def __init__(self, code: str, http_status: int | None = None, detail: str | None = None):
        super().__init__(code)
        self.code = code
        self.http_status = http_status
        self.detail = detail


def safe_provider_error(exc, key: str) -> tuple[str, str | None]:
    """Keep only short provider code/type and a redacted diagnostic message."""
    try:
        body = exc.read(4096)
        parsed = json.loads(body)
        entry = parsed.get("error", parsed) if isinstance(parsed, dict) else {}
        if not isinstance(entry, dict):
            return "provider_http_error", None
        code = entry.get("code") or entry.get("type") or "provider_http_error"
        code = re.sub(r"[^a-zA-Z0-9_.-]", "", str(code))[:64] or "provider_http_error"
        message = entry.get("message")
        if not isinstance(message, str):
            return code, None
        message = message.replace(key, "[redacted]")
        message = re.sub(r"(?i)Bearer\s+\S+|LLM_[A-Za-z0-9_-]+|https?://\S+", "[redacted]", message)
        return code, message[:160]
    except (ValueError, OSError):
        return "provider_http_error", None


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


def save_activity_frame(run_dir: Path, record: dict, jpeg: bytes):
    """Retain the exact frame bytes submitted to Meta for later review."""
    if not 0 < len(jpeg) <= 2_000_000 or hashlib.sha256(jpeg).hexdigest() != record.get("frame_sha256"):
        raise ValueError("Selected frame failed integrity validation")
    folder = activity_dir(run_dir)
    target = folder / (record["activity_id"] + ".jpg")
    if target.is_symlink() or target.exists():
        raise ValueError("Selected frame already exists")
    fd, name = tempfile.mkstemp(prefix=".frame-", dir=folder)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(jpeg)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def verified_activity_frame(run_dir: Path, record: dict) -> bytes:
    """Serve saved request bytes; re-decode only for older records without them."""
    digest = record.get("frame_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Selected frame hash unavailable")
    frame = run_dir / "meta_activity" / (record["activity_id"] + ".jpg")
    if frame.is_symlink():
        raise ValueError("Unsafe selected frame")
    if frame.exists():
        if not frame.is_file() or not 0 < frame.stat().st_size <= 2_000_000:
            raise ValueError("Selected frame size is invalid")
        jpeg = frame.read_bytes()
    else:
        jpeg, _ = source_frame(run_dir / "source_video.mp4", record["requested_time_s"])
    if hashlib.sha256(jpeg).hexdigest() != digest:
        raise ValueError("Selected frame lineage changed")
    return jpeg


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


def new_activity(run_id: str, model_kind: str, requested_time_s: float, *, objective: str | None = None) -> dict:
    if model_kind not in MODELS:
        raise ValueError("Unsupported Meta model")
    if type(requested_time_s) not in (int, float) or not math.isfinite(requested_time_s) or requested_time_s < 0:
        raise ValueError("A finite source-video time is required")
    if model_kind == "wonder" and (not isinstance(objective, str) or not 3 <= len(objective.strip()) <= 280):
        raise ValueError("A specific coach objective is required for Muse Wonder")
    return {
        "activity_id": "meta-" + uuid.uuid4().hex,
        "provider": "Meta Model API",
        "model": MODELS[model_kind],
        "model_kind": model_kind,
        "run_id": run_id,
        "status": "submitted",
        "runtime": "Touchline Studio server",
        "input_type": "coach_audio_file" if model_kind == "voice" else "source_video_frame",
        "requested_time_s": requested_time_s,
        **({"coach_objective": objective.strip()} if model_kind == "wonder" else {}),
        **({"input_origin": "coach_audio_source_unverified"} if model_kind == "voice" else {}),
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


def meta_payload(kind: str, jpeg: bytes, *, objective: str | None = None) -> dict:
    image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    if kind == "sam":
        prompt = "person"
    elif kind == "spark":
        prompt = (
            "Inspect this one football frame for a coach. Reply as a short JSON object with keys "
            "visible_observation, supporting_visual_evidence, uncertainty, suggested_coaching_question, "
            "coach_review_required. Describe only what can be seen. Do not identify real players, infer jersey "
            "numbers, passes, speed, distance, biomechanics, injury, or events outside this still image. "
            "Use null when evidence is insufficient and set coach_review_required to true."
        )
    elif kind == "wonder":
        if not isinstance(objective, str) or not 3 <= len(objective.strip()) <= 280:
            raise ValueError("Muse Wonder requires a coach objective")
        prompt = (
            f"Coach objective: {objective.strip()}\nThis is one still frame from one session. "
            "Explain in four brief sections: visible evidence; what this frame cannot establish; "
            "questions for a coach in a later session; the next observation to collect. "
            "Never assert longitudinal improvement, player identity, a pass, physical measurements, "
            "biomechanics, or medical findings from this image. State uncertainty."
        )
    else:
        raise ValueError("Unsupported selected-frame model")
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
        payload["max_output_tokens"] = 700
        payload["reasoning"] = {"effort": "minimal"}
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
        with sender(req, timeout=35 if payload.get("model") == MODELS["sam"] else 60) as response:
            if response.status != 200:
                raise RuntimeError("Meta request failed")
            body = response.read(3_000_001)
            if len(body) > 3_000_000:
                raise RuntimeError("Meta response exceeds safe bound")
            try:
                result = json.loads(body)
            except ValueError:
                raise MetaCallError("malformed_provider_response") from None
            if not isinstance(result, dict):
                raise RuntimeError("Meta response is not an object")
            return result, response.headers.get("x-request-id")
    except error.HTTPError as exc:
        code, detail = safe_provider_error(exc, key)
        raise MetaCallError(code, exc.code, detail) from None
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


def call_voice(wav_bytes: bytes, key: str, session_id: str, *, transport=None) -> tuple[dict, str | None, float]:
    """Submit one bounded, validated WAV file to Meta's documented file ASR API."""
    if not key:
        raise ValueError("Meta Model API credential unavailable")
    if not 44 <= len(wav_bytes) <= 8_000_000:
        raise ValueError("Audio outside accepted size")
    with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() not in (16000, 24000):
            raise ValueError("Expected mono 16-bit PCM WAV at 16 or 24 kHz")
        duration_s = audio.getnframes() / audio.getframerate()
        if not 0.1 <= duration_s <= 120:
            raise ValueError("Audio duration outside 0.1–120 seconds")
    boundary = "touchline-" + uuid.uuid4().hex
    settings = json.dumps({"mode": "ENDPOINTING", "model": MODELS["voice"], "audioEncoding": "WAV"}).encode()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
            "Content-Type: application/json\r\n\r\n").encode() + settings + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"coach.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = request.Request(f"{VOICE_URL}?sessionId={session_id}", body, {
        "Authorization": "Bearer " + key,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/json",
    }, method="POST")
    sender = transport or request.urlopen
    try:
        with sender(req, timeout=70) as response:
            if response.status != 200:
                raise MetaCallError("provider_http_error", response.status)
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise MetaCallError("provider_response_too_large")
            try:
                result = json.loads(raw)
            except ValueError:
                raise MetaCallError("malformed_provider_response") from None
            if not isinstance(result, dict):
                raise MetaCallError("malformed_provider_response")
            return result, response.headers.get("x-request-id"), duration_s
    except error.HTTPError as exc:
        code, detail = safe_provider_error(exc, key)
        raise MetaCallError(code, exc.code, detail) from None
    except (error.URLError, TimeoutError) as exc:
        raise MetaCallError("provider_transport_error") from exc


def execute_voice_activity(run_dir: Path, record: dict, key: str, audio_bytes: bytes, *, transport=None):
    """Normalize browser audio, make one call, and persist only reviewed-safe text."""
    from .audio import convert_audio_to_wav16k
    started = time.monotonic()
    record["status"] = "processing"
    record["processing_at"] = utc_now()
    save_activity(run_dir, record)
    fd, name = tempfile.mkstemp(prefix="touchline-voice-", suffix=".wav")
    os.close(fd)
    wav_path = Path(name)
    try:
        if not 100 <= len(audio_bytes) <= 8_000_000:
            raise ValueError("Audio outside accepted size")
        convert_audio_to_wav16k(audio_bytes, wav_path)
        wav_bytes = wav_path.read_bytes()
        record["audio_sha256"] = hashlib.sha256(wav_bytes).hexdigest()
        result, header_id, duration = call_voice(wav_bytes, key, record["activity_id"], transport=transport)
        record["audio_duration_s"] = round(duration, 3)
        record["provider_session_id"] = result.get("sessionId") if isinstance(result.get("sessionId"), str) else None
        record["request_id"] = header_id
        record["response_model"] = MODELS["voice"]
        transcript = result.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            raise MetaCallError("no_reviewable_output")
        record["output_preview"] = transcript[:12_000]
        record["output_truncated"] = len(transcript) > 12_000
        turns = result.get("turns")
        if isinstance(turns, list):
            record["turns"] = [
                {"startMs": turn["startMs"], "endMs": turn["endMs"], "transcript": turn["transcript"][:2000],
                 **({"speaker": turn["speaker"][:16]} if isinstance(turn.get("speaker"), str) else {})}
                for turn in turns[:100] if isinstance(turn, dict)
                and isinstance(turn.get("startMs"), (int, float)) and isinstance(turn.get("endMs"), (int, float))
                and isinstance(turn.get("transcript"), str)
                and 0 <= turn["startMs"] <= turn["endMs"] <= 120_000
            ]
        record["status"] = "response_received"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = "Audio preparation or Meta transcription failed; inspect the server configuration."
        record["error_code"] = exc.code if isinstance(exc, MetaCallError) else "local_or_response_error"
        if isinstance(exc, MetaCallError) and exc.http_status is not None:
            record["provider_http_status"] = exc.http_status
        if isinstance(exc, MetaCallError) and exc.detail:
            record["provider_error_detail"] = exc.detail
    finally:
        wav_path.unlink(missing_ok=True)
    record["responded_at"] = utc_now()
    record["latency_ms"] = round((time.monotonic() - started) * 1000)
    save_activity(run_dir, record)
    return record


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
        save_activity_frame(run_dir, record, jpeg)
        result, header_id = call_meta(meta_payload(record["model_kind"], jpeg, objective=record.get("coach_objective")), key, transport=transport)
        record["request_id"] = result.get("id") if isinstance(result.get("id"), str) else header_id
        record["response_model"] = result.get("model") if isinstance(result.get("model"), str) else None
        record["response_status"] = result.get("status") if isinstance(result.get("status"), str) else None
        details = result.get("incomplete_details")
        if isinstance(details, dict) and isinstance(details.get("reason"), str):
            record["incomplete_reason"] = details["reason"][:80]
        output = output_text(result)
        if record["response_status"] not in (None, "completed"):
            raise MetaCallError("provider_response_not_completed")
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
        if isinstance(exc, MetaCallError) and exc.detail:
            record["provider_error_detail"] = exc.detail
    record["responded_at"] = utc_now()
    record["latency_ms"] = round((time.monotonic() - started) * 1000)
    save_activity(run_dir, record)
    return record
