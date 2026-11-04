"""Coach Bluetooth Collar Mic audio normalization, Sarvam AI transcription, and video-relative timeline alignment.

Implements the end-to-end coach commentary workflow:
1. Converts browser-recorded audio blobs (`.webm`, `.ogg`, `.mp4`, `.wav`) into 16kHz mono `.wav` via PyAV.
2. Stores and retrieves the Sarvam AI API key strictly outside the Git repository (`<runs_root>/.sarvam_key` or `SARVAM_API_KEY`).
3. Calls Sarvam AI Speech-to-Text (`https://api.sarvam.ai/speech-to-text`) with `with_timestamps=true`.
4. Aligns every transcribed segment and live coach cue to both:
   - Audio/session coordinates (`audio_start_s`, `audio_end_s`, `monotonic_start_ms`)
   - Video-relative coordinates (`video_start_s`, `video_end_s`, `nearest_frame_pts`, `nearest_frame_index`)
   using explicit affine sync (`video_time_s = slope_a * audio_time_s + video_anchor_start_s`).
"""
from __future__ import annotations
from .diagnostics import diagnosed

from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request
import uuid
import wave


SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

META_TRANSCRIBE_URL = "https://api.meta.ai/v1/transcribe"
META_CHAT_URL = "https://api.meta.ai/v1/chat/completions"


def get_meta_key_path(runs_root: Path) -> Path:
    """Return the path outside the Git repository where the Meta API key is persisted."""
    return Path(runs_root).resolve() / ".meta_key"


def save_meta_key(runs_root: Path, api_key: str) -> Path:
    """Save Meta API key outside the Git repository and return the file path."""
    cleaned = (api_key or "").strip()
    if not cleaned:
        raise ValueError("Meta API key cannot be empty")
    runs_dir = Path(runs_root).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    key_file = get_meta_key_path(runs_dir)
    if key_file.is_symlink():
        raise ValueError("Refusing a symbolic-link credential file")
    key_file.write_text(cleaned + "\n", encoding="utf-8")
    return key_file


def load_meta_key(runs_root: Path | None = None) -> str | None:
    """Load Meta API key from environment variable `META_API_KEY` or `<runs_root>/.meta_key`."""
    env_key = os.environ.get("META_API_KEY", "").strip()
    if env_key:
        return env_key
    if runs_root is not None:
        candidate = get_meta_key_path(runs_root)
        if candidate.is_file() and not candidate.is_symlink():
            val = candidate.read_text(encoding="utf-8").strip()
            if val:
                return val
    return None



def get_sarvam_key_path(runs_root: Path) -> Path:
    """Return the path outside the Git repository where the Sarvam key is persisted."""
    return Path(runs_root).resolve() / ".sarvam_key"


def save_sarvam_key(runs_root: Path, api_key: str) -> Path:
    """Save Sarvam API key outside the Git repository and return the file path."""
    cleaned = (api_key or "").strip()
    if not cleaned:
        raise ValueError("Sarvam API key cannot be empty")
    runs_dir = Path(runs_root).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    key_file = get_sarvam_key_path(runs_dir)
    if key_file.is_symlink():
        raise ValueError("Refusing a symbolic-link credential file")
    key_file.write_text(cleaned + "\n", encoding="utf-8")
    return key_file


def load_sarvam_key(runs_root: Path | None = None) -> str | None:
    """Load Sarvam API key from environment variable `SARVAM_API_KEY` or `<runs_root>/.sarvam_key`."""
    env_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if env_key:
        return env_key
    if runs_root is not None:
        candidate = get_sarvam_key_path(runs_root)
        if candidate.is_file() and not candidate.is_symlink():
            val = candidate.read_text(encoding="utf-8").strip()
            if val:
                return val
    return None


def mask_api_key(api_key: str | None) -> str | None:
    """Return a masked representation (`sk_...abcd`) safe for UI status display."""
    if not api_key:
        return None
    s = api_key.strip()
    if len(s) <= 8:
        return "****"
    return f"{s[:4]}...{s[-4:]}"


@diagnosed("audio.normalize")
def convert_audio_to_wav16k(input_bytes: bytes, output_wav_path: Path) -> dict[str, Any]:
    """Decode any browser MediaRecorder audio container (`.webm`, `.ogg`, `.mp4`, `.wav`) to 16kHz mono PCM WAV."""
    if not input_bytes:
        raise ValueError("input_bytes cannot be empty")
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path if already a valid RIFF WAVE header and PyAV is not needed or fails
    try:
        import av

        in_buf = io.BytesIO(input_bytes)
        with av.open(in_buf, mode="r") as in_container:
            audio_streams = [s for s in in_container.streams if s.type == "audio"]
            if not audio_streams:
                raise ValueError("No audio stream found in uploaded media")
            in_stream = audio_streams[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            pcm_chunks: list[bytes] = []
            total_samples = 0
            for frame in in_container.decode(in_stream):
                out_frames = resampler.resample(frame)
                if out_frames is None:
                    continue
                if not isinstance(out_frames, list):
                    out_frames = [out_frames]
                for of in out_frames:
                    raw = bytes(of.planes[0])[:of.samples * 2]
                    pcm_chunks.append(raw)
                    total_samples += of.samples
            # Flush resampler
            flushed = resampler.resample(None)
            if flushed:
                if not isinstance(flushed, list):
                    flushed = [flushed]
                for of in flushed:
                    raw = bytes(of.planes[0])[:of.samples * 2]
                    pcm_chunks.append(raw)
                    total_samples += of.samples

        with wave.open(str(output_wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_chunks))

        duration_s = round(total_samples / 16000.0, 3)
        return {
            "wav_path": str(output_wav_path),
            "sample_rate_hz": 16000,
            "channels": 1,
            "duration_s": duration_s,
            "transcoded_with": "pyav",
        }
    except Exception as exc:
        # Fallback if input_bytes is already WAV
        if input_bytes[:4] == b"RIFF" and input_bytes[8:12] == b"WAVE":
            with wave.open(io.BytesIO(input_bytes), "rb") as wf:
                if (wf.getframerate() != 16000 or wf.getnchannels() != 1
                        or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE"):
                    raise ValueError("PyAV normalization failed; fallback requires 16kHz mono 16-bit PCM WAV") from exc
                frames = wf.getnframes()
                rate = wf.getframerate() or 16000
                channels = wf.getnchannels()
                duration_s = round(frames / float(rate), 3)
            output_wav_path.write_bytes(input_bytes)
            return {
                "wav_path": str(output_wav_path),
                "sample_rate_hz": rate,
                "channels": channels,
                "duration_s": duration_s,
                "transcoded_with": "wav_passthrough",
            }
        raise ValueError(f"Failed to decode uploaded audio into 16kHz WAV: {exc}") from exc


def _build_multipart_form(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "audio/wav",
) -> tuple[bytes, str]:
    boundary = f"----TouchlineSarvamBoundary{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for k, v in fields.items():
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        lines.append(f"{v}\r\n".encode("utf-8"))

    lines.append(f"--{boundary}\r\n".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    lines.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    lines.append(file_bytes)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"


@diagnosed("audio.align", failures_only=True)
def align_audio_segment_to_video(
    audio_start_s: float,
    audio_end_s: float,
    *,
    video_anchor_start_s: float = 0.0,
    slope_a: float = 1.0,
    monotonic_start_ms: float | None = None,
    frame_lookup: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map an audio-relative interval `[audio_start_s, audio_end_s]` to video-relative time and nearest frame PTS."""
    if not (math.isfinite(audio_start_s) and math.isfinite(audio_end_s)):
        raise ValueError("audio_start_s and audio_end_s must be finite")
    if audio_end_s < audio_start_s:
        audio_end_s = audio_start_s
    if not (math.isfinite(video_anchor_start_s) and math.isfinite(slope_a) and slope_a > 0):
        raise ValueError("video_anchor_start_s must be finite and slope_a must be positive")

    v_start = round(max(0.0, slope_a * audio_start_s + video_anchor_start_s), 3)
    v_end = round(max(v_start, slope_a * audio_end_s + video_anchor_start_s), 3)

    nearest_pts = None
    nearest_frame_index = None
    nearest_frame_time_s = None
    if frame_lookup:
        best = min(
            frame_lookup,
            key=lambda f: abs(float(f.get("timestamp_s", 0.0)) - v_start),
        )
        nearest_pts = best.get("pts")
        nearest_frame_index = best.get("index")
        nearest_frame_time_s = best.get("timestamp_s")

    return {
        "audio_start_s": round(audio_start_s, 3),
        "audio_end_s": round(audio_end_s, 3),
        "video_start_s": v_start,
        "video_end_s": v_end,
        "nearest_frame_pts": nearest_pts,
        "nearest_frame_index": nearest_frame_index,
        "nearest_frame_time_s": nearest_frame_time_s,
        "monotonic_offset_ms": (
            round(monotonic_start_ms + audio_start_s * 1000.0, 1)
            if monotonic_start_ms is not None and math.isfinite(monotonic_start_ms)
            else None
        ),
    }


def extract_frame_lookup(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a sorted deduplicated list of `{index, timestamp_s, pts}` from observations."""
    seen: dict[float, dict[str, Any]] = {}
    for obs in observations:
        f = obs.get("frame", {})
        ts = f.get("timestamp_s")
        if ts is not None and ts not in seen:
            seen[float(ts)] = {
                "index": f.get("index"),
                "timestamp_s": float(ts),
                "pts": f.get("pts"),
            }
    return [seen[k] for k in sorted(seen.keys())]


def parse_sarvam_response(
    raw_response: dict[str, Any],
    *,
    audio_duration_s: float,
    video_anchor_start_s: float = 0.0,
    slope_a: float = 1.0,
    monotonic_start_ms: float | None = None,
    frame_lookup: list[dict[str, Any]] | None = None,
    speaker_label: str = "Coach (Collar Mic — Unverified Speaker Turn)",
) -> list[dict[str, Any]]:
    """Normalize Sarvam AI STT response (`transcript` + optional `timestamps`) into video-aligned segments."""
    transcript_text = str(raw_response.get("transcript") or "").strip()
    ts_data = raw_response.get("timestamps")
    segments: list[dict[str, Any]] = []

    # Case 1: Sarvam returns `timestamps: {"words": [...], "start_time_seconds": [...], "end_time_seconds": [...]}`
    if isinstance(ts_data, dict):
        words = ts_data.get("words") or []
        starts = ts_data.get("start_time_seconds") or []
        ends = ts_data.get("end_time_seconds") or []
        if words and len(words) == len(starts) == len(ends):
            # Group words into readable ~4-second or punctuation-bounded coaching phrases
            chunk_words: list[str] = []
            chunk_start = float(starts[0])
            chunk_end = float(ends[0])
            for w, s_t, e_t in zip(words, starts, ends):
                s_f, e_f = float(s_t), float(e_t)
                if chunk_words and ((e_f - chunk_start) > 4.5 or (s_f - chunk_end) > 1.0):
                    aligned = align_audio_segment_to_video(
                        chunk_start,
                        chunk_end,
                        video_anchor_start_s=video_anchor_start_s,
                        slope_a=slope_a,
                        monotonic_start_ms=monotonic_start_ms,
                        frame_lookup=frame_lookup,
                    )
                    segments.append({
                        "segment_id": f"seg_{len(segments) + 1:03d}",
                        "speaker_label": speaker_label,
                        "speaker_verified": False,
                        "text": " ".join(chunk_words).strip(),
                        **aligned,
                    })
                    chunk_words = []
                    chunk_start = s_f
                if not chunk_words:
                    chunk_start = s_f
                chunk_words.append(str(w))
                chunk_end = e_f
                if str(w).endswith((".", "!", "?")):
                    aligned = align_audio_segment_to_video(
                        chunk_start,
                        chunk_end,
                        video_anchor_start_s=video_anchor_start_s,
                        slope_a=slope_a,
                        monotonic_start_ms=monotonic_start_ms,
                        frame_lookup=frame_lookup,
                    )
                    segments.append({
                        "segment_id": f"seg_{len(segments) + 1:03d}",
                        "speaker_label": speaker_label,
                        "speaker_verified": False,
                        "text": " ".join(chunk_words).strip(),
                        **aligned,
                    })
                    chunk_words = []
            if chunk_words:
                aligned = align_audio_segment_to_video(
                    chunk_start,
                    chunk_end,
                    video_anchor_start_s=video_anchor_start_s,
                    slope_a=slope_a,
                    monotonic_start_ms=monotonic_start_ms,
                    frame_lookup=frame_lookup,
                )
                segments.append({
                    "segment_id": f"seg_{len(segments) + 1:03d}",
                    "speaker_label": speaker_label,
                    "speaker_verified": False,
                    "text": " ".join(chunk_words).strip(),
                    **aligned,
                })

    # Case 2: Sarvam or custom payload returns list of segment dicts under `timestamps` or `segments`
    elif isinstance(ts_data, list):
        for idx, item in enumerate(ts_data):
            if isinstance(item, dict) and item.get("text"):
                s_f = float(item.get("start_time_seconds", item.get("start", 0.0)))
                e_f = float(item.get("end_time_seconds", item.get("end", s_f + 1.0)))
                aligned = align_audio_segment_to_video(
                    s_f,
                    e_f,
                    video_anchor_start_s=video_anchor_start_s,
                    slope_a=slope_a,
                    monotonic_start_ms=monotonic_start_ms,
                    frame_lookup=frame_lookup,
                )
                segments.append({
                    "segment_id": f"seg_{idx + 1:03d}",
                    "speaker_label": speaker_label,
                    "speaker_verified": False,
                    "text": str(item["text"]).strip(),
                    **aligned,
                })

    # Case 3: Plain transcript text without word timestamps — split sentences evenly across audio_duration_s
    if not segments and transcript_text:
        raw_parts = [p.strip() for p in transcript_text.replace("!", ".").replace("?", ".").split(".") if p.strip()]
        if not raw_parts:
            raw_parts = [transcript_text]
        dur = max(1.0, float(audio_duration_s))
        step = dur / len(raw_parts)
        for idx, part in enumerate(raw_parts):
            s_f = round(idx * step, 3)
            e_f = round(min(dur, (idx + 1) * step), 3)
            aligned = align_audio_segment_to_video(
                s_f,
                e_f,
                video_anchor_start_s=video_anchor_start_s,
                slope_a=slope_a,
                monotonic_start_ms=monotonic_start_ms,
                frame_lookup=frame_lookup,
            )
            segments.append({
                "segment_id": f"seg_{idx + 1:03d}",
                "speaker_label": speaker_label,
                "speaker_verified": False,
                "text": part,
                **aligned,
            })

    return segments


@diagnosed("audio.transcribe")
def transcribe_with_sarvam(
    wav_path: Path,
    api_key: str,
    *,
    model: str = "saarika:v2",
    language_code: str = "en-IN",
    video_anchor_start_s: float = 0.0,
    slope_a: float = 1.0,
    monotonic_start_ms: float | None = None,
    device_label: str = "Bluetooth Collar Mic",
    frame_lookup: list[dict[str, Any]] | None = None,
    manual_cues: list[dict[str, Any]] | None = None,
    opener: Any = None,
) -> dict[str, Any]:
    """Call Sarvam AI Speech-to-Text API on a 16kHz WAV file and return a structured `touchline.transcript.v1` ledger."""
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise ValueError(f"Audio file not found: {wav_path}")
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Sarvam API key is required for transcription")

    wav_bytes = wav_path.read_bytes()
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        audio_duration_s = round(wf.getnframes() / float(wf.getframerate() or 16000), 3)

    fields = {
        "model": model,
        "language_code": language_code,
        "with_timestamps": "true",
    }
    body, content_type = _build_multipart_form(
        fields=fields,
        file_field="file",
        filename=wav_path.name,
        file_bytes=wav_bytes,
        content_type="audio/wav",
    )

    req = urllib.request.Request(
        SARVAM_STT_URL,
        data=body,
        method="POST",
        headers={
            "api-subscription-key": cleaned_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        },
    )

    http_open = opener or urllib.request.urlopen
    try:
        with http_open(req, timeout=45) as resp:
            resp_bytes = resp.read()
            raw_json = json.loads(resp_bytes.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        # If `saarika:v2` is rejected in favor of `saarika:v2.5` or without `with_timestamps`, retry once cleanly
        if exc.code in (400, 422) and model == "saarika:v2" and opener is None:
            return transcribe_with_sarvam(
                wav_path,
                cleaned_key,
                model="saarika:v2.5",
                language_code=language_code,
                video_anchor_start_s=video_anchor_start_s,
                slope_a=slope_a,
                monotonic_start_ms=monotonic_start_ms,
                device_label=device_label,
                frame_lookup=frame_lookup,
                manual_cues=manual_cues,
                opener=opener,
            )
        raise RuntimeError(f"Sarvam AI API HTTP {exc.code}: {err_body or exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"Sarvam AI transcription request failed: {exc}") from exc

    segments = parse_sarvam_response(
        raw_json,
        audio_duration_s=audio_duration_s,
        video_anchor_start_s=video_anchor_start_s,
        slope_a=slope_a,
        monotonic_start_ms=monotonic_start_ms,
        frame_lookup=frame_lookup,
    )

    # Also align any live manual cue markers clicked during recording
    aligned_cues: list[dict[str, Any]] = []
    for idx, cue in enumerate(manual_cues or []):
        c_audio_s = float(cue.get("audio_offset_s", 0.0))
        aligned = align_audio_segment_to_video(
            c_audio_s,
            c_audio_s + 0.5,
            video_anchor_start_s=video_anchor_start_s,
            slope_a=slope_a,
            monotonic_start_ms=monotonic_start_ms,
            frame_lookup=frame_lookup,
        )
        aligned_cues.append({
            "cue_id": f"cue_{idx + 1:03d}",
            "note": str(cue.get("note") or "Coach live timestamp marker"),
            "target_track_id": cue.get("target_track_id"),
            **aligned,
        })

    return {
        "schema_version": "touchline.transcript.v1",
        "provider": "sarvam_ai",
        "model": model,
        "language_code": raw_json.get("language_code") or language_code,
        "device_label": device_label,
        "audio_duration_s": audio_duration_s,
        "alignment": {
            "sync_model": "video_time_s = slope_a * audio_time_s + video_anchor_start_s",
            "video_anchor_start_s": round(video_anchor_start_s, 3),
            "slope_a": round(slope_a, 6),
            "monotonic_start_ms": monotonic_start_ms,
        },
        "transcript_full": str(raw_json.get("transcript") or "").strip(),
        "segments": segments,
        "manual_cues": aligned_cues,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_meta_muse_response(
    raw_response: dict[str, Any],
    *,
    audio_duration_s: float,
    video_anchor_start_s: float = 0.0,
    slope_a: float = 1.0,
    monotonic_start_ms: float | None = None,
    frame_lookup: list[dict[str, Any]] | None = None,
    speaker_label: str = "Coach (Collar Mic — Unverified Speaker Turn)",
) -> list[dict[str, Any]]:
    """Normalize Meta Muse Voice Transcribe response into video-aligned segments."""
    transcript_text = str(raw_response.get("text") or raw_response.get("transcript") or "").strip()
    segments: list[dict[str, Any]] = []

    # Case 1: Word-level timestamps in `words` list
    words_data = raw_response.get("words")
    if not words_data and isinstance(raw_response.get("timestamps"), dict):
        w_list = raw_response["timestamps"].get("words") or []
        s_list = raw_response["timestamps"].get("start_time_seconds") or []
        e_list = raw_response["timestamps"].get("end_time_seconds") or []
        if len(w_list) == len(s_list) == len(e_list):
            words_data = [{"word": w, "start": s, "end": e} for w, s, e in zip(w_list, s_list, e_list)]

    if isinstance(words_data, list) and words_data:
        chunk_words: list[str] = []
        chunk_start = float(words_data[0].get("start", 0.0))
        chunk_end = float(words_data[0].get("end", chunk_start + 0.5))
        for item in words_data:
            w = str(item.get("word", "")).strip()
            s_f = float(item.get("start", chunk_end))
            e_f = float(item.get("end", s_f + 0.5))
            if chunk_words and ((e_f - chunk_start) > 4.5 or (s_f - chunk_end) > 1.0):
                aligned = align_audio_segment_to_video(
                    chunk_start,
                    chunk_end,
                    video_anchor_start_s=video_anchor_start_s,
                    slope_a=slope_a,
                    monotonic_start_ms=monotonic_start_ms,
                    frame_lookup=frame_lookup,
                )
                segments.append({
                    "segment_id": f"seg_{len(segments) + 1:03d}",
                    "speaker_label": speaker_label,
                    "speaker_verified": False,
                    "text": " ".join(chunk_words).strip(),
                    **aligned,
                })
                chunk_words = []
                chunk_start = s_f
            if not chunk_words:
                chunk_start = s_f
            chunk_words.append(w)
            chunk_end = e_f
            if w.endswith((".", "!", "?")):
                aligned = align_audio_segment_to_video(
                    chunk_start,
                    chunk_end,
                    video_anchor_start_s=video_anchor_start_s,
                    slope_a=slope_a,
                    monotonic_start_ms=monotonic_start_ms,
                    frame_lookup=frame_lookup,
                )
                segments.append({
                    "segment_id": f"seg_{len(segments) + 1:03d}",
                    "speaker_label": speaker_label,
                    "speaker_verified": False,
                    "text": " ".join(chunk_words).strip(),
                    **aligned,
                })
                chunk_words = []
        if chunk_words:
            aligned = align_audio_segment_to_video(
                chunk_start,
                chunk_end,
                video_anchor_start_s=video_anchor_start_s,
                slope_a=slope_a,
                monotonic_start_ms=monotonic_start_ms,
                frame_lookup=frame_lookup,
            )
            segments.append({
                "segment_id": f"seg_{len(segments) + 1:03d}",
                "speaker_label": speaker_label,
                "speaker_verified": False,
                "text": " ".join(chunk_words).strip(),
                **aligned,
            })

    # Case 2: Structured segments in `segments` list
    elif isinstance(raw_response.get("segments"), list) and raw_response["segments"]:
        for idx, item in enumerate(raw_response["segments"]):
            if isinstance(item, dict) and item.get("text"):
                s_f = float(item.get("start", item.get("start_time_seconds", 0.0)))
                e_f = float(item.get("end", item.get("end_time_seconds", s_f + 1.0)))
                aligned = align_audio_segment_to_video(
                    s_f,
                    e_f,
                    video_anchor_start_s=video_anchor_start_s,
                    slope_a=slope_a,
                    monotonic_start_ms=monotonic_start_ms,
                    frame_lookup=frame_lookup,
                )
                segments.append({
                    "segment_id": f"seg_{idx + 1:03d}",
                    "speaker_label": speaker_label,
                    "speaker_verified": False,
                    "text": str(item["text"]).strip(),
                    **aligned,
                })

    # Case 3: Sentence splitting across total audio duration
    if not segments and transcript_text:
        raw_parts = [p.strip() for p in transcript_text.replace("!", ".").replace("?", ".").split(".") if p.strip()]
        if not raw_parts:
            raw_parts = [transcript_text]
        dur = max(1.0, float(audio_duration_s))
        step = dur / len(raw_parts)
        for idx, part in enumerate(raw_parts):
            s_f = round(idx * step, 3)
            e_f = round(min(dur, (idx + 1) * step), 3)
            aligned = align_audio_segment_to_video(
                s_f,
                e_f,
                video_anchor_start_s=video_anchor_start_s,
                slope_a=slope_a,
                monotonic_start_ms=monotonic_start_ms,
                frame_lookup=frame_lookup,
            )
            segments.append({
                "segment_id": f"seg_{idx + 1:03d}",
                "speaker_label": speaker_label,
                "speaker_verified": False,
                "timing_provenance": "estimated_equal_duration",
                "alignment_review_required": True,
                "text": part,
                **aligned,
            })

    return segments


@diagnosed("audio.transcribe")
def transcribe_with_meta_muse(
    wav_path: Path,
    api_key: str,
    *,
    model: str = "muse-voice-transcribe",
    language_code: str = "en",
    video_anchor_start_s: float = 0.0,
    slope_a: float = 1.0,
    monotonic_start_ms: float | None = None,
    device_label: str = "Bluetooth Collar Mic",
    frame_lookup: list[dict[str, Any]] | None = None,
    manual_cues: list[dict[str, Any]] | None = None,
    endpoint_url: str = META_TRANSCRIBE_URL,
    opener: Any = None,
) -> dict[str, Any]:
    """Call Meta Muse Voice Transcribe API on a 16kHz WAV file and return a structured `touchline.transcript.v1` ledger."""
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise ValueError(f"Audio file not found: {wav_path}")
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Meta API key is required for Meta Muse transcription")

    wav_bytes = wav_path.read_bytes()
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        audio_duration_s = round(wf.getnframes() / float(wf.getframerate() or 16000), 3)

    fields = {
        "model": model,
        "language": language_code,
        "response_format": "verbose_json",
    }
    body, content_type = _build_multipart_form(
        fields=fields,
        file_field="file",
        filename=wav_path.name,
        file_bytes=wav_bytes,
        content_type="audio/wav",
    )

    req = urllib.request.Request(
        endpoint_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {cleaned_key}",
            "Content-Type": content_type,
            "Accept": "application/json",
        },
    )

    http_open = opener or urllib.request.urlopen
    try:
        with http_open(req, timeout=45) as resp:
            resp_bytes = resp.read()
            raw_json = json.loads(resp_bytes.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Meta speech HTTP {exc.code}") from None
    except Exception as exc:
        raise RuntimeError("Meta speech request failed; inspect safe provider diagnostics") from None

    segments = parse_meta_muse_response(
        raw_json,
        audio_duration_s=audio_duration_s,
        video_anchor_start_s=video_anchor_start_s,
        slope_a=slope_a,
        monotonic_start_ms=monotonic_start_ms,
        frame_lookup=frame_lookup,
    )

    aligned_cues: list[dict[str, Any]] = []
    for idx, cue in enumerate(manual_cues or []):
        c_audio_s = float(cue.get("audio_offset_s", 0.0))
        aligned = align_audio_segment_to_video(
            c_audio_s,
            c_audio_s + 0.5,
            video_anchor_start_s=video_anchor_start_s,
            slope_a=slope_a,
            monotonic_start_ms=monotonic_start_ms,
            frame_lookup=frame_lookup,
        )
        aligned_cues.append({
            "cue_id": f"cue_{idx + 1:03d}",
            "note": str(cue.get("note") or "Coach live timestamp marker"),
            "target_track_id": cue.get("target_track_id"),
            **aligned,
        })

    return {
        "schema_version": "touchline.transcript.v1",
        "provider": "meta_muse",
        "model": model,
        "language_code": raw_json.get("language") or language_code,
        "device_label": device_label,
        "audio_duration_s": audio_duration_s,
        "alignment": {
            "sync_model": "video_time_s = slope_a * audio_time_s + video_anchor_start_s",
            "video_anchor_start_s": round(video_anchor_start_s, 3),
            "slope_a": round(slope_a, 6),
            "monotonic_start_ms": monotonic_start_ms,
        },
        "transcript_full": str(raw_json.get("text") or raw_json.get("transcript") or "").strip(),
        "segments": segments,
        "manual_cues": aligned_cues,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def transcribe_coach_audio(
    wav_path: Path,
    api_key: str | None = None,
    *,
    provider: str = "meta",
    model: str | None = None,
    language_code: str = "en",
    video_anchor_start_s: float = 0.0,
    slope_a: float = 1.0,
    monotonic_start_ms: float | None = None,
    device_label: str = "Bluetooth Collar Mic",
    frame_lookup: list[dict[str, Any]] | None = None,
    manual_cues: list[dict[str, Any]] | None = None,
    runs_root: Path | None = None,
    endpoint_url: str | None = None,
    opener: Any = None,
) -> dict[str, Any]:
    """Unified coach transcriber with explicit provider selection; no silent fallback."""
    if provider == "meta":
        k = api_key or load_meta_key(runs_root)
        if not k:
            raise ValueError("Meta API key required for Meta Muse Voice Transcribe")
        return transcribe_with_meta_muse(
            wav_path,
            k,
            model=model or "muse-voice-transcribe",
            language_code=language_code,
            video_anchor_start_s=video_anchor_start_s,
            slope_a=slope_a,
            monotonic_start_ms=monotonic_start_ms,
            device_label=device_label,
            frame_lookup=frame_lookup,
            manual_cues=manual_cues,
            endpoint_url=endpoint_url or META_TRANSCRIBE_URL,
            opener=opener,
        )
    elif provider == "sarvam":
        k = api_key or load_sarvam_key(runs_root)
        if not k:
            raise ValueError("Sarvam API key required for Sarvam AI transcription")
        return transcribe_with_sarvam(
            wav_path,
            k,
            model=model or "saarika:v2",
            language_code=language_code if language_code != "en" else "en-IN",
            video_anchor_start_s=video_anchor_start_s,
            slope_a=slope_a,
            monotonic_start_ms=monotonic_start_ms,
            device_label=device_label,
            frame_lookup=frame_lookup,
            manual_cues=manual_cues,
            opener=opener,
        )
    else:
        raise ValueError(f"Unknown speech-to-text provider: {provider}. Supported: 'meta', 'sarvam'")


@diagnosed("provider.meta")
def generate_tactical_coach_analysis_with_meta_muse(
    player_name: str,
    physical_metrics: dict[str, Any],
    aligned_voice_notes: list[str],
    *,
    api_key: str,
    model: str = "muse-spark",
    endpoint_url: str = META_CHAT_URL,
    opener: Any = None,
) -> dict[str, Any]:
    """Call Meta Muse Wonder/Spark to synthesize tactical feedback & actionable training drills."""
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Meta API key required for Meta Muse tactical analysis")

    prompt = (
        f"You are a Premier League tactical analyst for {player_name}.\n"
        f"Physical match workload: {json.dumps(physical_metrics)}\n"
        f"Coach on-field live microphone cues: {json.dumps(aligned_voice_notes)}\n"
        "Provide a structured JSON response with:\n"
        "1. tactical_summary (2 sentences)\n"
        "2. key_strengths (list of 2 strings)\n"
        "3. areas_for_growth (list of 2 strings)\n"
        "4. training_drills (list of 3 specific drills with drill_name, objective, and reps)"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional football tactics and development coach. Respond in valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {cleaned_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    http_open = opener or urllib.request.urlopen
    try:
        with http_open(req, timeout=45) as resp:
            resp_bytes = resp.read()
            raw_json = json.loads(resp_bytes.decode("utf-8"))
            content = raw_json.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {"raw_tactical_notes": content}
            return {
                "status": "success",
                "model": model,
                "provider": "meta_muse",
                "tactical_assessment": parsed,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
    except Exception as exc:
        raise RuntimeError("Meta tactical request failed; inspect safe provider diagnostics") from None
