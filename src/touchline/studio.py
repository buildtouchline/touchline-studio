"""Touchline Review Studio — Professional Frame.io & Hudl Sportscode Football Video Suite.

Features:
- Fixed-Viewport Full-Screen Application (Zero window scroll, responsive 100vh app layout)
- Native Hardware-Accelerated Video Playback with HTTP 206 Partial Content Streaming
- Broadcast-Quality Player Tracking (Thin 1.5px outlines, dark frosted tags, smooth ground contact trails)
- Real-Time On-Screen Coach Subtitles Overlay directly on Video
- Bluetooth Collar Mic Audio Recording with direct Sarvam AI Transcription
- Clickable Voice-to-Video Navigation (Click any subtitle to jump to that moment)
- Burned-In Subtitle Video Export (.mp4) via PyAV & Pillow
- Full Subtitle Export (.srt, .vtt) & Video Download
- Inline Subtitle & Cue Editor in Any Desired Language (English, Hindi, Tamil, Telugu, Kannada, Malayalam)
- Tracked Players Workload Roster (Speed m/s, Distance m)
"""
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import urllib.parse
import uuid
import threading

from .voice_review import append_transcript, select_speech_provider
_TRANSCRIPT_LOCK = threading.RLock()

from .analytics import generate_match_analytics
from .audio import (
    convert_audio_to_wav16k,
    extract_frame_lookup,
    load_meta_key,
    load_sarvam_key,
    mask_api_key,
    save_sarvam_key,
    transcribe_coach_audio,
    transcribe_with_sarvam,
)
from .data.manifest import sha256_file
from .job_state import JOB_ID, read_job_status, write_job_status
from .metrics import Calibration
from .tracking import PlayerAssociation
from .pipeline import run_pipeline
from .video.decode import decode_video
from .diagnostics import diagnostic_session, diagnostic_span, record_failure, current_trace_id, DiagnosticUnavailable
from functools import wraps
from .video.sampling import FixedRateSampler
from .viewer import render_inspection_dashboard
from .meta_activity import MODELS as META_MODELS, execute_activity, list_activities, new_activity, save_activity, source_frame

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB bounded streaming upload
_UPLOAD_CHUNK_BYTES = 64 * 1024


def _new_panorama_job_id() -> str:
    return f"upload-{int(time.time())}-{uuid.uuid4().hex}"


def _open_run_artifact(root: Path, target: Path):
    """Open a regular artifact through no-follow directory descriptors."""
    parts = target.relative_to(root).parts
    if not parts or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("Safe run artifact opening is unsupported on this platform")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    directory = os.open(root, flags | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_directory = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        artifact = os.open(parts[-1], flags, dir_fd=directory)
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(artifact).st_mode):
        os.close(artifact)
        raise OSError("Run artifact is not a regular file")
    return os.fdopen(artifact, "rb")


def make_broadcast_calibration(width: int = 1920, height: int = 1080, duration_s: float = 3600.0) -> Calibration:
    """Preliminary demo scale: assumes 52m x 34m visible area (unvalidated estimate)."""
    sx = 52.0 / float(width)
    sy = 34.0 / float(height)
    return Calibration(
        matrix=(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0),
        independent_check_ref="PRELIMINARY_assumed_visible_area_no_survey",
        valid_from_s=0.0,
        valid_until_s=max(60.0, duration_s + 10.0),
        image_bounds=(0.0, 0.0, float(width), float(height)),
        source_geometry=(width, height),
        geometry_revision="studio_v1",
        calibration_revision="preliminary_demo_scale_unvalidated",
        ground_contact_convention="bottom_center",
        is_valid=False,
        invalidation_reason="Preliminary assumed scale has no independent surveyed calibration",
        units="metres",
    )


def ensure_calibrated_demo_run(runs_root: Path) -> Path:
    """Ensure `/runs/calibrated-snmot097` has active calibration and video available."""
    src_dir = runs_root / "live-video-review-snmot097"
    if not src_dir.is_dir():
        alt_src = runs_root / "soccernet-run-01" / "live-video-review-snmot097"
        if alt_src.is_dir():
            src_dir = alt_src
    cal_dir = runs_root / "calibrated-snmot097"
    if not src_dir.is_dir() and not cal_dir.is_dir():
        return cal_dir
    cal_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.is_dir():
        for fname in ("manifest.json", "observations.jsonl", "coverage.jsonl", "associations.json", "source_video.mp4"):
            s = src_dir / fname
            d = cal_dir / fname
            if s.is_file() and not d.is_file():
                shutil.copy2(s, d)

    manifest_file = cal_dir / "manifest.json"
    obs_file = cal_dir / "observations.jsonl"
    if manifest_file.is_file() and obs_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        observations = [
            json.loads(line)
            for line in obs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assoc_file = cal_dir / "associations.json"
        associations = json.loads(assoc_file.read_text(encoding="utf-8")) if assoc_file.is_file() else None
        cal = make_broadcast_calibration(1920, 1080, 120.0)
        analytics = generate_match_analytics(manifest, observations, calibration=cal, associations=associations)
        (cal_dir / "analytics.json").write_text(json.dumps(analytics, indent=2) + "\n", encoding="utf-8")
        transcript = _load_transcript(cal_dir)
        (cal_dir / "review.html").write_text(
            render_inspection_dashboard(manifest, analytics, observations, transcript=transcript),
            encoding="utf-8",
        )
    return cal_dir


def _safe_path(root: Path, relative: str) -> Path | None:
    """Resolve a public artifact path without traversal, hidden files or symlinks."""
    try:
        relative = urllib.parse.unquote(relative, errors="strict")
        if not relative or any(c in relative for c in ("\\", ":", "\0", "%")):
            return None
        parts = relative.split("/")
        if any(not part or part.startswith(".") for part in parts):
            return None
        root = root.resolve()
        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                return None
        target = current.resolve()
        if target == root or not target.is_relative_to(root):
            return None
        return target
    except (OSError, ValueError, UnicodeError):
        return None


def _find_run_dir(runs_root: Path, run_id: str) -> Path | None:
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return None
    c1 = (runs_root / run_id).resolve()
    if (_safe_path(runs_root, run_id) is not None and c1.is_dir()
            and _safe_path(c1, "manifest.json") is not None and (c1 / "manifest.json").is_file()):
        return c1
    c2 = (runs_root / "soccernet-run-01" / run_id).resolve()
    if (_safe_path(runs_root, "soccernet-run-01/" + run_id) is not None and c2.is_dir()
            and _safe_path(c2, "manifest.json") is not None and (c2 / "manifest.json").is_file()):
        return c2
    return None


def _load_run(run_dir: Path):
    manifest_file = _safe_path(run_dir, "manifest.json")
    observation_file = _safe_path(run_dir, "observations.jsonl")
    if manifest_file is None or observation_file is None:
        raise ValueError("Unsafe run artifact path")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    observations = [
        json.loads(line)
        for line in observation_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest, observations


def _load_transcript(run_dir: Path):
    tfile = _safe_path(run_dir, "transcript.json")
    if tfile is not None and tfile.is_file():
        try:
            return json.loads(tfile.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_srt_content(transcript: dict) -> str:
    segments = transcript.get("segments", [])
    cues = transcript.get("manual_cues", [])
    all_items = []
    for s in segments:
        all_items.append({
            "start": float(s.get("video_start_s", 0.0)),
            "end": float(s.get("video_end_s", float(s.get("video_start_s", 0.0)) + 2.0)),
            "text": str(s.get("text", "")).strip(),
        })
    for c in cues:
        all_items.append({
            "start": float(c.get("video_start_s", 0.0)),
            "end": float(c.get("video_end_s", float(c.get("video_start_s", 0.0)) + 2.0)),
            "text": str(c.get("text", "")).strip(),
        })
    all_items.sort(key=lambda x: x["start"])

    lines = []
    for idx, item in enumerate(all_items, 1):
        if not item["text"]:
            continue
        lines.append(str(idx))
        lines.append(f"{format_timestamp_srt(item['start'])} --> {format_timestamp_srt(item['end'])}")
        lines.append(item["text"])
        lines.append("")
    return "\n".join(lines)


def build_vtt_content(transcript: dict) -> str:
    srt = build_srt_content(transcript)
    if not srt.strip():
        return "WEBVTT\n\n"
    vtt = "WEBVTT\n\n" + srt.replace(",", ".")
    return vtt


def generate_video_with_subtitles(src_video: Path, transcript: dict, out_video: Path):
    """Burn subtitles directly onto the bottom of the video frames using PyAV and Pillow."""
    import av
    from PIL import ImageDraw

    segments = transcript.get("segments", []) + transcript.get("manual_cues", [])
    sub_intervals = []
    for s in segments:
        st = float(s.get("video_start_s", 0.0))
        en = float(s.get("video_end_s", st + 2.5))
        txt = str(s.get("text", "")).strip()
        if txt:
            sub_intervals.append((st, en, txt))

    input_container = av.open(str(src_video))
    in_stream = input_container.streams.video[0]
    fps = float(in_stream.average_rate) if in_stream.average_rate else 25.0
    w = in_stream.width
    h = in_stream.height

    out_temp = out_video.with_suffix(".tmp.mp4")
    output_container = av.open(str(out_temp), mode="w", format="mp4")
    out_stream = output_container.add_stream("h264", rate=int(fps))
    out_stream.width = w
    out_stream.height = h
    out_stream.pix_fmt = "yuv420p"

    time_base = in_stream.time_base
    for frame in input_container.decode(in_stream):
        curr_t = float(frame.pts * time_base) if frame.pts is not None and time_base else 0.0
        active_txt = None
        for st, en, txt in sub_intervals:
            if st <= curr_t <= en:
                active_txt = txt
                break

        img = frame.to_image()
        if active_txt:
            draw = ImageDraw.Draw(img)
            banner_text = f"COACH: {active_txt}"
            banner_h = 42
            banner_w = min(w - 40, max(240, len(banner_text) * 11 + 36))
            bx1 = (w - banner_w) // 2
            by1 = h - 68
            bx2 = bx1 + banner_w
            by2 = by1 + banner_h
            draw.rounded_rectangle([bx1, by1, bx2, by2], radius=8, fill=(12, 14, 20), outline=(255, 255, 255), width=1)
            draw.text((bx1 + 16, by1 + 14), banner_text, fill=(255, 255, 255))

        new_frame = av.VideoFrame.from_image(img)
        for packet in out_stream.encode(new_frame):
            output_container.mux(packet)

    for packet in out_stream.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    if out_temp.is_file():
        shutil.move(str(out_temp), str(out_video))


def list_all_sessions(runs_root: Path, active_job_ids: set | None = None):
    candidates = []
    if runs_root.is_dir():
        for p in runs_root.iterdir():
            if p.is_dir() and (p / "manifest.json").is_file():
                candidates.append(p)
        sn_dir = runs_root / "soccernet-run-01"
        if sn_dir.is_dir():
            for p in sn_dir.iterdir():
                if p.is_dir() and (p / "manifest.json").is_file() and p not in candidates:
                    candidates.append(p)

    found = []
    for c in candidates:
        if _safe_path(runs_root, c.relative_to(runs_root).as_posix()) is None:
            continue
        mfile = _safe_path(c, "manifest.json")
        if mfile is None:
            continue
        try:
            m = json.loads(mfile.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel_dir = str(c.relative_to(runs_root)).replace("\\", "/")
        afile = _safe_path(c, "analytics.json")
        is_cal = False
        tot_tracks = 0
        if afile is not None and afile.is_file():
            try:
                ad = json.loads(afile.read_text(encoding="utf-8"))
                calibration = ad.get("calibration", {})
                validation = calibration.get("validation", {})
                is_cal = (ad.get("synthetic") is not True
                          and calibration.get("status") == "verified"
                          and calibration.get("calibrated") is True
                          and bool(calibration.get("independent_check_ref"))
                          and validation.get("held_out_check_passed") is True
                          and validation.get("geometry_match") is True
                          and validation.get("independent_reference_verified") is True)
                tot_tracks = int(ad.get("quality_indicators", {}).get("total_tracks", 0))
            except Exception:
                pass
        has_video = (c / "source_video.mp4").is_file()
        has_transcript = (c / "transcript.json").is_file()

        label = c.name
        if c.name == "calibrated-snmot097":
            label = "SNMOT-097 legacy demo (scale unverified)"
        elif c.name == "live-video-review-snmot097":
            label = "SNMOT-097 Match Video (Pixels Mode)"
        elif c.name.startswith("upload-"):
            label = f"Uploaded Footage: {c.name[:18]}"
        else:
            label = f"Session: {c.name}"

        found.append({
            "id": c.name,
            "rel_dir": rel_dir,
            "name": label,
            "status": m.get("status", "unknown"),
            "synthetic": m.get("synthetic") is True,
            "calibrated": is_cal,
            "total_tracks": tot_tracks,
            "duration_s": m.get("duration_s", 0.0),
            "has_video": has_video,
            "has_transcript": has_transcript,
        })
    known_ids = {item["id"] for item in found}
    if runs_root.is_dir():
        for candidate in runs_root.iterdir():
            if candidate.name in known_ids or not JOB_ID.fullmatch(candidate.name):
                continue
            job = read_job_status(runs_root, candidate.name, active_job_ids=active_job_ids)
            if job is None:
                continue
            found.append({"id": candidate.name, "rel_dir": candidate.name,
                          "name": f"Uploaded Footage: {candidate.name[:18]}",
                          "status": job["status"], "synthetic": False,
                          "calibrated": False, "total_tracks": 0, "duration_s": 0.0,
                          "has_video": (candidate / "source_video.mp4").is_file(),
                          "has_transcript": False})
    found.sort(key=lambda x: (not x["calibrated"], not x["has_video"], x["id"]))
    return found


WEB_DIR = Path(__file__).parent / "web"
PUBLIC_WEB_FILES = frozenset({
    "index.html", "styles.css", "app.js", "request.js", "timeline.js",
    "development.js", "development-storage.js", "player-development-api.js",
    "parent-preview.html", "parent-preview.css", "parent-preview.js", "upload.js",
    "vikzar-icon.png", "api-client.js",
})
PUBLIC_RUN_FILES = frozenset({"source_video.mp4", "overlay24.mp4", "video_with_subtitles.mp4", "review.html"})


def diagnostic_http(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        self._diagnostic_response_started = False
        try:
            with diagnostic_session():
                try:
                    with diagnostic_span('studio.request'):
                        return function(self, *args, **kwargs)
                except DiagnosticUnavailable:
                    raise
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                except Exception as exc:
                    record_failure(exc, 'studio.request')
                    if self._diagnostic_response_started:
                        self.close_connection = True
                    else:
                        self._json(400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 500, {})
        except DiagnosticUnavailable:
            # No raw traceback/provider exception is delegated to HTTPServer.
            if not self._diagnostic_response_started:
                body = b'{"error":"Diagnostics unavailable; execution blocked","code":"DIAGNOSTICS_UNAVAILABLE"}'
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if self.command != 'HEAD':
                    self.wfile.write(body)
            self.close_connection = True
    return wrapped


def run_studio_server(
    runs_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    settings=None,
    speech_executor=None,
    worker_adapter=None,
    auth_token: str | None = None,
    max_upload_bytes: int = _MAX_UPLOAD_BYTES,
    run_access_policy=None,
    roster_authorizer=None,
    reviewer_identity: str | None = None,
):
    if auth_token is not None and (not isinstance(auth_token, str) or not auth_token.strip()):
        raise ValueError("auth_token must be a non-empty string when provided")
    normalized_host = (host or "").strip().lower()
    if normalized_host not in _LOOPBACK_HOSTS and not auth_token:
        raise ValueError(
            "Refusing to bind Studio server to non-loopback host without an explicit auth_token; "
            "keep local development isolated on 127.0.0.1/localhost"
        )
    if worker_adapter is not None and getattr(worker_adapter, "allow_live_dispatch", False) and not auth_token:
        raise ValueError("Live Panorama dispatch requires server-side Studio authentication")
    if worker_adapter is not None and getattr(worker_adapter, "allow_live_dispatch", False) and (
        run_access_policy is None or roster_authorizer is None or not reviewer_identity
    ):
        raise ValueError("Live Panorama dispatch requires run, roster and reviewer authorization")
    runtime_checkpoint = settings.checkpoint_path if settings is not None else None
    runtime_device = settings.device if settings is not None else 'cpu'
    if settings is not None and (runtime_checkpoint is None or not runtime_checkpoint.is_file() or runtime_checkpoint.suffix.lower() not in ('.pt', '.pth')):
        raise ValueError('Configured Studio needs existing trained .pt/.pth checkpoint weights')
    runs_root = runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    if worker_adapter is None:
        from .adapters.panorama_worker import PanoramaWorkerAdapter
        worker_adapter = PanoramaWorkerAdapter(runs_root)
    active_job_ids = set()
    meta_activity_lock = threading.RLock()
    # Legacy synthetic calibration is never generated on Studio startup.

    class StudioHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def end_headers(self):
            trace_id = current_trace_id()
            if trace_id:
                self.send_header('X-Touchline-Trace-ID', trace_id)
            self._diagnostic_response_started = True
            super().end_headers()

        def _json(self, code: int, payload: dict | list):
            if code >= 400:
                if self.command == "POST":
                    # Rejected request bodies may still be unread on HTTP/1.1.
                    self.close_connection = True
                problem = record_failure(ValueError(), 'studio.request', status=code)
                payload = {"error": problem['message'], "code": problem['code'], "trace_id": problem['trace_id']}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def send_error(self, code, message=None, explain=None):
            self._json(code, {})

        def _check_auth(self) -> bool:
            if not auth_token:
                return True
            header = self.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                self._json(401, {"error": "Bearer authentication required"})
                return False
            token = header[len("Bearer "):].strip()
            if not hmac.compare_digest(token.encode("utf-8"), auth_token.encode("utf-8")):
                self._json(401, {"error": "Invalid authentication token"})
                return False
            return True

        def _check_run_access(self, run_id: str, action: str) -> bool:
            if run_access_policy is None:
                return True  # Explicitly loopback-only local development mode.
            try:
                allowed = run_access_policy(run_id, action) is True
            except Exception:
                allowed = False
            if not allowed:
                self._json(403, {})
            return allowed

        @diagnostic_http
        def do_HEAD(self):
            self._serve_request(is_head=True)

        @diagnostic_http
        def do_GET(self):
            self._serve_request(is_head=False)

        def _serve_request(self, is_head: bool = False):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                target = _safe_path(WEB_DIR, "index.html")
                if target is not None and target.is_file():
                    body = target.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if not is_head:
                        self.wfile.write(body)
                    return

            if path in ("/styles.css", "/app.js") or path.startswith("/web/"):
                rel = path[len("/web/"):] if path.startswith("/web/") else path.lstrip("/")
                target = _safe_path(WEB_DIR, rel)
                if (target is not None and target.name in PUBLIC_WEB_FILES
                        and target.parent == WEB_DIR.resolve() and target.is_file()):
                    ctype = "text/plain"
                    if target.suffix == ".css":
                        ctype = "text/css; charset=utf-8"
                    elif target.suffix == ".js":
                        ctype = "application/javascript; charset=utf-8"
                    elif target.suffix == ".html":
                        ctype = "text/html; charset=utf-8"
                    elif target.suffix == ".svg":
                        ctype = "image/svg+xml"
                    elif target.suffix == ".png":
                        ctype = "image/png"
                    body = target.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if not is_head:
                        self.wfile.write(body)
                    return

            if (path.startswith("/api/") or path.startswith("/runs/")) and not self._check_auth():
                return
            if path == "/api/runs" and not self._check_run_access("*", "list"):
                return
            if path.startswith(("/api/job/", "/api/jobs/")):
                rid = path.rsplit("/", 1)[-1]
                if not self._check_run_access(rid, "read"):
                    return
            if path.startswith(("/api/run/", "/api/runs/", "/runs/")):
                rid = path.split("/")[3] if path.startswith("/api/") else path.split("/")[2]
                if not self._check_run_access(rid, "read"):
                    return

            if path == "/api/meta-key":
                self._json(200, {"configured": bool(os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")),
                                 "server_managed": True})
                return

            if path == "/api/sarvam-key":
                key = load_sarvam_key(runs_root)
                self._json(200, {"configured": bool(key), "masked": mask_api_key(key)})
                return

            if path == "/api/runs":
                sessions = list_all_sessions(runs_root, active_job_ids=active_job_ids)
                if run_access_policy is not None:
                    def visible(session):
                        try:
                            return run_access_policy(session["id"], "read") is True
                        except Exception:
                            return False
                    sessions = [session for session in sessions if visible(session)]
                self._json(200, sessions)
                return

            meta_frame_image_match = re.fullmatch(r"/api/runs?/([^/]+)/meta-activity/(meta-[0-9a-f]{32})/frame\.jpg", path)
            if meta_frame_image_match:
                rid, activity_id = meta_frame_image_match.groups()
                rdir = _find_run_dir(runs_root, rid)
                record = next((row for row in list_activities(rdir) if row["activity_id"] == activity_id), None) if rdir else None
                if record is None or not record.get("frame_sha256") or not (rdir / "source_video.mp4").is_file():
                    self._json(404, {})
                    return
                try:
                    jpeg, _ = source_frame(rdir / "source_video.mp4", record["requested_time_s"])
                    if hashlib.sha256(jpeg).hexdigest() != record["frame_sha256"]:
                        raise ValueError("Frame lineage changed")
                except (OSError, ValueError, RuntimeError):
                    self._json(409, {})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "private, no-store")
                self.end_headers()
                if not is_head:
                    self.wfile.write(jpeg)
                return

            meta_list_match = re.fullmatch(r"/api/runs?/([^/]+)/meta-activity", path)
            if meta_list_match:
                rid = meta_list_match.group(1)
                rdir = _find_run_dir(runs_root, rid)
                if rdir is None:
                    self._json(404, {})
                    return
                configured_limit = os.environ.get("TOUCHLINE_META_MAX_CALLS", "0")
                limit = int(configured_limit) if configured_limit.isdigit() and 0 < int(configured_limit) <= 4 else 0
                used = sum(len(list_activities(folder)) for folder in runs_root.iterdir()
                           if folder.is_dir() and not folder.is_symlink() and (folder / "manifest.json").is_file())
                self._json(200, {"run_id": rid, "activities": list_activities(rdir),
                                 "configured": bool(os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")),
                                 "enabled": limit > 0, "remaining_calls": max(0, limit - used)})
                return

            if path == "/api/analyze/readiness":
                adapter_ready = bool(worker_adapter is not None and getattr(worker_adapter, "is_ready", lambda: False)())
                if adapter_ready:
                    self._json(200, {
                        "ready": True,
                        "upload_endpoint": "/api/runs",
                        "environment": "local_development" if normalized_host in _LOOPBACK_HOSTS else "authenticated_studio",
                        "authenticated": bool(auth_token),
                        "message": "Panorama ResNet50 Faster R-CNN worker adapter ready for upload.",
                    })
                else:
                    self._json(200, {"ready": False, "message":
                        "Authenticated full-recording upload and Sandbox dispatch are not deployed."})
                return

            if path.startswith("/api/job/") or path.startswith("/api/jobs/"):
                prefix = "/api/jobs/" if path.startswith("/api/jobs/") else "/api/job/"
                jid = path[len(prefix):]
                self._json(200, read_job_status(runs_root, jid, active_job_ids=active_job_ids) or {"status": "not_found"})
                return

            if path.startswith("/api/run/") and path.endswith("/subtitles.srt"):
                rid = path[len("/api/run/"):-len("/subtitles.srt")]
                rdir = _find_run_dir(runs_root, rid)
                transcript = _load_transcript(rdir) if rdir else {}
                srt_text = build_srt_content(transcript or {})
                body = srt_text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="coach_subtitles_{rid}.srt"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not is_head:
                    self.wfile.write(body)
                return

            if path.startswith("/api/run/") and path.endswith("/subtitles.vtt"):
                rid = path[len("/api/run/"):-len("/subtitles.vtt")]
                rdir = _find_run_dir(runs_root, rid)
                transcript = _load_transcript(rdir) if rdir else {}
                vtt_text = build_vtt_content(transcript or {})
                body = vtt_text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/vtt; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="coach_subtitles_{rid}.vtt"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not is_head:
                    self.wfile.write(body)
                return

            if path.startswith("/api/run/") and path.endswith("/video_with_subtitles.mp4"):
                rid = path[len("/api/run/"):-len("/video_with_subtitles.mp4")]
                rdir = _find_run_dir(runs_root, rid)
                if not rdir or not (rdir / "source_video.mp4").is_file():
                    self.send_error(404, "Source video not found")
                    return
                src_v = _safe_path(rdir, "source_video.mp4")
                sub_v = _safe_path(rdir, "video_with_subtitles.mp4")
                if src_v is None or sub_v is None:
                    self.send_error(404, "Not found")
                    return
                transcript = _load_transcript(rdir) or {}
                tfile = rdir / "transcript.json"
                needs_gen = not sub_v.is_file()
                if not needs_gen and tfile.is_file():
                    needs_gen = tfile.stat().st_mtime > sub_v.stat().st_mtime

                if needs_gen:
                    try:
                        generate_video_with_subtitles(src_v, transcript, sub_v)
                    except Exception:
                        sub_v = src_v

                self._serve_file_range(sub_v, f"match_{rid}_with_subtitles.mp4", is_head=is_head, attachment=True)
                return

            if (path.startswith("/api/run/") or path.startswith("/api/runs/")) and path.endswith("/manifest"):
                prefix = "/api/runs/" if path.startswith("/api/runs/") else "/api/run/"
                rid = path[len(prefix):-len("/manifest")]
                rdir = _find_run_dir(runs_root, rid)
                if not rdir:
                    self._json(404, {"error": f"run not found: {rid}"})
                    return
                mfile = _safe_path(rdir, "manifest.json")
                if mfile is None or not mfile.is_file():
                    self._json(404, {"error": "Manifest unavailable"})
                    return
                try:
                    self._json(200, json.loads(mfile.read_text(encoding="utf-8")))
                except Exception:
                    self._json(500, {"error": "Manifest parse error"})
                return

            if (path.startswith("/api/run/") or path.startswith("/api/runs/")) and path.endswith("/observations"):
                prefix = "/api/runs/" if path.startswith("/api/runs/") else "/api/run/"
                rid = path[len(prefix):-len("/observations")]
                rdir = _find_run_dir(runs_root, rid)
                if not rdir:
                    self._json(404, {})
                    return
                from .observation_pages import observation_page
                query = urllib.parse.parse_qs(parsed.query)
                page = observation_page(rdir / "observations.jsonl", offset=int(query.get("offset", ["0"])[0]), limit=int(query.get("limit", ["2000"])[0]))
                self._json(200, page)
                return

            if path.startswith("/api/run/") or (path.startswith("/api/runs/") and path != "/api/runs"):
                prefix = "/api/runs/" if path.startswith("/api/runs/") else "/api/run/"
                rid = path[len(prefix):]
                rdir = _find_run_dir(runs_root, rid)
                if not rdir:
                    self._json(404, {"error": f"run not found: {rid}"})
                    return
                afile = _safe_path(rdir, "analytics.json")
                if afile is None:
                    self._json(404, {"error": "Unsafe run artifact path"})
                    return
                try:
                    manifest, observations = _load_run(rdir)
                    analytics = json.loads(afile.read_text(encoding="utf-8")) if afile.is_file() else {}
                except (OSError, ValueError):
                    self._json(404, {"error": "Run artifacts unavailable"})
                    return
                transcript = _load_transcript(rdir)
                rel_dir = str(rdir.relative_to(runs_root)).replace("\\", "/")
                video_url = f"/runs/{rel_dir}/source_video.mp4" if (rdir / "source_video.mp4").is_file() else None
                frame_map = None
                map_path = _safe_path(rdir, "normalized/frame-map.jsonl")
                if (map_path is not None and map_path.is_file() and
                        map_path.stat().st_size <= 64 * 1024**2 and
                        manifest.get("frame_map_sha256") == sha256_file(map_path)):
                    try:
                        frame_map = [json.loads(line) for line in map_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    except (OSError, ValueError):
                        frame_map = None
                overlay_path = _safe_path(rdir, "overlay24.mp4")
                artifacts = manifest.get("artifacts")
                overlay_record = artifacts.get("annotated_video", {}) if isinstance(artifacts, dict) else {}
                if not isinstance(overlay_record, dict):
                    overlay_record = {}
                overlay_verified = (overlay_path is not None and overlay_path.is_file()
                                    and overlay_record.get("filename") == "overlay24.mp4"
                                    and overlay_record.get("sha256") == sha256_file(overlay_path)
                                    and frame_map is not None)
                self._json(200, {
                    "run_id": rid,
                    "rel_dir": rel_dir,
                    "manifest": manifest,
                    "analytics": analytics,
                    "observations": observations[:1200],
                    "observation_count": len(observations),
                    "observations_next_offset": 1200 if len(observations) > 1200 else None,
                    "transcript": transcript,
                    "video_url": video_url,
                    "annotated_video_url": f"/runs/{rel_dir}/overlay24.mp4" if overlay_verified else None,
                    "annotated_video_verified": overlay_verified,
                    "frame_map": frame_map,
                })
                return

            # Static file serving with HTTP 206 Partial Content (Range requests)
            if path.startswith("/runs/"):
                rel = path[len("/runs/"):]
                target = _safe_path(runs_root, rel)
                if target is not None and target.parent.name == "calibrated-snmot097" and target.name == "review.html":
                    self._json(410, {})
                    return
                if target is not None and not target.is_file():
                    target = _safe_path(runs_root, "soccernet-run-01/" + rel)
                if (target is not None and target.is_file()
                        and target.name in PUBLIC_RUN_FILES and target.parent != runs_root
                        and _safe_path(target.parent, "manifest.json") is not None
                        and (target.parent / "manifest.json").is_file()):
                    self._serve_file_range(target, target.name, is_head=is_head, attachment=False)
                    return

            self.send_error(404, "Not found")

        def _serve_file_range(self, target: Path, filename: str, is_head: bool = False, attachment: bool = False):
            try:
                relative = target.relative_to(runs_root).as_posix()
            except ValueError:
                self.send_error(404, "Not found")
                return
            safe = _safe_path(runs_root, relative)
            if safe is None or safe.name not in PUBLIC_RUN_FILES or not safe.is_file():
                self.send_error(404, "Not found")
                return
            target = safe
            ctype = "application/octet-stream"
            if target.suffix == ".html":
                ctype = "text/html; charset=utf-8"
            elif target.suffix == ".json":
                ctype = "application/json; charset=utf-8"
            elif target.suffix == ".mp4":
                ctype = "video/mp4"
            elif target.suffix == ".wav":
                ctype = "audio/wav"

            try:
                stream = _open_run_artifact(runs_root, target)
            except (OSError, ValueError):
                self.send_error(404, "Not found")
                return
            with stream:
                file_size = os.fstat(stream.fileno()).st_size
                range_header = self.headers.get("Range")
                m = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header or "")
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start >= file_size or end < start:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges", "bytes")
                    if attachment:
                        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                    self.end_headers()
                    if not is_head:
                        stream.seek(start)
                        self.wfile.write(stream.read(length))
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                if attachment:
                    self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                if not is_head:
                    shutil.copyfileobj(stream, self.wfile)

        def _handle_sarvam_key(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                save_sarvam_key(runs_root, data.get("api_key", ""))
                key = load_sarvam_key(runs_root)
                self._json(200, {"configured": True, "masked": mask_api_key(key)})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        @diagnostic_http
        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if not self._check_auth():
                return
            meta_frame_match = re.fullmatch(r"/api/runs?/([^/]+)/meta-frame", path)
            meta_review_match = re.fullmatch(r"/api/runs?/([^/]+)/meta-activity/(meta-[0-9a-f]{32})/review", path)
            if meta_frame_match or meta_review_match:
                rid = (meta_frame_match or meta_review_match).group(1)
                if not self._check_run_access(rid, "write"):
                    return
                rdir = _find_run_dir(runs_root, rid)
                if rdir is None:
                    self._json(404, {})
                    return
                length_text = self.headers.get("Content-Length", "")
                if not length_text.isdigit() or not 0 < int(length_text) <= 2048 or self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                    self._json(400, {})
                    return
                try:
                    body = json.loads(self.rfile.read(int(length_text)))
                except (ValueError, UnicodeError):
                    self._json(400, {})
                    return
                if not isinstance(body, dict):
                    self._json(400, {})
                    return
                if meta_review_match:
                    activity_id = meta_review_match.group(2)
                    decision = body.get("decision")
                    if decision not in ("confirmed_observation", "rejected_observation"):
                        self._json(400, {})
                        return
                    with meta_activity_lock:
                        record = next((row for row in list_activities(rdir) if row["activity_id"] == activity_id), None)
                        if record is None or record["status"] != "response_received" or record.get("review") is not None:
                            self._json(409, {})
                            return
                        record["review"] = {"decision": decision, "reviewed_at": datetime.now(timezone.utc).isoformat(),
                                            "reviewer": reviewer_identity or "local_unauthenticated_coach"}
                        save_activity(rdir, record)
                    self._json(200, record)
                    return
                model_kind = body.get("model")
                selected_time = body.get("timestamp_s")
                if model_kind not in META_MODELS or type(selected_time) not in (int, float) or not math.isfinite(selected_time) or selected_time < 0:
                    self._json(400, {})
                    return
                if not (rdir / "source_video.mp4").is_file():
                    self._json(409, {})
                    return
                key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
                configured_limit = os.environ.get("TOUCHLINE_META_MAX_CALLS", "0")
                if not key or not configured_limit.isdigit() or not 0 < int(configured_limit) <= 4:
                    self._json(503, {})
                    return
                with meta_activity_lock:
                    used = sum(len(list_activities(folder)) for folder in runs_root.iterdir()
                               if folder.is_dir() and not folder.is_symlink() and (folder / "manifest.json").is_file())
                    if used >= int(configured_limit):
                        self._json(429, {})
                        return
                    if any(row.get("model_kind") == model_kind for row in list_activities(rdir)):
                        self._json(409, {})
                        return
                    record = new_activity(rid, model_kind, selected_time)
                    save_activity(rdir, record)
                threading.Thread(target=execute_activity, args=(rdir, record, key), daemon=True,
                                 name="meta-selected-frame").start()
                self._json(202, {"activity_id": record["activity_id"], "status": "submitted", "run_id": rid})
                return
            if path in ("/api/runs", "/api/run") and not self._check_run_access("*", "create"):
                return
            if path.startswith("/api/run/") and path.endswith("/transcript"):
                rid = path[len("/api/run/"):-len("/transcript")]
                if not self._check_run_access(rid, "write"):
                    return
            if path == "/api/audio/transcribe":
                query = urllib.parse.parse_qs(parsed.query)
                rid = query.get("run_id", [""])[0]
                if not rid:
                    self._json(400, {"error": "run_id is required"})
                    return
                if not self._check_run_access(rid, "write"):
                    return
            if path.startswith(("/api/run/", "/api/runs/")) and path.endswith("/corrections"):
                rid = path.split("/", 4)[3]
                if not self._check_run_access(rid, "correct"):
                    return

            if path == "/api/meta-key":
                # Provider credentials must be configured by the trusted server operator.
                # _json closes a rejected POST before unread bytes can become a new request.
                self._json(410, {})
                return

            if path == "/api/sarvam-key":
                self._handle_sarvam_key()
                return

            if path.startswith("/api/run/") and path.endswith("/transcript"):
                rid = path[len("/api/run/"):-len("/transcript")]
                rdir = _find_run_dir(runs_root, rid)
                if not rdir:
                    self._json(404, {"error": f"run not found: {rid}"})
                    return
                content_len = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(content_len).decode("utf-8") or "{}")
                tfile = _safe_path(rdir, "transcript.json")
                if tfile is None:
                    self._json(404, {"error": "Unsafe run artifact path"})
                    return
                tfile.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
                self._json(200, {"status": "ok", "transcript": body})
                return

            if path == "/api/audio/transcribe":
                if speech_executor is None:
                    self._json(503, {'error': 'A budgeted, authorized speech executor is required'})
                    return
                qs = urllib.parse.parse_qs(parsed.query)
                run_id = qs.get("run_id", [""])[0]
                lang = qs.get("language_code", ["en-IN"])[0]
                video_anchor = float(qs.get("video_anchor_start_s", ["0.0"])[0])
                mono_start = float(qs.get("monotonic_start_ms", ["0.0"])[0])
                rdir = _find_run_dir(runs_root, run_id) if run_id else None
                if not rdir:
                    self._json(404, {"error": f"run not found: {run_id}"})
                    return

                content_len = int(self.headers.get("Content-Length", "0"))
                audio_bytes = self.rfile.read(content_len)

                tmp_wav = Path(tempfile.gettempdir()) / f"coach_{uuid.uuid4().hex[:8]}.wav"
                try:
                    # convert_audio_to_wav16k takes raw bytes directly
                    convert_audio_to_wav16k(audio_bytes, tmp_wav)
                    manifest, observations = _load_run(rdir)
                    lookup = extract_frame_lookup(observations)
                    provider_req = qs.get("provider", ["meta"])[0]
                    meta_k = load_meta_key(runs_root)
                    sarvam_k = load_sarvam_key(runs_root)

                    active_provider, active_key = select_speech_provider(provider_req, meta_key=meta_k, sarvam_key=sarvam_k)

                    transcript_data = speech_executor(
                        wav_path=tmp_wav,
                        api_key=active_key,
                        provider=active_provider,
                        video_anchor_start_s=video_anchor,
                        monotonic_start_ms=mono_start,
                        frame_lookup=lookup,
                        language_code=lang,
                    )
                    tfile = _safe_path(rdir, "transcript.json")
                    if tfile is None:
                        raise ValueError('Unsafe transcript destination')
                    with _TRANSCRIPT_LOCK:
                        old = json.loads(tfile.read_text(encoding='utf-8')) if tfile.is_file() else {}
                        transcript_data = append_transcript(old, transcript_data)
                        temporary = rdir / ('transcript-' + uuid.uuid4().hex + '.tmp')
                        temporary.write_text(json.dumps(transcript_data, indent=2) + '\n', encoding='utf-8')
                        temporary.replace(tfile)
                    self._json(200, {"status": "ok", "transcript": transcript_data})
                except Exception as exc:
                    self._json(500, {"error": str(exc)})
                finally:
                    tmp_wav.unlink(missing_ok=True)
                return

            if (path.startswith("/api/run/") or path.startswith("/api/runs/")) and path.endswith("/corrections"):
                prefix = "/api/runs/" if path.startswith("/api/runs/") else "/api/run/"
                rid = path[len(prefix):-len("/corrections")]
                rdir = _find_run_dir(runs_root, rid)
                if not rdir:
                    self._json(404, {"error": f"run not found: {rid}"})
                    return

                length_header = self.headers.get("Content-Length", "")
                if not length_header.isdigit() or not 0 < int(length_header) <= 64 * 1024:
                    self._json(400, {"error": "Correction payload outside accepted bounds"})
                    return
                try:
                    body = json.loads(self.rfile.read(int(length_header)).decode("utf-8"))
                    if not isinstance(body, dict):
                        raise ValueError("Correction payload must be an object")
                except Exception as exc:
                    self._json(400, {"error": f"Invalid correction payload: {exc}"})
                    return

                local_track_id = body.get("local_track_id")
                player_id = body.get("player_id")
                decision = body.get("decision", "confirm")
                reviewer = reviewer_identity or body.get("reviewer", "coach")
                note = body.get("note", "")

                if local_track_id is None or isinstance(local_track_id, bool) or not isinstance(local_track_id, int) or local_track_id <= 0:
                    self._json(400, {"error": "Positive integer local_track_id required"})
                    return
                if decision not in ("confirm", "reject"):
                    self._json(400, {"error": "decision must be 'confirm' or 'reject'"})
                    return
                if not isinstance(reviewer, str) or not reviewer.strip():
                    self._json(400, {"error": "Non-empty reviewer required"})
                    return
                reviewer = reviewer.strip()

                p_manifest_file = _safe_path(rdir, "manifest.json")
                p_obs_file = _safe_path(rdir, "observations.jsonl")
                if p_manifest_file is None or not p_manifest_file.is_file() or p_obs_file is None or not p_obs_file.is_file():
                    self._json(404, {"error": "Parent run artifacts incomplete"})
                    return

                p_manifest = json.loads(p_manifest_file.read_text(encoding="utf-8"))
                observed_track = False
                with p_obs_file.open(encoding="utf-8") as source:
                    for line in source:
                        if json.loads(line).get("local_track_id") == local_track_id:
                            observed_track = True
                            break
                if not observed_track:
                    self._json(400, {"error": "Track is absent from the parent evidence"})
                    return
                child_id = _new_panorama_job_id()

                resolved_player_id: str | None = None
                if decision == "confirm":
                    if not isinstance(player_id, str) or not player_id.strip():
                        self._json(400, {"error": "Non-empty player_id required when confirming an association"})
                        return
                    try:
                        assoc = PlayerAssociation(
                            run_id=child_id,
                            local_track_id=local_track_id,
                            player_id=player_id.strip(),
                            reviewed_by=reviewer,
                        )
                        resolved_player_id = assoc.player_id
                    except ValueError as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    if roster_authorizer is None or roster_authorizer(rid, resolved_player_id) is not True:
                        self._json(403, {"error": "Verified roster authorization required"})
                        return

                child_dir = runs_root / child_id
                child_dir.mkdir(parents=True, exist_ok=False)

                correction_ts = time.time()
                review_ref = f"correction:{child_id}:track-{local_track_id}"
                corrections = list(p_manifest.get("corrections", []))
                corrections.append({
                    "local_track_id": local_track_id,
                    "player_id": resolved_player_id,
                    "decision": decision,
                    "reviewer": reviewer,
                    "review_ref": review_ref,
                    "note": note,
                    "timestamp": correction_ts,
                })
                c_manifest = dict(p_manifest)
                c_manifest["run_id"] = child_id
                c_manifest["parent_run_id"] = rid
                c_manifest["revision"] = p_manifest.get("revision", 1) + 1
                c_manifest["identity_confirmed"] = False
                c_manifest["corrections"] = corrections

                c_obs_file = child_dir / "observations.jsonl"
                obs_count = 0
                with p_obs_file.open(encoding="utf-8") as src, c_obs_file.open("w", encoding="utf-8") as dst:
                    for line in src:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        row["run_id"] = child_id
                        if row.get("local_track_id") == local_track_id:
                            row["player_id"] = resolved_player_id
                            row["association_state"] = "confirmed" if decision == "confirm" else "rejected"
                        else:
                            if row.get("association_state") not in ("confirmed", "rejected"):
                                row["player_id"] = None
                                row["association_state"] = "unreviewed"
                        dst.write(json.dumps(row) + "\n")
                        obs_count += 1
                c_manifest["observation_count"] = obs_count

                associations = {}
                p_assoc = rdir / "associations.json"
                if p_assoc.is_file():
                    try:
                        associations = json.loads(p_assoc.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                associations[str(local_track_id)] = {
                    "player_id": resolved_player_id,
                    "decision": decision,
                    "reviewer": reviewer,
                    "review_ref": review_ref,
                    "note": note,
                    "timestamp": correction_ts,
                }
                (child_dir / "associations.json").write_text(json.dumps(associations, indent=2) + "\n", encoding="utf-8")

                for f_name in ("analytics.json", "source_video.mp4", "overlay24.mp4", "transcript.json", "motion_predictions.jsonl", "coverage.jsonl"):
                    s = rdir / f_name
                    if s.is_file():
                        shutil.copy2(s, child_dir / f_name)

                c_analytics_file = child_dir / "analytics.json"
                if c_analytics_file.is_file():
                    try:
                        c_analytics = json.loads(c_analytics_file.read_text(encoding="utf-8"))
                        c_analytics["run_id"] = child_id
                        roster = c_analytics.get("player_roster")
                        if isinstance(roster, list):
                            for entry in roster:
                                if not isinstance(entry, dict):
                                    continue
                                if entry.get("local_track_id") == local_track_id:
                                    entry["confirmed_player_id"] = resolved_player_id
                                    entry["association_state"] = "confirmed" if decision == "confirm" else "rejected"
                                    entry["reviewed_by"] = reviewer
                                    entry["review_ref"] = review_ref
                                elif entry.get("association_state") not in ("confirmed", "rejected"):
                                    entry["confirmed_player_id"] = None
                                    entry["association_state"] = "unreviewed"
                        c_analytics_file.write_text(json.dumps(c_analytics, indent=2) + "\n", encoding="utf-8")
                    except Exception:
                        pass

                p_norm = rdir / "normalized"
                if p_norm.is_dir():
                    c_norm = child_dir / "normalized"
                    c_norm.mkdir(exist_ok=True)
                    for f in p_norm.iterdir():
                        if f.is_file():
                            shutil.copy2(f, c_norm / f.name)

                (child_dir / "manifest.json").write_text(json.dumps(c_manifest, indent=2) + "\n", encoding="utf-8")
                (child_dir / "run-record.json").write_text(
                    json.dumps({
                        "run_id": child_id,
                        "parent_run_id": rid,
                        "status": "completed",
                        "stage": "complete",
                        "revision": c_manifest["revision"],
                        "observation_count": obs_count,
                        "identity_confirmed": False,
                        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                write_job_status(child_dir, child_id, "completed", stage="complete", observation_count=obs_count, parent_run_id=rid)

                self._json(201, {
                    "child_run_id": child_id,
                    "parent_run_id": rid,
                    "status": "completed",
                    "corrections_count": len(corrections),
                    "associations": associations,
                })
                return

            if path in ("/api/runs", "/api/run"):
                content_len_str = self.headers.get("Content-Length")
                if not content_len_str or not content_len_str.isdigit():
                    self._json(400, {"error": "Content-Length header required"})
                    return
                content_len = int(content_len_str)
                if content_len <= 0 or content_len > max_upload_bytes:
                    self._json(400, {"error": "Content length outside accepted bounds"})
                    return

                ctype = (self.headers.get("Content-Type") or "").lower()
                if "application/json" in ctype:
                    # Security fix: never copy arbitrary server filesystem paths from a JSON body
                    # or write fabricated placeholder video bytes.
                    self.close_connection = True
                    self._json(400, {"error": "JSON path references are not accepted; upload raw video bytes"})
                    return

                if ctype.split(";", 1)[0].strip() != "video/mp4":
                    self._json(415, {"error": "An MP4 upload is required"})
                    return
                if not worker_adapter.is_ready():
                    self._json(503, {"error": "Panorama processing is not ready"})
                    return

                job_id = _new_panorama_job_id()
                job_dir = runs_root / job_id
                try:
                    job_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
                except FileExistsError:
                    self._json(409, {})
                    return

                video_file = job_dir / "source_video.mp4"
                tmp_upload = job_dir / f".upload-{uuid.uuid4().hex}.tmp"
                remaining = content_len
                try:
                    with tmp_upload.open("wb") as out_f:
                        while remaining > 0:
                            chunk = self.rfile.read(min(_UPLOAD_CHUNK_BYTES, remaining))
                            if not chunk:
                                raise ValueError("Unexpected EOF while reading upload stream")
                            out_f.write(chunk)
                            remaining -= len(chunk)
                    tmp_upload.replace(video_file)
                except Exception:
                    tmp_upload.unlink(missing_ok=True)
                    # Keep the uniquely created job directory for operator
                    # reconciliation; never recursively delete by a path that
                    # another local process could replace after creation.
                    raise

                digest = sha256_file(video_file)
                queued_ts = time.time()
                (job_dir / "run-record.json").write_text(
                    json.dumps({
                        "run_id": job_id,
                        "job_id": job_id,
                        "status": "processing",
                        "stage": "queued",
                        "source_sha256": digest,
                        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                write_job_status(
                    job_dir,
                    job_id,
                    "processing",
                    stage="queued",
                    progress=0,
                    source_sha256=digest,
                    events=[{"stage": "queued", "timestamp": queued_ts}],
                )

                active_job_ids.add(job_id)

                def _run_worker(spec):
                    try:
                        if spec is None:
                            write_job_status(job_dir, job_id, "processing", stage="preparing", source_sha256=digest)
                            spec = worker_adapter.prepare_job_spec(job_dir, job_id, digest)
                        worker_adapter.run_job_sync(job_dir, job_id, spec)
                    except Exception as exc:
                        # run_job_sync records its own failures; preparation has not
                        # entered that lifecycle yet and must never stay processing.
                        current = read_job_status(runs_root, job_id, active_job_ids=active_job_ids)
                        if current is None or current.get("status") == "processing":
                            write_job_status(job_dir, job_id, "failed", stage="preparing",
                                             error_code=type(exc).__name__,
                                             message="Recording preparation or worker execution failed")
                    finally:
                        active_job_ids.discard(job_id)

                if worker_adapter is not None:
                    job_spec = None
                    if self.headers.get("X-Touchline-Sync-Execution") == "true" and worker_adapter.runner is not None:
                        _run_worker(job_spec)
                    else:
                        t = threading.Thread(target=_run_worker, args=(job_spec,), name=f"worker-{job_id}", daemon=True)
                        t.start()

                current_job = read_job_status(runs_root, job_id, active_job_ids=active_job_ids) or {}
                self._json(202, {
                    "job_id": job_id,
                    "run_id": job_id,
                    "status": current_job.get("status", "processing"),
                    "stage": current_job.get("stage", "queued"),
                    "source_sha256": digest,
                })
                return

            if path != "/api/analyze":
                self.send_error(404, "Not found")
                return

            # The selected full-recording Sandbox is an operator workflow, not a
            # deployed authenticated upload service. Close before unread request
            # bytes can be parsed as another HTTP/1.1 request.
            self.close_connection = True
            self._json(503, {})

    server = ThreadingHTTPServer((host, port), StudioHandler)
    print(f"Touchline Review Studio running at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
