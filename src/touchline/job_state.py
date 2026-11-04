"""Small durable status record for the loopback Studio analysis job.

This is not the authenticated, multi-tenant job service proposed for /api/v1.
"""
import json
import os
from pathlib import Path
import re
import time
import uuid


JOB_ID = re.compile(r"upload-[0-9]+-[0-9a-f]{6,32}\Z")
STATUSES = frozenset({"processing", "partial", "completed", "failed"})
STATUS_FILE = "job_status.json"


def write_job_status(job_dir: Path, job_id: str, status: str, **details):
    """Replace a run-owned status record atomically without storing provider errors."""
    if not JOB_ID.fullmatch(job_id) or job_dir.name != job_id or status not in STATUSES:
        raise ValueError("Invalid Studio job state")
    record = {**details, "job_id": job_id, "status": status}
    temporary = job_dir / f".job-status-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, job_dir / STATUS_FILE)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def read_job_status(runs_root: Path, job_id: str, active_job_ids: set | frozenset | None = None):
    """Return a status for a known job; never trust arbitrary URL path segments."""
    if not JOB_ID.fullmatch(job_id):
        return None
    job_dir = runs_root / job_id
    status_file = job_dir / STATUS_FILE
    if job_dir.is_symlink() or status_file.is_symlink() or not status_file.is_file():
        return None
    try:
        record = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(record, dict) or record.get("job_id") != job_id
            or record.get("status") not in STATUSES):
        return None
    manifest_file = job_dir / "manifest.json"
    analytics_file = job_dir / "analytics.json"
    observations_file = job_dir / "observations.jsonl"
    manifest = None
    if (not any(path.is_symlink() for path in (manifest_file, analytics_file, observations_file))
            and manifest_file.is_file() and analytics_file.is_file()
            and observations_file.is_file()):
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    published = (isinstance(manifest, dict) and manifest.get("run_id") == job_id
                 and manifest.get("status") in ("partial", "completed"))
    if record["status"] in ("partial", "completed") and not published:
        return {"job_id": job_id, "status": "unavailable",
                "message": "Saved analysis artifacts are incomplete; ask the operator to inspect the run."}
    if record["status"] == "processing":
        if published:
            return {"job_id": job_id, "status": manifest["status"],
                    "observation_count": manifest.get("observation_count", 0)}
        if active_job_ids is not None and job_id in active_job_ids:
            return record
        # An in-memory worker is the only source of live progress. On restart,
        # never claim a prior dispatch completed or failed.
        return {"job_id": job_id, "status": "unavailable",
                "message": "Worker status is unknown after Studio restart; ask the operator to reconcile it before retrying."}
    return record
