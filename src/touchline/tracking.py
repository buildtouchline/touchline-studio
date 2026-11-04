"""Person tracking, explicit player association, and tracking event ledger.

Implements:
- IoU-based bounding box tracking baseline with crossing/ambiguity detection.
- Tracking lifecycle event ledger (started, matched, ambiguous crossing, lost, terminated).
- Explicit player association and target player selection (TL-004, TL-004.1).
- Run/session-scoped identities to prevent cross-session identity bleed.
- Revisioned correction tracking (TL-S05) ensuring auditability of reassigned tracks.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
from typing import Any

from .contracts import Box, Detection


@dataclass
class Track:
    track_id: int
    box: Box
    last_time: float


@dataclass(frozen=True)
class TrackingEvent:
    timestamp_s: float
    local_track_id: int
    event_type: str  # "started", "matched", "crossing_ambiguous", "lost", "terminated"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 4),
            "local_track_id": self.local_track_id,
            "event_type": self.event_type,
            "details": self.details,
        }


@dataclass(frozen=True)
class PlayerAssociation:
    run_id: str
    local_track_id: int
    player_id: str
    player_name: str | None = None
    confirmed: bool = True
    is_target: bool = False
    reviewed_by: str | None = None
    revision: int = 1
    notes: str | None = None

    def __post_init__(self):
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("association requires a non-empty string run_id")
        if not isinstance(self.local_track_id, int) or self.local_track_id <= 0:
            raise ValueError("local_track_id must be a positive integer")
        if not self.player_id or not isinstance(self.player_id, str):
            raise ValueError("confirmed player_id must be a non-empty string")
        # Enforce contract: raw track ID cannot be adopted as player_id without review
        normalized = self.player_id.strip().lower()
        forbidden = (
            str(self.local_track_id),
            f"track_{self.local_track_id}",
            f"track_{self.local_track_id:02d}",
            f"local_{self.local_track_id}",
        )
        if normalized in forbidden:
            raise ValueError(
                f"local track ID ({self.local_track_id}) cannot be used directly as confirmed player_id"
            )


def reviewed_association(association, *, run_id, local_track_id):
    """Validate a local review record; this does not provide server authentication.

    A bare player ID or generated track label is not review evidence. Consumers
    must supply the actual run-scoped ledger record and preserve its revision.
    """
    if isinstance(association, PlayerAssociation):
        association = asdict(association)
    if not isinstance(association, dict) or not isinstance(run_id, str) or not run_id.strip():
        return None
    if (association.get("run_id") != run_id
            or type(association.get("local_track_id")) is not int
            or association["local_track_id"] != local_track_id
            or association.get("confirmed") is not True
            or association.get("decision", "confirmed") != "confirmed"
            or not isinstance(association.get("player_id"), str)
            or not association["player_id"].strip()
            or type(association.get("revision")) is not int
            or association["revision"] < 1
            or not isinstance(association.get("reviewed_by"), str)
            or not association["reviewed_by"].strip()):
        return None
    try:
        PlayerAssociation(run_id, local_track_id, association.get("player_id"))
    except (ValueError, TypeError):
        return None
    record = {key: association.get(key) for key in
              ("run_id", "local_track_id", "player_id", "revision", "reviewed_by")}
    record["confirmed"] = True
    ref = "association-sha256:" + hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if association.get("review_ref", ref) != ref:
        return None
    return {**record, "review_ref": ref}


class PlayerAssociationLedger:
    """Manages run-scoped player associations, target selection, and revisioned corrections.

    Strictly enforces:
    - Run/session scoping: Track 7 in session A cannot be merged with Track 7 in session B.
    - Correction tracking: Reassigning an association creates an explicit correction record.
    """

    def __init__(self, run_id: str):
        if not run_id or not isinstance(run_id, str):
            raise ValueError("valid run_id required for association ledger")
        self.run_id = run_id
        self.associations: dict[int, PlayerAssociation] = {}
        self.corrections: list[dict[str, Any]] = []

    def associate(
        self,
        local_track_id: int,
        player_id: str,
        player_name: str | None = None,
        is_target: bool = False,
        reviewed_by: str | None = None,
        notes: str | None = None,
    ) -> PlayerAssociation:
        """Associate a local track ID with a confirmed player identity."""
        prior = self.associations.get(local_track_id)
        revision = (prior.revision + 1) if prior else 1
        assoc = PlayerAssociation(
            run_id=self.run_id,
            local_track_id=local_track_id,
            player_id=player_id,
            player_name=player_name,
            confirmed=True,
            is_target=is_target,
            reviewed_by=reviewed_by,
            revision=revision,
            notes=notes,
        )
        if prior and prior.player_id != player_id:
            # Record explicit correction per TL-S05
            self.corrections.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "local_track_id": local_track_id,
                "previous_player_id": prior.player_id,
                "new_player_id": player_id,
                "previous_revision": prior.revision,
                "new_revision": revision,
                "reviewed_by": reviewed_by,
                "reason": notes or "manual re-association",
                "stale_derivatives": ["analytics.json", "review.html"],
            })
        self.associations[local_track_id] = assoc
        return assoc

    def get_association(self, local_track_id: int) -> PlayerAssociation | None:
        return self.associations.get(local_track_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "touchline.association-ledger.v1",
            "run_id": self.run_id,
            "target_tracks": [tid for tid, a in self.associations.items() if a.is_target],
            "associations": {
                str(tid): {
                    "local_track_id": a.local_track_id,
                    "player_id": a.player_id,
                    "player_name": a.player_name,
                    "confirmed": a.confirmed,
                    "is_target": a.is_target,
                    "reviewed_by": a.reviewed_by,
                    "revision": a.revision,
                    "notes": a.notes,
                }
                for tid, a in self.associations.items()
            },
            "corrections": self.corrections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerAssociationLedger":
        if not isinstance(data, dict) or "run_id" not in data:
            raise ValueError("invalid association ledger format: missing run_id")
        ledger = cls(run_id=data["run_id"])
        for tid_str, a in data.get("associations", {}).items():
            tid = int(tid_str)
            ledger.associations[tid] = PlayerAssociation(
                run_id=ledger.run_id,
                local_track_id=a["local_track_id"],
                player_id=a["player_id"],
                player_name=a.get("player_name"),
                confirmed=a.get("confirmed", True),
                is_target=a.get("is_target", False),
                reviewed_by=a.get("reviewed_by"),
                revision=a.get("revision", 1),
                notes=a.get("notes"),
            )
        ledger.corrections = list(data.get("corrections", []))
        return ledger


class IoUTracker:
    def __init__(self, threshold: float = 0.3, max_gap_s: float = 0.75, run_id: str | None = None):
        if not 0 < threshold <= 1 or not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("invalid tracker configuration")
        self.threshold = threshold
        self.max_gap_s = max_gap_s
        self.run_id = run_id or "default_session"
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.last_time: float | None = None
        self.events: list[TrackingEvent] = []
        self.terminated_tracks: dict[int, dict[str, Any]] = {}

    def update(self, timestamp_s: float, detections: list[Detection]) -> list[dict[str, Any]]:
        if not math.isfinite(timestamp_s) or (self.last_time is not None and timestamp_s <= self.last_time):
            raise ValueError("tracker timestamps must increase")
        self.last_time = timestamp_s

        # Identify tracks that exceeded max_gap_s and record termination event
        active_tracks = {}
        for k, t in self.tracks.items():
            gap = timestamp_s - t.last_time
            if gap <= self.max_gap_s:
                active_tracks[k] = t
            else:
                self.events.append(
                    TrackingEvent(
                        timestamp_s=timestamp_s,
                        local_track_id=k,
                        event_type="terminated",
                        details={"last_seen_s": t.last_time, "gap_s": round(gap, 4)},
                    )
                )
                self.terminated_tracks[k] = {"terminated_at_s": timestamp_s, "last_box": t.box.values()}
        self.tracks = active_tracks

        candidates = [
            (track.box.iou(det.box), tid, i)
            for tid, track in self.tracks.items()
            for i, det in enumerate(detections)
            if track.box.iou(det.box) >= self.threshold
        ]

        # Multiple plausible links are flagged, even if greedy assignment succeeds
        ambiguous = {i for _, _, i in candidates if sum(j == i for _, _, j in candidates) > 1}
        contested = {tid for _, tid, _ in candidates if sum(t == tid for _, t, _ in candidates) > 1}
        ambiguous |= {i for _, tid, i in candidates if tid in contested}

        for tid in contested:
            self.events.append(
                TrackingEvent(
                    timestamp_s=timestamp_s,
                    local_track_id=tid,
                    event_type="crossing_ambiguous",
                    details={"contested": True},
                )
            )

        used_tracks, assignments = set(), {}
        for score, tid, i in sorted(candidates, reverse=True):
            if tid not in used_tracks and i not in assignments:
                assignments[i] = tid
                used_tracks.add(tid)

        # Flag active tracks that received no match this frame (lost/occluded)
        if detections:
            for tid, track in self.tracks.items():
                if tid not in used_tracks:
                    self.events.append(
                        TrackingEvent(
                            timestamp_s=timestamp_s,
                            local_track_id=tid,
                            event_type="lost",
                            details={"last_seen_s": track.last_time},
                        )
                    )

        results = []
        for i, det in enumerate(detections):
            tid = assignments.get(i)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.events.append(
                    TrackingEvent(
                        timestamp_s=timestamp_s,
                        local_track_id=tid,
                        event_type="started",
                        details={"bbox_xyxy": det.box.values(), "detector_score": det.score},
                    )
                )
            else:
                self.events.append(
                    TrackingEvent(
                        timestamp_s=timestamp_s,
                        local_track_id=tid,
                        event_type="matched",
                        details={"detector_score": det.score},
                    )
                )

            self.tracks[tid] = Track(tid, det.box, timestamp_s)
            results.append({
                "local_track_id": tid,
                "bbox_xyxy": det.box.values(),
                "detector_score": det.score,
                "association_state": "ambiguous" if i in ambiguous else "tentative",
                "player_id": None,
            })
        return results

    def get_event_ledger(self) -> list[dict[str, Any]]:
        """Return all tracked lifecycle events."""
        return [e.to_dict() for e in self.events]
