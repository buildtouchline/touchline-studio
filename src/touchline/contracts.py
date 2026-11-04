from dataclasses import dataclass, fields
import json
import math
from typing import Any, Protocol

SCHEMA_VERSION = "touchline.observation.v1"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("box coordinates must be finite")
        if not (0 <= self.x1 < self.x2 and 0 <= self.y1 < self.y2):
            raise ValueError("box must be nonempty, nonnegative source-pixel XYXY")

    def fits(self, width, height):
        return self.x2 <= width and self.y2 <= height

    def values(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def iou(self, other):
        intersection = max(0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        area = (self.x2 - self.x1) * (self.y2 - self.y1)
        other_area = (other.x2 - other.x1) * (other.y2 - other.y1)
        return intersection / (area + other_area - intersection)

    def ground_contact_point(self, convention="bottom_center") -> tuple[float, float]:
        """Return 2D image coordinates for pitch ground contact.

        Default convention 'bottom_center' returns ((x1 + x2) / 2, y2).
        Box centre ((x1 + x2) / 2, (y1 + y2) / 2) is strictly rejected because the box centre
        does not represent where an upright or running player contacts the pitch surface.
        """
        if convention in ("box_center", "center"):
            raise ValueError(
                "box centre cannot be silently treated as a ground contact point; "
                "physical ground mapping requires 'bottom_center' or an explicit ground mark"
            )
        if convention != "bottom_center":
            raise ValueError(f"unsupported ground contact convention: {convention}")
        return ((self.x1 + self.x2) / 2.0, float(self.y2))


@dataclass(frozen=True)
class Frame:
    source_id: str
    index: int
    pts: int
    time_base_num: int
    time_base_den: int
    timestamp_s: float
    width: int
    height: int
    image: Any = None

    def __post_init__(self):
        if not self.source_id or type(self.index) is not int or self.index < 0:
            raise ValueError("source ID and exact nonnegative frame index required")
        if type(self.pts) is not int or self.time_base_num <= 0 or self.time_base_den <= 0:
            raise ValueError("original PTS and positive rational time base required")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("invalid source time or geometry")

    def metadata(self):
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name != "image"}


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float

    def __post_init__(self):
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("detector score must be in [0,1]; it is not identity confidence")


@dataclass(frozen=True)
class CapabilityEnvelope:
    name: str
    supported: bool
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class ObservationRecord:
    schema_version: str
    run_id: str
    frame: Frame
    local_track_id: int
    box: Box
    detector_score: float
    association_state: str
    player_id: str | None = None
    synthetic: bool = False

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported observation schema version: {self.schema_version}")
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("observation requires non-empty string run_id")
        if not isinstance(self.local_track_id, int) or self.local_track_id < 0:
            raise ValueError("local_track_id must be a non-negative integer")
        # Enforce that local track ID cannot be automatically adopted as player_id without confirmed review
        if self.player_id is not None:
            normalized_player = str(self.player_id).strip().lower()
            forbidden = (
                str(self.local_track_id),
                f"track_{self.local_track_id}",
                f"track_{self.local_track_id:02d}",
                f"local_{self.local_track_id}",
            )
            if normalized_player in forbidden:
                raise ValueError(
                    f"local track ID ({self.local_track_id}) cannot be used directly as a confirmed roster player_id without review"
                )


def deserialize_observation(data: dict | str) -> ObservationRecord:
    """Deserialize and validate an observation in a provider-independent inspection path without tensor dependencies."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as exc:
            raise ValueError(f"invalid JSON observation: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("observation must be a dictionary")

    schema_ver = data.get("schema_version")
    if schema_ver != SCHEMA_VERSION:
        raise ValueError(f"expected schema '{SCHEMA_VERSION}', got '{schema_ver}'")

    raw_frame = data.get("frame")
    if not isinstance(raw_frame, dict):
        raise ValueError("observation missing required 'frame' metadata dictionary")

    frame = Frame(
        source_id=raw_frame["source_id"],
        index=raw_frame["index"],
        pts=raw_frame["pts"],
        time_base_num=raw_frame["time_base_num"],
        time_base_den=raw_frame["time_base_den"],
        timestamp_s=raw_frame["timestamp_s"],
        width=raw_frame["width"],
        height=raw_frame["height"],
    )

    raw_box = data.get("bbox_xyxy")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw_box):
        raise ValueError("observation requires 4-element sequence of numbers for 'bbox_xyxy'")
    box = Box(*raw_box)

    score = data.get("detector_score")
    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
        raise ValueError("observation requires float detector_score in [0, 1]")

    return ObservationRecord(
        schema_version=schema_ver,
        run_id=data.get("run_id", ""),
        frame=frame,
        local_track_id=data.get("local_track_id", 0),
        box=box,
        detector_score=float(score),
        association_state=data.get("association_state", "unrecorded"),
        player_id=data.get("player_id"),
        synthetic=bool(data.get("synthetic", False)),
    )


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...
    def provenance(self) -> dict: ...
