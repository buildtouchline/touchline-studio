"""Measured-trajectory arithmetic and pitch calibration.

Estimates stay estimates; no validation is inferred.
Unsupported or invalid calibration suppresses dependent physical metrics.
"""
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class Point:
    time_s: float
    x_px: float
    y_px: float
    segment_id: str
    identity_review_ref: str | None = None
    contact_convention: str = "bottom_center"
    geometry_revision: str | None = None

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.time_s, self.x_px, self.y_px)):
            raise ValueError("finite positions/timestamps required")
        if self.contact_convention in ("box_center", "center"):
            raise ValueError("box centre cannot be treated as a ground contact point")

    @classmethod
    def from_box(
        cls,
        time_s: float,
        box: Any,
        segment_id: str,
        identity_review_ref: str | None = None,
        convention: str = "bottom_center",
        geometry_revision: str | None = None,
    ) -> "Point":
        """Construct a Point using a Box's declared ground contact convention.

        Rejects box centre ('box_center') because the centre of a person box
        does not represent contact with the pitch surface.
        """
        if convention in ("box_center", "center"):
            raise ValueError("box centre cannot be silently treated as a ground contact point; ground contact requires 'bottom_center'")
        x, y = box.ground_contact_point(convention=convention)
        return cls(
            time_s=time_s,
            x_px=x,
            y_px=y,
            segment_id=segment_id,
            identity_review_ref=identity_review_ref,
            contact_convention=convention,
            geometry_revision=geometry_revision,
        )


@dataclass(frozen=True)
class Calibration:
    matrix: tuple[float, ...]
    independent_check_ref: str | None
    valid_from_s: float
    valid_until_s: float
    image_bounds: tuple[float, float, float, float]
    source_geometry: tuple[int, int] | None = None
    geometry_revision: str | None = None
    calibration_revision: str | None = None
    ground_contact_convention: str = "bottom_center"
    is_valid: bool = True
    invalidation_reason: str | None = None
    units: str = "metres"

    def __post_init__(self):
        object.__setattr__(self, "matrix", tuple(self.matrix))
        object.__setattr__(self, "image_bounds", tuple(self.image_bounds))
        if self.source_geometry is not None:
            object.__setattr__(self, "source_geometry", tuple(self.source_geometry))
        if len(self.matrix) != 9 or not all(math.isfinite(v) for v in self.matrix):
            raise ValueError("finite 3x3 ground-plane mapping required")
        if not (math.isfinite(self.valid_from_s) and math.isfinite(self.valid_until_s) and self.valid_from_s < self.valid_until_s):
            raise ValueError("invalid calibration validity interval")
        a, b, c, d, e, f, g, h, i = self.matrix
        if abs(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)) < 1e-12:
            raise ValueError("singular calibration")
        x1, y1, x2, y2 = self.image_bounds
        if not all(math.isfinite(v) for v in self.image_bounds) or not (x1 < x2 and y1 < y2):
            raise ValueError("invalid calibrated image bounds")
        if self.source_geometry is not None:
            gw, gh = self.source_geometry
            if gw <= 0 or gh <= 0:
                raise ValueError("source geometry width and height must be positive integers")

    def invalidate(self, reason: str) -> "Calibration":
        """Return a copy of this calibration marked as invalid with an explicit reason."""
        return Calibration(
            matrix=self.matrix,
            independent_check_ref=self.independent_check_ref,
            valid_from_s=self.valid_from_s,
            valid_until_s=self.valid_until_s,
            image_bounds=self.image_bounds,
            source_geometry=self.source_geometry,
            geometry_revision=self.geometry_revision,
            calibration_revision=self.calibration_revision,
            ground_contact_convention=self.ground_contact_convention,
            is_valid=False,
            invalidation_reason=reason,
            units=self.units,
        )

    def check_geometry_compatibility(self, width: int, height: int, geometry_revision: str | None = None):
        """Verify camera/source geometry matches this calibration."""
        if self.source_geometry is not None and (width, height) != tuple(self.source_geometry):
            raise ValueError(
                f"source geometry changed from {self.source_geometry} to ({width}, {height}); "
                "geometry change invalidates pitch calibration"
            )
        if self.geometry_revision is not None and geometry_revision is not None:
            if self.geometry_revision != geometry_revision:
                raise ValueError(
                    f"geometry revision changed from '{self.geometry_revision}' to '{geometry_revision}'; "
                    "calibration mapping invalidated"
                )

    def project(self, point: Point) -> tuple[float, float]:
        """Project an image point to ground-plane coordinates.

        Raises:
            ValueError: If calibration is invalid, unverified, out of bounds, out of time interval,
                or uses an unsupported ground-contact convention.
        """
        if not self.is_valid:
            reason = self.invalidation_reason or "calibration marked invalid"
            raise ValueError(f"calibration is invalid: {reason}")
        if not self.independent_check_ref:
            raise ValueError("calibration has no independent-check reference")
        if not (self.valid_from_s <= point.time_s < self.valid_until_s):
            raise ValueError(
                f"timestamp {point.time_s}s is outside calibration validity interval "
                f"[{self.valid_from_s}s, {self.valid_until_s}s)"
            )
        if point.contact_convention in ("box_center", "center"):
            raise ValueError("box centre is not a ground contact point; physical mapping requires verified ground contact (bottom_center)")
        if self.geometry_revision is not None and point.geometry_revision is not None:
            if self.geometry_revision != point.geometry_revision:
                raise ValueError(
                    f"point geometry revision '{point.geometry_revision}' does not match "
                    f"calibration revision '{self.geometry_revision}'"
                )
        x1, y1, x2, y2 = self.image_bounds
        if not (x1 <= point.x_px <= x2 and y1 <= point.y_px <= y2):
            raise ValueError("point outside calibrated image region")
        a, b, c, d, e, f, g, h, i = self.matrix
        denominator = g * point.x_px + h * point.y_px + i
        if abs(denominator) < 1e-12:
            raise ValueError("point projects to infinity")
        return ((a * point.x_px + b * point.y_px + c) / denominator, (d * point.x_px + e * point.y_px + f) / denominator)


def movement_summary(points, calibration=None, max_gap_s=0.75, peak_window_s=1.0):
    if not math.isfinite(max_gap_s) or max_gap_s <= 0 or not math.isfinite(peak_window_s) or peak_window_s <= 0:
        raise ValueError("positive finite gap/window parameters required")
    points = list(points)
    for index, point in enumerate(points):
        if not all(math.isfinite(v) for v in (point.time_s, point.x_px, point.y_px)):
            raise ValueError("finite positions/timestamps required")
        if index and point.time_s <= points[index - 1].time_s:
            raise ValueError("point timestamps must strictly increase")
    result = {
        "status": "unavailable",
        "distance_m": None,
        "average_speed_mps": None,
        "maximum_windowed_speed_mps": None,
        "peak_window_s": peak_window_s,
        "eligible_duration_s": 0.0,
        "excluded_segments": [],
        "reason": None,
    }
    if calibration is None:
        result["reason"] = "calibration and independent check required"
        return result
    if not calibration.is_valid:
        result["reason"] = f"calibration invalid: {calibration.invalidation_reason or 'unspecified reason'}"
        return result
    if not calibration.independent_check_ref:
        result["reason"] = "calibration and independent check required"
        return result

    segments, group = [], 0
    for left, right in zip(points, points[1:]):
        reason = None
        if not left.identity_review_ref or not right.identity_review_ref:
            reason = "identity not reviewed"
        elif left.segment_id != right.segment_id:
            reason = "track/association segment changed"
        elif right.time_s - left.time_s > max_gap_s:
            reason = "gap exceeds eligibility limit"
        try:
            a, b = calibration.project(left), calibration.project(right)
        except ValueError as exc:
            reason = str(exc)
        if reason:
            result["excluded_segments"].append({"start_s": left.time_s, "end_s": right.time_s, "reason": reason})
            group += 1
            continue
        duration = right.time_s - left.time_s
        distance = math.dist(a, b)
        segments.append((left.time_s, right.time_s, distance / duration, group))
    if not segments:
        result["reason"] = "no eligible observed trajectory segments"
        return result
    duration = sum(end - start for start, end, _, _ in segments)
    distance = sum((end - start) * speed for start, end, speed, _ in segments)
    peaks = []
    for _, end, _, group in segments:
        start = end - peak_window_s
        selected = [(a, b, v) for a, b, v, g in segments if g == group and b > start and a < end]
        covered = sum(max(0, min(b, end) - max(a, start)) for a, b, _ in selected)
        if math.isclose(covered, peak_window_s, abs_tol=1e-9):
            peaks.append(sum(max(0, min(b, end) - max(a, start)) * v for a, b, v in selected) / peak_window_s)
    result.update(
        status="estimated",
        distance_m=distance,
        average_speed_mps=distance / duration,
        maximum_windowed_speed_mps=max(peaks) if peaks else None,
        eligible_duration_s=duration,
        reason="No complete peak window" if not peaks else None,
    )
    result["limitations"] = "Piecewise-linear observed trajectory; no jitter filter, gait model or real-world metric validation supplied. Partial exposure only."
    return result


DEFAULT_ACTIVITY_BANDS = (
    ("standing", 0.0, 0.5),
    ("walking", 0.5, 2.0),
    ("jogging", 2.0, 4.0),
    ("running", 4.0, 7.0),
    ("sprinting", 7.0, float("inf")),
)

DEFAULT_INTENSITY_ZONES = (
    ("zone_1_low", 0.0, 2.0),
    ("zone_2_moderate", 2.0, 4.0),
    ("zone_3_high_speed_running", 4.0, 5.5),
    ("zone_4_very_high_speed", 5.5, 7.0),
    ("zone_5_sprint_band", 7.0, float("inf")),
)


def activity_summary(
    points,
    calibration=None,
    max_gap_s: float = 0.75,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_ACTIVITY_BANDS,
) -> dict[str, Any]:
    """Compute speed-rule activity classification estimates (TL-008, TL-008.2).

    Enforces:
    - Versioned class definitions and speed-rule estimate disclosure.
    - Missing/off-camera/unreviewed/gap intervals remain explicitly 'unknown' and
      are NEVER counted as standing or inactivity.
    - Variable sampling timestamps aggregate exact elapsed time without double counting.
    """
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("positive finite max_gap_s required")
    points = list(points)
    for index, point in enumerate(points):
        if not all(math.isfinite(v) for v in (point.time_s, point.x_px, point.y_px)):
            raise ValueError("finite positions/timestamps required")
        if index and point.time_s <= points[index - 1].time_s:
            raise ValueError("point timestamps must strictly increase")

    band_defs = [
        {"class": name, "min_speed_mps": low, "max_speed_mps": None if math.isinf(high) else high}
        for name, low, high in bands
    ]
    base_result = {
        "schema_version": "touchline.activity_summary.v1",
        "policy_version": "touchline.activity_speed_rules.v1",
        "method": "speed_rule_estimate",
        "status": "unavailable",
        "supported_classes": [name for name, _, _ in bands] + ["unknown"],
        "unimplemented_classes": ["jumping", "tackling", "backpedaling", "side_shuffling"],
        "band_definitions_mps": band_defs,
        "eligible_duration_s": 0.0,
        "unknown_duration_s": 0.0,
        "total_span_s": 0.0,
        "class_durations_s": None,
        "intervals": [],
        "reason": None,
    }

    if calibration is None:
        base_result["reason"] = "calibration and independent check required"
        return base_result
    if not calibration.is_valid:
        base_result["reason"] = f"calibration invalid: {calibration.invalidation_reason or 'unspecified reason'}"
        return base_result
    if not calibration.independent_check_ref:
        base_result["reason"] = "calibration and independent check required"
        return base_result
    if len(points) < 2:
        base_result["reason"] = "at least two timestamped points required"
        return base_result

    class_durations = {name: 0.0 for name, _, _ in bands}
    unknown_duration = 0.0
    eligible_duration = 0.0
    intervals = []

    for left, right in zip(points, points[1:]):
        dt = right.time_s - left.time_s
        reason = None
        if not left.identity_review_ref or not right.identity_review_ref:
            reason = "identity not reviewed"
        elif left.segment_id != right.segment_id:
            reason = "track/association segment changed"
        elif dt > max_gap_s:
            reason = "gap exceeds eligibility limit"
        else:
            try:
                a, b = calibration.project(left), calibration.project(right)
            except ValueError as exc:
                reason = str(exc)

        if reason is not None:
            unknown_duration += dt
            intervals.append({
                "start_s": left.time_s,
                "end_s": right.time_s,
                "duration_s": round(dt, 6),
                "activity": "unknown",
                "speed_mps": None,
                "reason": reason,
            })
            continue

        dist = math.dist(a, b)
        speed = dist / dt
        matched_class = bands[-1][0]
        for name, low, high in bands:
            if low <= speed < high:
                matched_class = name
                break

        class_durations[matched_class] += dt
        eligible_duration += dt
        intervals.append({
            "start_s": left.time_s,
            "end_s": right.time_s,
            "duration_s": round(dt, 6),
            "activity": matched_class,
            "speed_mps": round(speed, 4),
            "reason": None,
        })

    if eligible_duration <= 0.0:
        base_result["unknown_duration_s"] = round(unknown_duration, 6)
        base_result["total_span_s"] = round(unknown_duration, 6)
        base_result["intervals"] = intervals
        base_result["reason"] = "no eligible observed trajectory segments"
        return base_result

    base_result.update(
        status="estimated",
        eligible_duration_s=round(eligible_duration, 6),
        unknown_duration_s=round(unknown_duration, 6),
        total_span_s=round(eligible_duration + unknown_duration, 6),
        class_durations_s={k: round(v, 6) for k, v in class_durations.items()},
        intervals=intervals,
        reason=None,
    )
    return base_result


def evaluate_activity_intervals(
    predicted_intervals: list[dict[str, Any]],
    reference_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate predicted activity intervals against independent reviewed labels (TL-008.2).

    Reports confusion matrix counts and usable coverage duration without counting
    unknown/missing intervals as standing.
    """
    confusion: dict[str, dict[str, float]] = {}
    evaluated_duration_s = 0.0
    agreed_duration_s = 0.0
    unknown_predicted_s = 0.0
    unknown_reference_s = 0.0

    for ref in reference_intervals:
        r_start, r_end = float(ref["start_s"]), float(ref["end_s"])
        r_act = str(ref["activity"])
        if r_end <= r_start:
            continue
        if r_act == "unknown":
            unknown_reference_s += (r_end - r_start)
            continue

        for pred in predicted_intervals:
            p_start, p_end = float(pred["start_s"]), float(pred["end_s"])
            overlap = max(0.0, min(r_end, p_end) - max(r_start, p_start))
            if overlap <= 0.0:
                continue
            p_act = str(pred["activity"])
            confusion.setdefault(r_act, {})
            confusion[r_act][p_act] = round(confusion[r_act].get(p_act, 0.0) + overlap, 6)
            if p_act == "unknown":
                unknown_predicted_s += overlap
            else:
                evaluated_duration_s += overlap
                if p_act == r_act:
                    agreed_duration_s += overlap

    total_ref_known = evaluated_duration_s + unknown_predicted_s
    return {
        "schema_version": "touchline.activity_evaluation.v1",
        "evaluated_duration_s": round(evaluated_duration_s, 6),
        "agreed_duration_s": round(agreed_duration_s, 6),
        "unknown_predicted_duration_s": round(unknown_predicted_s, 6),
        "unknown_reference_duration_s": round(unknown_reference_s, 6),
        "usable_coverage_ratio": round(evaluated_duration_s / total_ref_known, 4) if total_ref_known > 0 else None,
        "agreement_rate": round(agreed_duration_s / evaluated_duration_s, 4) if evaluated_duration_s > 0 else None,
        "confusion_duration_s": confusion,
    }


def intensity_zone_summary(
    points,
    calibration=None,
    max_gap_s: float = 0.75,
    zones: tuple[tuple[str, float, float], ...] = DEFAULT_INTENSITY_ZONES,
) -> dict[str, Any]:
    """Compute observed time and distance across declared speed intensity zones (TL-009, TL-009.1).

    Enforces:
    - Suppressed/unavailable when prerequisite calibration or speed eligibility fails.
    - Half-open [low, high) band boundaries prevent double counting at boundaries.
    - High-speed zone totals are explicitly marked as band exposure, never validated sprint-event counts.
    """
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("positive finite max_gap_s required")
    points = list(points)
    for index, point in enumerate(points):
        if not all(math.isfinite(v) for v in (point.time_s, point.x_px, point.y_px)):
            raise ValueError("finite positions/timestamps required")
        if index and point.time_s <= points[index - 1].time_s:
            raise ValueError("point timestamps must strictly increase")

    zone_defs = [
        {"zone": name, "min_speed_mps": low, "max_speed_mps": None if math.isinf(high) else high}
        for name, low, high in zones
    ]
    result = {
        "schema_version": "touchline.intensity_zones.v1",
        "policy_version": "touchline.intensity_zones.v1",
        "status": "unavailable",
        "units": "metres_and_seconds",
        "zone_definitions": zone_defs,
        "eligible_duration_s": 0.0,
        "excluded_duration_s": 0.0,
        "total_eligible_distance_m": None,
        "high_speed_running_time_s": None,
        "high_speed_running_distance_m": None,
        "sprint_event_counts": "unavailable: zone totals measure time/distance in speed bands, not validated discrete sprint events",
        "zones": None,
        "excluded_segments": [],
        "reason": None,
    }

    if calibration is None:
        result["reason"] = "calibration and independent check required"
        return result
    if not calibration.is_valid:
        result["reason"] = f"calibration invalid: {calibration.invalidation_reason or 'unspecified reason'}"
        return result
    if not calibration.independent_check_ref:
        result["reason"] = "calibration and independent check required"
        return result
    if len(points) < 2:
        result["reason"] = "at least two timestamped points required"
        return result

    zone_Buckets = {name: {"time_s": 0.0, "distance_m": 0.0} for name, _, _ in zones}
    eligible_duration = 0.0
    excluded_duration = 0.0
    total_distance = 0.0

    for left, right in zip(points, points[1:]):
        dt = right.time_s - left.time_s
        reason = None
        if not left.identity_review_ref or not right.identity_review_ref:
            reason = "identity not reviewed"
        elif left.segment_id != right.segment_id:
            reason = "track/association segment changed"
        elif dt > max_gap_s:
            reason = "gap exceeds eligibility limit"
        else:
            try:
                a, b = calibration.project(left), calibration.project(right)
            except ValueError as exc:
                reason = str(exc)

        if reason is not None:
            excluded_duration += dt
            result["excluded_segments"].append({
                "start_s": left.time_s,
                "end_s": right.time_s,
                "duration_s": round(dt, 6),
                "reason": reason,
            })
            continue

        dist = math.dist(a, b)
        speed = dist / dt
        matched_zone = zones[-1][0]
        for name, low, high in zones:
            if low <= speed < high:
                matched_zone = name
                break

        zone_Buckets[matched_zone]["time_s"] += dt
        zone_Buckets[matched_zone]["distance_m"] += dist
        eligible_duration += dt
        total_distance += dist

    if eligible_duration <= 0.0:
        result["excluded_duration_s"] = round(excluded_duration, 6)
        result["reason"] = "no eligible observed trajectory segments"
        return result

    hsr_time = sum(
        zone_Buckets[name]["time_s"]
        for name, low, _ in zones
        if low >= 4.0
    )
    hsr_dist = sum(
        zone_Buckets[name]["distance_m"]
        for name, low, _ in zones
        if low >= 4.0
    )

    result.update(
        status="estimated",
        eligible_duration_s=round(eligible_duration, 6),
        excluded_duration_s=round(excluded_duration, 6),
        total_eligible_distance_m=round(total_distance, 4),
        high_speed_running_time_s=round(hsr_time, 6),
        high_speed_running_distance_m=round(hsr_dist, 4),
        zones={
            k: {"time_s": round(v["time_s"], 6), "distance_m": round(v["distance_m"], 4)}
            for k, v in zone_Buckets.items()
        },
        reason=None,
    )
    return result


def compare_zone_summaries(
    predicted_summary: dict[str, Any],
    reference_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare estimated intensity zone summary against reference measurements (TL-009)."""
    if predicted_summary.get("status") == "unavailable" or reference_summary.get("status") == "unavailable":
        return {
            "status": "unavailable",
            "reason": "predicted or reference zone summary is unavailable",
            "per_zone_error": None,
        }
    pred_zones = predicted_summary.get("zones") or {}
    ref_zones = reference_summary.get("zones") or {}
    per_zone = {}
    for z_name in sorted(set(pred_zones) | set(ref_zones)):
        p_t = pred_zones.get(z_name, {}).get("time_s", 0.0)
        r_t = ref_zones.get(z_name, {}).get("time_s", 0.0)
        p_d = pred_zones.get(z_name, {}).get("distance_m", 0.0)
        r_d = ref_zones.get(z_name, {}).get("distance_m", 0.0)
        per_zone[z_name] = {
            "predicted_time_s": p_t,
            "reference_time_s": r_t,
            "abs_time_error_s": round(abs(p_t - r_t), 6),
            "predicted_distance_m": p_d,
            "reference_distance_m": r_d,
            "abs_distance_error_m": round(abs(p_d - r_d), 4),
        }
    return {
        "status": "compared",
        "eligible_duration_s": predicted_summary.get("eligible_duration_s"),
        "per_zone_error": per_zone,
    }
