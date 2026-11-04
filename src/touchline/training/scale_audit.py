"""Effective person bounding-box size audit under model transform resize (TL-030.2).

CPU-only audit that calculates the effective person bounding-box dimensions after
applying declared model preprocessing (e.g. Faster R-CNN min_size/max_size resize).
Counts person boxes falling below diagnostic thresholds (<8px and <16px) per split
and capture group without running GPU inference or altering model weights.
"""
from ..diagnostics import diagnosed

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MIN_SIZE = 320
DEFAULT_MAX_SIZE = 640


def compute_resize_scale(
    width: int,
    height: int,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> float:
    """Compute the deterministic aspect-preserving scale factor used by Torchvision GeneralizedRCNNTransform.

    Torchvision detection transform logic:
        im_shape = (height, width)
        min_dim = min(im_shape)
        max_dim = max(im_shape)
        scale = min(min_size / min_dim, max_size / max_dim)
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive integers, got {width}x{height}")
    if min_size <= 0 or max_size <= 0:
        raise ValueError(f"resize bounds must be positive, got min_size={min_size}, max_size={max_size}")
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot exceed max_size ({max_size})")

    min_dim = float(min(width, height))
    max_dim = float(max(width, height))
    return min(min_size / min_dim, max_size / max_dim)


def compute_effective_box(
    box_xyxy: list[float] | tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    """Compute effective box dimensions (w, h, min_dimension, area) after scaling."""
    if len(box_xyxy) != 4:
        raise ValueError(f"box must have 4 coordinates, got {box_xyxy}")
    x1, y1, x2, y2 = box_xyxy
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        raise ValueError(f"box coordinates must be finite numbers: {box_xyxy}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box coordinates must be non-inverted (x2 > x1, y2 > y1): {box_xyxy}")
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(f"scale must be a positive finite number, got {scale}")

    effective_w = (x2 - x1) * scale
    effective_h = (y2 - y1) * scale
    min_dim = min(effective_w, effective_h)
    area = effective_w * effective_h
    return (effective_w, effective_h, min_dim, area)


def audit_scale(
    manifest_data: dict[str, Any],
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> dict[str, Any]:
    """Run CPU-only effective person-size audit on a dataset release manifest.

    Groups diagnostic counts (<8px, <16px) and effective size metrics by:
    - split (e.g. train, val, test)
    - capture_group_id
    """
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest data must be a dictionary")
    samples = manifest_data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest data must contain a non-empty 'samples' list")

    total_person_boxes = 0
    total_images = len(samples)

    # Buckets structure:
    # count, below_8px, below_16px, sum_effective_w, sum_effective_h, min_w, max_w, min_h, max_h
    def create_bucket():
        return {
            "box_count": 0,
            "below_8px": 0,
            "below_16px": 0,
            "sum_w": 0.0,
            "sum_h": 0.0,
            "min_effective_w": None,
            "max_effective_w": None,
            "min_effective_h": None,
            "max_effective_h": None,
            "min_effective_dim": None,
        }

    overall = create_bucket()
    by_split = defaultdict(create_bucket)
    by_capture = defaultdict(create_bucket)

    for sample in samples:
        img_w = sample.get("width")
        img_h = sample.get("height")
        if not (isinstance(img_w, int) and isinstance(img_h, int) and img_w > 0 and img_h > 0):
            raise ValueError(f"sample {sample.get('sample_id')} has invalid dimensions: {img_w}x{img_h}")

        split = sample.get("split", "unknown")
        capture_group = sample.get("capture_group_id", "unknown")
        scale = compute_resize_scale(img_w, img_h, min_size=min_size, max_size=max_size)

        objects = sample.get("objects", [])
        for obj in objects:
            # Person class in Touchline is class_id=1
            if obj.get("class_id") != 1:
                continue

            box = obj.get("bbox_xyxy")
            if not box:
                continue

            eff_w, eff_h, min_dim, _ = compute_effective_box(box, scale)
            total_person_boxes += 1

            for bucket in (overall, by_split[split], by_capture[capture_group]):
                bucket["box_count"] += 1
                if min_dim < 8.0:
                    bucket["below_8px"] += 1
                if min_dim < 16.0:
                    bucket["below_16px"] += 1

                bucket["sum_w"] += eff_w
                bucket["sum_h"] += eff_h

                if bucket["min_effective_w"] is None or eff_w < bucket["min_effective_w"]:
                    bucket["min_effective_w"] = eff_w
                if bucket["max_effective_w"] is None or eff_w > bucket["max_effective_w"]:
                    bucket["max_effective_w"] = eff_w

                if bucket["min_effective_h"] is None or eff_h < bucket["min_effective_h"]:
                    bucket["min_effective_h"] = eff_h
                if bucket["max_effective_h"] is None or eff_h > bucket["max_effective_h"]:
                    bucket["max_effective_h"] = eff_h

                if bucket["min_effective_dim"] is None or min_dim < bucket["min_effective_dim"]:
                    bucket["min_effective_dim"] = min_dim

    def format_summary(bucket: dict[str, Any]) -> dict[str, Any]:
        count = bucket["box_count"]
        return {
            "person_box_count": count,
            "below_8px_count": bucket["below_8px"],
            "below_8px_ratio": round(bucket["below_8px"] / count, 4) if count > 0 else 0.0,
            "below_16px_count": bucket["below_16px"],
            "below_16px_ratio": round(bucket["below_16px"] / count, 4) if count > 0 else 0.0,
            "mean_effective_width": round(bucket["sum_w"] / count, 2) if count > 0 else None,
            "mean_effective_height": round(bucket["sum_h"] / count, 2) if count > 0 else None,
            "min_effective_width": round(bucket["min_effective_w"], 2) if bucket["min_effective_w"] is not None else None,
            "max_effective_width": round(bucket["max_effective_w"], 2) if bucket["max_effective_w"] is not None else None,
            "min_effective_height": round(bucket["min_effective_h"], 2) if bucket["min_effective_h"] is not None else None,
            "max_effective_height": round(bucket["max_effective_h"], 2) if bucket["max_effective_h"] is not None else None,
            "min_effective_dimension": round(bucket["min_effective_dim"], 2) if bucket["min_effective_dim"] is not None else None,
        }

    return {
        "schema_version": "touchline.scale_audit.v1",
        "release_id": manifest_data.get("release_id"),
        "dataset_version": manifest_data.get("dataset_version"),
        "image_count": total_images,
        "transform": {
            "model_architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "min_size": min_size,
            "max_size": max_size,
        },
        "overall": format_summary(overall),
        "by_split": {split: format_summary(b) for split, b in sorted(by_split.items())},
        "by_capture_group": {grp: format_summary(b) for grp, b in sorted(by_capture.items())},
    }


@diagnosed("audit")
def audit_scale_file(
    manifest_path: str | Path,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> dict[str, Any]:
    """Load manifest JSON file and perform scale audit."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return audit_scale(data, min_size=min_size, max_size=max_size)
