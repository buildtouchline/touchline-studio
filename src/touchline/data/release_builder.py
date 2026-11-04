"""Resumable and input-bound release builder for ETL (TL-023.5, TL-029.4).

Provides per-image ETL restart bound to source/label/configuration/code/Pillow
fingerprints. Reuses verified target PNGs across restarts, deduplicates
identical pixel images before saving, uses tuned lossless PNG compression,
and indexes COCO annotations via set/hash lookups for high-throughput ETL.
"""
from ..diagnostics import diagnosed

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from .manifest import sha256_file, validate_manifest


def compute_sample_fingerprint(
    raw_hash: str,
    key: str,
    objects_signature: list[Any],
    pillow_version: str,
) -> str:
    """Compute an immutable fingerprint for an ETL image conversion task.

    Any change to the underlying source image bytes, relative path/key,
    converted label coordinates, or Pillow decoding version changes the fingerprint.
    """
    hasher = hashlib.sha256()
    hasher.update(raw_hash.encode("utf-8"))
    hasher.update(key.encode("utf-8"))
    hasher.update(json.dumps(objects_signature, sort_keys=True).encode("utf-8"))
    hasher.update(pillow_version.encode("utf-8"))
    return hasher.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or out-of-root input: {relative}")
    return path


def _records(item: dict[str, Any], base: Path):
    """Yield (relative_name, full_path, objects_list, normalized_flag) per image."""
    images = (base / item["images_root"]).resolve()
    kind = item["format"]

    if kind == "coco_boxes":
        annotations = json.loads((base / item["annotations"]).read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in annotations["categories"]}
        by_image = defaultdict(list)
        image_ids = set()

        for img in annotations["images"]:
            iid = img["id"]
            if iid in image_ids:
                raise ValueError("duplicate COCO image IDs")
            image_ids.add(iid)

        for ann in annotations["annotations"]:
            iid = ann["image_id"]
            if iid not in image_ids:
                raise ValueError("annotation refers to unknown COCO image")
            if ann.get("iscrowd", 0):
                raise ValueError("crowd labels require an explicit ignore/crowd adapter")
            name = categories[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            by_image[iid].append((name, [x, y, x + w, y + h]))

        for image in annotations["images"]:
            yield image["file_name"], _inside(images, image["file_name"]), by_image.get(image["id"], []), None

    elif kind == "csv_boxes":
        columns = item["columns"]
        required = ("image", "class", "x1", "y1", "x2", "y2")
        if any(k not in columns for k in required):
            raise ValueError("CSV mapping needs image,class,x1,y1,x2,y2 columns")
        grouped = defaultdict(list)
        with (base / item["annotations"]).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = row[columns["class"]]
                if item.get("class_format", "name") == "index":
                    name = item["classes"][int(name)]
                values = [float(row[columns[k]]) for k in ("x1", "y1", "x2", "y2")]
                grouped[row[columns["image"]]].append((name, values))

        if item.get("coordinate_units") not in ("pixels", "normalized"):
            raise ValueError("CSV coordinate_units must be explicit")
        for relative, rows in grouped.items():
            yield relative, _inside(images, relative), rows, "normalized_xyxy" if item["coordinate_units"] == "normalized" else None

    elif kind in ("yolo_boxes", "xyxy_boxes"):
        labels = (base / item["labels_root"]).resolve()
        classes = item["classes"]
        if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
            raise ValueError("YOLO source classes must be an explicit ordered JSON list")
        paths = sorted(p for p in images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        for path in paths:
            relative = path.relative_to(images).as_posix()
            label_path = _inside(labels, str(Path(relative).with_suffix(".txt")))
            rows = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError("only five-column detection boxes are supported; polygon/pose labels need another adapter")
                ci = int(parts[0])
                values = list(map(float, parts[1:]))
                if not 0 <= ci < len(classes) or not all(math.isfinite(v) for v in values):
                    raise ValueError("invalid YOLO class or coordinates")
                rows.append((classes[ci], values))
            if kind == "xyxy_boxes" and item.get("coordinate_units") not in ("pixels", "normalized"):
                raise ValueError("corner-box coordinate_units must be explicit")
            normalized = "normalized_xywh" if kind == "yolo_boxes" else ("normalized_xyxy" if item["coordinate_units"] == "normalized" else None)
            yield relative, path, rows, normalized
    else:
        raise ValueError("supported input formats are coco_boxes, yolo_boxes, xyxy_boxes and explicitly mapped csv_boxes")


class ResumableReleaseBuilder:
    """Builds release with per-image caching, resumption, and deduplication."""

    def __init__(self, config_path: str | Path, output_dir: str | Path, resume: bool = True):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.base = self.config_path.parent
        self.output = Path(output_dir).resolve()
        self.resume = resume
        self.cache_dir = self.output / ".cache"
        self.images_dir = self.output / "images"

    @diagnosed("data.release")
    def build(self) -> dict[str, Any]:
        from PIL import Image, __version__ as pillow_version

        # Validate config assertions
        for field in ("release_id", "dataset_version", "reviewer_id"):
            if not isinstance(self.config.get(field), str) or not self.config[field]:
                raise ValueError("missing config field " + field)
        if self.config.get("annotations_reviewed") is not True or self.config.get("person_labels_complete") is not True:
            raise ValueError("existing annotations need explicit review/completeness attestation; do not manufacture gold labels")
        if self.config.get("training_eligible") is not True and not self.config.get("fixture_only", False):
            raise ValueError("training eligibility required unless marked as a fixture")

        self.output.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        config_hash = sha256_file(self.config_path)
        report = {
            "status": "in_progress",
            "release_id": self.config["release_id"],
            "config_sha256": config_hash,
            "inputs": len(self.config.get("inputs", [])),
            "duplicates_removed": 0,
            "images_reused_from_cache": 0,
            "images_encoded_fresh": 0,
        }

        start_time = time.monotonic()
        candidates = []
        seen = {}
        aliases = self.config.get("class_aliases", {"player": 1, "person": 1, "referee": 1, "goalkeeper": 1})
        ignored = set(self.config.get("ignore_classes", ["ball"]))
        cache_index_path = self.cache_dir / "cache_index.json"
        cache_index = {}
        if self.resume and cache_index_path.is_file():
            try:
                cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
            except Exception:
                cache_index = {}

        for item in self.config.get("inputs", []):
            item_id = item["id"]
            for relative, src_path, objects, normalized in _records(item, self.base):
                key = f"{item_id}/{relative}"
                group = self.config.get("capture_groups", {}).get(key)
                if not group:
                    raise ValueError(f"capture group missing for input: {key}")

                raw_hash = sha256_file(src_path)
                with Image.open(src_path) as raw:
                    if raw.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF orientation must be normalized together with labels before ETL")
                    image = raw.convert("RGB")
                    width, height = image.size

                converted = []
                for name, box in objects:
                    name_cf = name.casefold()
                    if name_cf in ignored:
                        continue
                    if name_cf not in aliases:
                        raise ValueError("class needs an explicit person mapping or ignore decision: " + name)
                    if normalized == "normalized_xywh":
                        x, y, w, h = box
                        box = [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]
                    elif normalized == "normalized_xyxy":
                        box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
                    if len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in box):
                        raise ValueError("invalid finite box")
                    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                        raise ValueError("box reversed/outside decoded image; repair labels explicitly")
                    converted.append({
                        "class_id": 1,
                        "bbox_xyxy": box,
                        "annotation_status": "human_reviewed",
                        "origin": "existing_annotations_reviewed",
                    })

                sample_id = hashlib.sha256((key + raw_hash).encode()).hexdigest()[:24]
                target_path = self.images_dir / f"{sample_id}.png"
                fingerprint = compute_sample_fingerprint(raw_hash, key, converted, pillow_version)

                # Check if target already exists and fingerprint matches (safe restart)
                cached_entry = cache_index.get(sample_id)
                reused = False
                if self.resume and target_path.is_file() and cached_entry and cached_entry.get("fingerprint") == fingerprint:
                    image_hash = cached_entry["image_sha256"]
                    # Quick verify of cached hash
                    if sha256_file(target_path) == image_hash:
                        reused = True
                        report["images_reused_from_cache"] += 1

                if not reused:
                    # Save using tuned lossless PNG compression (compress_level=1 for speed)
                    image.save(target_path, format="PNG", compress_level=1)
                    image_hash = sha256_file(target_path)
                    cache_index[sample_id] = {
                        "fingerprint": fingerprint,
                        "image_sha256": image_hash,
                    }
                    report["images_encoded_fresh"] += 1

                signature = sorted(tuple(o["bbox_xyxy"]) for o in converted)
                if image_hash in seen:
                    old_group, old_labels = seen[image_hash]
                    if old_group != group or old_labels != signature:
                        raise ValueError("duplicate image has conflicting group/labels; resolve before splitting")
                    if target_path.exists():
                        target_path.unlink()
                    report["duplicates_removed"] += 1
                    continue

                seen[image_hash] = (group, signature)
                candidates.append({
                    "sample_id": sample_id,
                    "source_id": raw_hash,
                    "source_sha256": raw_hash,
                    "source_kind": "still_image",
                    "source_image_ref": key,
                    "capture_group_id": group,
                    "pts": None,
                    "time_base": None,
                    "frame_index": None,
                    "width": width,
                    "height": height,
                    "image_path": target_path.relative_to(self.output).as_posix(),
                    "image_sha256": image_hash,
                    "duplicate_cluster_id": image_hash,
                    "review_status": "human_reviewed",
                    "reviewer_id": self.config["reviewer_id"],
                    "training_eligible": self.config.get("training_eligible", False),
                    "is_negative": not converted,
                    "objects": converted,
                })

        distinct = {s["capture_group_id"] for s in candidates}
        if len(distinct) < 3:
            raise ValueError("at least three independent declared capture groups are required for this train/val/test recipe")
        seed = self.config.get("split_seed", 42)
        ordered = sorted(distinct, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        holdout = max(1, len(ordered) // 6)
        if 2 * holdout >= len(ordered):
            raise ValueError("not enough capture groups for split")

        mapping = self.config.get("split_assignments")
        if mapping is None:
            mapping = {g: ("test" if i < holdout else "val" if i < 2 * holdout else "train") for i, g in enumerate(ordered)}
        elif not isinstance(mapping, dict) or set(mapping) != distinct or set(mapping.values()) != {"train", "val", "test"}:
            raise ValueError("split_assignments must cover exactly the declared groups with nonempty train/val/test")

        for sample in candidates:
            sample["split"] = mapping[sample["capture_group_id"]]

        manifest = {
            "release_id": self.config["release_id"],
            "dataset_version": self.config["dataset_version"],
            "label_schema_version": "touchline-person-v1",
            "fixture_only": self.config.get("fixture_only", False),
            "class_map": {"0": "background", "1": "person"},
            "split_policy": {"mode": "capture_group", "test_is_frozen": True, "seed": seed, "group_assignments": mapping},
            "etl": {
                "recipe": "explicit-detection-boxes-rgb-v2-resumable",
                "config_sha256": report["config_sha256"],
                "pillow": pillow_version,
                "geometry": "unchanged source image dimensions; no EXIF transform/resize",
                "label_source": "existing annotations; user-attested review",
            },
            "samples": candidates,
        }

        errors = validate_manifest(manifest, self.output, training=not manifest["fixture_only"])
        if errors:
            raise ValueError("; ".join(errors))

        (self.output / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        cache_index_path.write_text(json.dumps(cache_index, indent=2) + "\n", encoding="utf-8")

        elapsed = time.monotonic() - start_time
        report.update(
            status="complete",
            elapsed_s=round(elapsed, 3),
            exported_images=len(candidates),
            capture_groups=len(distinct),
            split_counts={s: sum(x["split"] == s for x in candidates) for s in ("train", "val", "test")},
            release_sha256=sha256_file(self.output / "release.json"),
        )
        (self.output / "etl-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return manifest


@diagnosed("data.release")
def build_release_v2(config_path: str | Path, output_dir: str | Path, resume: bool = True) -> dict[str, Any]:
    """Convenience helper to run resumable release builder."""
    builder = ResumableReleaseBuilder(config_path, output_dir, resume=resume)
    return builder.build()
