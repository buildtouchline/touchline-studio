"""Fail-closed metadata checks for the Torchvision detector handoff.

These checks do not run inference or establish field/model quality. They make
incompatible exports visible before a consumer loads a checkpoint.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
CLASS_MAP = {"0": "background", "1": "person"}
RECIPE = "starter-heads-fp32-batch1-v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EVALUATION_METRICS = (
    "map_50_95", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
    "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
)
REQUIRED_OUTCOMES = ("map_50_95", "ap50", "ar_100")


def _hash(value, field):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _version_pair(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} version is missing")
    match = re.match(r"^(\d+)\.(\d+)(?:\.|\+|$)", value)
    if not match:
        raise ValueError(f"{field} version is invalid")
    return tuple(int(part) for part in match.groups())


def validate_candidate_metadata(metadata, *, runtime_torch, runtime_torchvision):
    """Reject incompatible class, architecture, lineage and runtime metadata."""
    if not isinstance(metadata, dict):
        raise ValueError("candidate metadata must be an object")
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported candidate architecture")
    if type(metadata.get("num_classes")) is not int or metadata["num_classes"] != 2:
        raise ValueError("candidate requires a two-class head")
    if metadata.get("class_map") != CLASS_MAP:
        raise ValueError("candidate requires background=0 and person=1")
    if metadata.get("backbone_norm") != "frozen_batchnorm2d":
        raise ValueError("candidate backbone normalization is incompatible")
    if metadata.get("recipe") != RECIPE:
        if metadata.get('recipe')!='heads-fp32-batch1-reviewed-resize-v2' or metadata.get('training_transform')!={'min_size':[720],'max_size':1280}:
            raise ValueError("candidate preprocessing recipe is unknown")
        _hash(metadata.get('approved_recipe_sha256'),'approved_recipe_sha256')
    if metadata.get("promotion_state") != "unvalidated_candidate":
        raise ValueError("unexpected candidate promotion state")
    for field in ("release_id", "dataset_version"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"candidate {field} is missing")
    _hash(metadata.get("dataset_manifest_sha256"), "dataset_manifest_sha256")
    _hash(metadata.get("initial_weights_sha256"), "initial_weights_sha256")
    for field, runtime in (("torch", runtime_torch), ("torchvision", runtime_torchvision)):
        if _version_pair(metadata.get(field), field) != _version_pair(runtime, f"runtime {field}"):
            raise ValueError(f"candidate {field} major/minor differs from runtime")


def assess_handoff(recipe, result, baseline, candidate, *, checkpoint_path=None):
    """Check metadata and evaluation comparability; never promote a candidate."""
    issues = []
    if not all(isinstance(item, dict) for item in (recipe, result, baseline, candidate)):
        return {"ready_for_runtime_test": False, "issues": ["all records must be JSON objects"]}
    try:
        validate_candidate_metadata(recipe, runtime_torch=recipe.get("torch"), runtime_torchvision=recipe.get("torchvision"))
        expected = _hash(recipe.get("dataset_manifest_sha256"), "dataset_manifest_sha256")
        candidate_hash = _hash(result.get("checkpoint_sha256"), "checkpoint_sha256")
    except ValueError as exc:
        return {"ready_for_runtime_test": False, "issues": [str(exc)]}
    if result.get("promotion_state") != "unvalidated_candidate":
        issues.append("result promotion state is not the unvalidated candidate")
    if type(result.get("steps")) is not int or result["steps"] < 2:
        issues.append("two completed optimizer steps are not evidenced")
    if result.get("status") not in ("completed", "capped"):
        issues.append("training completion status is missing")
    for name, evaluation in (("baseline", baseline), ("candidate", candidate)):
        if evaluation.get("manifest_sha256") != expected:
            issues.append(f"{name} uses a different dataset manifest")
        if evaluation.get("split") != "val":
            issues.append(f"{name} is not a validation evaluation")
        ids = evaluation.get("sample_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            issues.append(f"{name} sample IDs are missing or duplicated")
        if type(evaluation.get("images")) is not int or evaluation["images"] != len(ids if isinstance(ids, list) else []):
            issues.append(f"{name} image count differs from sample IDs")
        if evaluation.get("subset") is not False:
            issues.append(f"{name} is not full validation")
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, dict) or any(metric not in metrics for metric in EVALUATION_METRICS):
            issues.append(f"{name} evaluation metrics are missing")
        else:
            for metric in EVALUATION_METRICS:
                value = metrics[metric]
                if value is None and metric not in REQUIRED_OUTCOMES:
                    continue
                if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
                    issues.append(f"{name} {metric} is invalid")
    if baseline.get("sample_ids") != candidate.get("sample_ids"):
        issues.append("baseline and candidate use different validation IDs")
    baseline_model, candidate_model = baseline.get("model"), candidate.get("model")
    if not isinstance(baseline_model, dict) or not isinstance(candidate_model, dict):
        issues.append("evaluation model provenance is missing")
    else:
        for name, model in (("baseline", baseline_model), ("candidate", candidate_model)):
            try:
                _hash(model.get("checkpoint"), f"{name} checkpoint")
            except ValueError as exc:
                issues.append(str(exc))
            if model.get("adapter") != "torchvision-detector-v1":
                issues.append(f"{name} adapter is incompatible")
            threshold = model.get("threshold")
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                issues.append(f"{name} score threshold is invalid")
        if candidate_model.get("checkpoint") != candidate_hash:
            issues.append("candidate evaluation checkpoint differs from training result")
        if baseline_model.get("architecture") != recipe.get("architecture") or candidate_model.get("architecture") != recipe.get("architecture"):
            issues.append("evaluation architecture differs from recipe")
        if baseline_model.get("threshold") != candidate_model.get("threshold"):
            issues.append("baseline and candidate use different score thresholds")
    if checkpoint_path is None:
        issues.append("checkpoint bytes have not been hashed by this consumer")
    else:
        digest = hashlib.sha256()
        with Path(checkpoint_path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != candidate_hash:
            issues.append("checkpoint bytes differ from training result")
    return {"ready_for_runtime_test": not issues, "issues": issues}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check candidate handoff metadata before runtime inference")
    for name in ("recipe", "result", "baseline", "candidate"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in (args.recipe, args.result, args.baseline, args.candidate)]
    report = assess_handoff(*records, checkpoint_path=args.checkpoint)
    print(json.dumps(report, indent=2))
    return 0 if report["ready_for_runtime_test"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
