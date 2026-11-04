from ..diagnostics import diagnosed
import hashlib
import json
import math
from pathlib import Path
import re


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(data, root=None, training=False):
    """Return errors. Training mode checks actual images; no label-quality claim."""
    errors = []
    if not isinstance(data,dict):
        return ["manifest must be an object"]
    if data.get("class_map") != {"0":"background","1":"person"}:
        errors.append("class map must be background=0/person=1")
    if data.get("split_policy",{}).get("mode") != "capture_group":
        errors.append("only capture-group releases are supported")
    if not data.get("release_id") or not data.get("dataset_version"):
        errors.append("release ID/version required")
    if training and data.get("fixture_only",True):
        errors.append("synthetic/unspecified fixture cannot be used for training")
    samples = data.get("samples",[])
    if not isinstance(samples,list) or not samples:
        return errors+["samples must be nonempty"]
    seen, splits, mappings, source_hashes = set(),set(),{},{}
    for index,sample in enumerate(samples):
        label = f"sample[{index}]"
        if not isinstance(sample,dict):
            errors.append(label+": must be an object")
            continue
        sid = sample.get("sample_id")
        if not isinstance(sid,str) or not sid or sid in seen:
            errors.append(label+": unique sample_id required")
        else:
            seen.add(sid)
        split = sample.get("split")
        if split not in ("train","val","test"):
            errors.append(label+": invalid split")
        else:
            splits.add(split)
        for key in ("source_id","capture_group_id","reviewer_id"):
            if not isinstance(sample.get(key),str) or not sample[key]:
                errors.append(label+": missing "+key)
        for key in ("source_sha256","image_sha256"):
            if not isinstance(sample.get(key),str) or not re.fullmatch(r"[a-f0-9]{64}",sample[key]):
                errors.append(label+": invalid "+key)
        for key in ("capture_group_id","source_sha256","image_sha256","duplicate_cluster_id"):
            value=sample.get(key)
            if isinstance(value,str) and value:
                marker=(key,value)
                if marker in mappings and mappings[marker] != split:
                    errors.append(label+": cross-split leakage in "+key)
                mappings[marker]=split
        source=sample.get("source_id")
        if isinstance(source,str):
            if source in source_hashes and source_hashes[source] != sample.get("source_sha256"):
                errors.append(label+": source ID changed hash")
            source_hashes[source]=sample.get("source_sha256")
        width,height=sample.get("width"),sample.get("height")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            errors.append(label+": invalid dimensions")
            width=height=0
        tb=sample.get("time_base")
        if sample.get("source_kind")=="still_image":
            if sample.get("pts") is not None or tb is not None or sample.get("frame_index") is not None:
                errors.append(label+": still images must not invent video timestamps/indices")
        elif type(sample.get("pts")) is not int or not isinstance(tb,list) or len(tb)!=2 or any(type(v) is not int or v<=0 for v in tb):
            errors.append(label+": original PTS/time base required")
        if sample.get("review_status") != "human_reviewed":
            errors.append(label+": sample not human reviewed")
        objects=sample.get("objects")
        if not isinstance(objects,list):
            errors.append(label+": objects must be a list")
            objects=[]
        if type(sample.get("is_negative")) is not bool or sample.get("is_negative") != (not objects):
            errors.append(label+": negative flag disagrees with objects")
        for obj in objects:
            if not isinstance(obj,dict):
                errors.append(label+": invalid object")
                continue
            box=obj.get("bbox_xyxy")
            if obj.get("class_id") != 1 or type(obj.get("class_id")) is not int or obj.get("annotation_status") != "human_reviewed":
                errors.append(label+": unreviewed/invalid person label")
            if obj.get("origin") not in ("manual","sam3_proposal_reviewed","existing_annotations_reviewed"):
                errors.append(label+": label origin missing")
            if not isinstance(box,list) or len(box)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box):
                errors.append(label+": finite XYXY box required")
            elif not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                errors.append(label+": box outside geometry or reversed")
        if training:
            if sample.get("training_eligible") is not True:
                errors.append(label+": training use not eligible")
            try:
                if root is None or not isinstance(sample.get("image_path"),str):
                    raise ValueError("dataset root/image path required")
                base=Path(root).resolve()
                image=(base/sample["image_path"]).resolve()
                if not image.is_relative_to(base) or not image.is_file():
                    raise ValueError("image missing or outside release")
                if sha256_file(image)!=sample.get("image_sha256"):
                    raise ValueError("image hash mismatch")
                from PIL import Image
                with Image.open(image) as decoded:
                    if decoded.size != (width,height):
                        raise ValueError("decoded image dimensions disagree")
                    decoded.verify()
            except (OSError,ValueError,ImportError) as exc:
                errors.append(label+": "+str(exc))
    if splits != {"train","val","test"}:
        errors.append("nonempty train/val/test groups required")
    return errors


@diagnosed("data.validate")
def read_manifest(path, training=False):
    path=Path(path)
    data=json.loads(path.read_text(encoding="utf-8"))
    errors=validate_manifest(data,path.parent,training)
    if errors:
        raise ValueError("\n".join(errors))
    return data
