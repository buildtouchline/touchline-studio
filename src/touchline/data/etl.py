"""Explicit COCO, YOLO, corner-box and mapped CSV ETL; no arbitrary JSON/masks."""
from ..diagnostics import diagnosed, failure_details, current_trace_id
import hashlib
import csv
import json
import math
from pathlib import Path
from .manifest import sha256_file, validate_manifest


def _inside(root, relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or out-of-root input: {relative}")
    return path


def _records(item, base):
    images=(base/item["images_root"]).resolve()
    kind=item["format"]
    if kind=="coco_boxes":
        annotations=json.loads((base/item["annotations"]).read_text(encoding="utf-8"))
        categories={c["id"]:c["name"] for c in annotations["categories"]}
        by_image={}
        image_ids=[i["id"] for i in annotations["images"]]
        if len(set(image_ids))!=len(image_ids):raise ValueError("duplicate COCO image IDs")
        for ann in annotations["annotations"]:
            if ann["image_id"] not in image_ids:raise ValueError("annotation refers to unknown COCO image")
            if ann.get("iscrowd",0):raise ValueError("crowd labels require an explicit ignore/crowd adapter")
            name=categories[ann["category_id"]]
            x,y,w,h=ann["bbox"]
            by_image.setdefault(ann["image_id"],[]).append((name,[x,y,x+w,y+h]))
        for image in annotations["images"]:
            yield image["file_name"],_inside(images,image["file_name"]),by_image.get(image["id"],[]),None
    elif kind=="csv_boxes":
        columns=item['columns'];required=('image','class','x1','y1','x2','y2')
        if any(k not in columns for k in required):raise ValueError('CSV mapping needs image,class,x1,y1,x2,y2 columns')
        grouped={}
        with (base/item['annotations']).open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                name=row[columns['class']]
                if item.get('class_format','name')=='index':name=item['classes'][int(name)]
                values=[float(row[columns[k]]) for k in ('x1','y1','x2','y2')]
                grouped.setdefault(row[columns['image']],[]).append((name,values))
        if item.get('coordinate_units') not in ('pixels','normalized'):raise ValueError('CSV coordinate_units must be explicit')
        for relative,rows in grouped.items():
            yield relative,_inside(images,relative),rows,'normalized_xyxy' if item['coordinate_units']=='normalized' else None
    elif kind in ("yolo_boxes","xyxy_boxes"):
        labels=(base/item["labels_root"]).resolve();classes=item["classes"]
        if not isinstance(classes,list) or not classes or not all(isinstance(v,str) for v in classes):
            raise ValueError("YOLO source classes must be an explicit ordered JSON list")
        paths=sorted(p for p in images.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp'))
        for path in paths:
            relative=path.relative_to(images).as_posix()
            label_path=_inside(labels,str(Path(relative).with_suffix('.txt')))
            rows=[]
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():continue
                parts=line.split()
                if len(parts)!=5:raise ValueError("only five-column detection boxes are supported; polygon/pose labels need another adapter")
                ci=int(parts[0]);values=list(map(float,parts[1:]))
                if not 0<=ci<len(classes) or not all(math.isfinite(v) for v in values):raise ValueError("invalid YOLO class or coordinates")
                rows.append((classes[ci],values))
            if kind=='xyxy_boxes' and item.get('coordinate_units') not in ('pixels','normalized'):
                raise ValueError('corner-box coordinate_units must be explicit')
            normalized='normalized_xywh' if kind=='yolo_boxes' else ('normalized_xyxy' if item['coordinate_units']=='normalized' else None)
            yield relative,path,rows,normalized
    else:
        raise ValueError("supported input formats are coco_boxes, yolo_boxes, xyxy_boxes and explicitly mapped csv_boxes")


@diagnosed("data.etl")
def build_release(config_path, output):
    from PIL import Image, __version__ as pillow_version
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text(encoding="utf-8"))
    base=config_path.parent
    for field in ("release_id","dataset_version","reviewer_id"):
        if not isinstance(config.get(field),str) or not config[field]:raise ValueError("missing config field "+field)
    if config.get("annotations_reviewed") is not True or config.get("person_labels_complete") is not True:
        raise ValueError("existing annotations need explicit review/completeness attestation; do not manufacture gold labels")
    if config.get("training_eligible") is not True and not config.get("fixture_only",False):
        raise ValueError("training data eligibility must be explicitly confirmed")
    aliases={str(k).casefold():v for k,v in config["class_aliases"].items()}
    ignored={str(k).casefold() for k in config.get("ignored_classes",[])}
    if not aliases or any(type(v) is not int or v!=1 for v in aliases.values()) or set(aliases)&ignored:
        raise ValueError("explicit nonoverlapping person-class aliases and ignored classes required")
    groups=config["capture_groups"]
    if not isinstance(groups,dict):raise ValueError("capture_groups must map input-id/relative-image to a capture/session group")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/"images").mkdir()
    report={"status":"running","config_sha256":sha256_file(config_path),"input_images":0,"exported_images":0,"duplicates_removed":0,
            "format_scope":"person detection boxes only; original geometry retained; RGB PNG normalization; no masks, geometric augmentation or synthetic annotation review",
            "review_note":"review/eligibility fields are user attestations, not automatically verified visual label correctness"}
    candidates=[];seen={};input_ids=set()
    try:
        for item in config["inputs"]:
            input_id=item["id"]
            if input_id in input_ids:raise ValueError("duplicate input ID")
            input_ids.add(input_id)
            for relative,path,objects,normalized in _records(item,base):
                report["input_images"]+=1
                key=f"{input_id}/{relative}"
                group=groups.get(key)
                if not isinstance(group,str) or not group:raise ValueError("capture/session group missing for "+key)
                raw_hash=sha256_file(path)
                with Image.open(path) as raw:
                    if raw.getexif().get(274,1)!=1:raise ValueError("EXIF orientation must be normalized together with labels before ETL")
                    image=raw.convert("RGB");width,height=image.size
                    converted=[]
                    for name,box in objects:
                        name=name.casefold()
                        if name in ignored:continue
                        if name not in aliases:raise ValueError("class needs an explicit person mapping or ignore decision: "+name)
                        if normalized=='normalized_xywh':
                            x,y,w,h=box
                            box=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]
                        elif normalized=='normalized_xyxy':
                            box=[box[0]*width,box[1]*height,box[2]*width,box[3]*height]
                        if len(box)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box):raise ValueError("invalid finite box")
                        if not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError("box reversed/outside decoded image; repair labels explicitly")
                        converted.append({"class_id":1,"bbox_xyxy":box,"annotation_status":"human_reviewed","origin":"existing_annotations_reviewed"})
                    sample_id=hashlib.sha256((key+raw_hash).encode()).hexdigest()[:24]
                    target=output/"images"/(sample_id+".png");image.save(target)
                image_hash=sha256_file(target)
                signature=sorted(tuple(o["bbox_xyxy"]) for o in converted)
                if image_hash in seen:
                    old_group,old_labels=seen[image_hash]
                    if old_group!=group or old_labels!=signature:raise ValueError("duplicate image has conflicting group/labels; resolve before splitting")
                    target.unlink();report["duplicates_removed"]+=1;continue
                seen[image_hash]=(group,signature)
                candidates.append({"sample_id":sample_id,"source_id":raw_hash,"source_sha256":raw_hash,"source_kind":"still_image",
                  "source_image_ref":key,"capture_group_id":group,"pts":None,"time_base":None,"frame_index":None,
                  "width":width,"height":height,"image_path":target.relative_to(output).as_posix(),"image_sha256":image_hash,
                  "duplicate_cluster_id":image_hash,"review_status":"human_reviewed","reviewer_id":config["reviewer_id"],
                  "training_eligible":config.get("training_eligible",False),"is_negative":not converted,"objects":converted})
        distinct={s["capture_group_id"] for s in candidates}
        if len(distinct)<3:raise ValueError("at least three independent declared capture groups are required for this train/val/test recipe")
        seed=config.get("split_seed",42)
        ordered=sorted(distinct,key=lambda g:hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        holdout=max(1,len(ordered)//6)
        if 2*holdout>=len(ordered):raise ValueError("not enough capture groups for split")
        mapping=config.get('split_assignments')
        if mapping is None:
            mapping={g:("test" if i<holdout else "val" if i<2*holdout else "train") for i,g in enumerate(ordered)}
        elif not isinstance(mapping,dict) or set(mapping)!=distinct or set(mapping.values())!={'train','val','test'}:
            raise ValueError('split_assignments must cover exactly the declared groups with nonempty train/val/test')
        for sample in candidates:sample["split"]=mapping[sample["capture_group_id"]]
        manifest={"release_id":config["release_id"],"dataset_version":config["dataset_version"],"label_schema_version":"touchline-person-v1",
          "fixture_only":config.get("fixture_only",False),"class_map":{"0":"background","1":"person"},
          "split_policy":{"mode":"capture_group","test_is_frozen":True,"seed":seed,"group_assignments":mapping},
          "etl":{"recipe":"explicit-detection-boxes-rgb-v1","config_sha256":report["config_sha256"],"pillow":pillow_version,
                 "geometry":"unchanged source image dimensions; no EXIF transform/resize","label_source":"existing annotations; user-attested review"},"samples":candidates}
        errors=validate_manifest(manifest,output,training=not manifest["fixture_only"])
        if errors:raise ValueError("; ".join(errors))
        (output/"release.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        report.update(status="complete",exported_images=len(candidates),capture_groups=len(distinct),
          split_counts={s:sum(x["split"]==s for x in candidates) for s in ("train","val","test")},release_sha256=sha256_file(output/"release.json"))
        return manifest
    except Exception as exc:
        report.update(status="failed",error=failure_details(exc,"data.etl"),diagnostic_trace_id=current_trace_id());raise
    finally:
        (output/"etl-report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
