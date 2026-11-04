"""Bounded local metadata inventory. No downloads and no automatic label-format guesses."""
from ..diagnostics import diagnosed
from collections import Counter
import csv
import json
from pathlib import Path


@diagnosed("data.inventory")
def inventory(root, sample_limit=30):
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError("inventory root must be an existing directory")
    counts=Counter();samples=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        extension=path.suffix.lower();counts[extension or '(none)']+=1
        if len(samples)>=sample_limit or extension not in ('.json','.csv','.txt','.yaml','.yml'):continue
        item={"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"status":"needs_mapping"}
        try:
            if path.stat().st_size>5_000_000:
                item['note']='not parsed in this bounded preview; inspect separately'
            elif extension=='.json':
                value=json.loads(path.read_text(encoding='utf-8-sig'))
                item['json_type']=type(value).__name__
                if isinstance(value,dict):
                    item['top_level_keys']=list(value)[:20]
                    if {'images','annotations','categories'}<=set(value):item['candidate']='COCO detection; inspect bbox/category semantics'
            elif extension=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as handle:item['columns']=next(csv.reader(handle),[])
                item['note']='CSV may contain events/stats rather than image boxes; explicit column and unit mapping required'
            elif extension=='.txt':
                nonempty=[line for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()][:3]
                item['sample_column_counts']=[len(line.split()) for line in nonempty]
                item['note']='Five columns can mean normalized YOLO or corner boxes; never auto-select by count'
            else:item['note']='YAML is not auto-executed or parsed; record source classes and paths explicitly'
        except (OSError,ValueError,UnicodeError) as exc:item['error']=str(exc)
        samples.append(item)
    return {"root":str(root),"extension_counts":dict(counts),"metadata_preview":samples,
            "training_ready":False,"note":"Local inventory only; no annotation review, rights, image validity, source split or coordinate units established."}
