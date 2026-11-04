"""Held-out person detection evaluation using the official COCO evaluator."""
from ..diagnostics import diagnosed
import json
from pathlib import Path
from ..data.manifest import read_manifest, sha256_file


@diagnosed("evaluation")
def evaluate_detector(manifest, output, checkpoint=None, split='val', device='cuda', max_images=None, allow_download=False,*,min_size=None,max_size=None):
    if (min_size is None)!=(max_size is None):raise ValueError('Specify both inference resize bounds')
    if min_size is not None and (type(min_size) is not int or type(max_size) is not int or not 0<min_size<=max_size):
        raise ValueError('Positive ordered integer inference resize bounds required')
    if split not in ('val', 'test'):
        raise ValueError('evaluation split must be val or test')
    if max_images is not None and (type(max_images) is not int or max_images <= 0):
        raise ValueError('max_images must be positive')
    data = read_manifest(manifest, training=True)
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ..adapters.torchvision import TorchvisionDetector
    resize={} if min_size is None else {'min_size':min_size,'max_size':max_size}
    detector = TorchvisionDetector(checkpoint, device, threshold=0.0, allow_download=allow_download,**resize)
    samples = sorted((s for s in data['samples'] if s['split'] == split), key=lambda s:s['sample_id'])
    if max_images:
        samples = samples[:max_images]
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    gt = {'info': {}, 'images': [], 'categories': [{'id':1, 'name':'person'}], 'annotations': []}
    predictions = []; annotation_id = 0
    for image_id, sample in enumerate(samples, 1):
        gt['images'].append({'id':image_id,'width':sample['width'],'height':sample['height']})
        for obj in sample['objects']:
            x1,y1,x2,y2 = obj['bbox_xyxy']; annotation_id += 1
            gt['annotations'].append({'id':annotation_id,'image_id':image_id,'category_id':1,
                'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
        with Image.open(Path(manifest).parent/sample['image_path']) as image:
            tensor = to_tensor(image.convert('RGB')).to(device)
        with detector.torch.inference_mode():
            pred = detector.model([tensor])[0]
        for box,label,score in zip(pred['boxes'].cpu().tolist(),pred['labels'].cpu().tolist(),pred['scores'].cpu().tolist()):
            if label == 1:
                x1,y1,x2,y2 = box
                predictions.append({'image_id':image_id,'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
    if not gt['annotations']:
        raise ValueError('held-out subset contains no person ground truth; expand it before evaluation')
    coco = COCO(); coco.dataset = gt; coco.createIndex()
    if predictions:
        detected = coco.loadRes(predictions)
    else:
        detected = COCO(); detected.dataset = {**gt, 'annotations': []}; detected.createIndex()
    evaluator = COCOeval(coco, detected, 'bbox')
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    names = ['map_50_95','ap50','ap75','ap_small','ap_medium','ap_large','ar_1','ar_10','ar_100','ar_small','ar_medium','ar_large']
    metrics = {name:float(value) if value >= 0 else None for name,value in zip(names,evaluator.stats)}
    result = {'split':split,'images':len(samples),'subset':max_images is not None,'sample_ids':[s['sample_id'] for s in samples],
        'manifest_sha256':sha256_file(manifest),'model':detector.provenance(),'metrics':metrics,
        'inference_transform':{'min_size':list(detector.model.transform.min_size),'max_size':detector.model.transform.max_size},
        'selection_policy':'Use validation for model selection; evaluate frozen test only after selection. No automatic promotion.',
        'limitations':'Detection boxes only; not identity/tracking accuracy or physical metric validation. Small means COCO pixel-area bins.'}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps(predictions)+'\n',encoding='utf-8')
    return result
