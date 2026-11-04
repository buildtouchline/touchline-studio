"""Bounded candidate training. GPU path must be exercised in the actual Colab runtime."""
from ..diagnostics import diagnosed, diagnostic_progress, failure_details, current_trace_id
import json
import math
from pathlib import Path
import random
import time
from datetime import datetime,timezone
from ..data.manifest import read_manifest,sha256_file
from ..run_records import new_record,write_record


def _train_detector(manifest_path,output,epochs=2,max_steps=100,max_seconds=600,device="cuda",allow_download=False,run_record=None,initial_checkpoint=None,learning_rate=.001,weight_decay=.0005,reviewed_recipe=None):
    if type(epochs) is not int or epochs<=0 or type(max_steps) is not int or max_steps<=0 or not math.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError("positive epochs/step/time caps required")
    if not allow_download and initial_checkpoint is None:
        raise ValueError("this starting recipe needs explicit --allow-weight-download for pretrained initialization")
    data=read_manifest(manifest_path,training=True)
    try:
        import torch
        import torchvision
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except (ImportError,RuntimeError) as exc:
        raise RuntimeError("A tested Torch/Torchvision/Pillow environment is required") from exc
    output=Path(output)
    manifest_path=Path(manifest_path).resolve()
    samples=[s for s in data["samples"] if s["split"]=="train"]
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):return len(samples)
        def __getitem__(self,index):
            sample=samples[index]
            with Image.open(manifest_path.parent/sample["image_path"]) as image:
                tensor=to_tensor(image.convert("RGB"))
            boxes=torch.tensor([o["bbox_xyxy"] for o in sample["objects"]],dtype=torch.float32).reshape(-1,4)
            return tensor,{"boxes":boxes,"labels":torch.ones(len(boxes),dtype=torch.int64),"image_id":torch.tensor(index),
                           "area":(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),"iscrowd":torch.zeros(len(boxes),dtype=torch.int64)}
    seed=42
    random.seed(seed);torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(Dataset(),batch_size=1,shuffle=True,generator=generator,num_workers=0,collate_fn=lambda batch:tuple(zip(*batch)))
    initial_weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    if initial_checkpoint is None:
        model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=initial_weights)
        model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
        initialization_hash=sha256_file(Path(torch.hub.get_dir())/'checkpoints'/initial_weights.url.rsplit('/',1)[-1])
    else:
        from ..adapters.handoff import validate_candidate_metadata
        from .model import empty_candidate
        bundle=torch.load(str(initial_checkpoint),map_location='cpu',weights_only=True)
        validate_candidate_metadata(bundle.get('metadata'),runtime_torch=torch.__version__,runtime_torchvision=torchvision.__version__)
        model=empty_candidate();model.load_state_dict(bundle['model_state'],strict=True)
        initialization_hash=sha256_file(initial_checkpoint)
    if reviewed_recipe is not None:
        from .recipes import validate_resize_recipe
        plan=validate_resize_recipe(reviewed_recipe,initialization_hash)
        model.transform.min_size=(plan['training_min_size'],);model.transform.max_size=plan['training_max_size']
    # Keep the starting recipe small; train RPN/ROI heads, freeze backbone.
    for parameter in model.backbone.parameters():parameter.requires_grad=False
    model.to(device)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=learning_rate,momentum=0.9,weight_decay=weight_decay)
    metadata={"run_id":run_record['run_id'],"execution_backend":run_record['backend'],"parent_run_id":run_record['parent_run_id'],
              "architecture":"fasterrcnn_mobilenet_v3_large_320_fpn","num_classes":2,"class_map":data["class_map"],
              "release_id":data["release_id"],"dataset_version":data["dataset_version"],"dataset_manifest_sha256":sha256_file(manifest_path),
              "torch":str(torch.__version__),"torchvision":str(torchvision.__version__),"seed":seed,"device":device,
              "backbone_norm":"frozen_batchnorm2d","initial_weights":str(initial_weights) if initial_checkpoint is None else 'reviewed-candidate-warm-start',
              "initial_weights_sha256":initialization_hash,
              "initial_checkpoint_sha256":initialization_hash if initial_checkpoint is not None else None,
              "optimizer_resume":False,
              "learning_rate":learning_rate,"weight_decay":weight_decay,
              "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
              "recipe":"starter-heads-fp32-batch1-v1","promotion_state":"unvalidated_candidate","resume_supported":False,
              "caps":{"epochs":epochs,"max_steps":max_steps,"max_seconds_cooperative":max_seconds},
              "limitations":"No automatic promotion, augmentation, mixed precision or optimizer-resume command supplied. Time cap checked between steps; not a billing guarantee. Separate evaluation command required."}
    if reviewed_recipe is not None:
        from .recipes import RESIZE_RECIPE
        metadata.update(recipe=RESIZE_RECIPE,training_transform={'min_size':list(model.transform.min_size),'max_size':model.transform.max_size},approved_recipe_sha256=reviewed_recipe['proposal_sha256'])
    (output/"recipe.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    start=time.monotonic();step=0;stopped=False
    try:
        with (output/"losses.jsonl").open("w",encoding="utf-8") as log:
            for epoch in range(epochs):
                model.train()
                for images,targets in loader:
                    if step>=max_steps or time.monotonic()-start>=max_seconds:
                        stopped=True;break
                    images=[i.to(device) for i in images]
                    targets=[{k:v.to(device) for k,v in target.items()} for target in targets]
                    losses=model(images,targets);loss=sum(losses.values())
                    if not torch.isfinite(loss):raise ValueError("non-finite loss")
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();step+=1
                    log.write(json.dumps({"step":step,"epoch":epoch,"loss":float(loss.detach().cpu())})+"\n");log.flush()
                    if step==1 or step%25==0:
                        diagnostic_progress("training",step=step,epoch=epoch,loss=float(loss.detach().cpu()))
                if stopped:break
        if step==0:raise ValueError("no training step completed")
        bundle={"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"metadata":metadata,
                "training_state":{"steps":step,"optimizer":optimizer.state_dict()}}
        temporary=output/"candidate.pt.tmp";torch.save(bundle,temporary);temporary.replace(output/"candidate.pt")
        result={"run_id":run_record['run_id'],"execution_backend":run_record['backend'],"status":"capped" if stopped else "completed","steps":step,"elapsed_s":time.monotonic()-start,
                "checkpoint_sha256":sha256_file(output/"candidate.pt"),"promotion_state":"unvalidated_candidate"}
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        return result
    except Exception as exc:
        (output/"failure.json").write_text(json.dumps({"error":failure_details(exc,"training"),"trace_id":current_trace_id(),"successful_steps":step,
          "warning":"No resumable training checkpoint is promised by this starter; preserve logs and restart under a new run."},indent=2)+"\n",encoding="utf-8")
        raise


@diagnosed('training')
def train_detector(manifest_path,output,epochs=2,max_steps=100,max_seconds=600,device='cuda',allow_download=False,*,run_id=None,execution_backend='python',parent_run_id=None,initial_checkpoint=None,learning_rate=.001,weight_decay=.0005,reviewed_recipe=None):
    if type(epochs) is not int or epochs<=0 or type(max_steps) is not int or max_steps<=0 or not math.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('Positive epochs/step/time caps required')
    if not allow_download and initial_checkpoint is None:raise ValueError('Explicit pretrained initialization consent or selected checkpoint required')
    for value,low,high in ((learning_rate,.0005,.002),(weight_decay,.00025,.001)):
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Minor tuning must remain within 0.5x to 2x the registered baseline')
    if initial_checkpoint is not None and (Path(initial_checkpoint).suffix.lower() not in ('.pt','.pth') or not Path(initial_checkpoint).is_file()):
        raise ValueError('Existing candidate checkpoint required')
    if reviewed_recipe is not None:
        from .recipes import validate_resize_recipe
        if initial_checkpoint is None:raise ValueError('Resize trial requires reviewed warm-start weights')
        plan=validate_resize_recipe(reviewed_recipe,sha256_file(initial_checkpoint))
        if (epochs,max_steps,max_seconds,learning_rate,weight_decay)!=(plan['epochs'],plan['max_steps'],plan['max_seconds'],plan['learning_rate'],plan['weight_decay']):raise ValueError('Trial parameters differ from approved recipe')
    record=new_record('training',backend=execution_backend,run_id=run_id,parent_run_id=parent_run_id)
    record['diagnostic_trace_id']=current_trace_id()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_record(output,record)
    try:
        record['status']='running';write_record(output,record)
        result=_train_detector(manifest_path,output,epochs,max_steps,max_seconds,device,allow_download,run_record=record,initial_checkpoint=initial_checkpoint,learning_rate=learning_rate,weight_decay=weight_decay,reviewed_recipe=reviewed_recipe)
        result.update(run_id=record['run_id'],execution_backend=record['backend'])
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        record['status']='partial' if result['status']=='capped' else 'completed'
        record['checkpoint_sha256']=result['checkpoint_sha256']
        return result
    except BaseException as exc:
        record['status']='cancelled' if isinstance(exc,(KeyboardInterrupt,GeneratorExit)) else 'failed'
        record['error']=failure_details(exc,'training')
        raise
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_record(output,record)
