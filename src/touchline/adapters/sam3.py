"""Local image-box proposals only; no training, API call or automatic human review."""
from ..diagnostics import diagnosed
from pathlib import Path
from ..data.manifest import sha256_file


@diagnosed("vision.sam")
def propose_people(image_path, checkpoint, device="cuda", prompt="person"):
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("an authorized local SAM 3 checkpoint is required")
    try:
        import torch
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError("Install the pinned official SAM 3 checkout in the tested runtime first") from exc
    model=build_sam3_image_model(checkpoint_path=str(checkpoint),load_from_HF=False,device=device)
    processor=Sam3Processor(model,device=device)
    with Image.open(image_path) as raw:
        image=raw.convert("RGB")
    with torch.inference_mode():
        state=processor.set_image(image)
        output=processor.set_text_prompt(state=state,prompt=prompt)
    proposals=[]
    for box,score in zip(output["boxes"].detach().cpu().tolist(),output["scores"].detach().cpu().tolist()):
        proposals.append({"bbox_xyxy":box,"proposal_score":score,"annotation_status":"model_proposal",
                          "origin":"sam3","concept_prompt":prompt})
    return {"schema_version":"touchline.annotation-proposals.v1","image_sha256":sha256_file(image_path),
            "width":image.width,"height":image.height,"checkpoint_sha256":sha256_file(checkpoint),
            "review_status":"pending","training_eligible":False,"objects":proposals,
            "limitations":"Box proposals only. Review all people and negatives. Export/review into the release schema before training."}
