from ..diagnostics import diagnosed
from pathlib import Path
from time import perf_counter
from ..contracts import Box, Detection
from ..data.manifest import sha256_file
from .handoff import validate_candidate_metadata


class TorchvisionDetector:
    architecture = "fasterrcnn_mobilenet_v3_large_320_fpn"

    @diagnosed("detector.load")
    def __init__(
        self,
        checkpoint=None,
        device="cpu",
        threshold=0.5,
        allow_download=False,
        min_size=None,
        max_size=None,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("score threshold must be in [0,1]")
        if checkpoint is None and not allow_download:
            raise ValueError("provide a local model bundle or explicitly allow pretrained weight download")
        if (min_size is None) != (max_size is None):
            raise ValueError("min_size and max_size must both be provided or both be None")
        if min_size is not None:
            if type(min_size) is not int or type(max_size) is not int or min_size <= 0 or max_size <= 0 or min_size > max_size:
                raise ValueError("min_size and max_size must be positive integers with min_size <= max_size")
        try:
            import torch
            import torchvision
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Install a tested compatible Torch/Torchvision pair; see docs/COLAB_RUNBOOK.md") from exc
        self.torch, self.device, self.threshold = torch, device, threshold
        if checkpoint is None:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
            checkpoint_ref = sha256_file(Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1])
        else:
            bundle = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            metadata = bundle.get("metadata", {})
            validate_candidate_metadata(metadata, runtime_torch=torch.__version__, runtime_torchvision=torchvision.__version__)
            from ..training.model import empty_candidate
            self.model = empty_candidate()
            self.model.load_state_dict(bundle["model_state"], strict=True)
            checkpoint_ref = sha256_file(Path(checkpoint))
        if min_size is not None:
            self.model.transform.min_size = (min_size,)
            self.model.transform.max_size = max_size
        self.model.to(device).eval()
        self.actual_device = str(next(self.model.parameters()).device)
        if str(device).startswith('cuda') and not self.actual_device.startswith('cuda'):
            raise RuntimeError('CUDA was requested but the detector parameters are not on CUDA')
        self._inference_calls = 0
        self._input_device = None
        self._detector_wall_seconds = 0.0
        self._provenance = {
            "adapter": "torchvision-detector-v1",
            "architecture": self.architecture,
            "checkpoint": checkpoint_ref,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": device,
            "model_parameter_device": self.actual_device,
            "threshold": threshold,
        }
        if min_size is not None:
            self._provenance["min_size"] = min_size
            self._provenance["max_size"] = max_size
        if checkpoint is not None:
            for key in ('run_id','execution_backend','parent_run_id','release_id','dataset_version','dataset_manifest_sha256'):
                if key in metadata:self._provenance['training_'+key]=metadata[key]

    def provenance(self):
        return dict(self._provenance)

    def runtime_stats(self):
        result = {'requested_device': str(self.device), 'model_parameter_device': self.actual_device,
            'last_input_device': self._input_device, 'inference_calls': self._inference_calls,
            'detector_wall_seconds': self._detector_wall_seconds,
            'timing_scope': 'image/tensor conversion, model forward and result transfer; not GPU utilization'}
        if self.actual_device.startswith('cuda'):
            result['gpu_name'] = self.torch.cuda.get_device_name(self.actual_device)
            result['gpu_peak_allocated_bytes'] = self.torch.cuda.max_memory_allocated(self.actual_device)
        return result

    @diagnosed("detector.infer", failures_only=True)
    def detect_with_proposals(self, frame):
        from torchvision.transforms.functional import to_tensor
        started = perf_counter()
        image = frame.image.to_image().convert("RGB")
        with self.torch.inference_mode():
            tensor = to_tensor(image).to(self.device)
            self._input_device = str(tensor.device)
            if self._input_device != self.actual_device:
                raise RuntimeError('Detector input and model devices differ')
            prediction = self.model([tensor])[0]
        detections = []
        proposals = []
        for box, label, score in zip(
            prediction["boxes"].detach().cpu().tolist(),
            prediction["labels"].detach().cpu().tolist(),
            prediction["scores"].detach().cpu().tolist(),
        ):
            if label != 1:
                continue
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.width, x2), min(frame.height, y2)
            if x1 < x2 and y1 < y2:
                proposals.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "score": score,
                    "label": label,
                })
                if score >= self.threshold:
                    detections.append(Detection(Box(x1, y1, x2, y2), score))
        self._inference_calls += 1
        self._detector_wall_seconds += perf_counter() - started
        return detections, proposals

    def detect(self, frame):
        detections, _ = self.detect_with_proposals(frame)
        return detections
