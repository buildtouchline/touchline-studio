"""Deployment factory for the Torchvision detector, not a managed LLM Endpoint.

Prepare only: importing this file creates no Modal resources. The operator must
stage approved weights in the named Volume and authorize deployment/cost.
"""
from pathlib import Path
import re


def build_app(image_id,checkpoint_sha256,*,volume_name='touchline-approved-models'):
    if not re.fullmatch('im-[a-zA-Z0-9_-]{1,150}',image_id):raise ValueError('Pinned image required')
    if not re.fullmatch('[a-f0-9]{64}',checkpoint_sha256):raise ValueError('Approved checkpoint hash required')
    import modal
    app=modal.App('touchline-detector-api')
    volume=modal.Volume.from_name(volume_name,create_if_missing=False)
    # Copy only the Python package into the image. Never upload the checkout,
    # .env, artifacts or a mutable serving-code directory on the model Volume.
    image=modal.Image.from_id(image_id).add_local_python_source('touchline',copy=True)

    @app.cls(image=image,gpu=None,cpu=4,memory=8192,
             timeout=120,min_containers=0,max_containers=1,scaledown_window=30,
             volumes={'/models':volume})
    class DetectorService:
        @modal.enter()
        def load(self):
            from touchline.data.manifest import sha256_file
            from touchline.adapters.torchvision import TorchvisionDetector
            checkpoint=Path('/models')/checkpoint_sha256/'candidate.pt'
            if sha256_file(checkpoint)!=checkpoint_sha256:raise ValueError('Serving checkpoint changed')
            self.detector=TorchvisionDetector(checkpoint,device='cpu',min_size=1080,max_size=1920)

        @modal.asgi_app(requires_proxy_auth=True)
        def api(self):
            from touchline.detector_api import create_app
            return create_app(self.detector)
    return app
