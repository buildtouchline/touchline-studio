import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

spec=importlib.util.spec_from_file_location('cpu_detector_factory_test',Path(__file__).resolve().parents[1]/'tools/modal_detector_endpoint.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)


class CpuDetectorFactoryTests(unittest.TestCase):
    def fixture(self):
        app=Mock();captured=[]
        def register(**kwargs):
            def capture(cls):captured.append(cls);return cls
            return capture
        app.cls.side_effect=register
        modal=SimpleNamespace(App=Mock(return_value=app),Volume=SimpleNamespace(from_name=Mock(return_value='volume')),
            Image=SimpleNamespace(from_id=Mock(return_value=Mock())),enter=Mock(return_value=lambda f:f),
            asgi_app=Mock(return_value=lambda f:f))
        with patch.dict(sys.modules,{'modal':modal}):factory.build_app('im-fixture','a'*64)
        return app,modal,captured[0]

    def test_prepared_inference_service_is_cpu_and_keeps_proxy_auth(self):
        app,modal,_=self.fixture()
        options=app.cls.call_args.kwargs
        self.assertIsNone(options['gpu']);self.assertEqual(options['cpu'],4)
        self.assertEqual(options['min_containers'],0);self.assertEqual(options['max_containers'],1)
        modal.asgi_app.assert_called_once_with(requires_proxy_auth=True)
        modal.Volume.from_name.assert_called_once_with('touchline-approved-models',create_if_missing=False)

    def test_checkpoint_load_uses_cpu_without_allocating_gpu(self):
        _,_,service=self.fixture()
        detector=Mock()
        with patch('touchline.data.manifest.sha256_file',return_value='a'*64),patch('touchline.adapters.torchvision.TorchvisionDetector',detector):
            service().load()
        self.assertEqual(detector.call_args.kwargs,{'device':'cpu','min_size':1080,'max_size':1920})
