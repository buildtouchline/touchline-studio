import asyncio
import base64
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from PIL import Image
from touchline.contracts import Box,Detection
from touchline.detector_api import create_app,MAX_BODY


class DetectorApiTests(unittest.IsolatedAsyncioTestCase):
    async def request(self,detector,body,path='/v1/detect',method='POST'):
        messages=[]
        async def receive():return {'type':'http.request','body':body,'more_body':False}
        async def send(message):messages.append(message)
        await create_app(detector)({'type':'http','path':path,'method':method},receive,send)
        return messages[0]['status'],json.loads(messages[1]['body'])

    def fixture(self):
        stream=io.BytesIO();Image.new('RGB',(64,32)).save(stream,format='PNG');raw=stream.getvalue()
        return {'image_base64':base64.b64encode(raw).decode(),'image_sha256':hashlib.sha256(raw).hexdigest(),
            'frame_index':24,'timestamp_s':1.}

    async def test_detector_returns_source_geometry_and_weights_without_training_or_identity(self):
        detector=Mock();detector.provenance.return_value={'checkpoint':'a'*64}
        detector.detect.return_value=[Detection(Box(1,2,10,20),.9)]
        status,result=await self.request(detector,json.dumps(self.fixture()).encode())
        self.assertEqual(status,200);self.assertEqual(result['detections'][0]['bbox_xyxy'],[1,2,10,20])
        self.assertEqual((result['width'],result['height']),(64,32))
        self.assertFalse(result['training_performed']);self.assertFalse(result['identity_inferred'])

    async def test_bad_hash_timestamp_and_extra_instructions_never_reach_model(self):
        for change in ({'image_sha256':'b'*64},{'timestamp_s':1.5},{'train':True}):
            detector=Mock();request={**self.fixture(),**change}
            status,_=await self.request(detector,json.dumps(request).encode())
            self.assertEqual(status,400);detector.detect.assert_not_called()

    async def test_large_requests_and_training_routes_are_rejected(self):
        detector=Mock()
        self.assertEqual((await self.request(detector,b'x'*(MAX_BODY+1)))[0],413)
        self.assertEqual((await self.request(detector,b'{}',path='/v1/train'))[0],404)
        detector.detect.assert_not_called()

    async def test_errors_do_not_disclose_exception_text(self):
        detector=Mock();detector.detect.side_effect=RuntimeError('fixture-private-key')
        status,result=await self.request(detector,json.dumps(self.fixture()).encode())
        self.assertEqual(status,503);self.assertNotIn('fixture',json.dumps(result))

from touchline.detector_api import DetectorApiClient
from touchline.contracts import Frame


class _FakeHttpResp:
    def __init__(self, data, status=200):
        self._body = json.dumps(data).encode()
        self.status = status

    def read(self, size=-1):
        return self._body if size < 0 else self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DetectorApiClientTests(unittest.TestCase):
    def test_client_headers_support_modal_tokens_and_bearer(self):
        client = DetectorApiClient(
            "https://test.modal.run",
            token_id="modal_id_123",
            token_secret="modal_sec_456",
        )
        headers = client._get_headers()
        self.assertEqual(headers["Modal-Token-Id"], "modal_id_123")
        self.assertEqual(headers["Modal-Token-Secret"], "modal_sec_456")

        bearer_client = DetectorApiClient(
            "https://test.modal.run",
            bearer_token="bearer_xyz",
        )
        self.assertEqual(bearer_client._get_headers()["Authorization"], "Bearer bearer_xyz")

    def test_client_health_check_success_and_provenance(self):
        chk = "a" * 64
        def fake_opener(req, timeout=30.0):
            self.assertEqual(req.full_url, "https://test.modal.run/healthz")
            return _FakeHttpResp({"status": "ready", "checkpoint_sha256": chk})

        client = DetectorApiClient(
            "https://test.modal.run",
            expected_checkpoint_sha256=chk,
            opener=fake_opener,
        )
        health = client.health()
        self.assertEqual(health["status"], "ready")
        prov = client.provenance()
        self.assertEqual(prov["checkpoint"], chk)
        self.assertEqual(prov["backend"], "modal_asgi")

    def test_client_health_check_mismatch_raises(self):
        def fake_opener(req, timeout=30.0):
            return _FakeHttpResp({"status": "ready", "checkpoint_sha256": "b" * 64})

        client = DetectorApiClient(
            "https://test.modal.run",
            expected_checkpoint_sha256="a" * 64,
            opener=fake_opener,
        )
        with self.assertRaises(ValueError):
            client.health()

    def test_client_detect_converts_frame_to_detections(self):
        chk = "a" * 64
        img = Image.new("RGB", (64, 32))
        frame = Frame("src_1", 24, 1000, 1, 24, 1.0, 64, 32, SimpleNamespace(to_image=lambda: img))

        def fake_opener(req, timeout=30.0):
            if req.full_url.endswith("/v1/detect"):
                body = json.loads(req.data.decode())
                self.assertEqual(body["frame_index"], 24)
                self.assertEqual(body["timestamp_s"], 1.0)
                return _FakeHttpResp({
                    "schema": "touchline.detector-response.v1",
                    "frame_index": 24,
                    "timestamp_s": 1.0,
                    "width": 64,
                    "height": 32,
                    "checkpoint_sha256": chk,
                    "detections": [
                        {"class_id": 1, "bbox_xyxy": [5.0, 10.0, 25.0, 30.0], "score": 0.88}
                    ],
                })
            return _FakeHttpResp({"status": "ready", "checkpoint_sha256": chk})

        client = DetectorApiClient(
            "https://test.modal.run",
            opener=fake_opener,
        )
        detections = client.detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box.values(), [5.0, 10.0, 25.0, 30.0])
        self.assertEqual(detections[0].score, 0.88)
