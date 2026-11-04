"""Small custom ASGI detector API; host behind mandatory Modal proxy auth.

This module neither loads weights nor credentials on import. No training routes.
"""
import asyncio
import base64
import hashlib
import io
import json
import math
from types import SimpleNamespace

MAX_BODY=8*1024*1024


def create_app(detector):
    semaphore=asyncio.Semaphore(1)
    async def app(scope,receive,send):
        if scope['type']!='http':return
        async def respond(status,data):
            body=json.dumps(data,allow_nan=False).encode()
            await send({'type':'http.response.start','status':status,'headers':[(b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
            await send({'type':'http.response.body','body':body})
        if scope['path']=='/healthz' and scope['method']=='GET':
            return await respond(200,{'status':'ready','checkpoint_sha256':detector.provenance()['checkpoint']})
        if scope['path']!='/v1/detect' or scope['method']!='POST':return await respond(404,{'error':'route_not_found'})
        body=bytearray()
        while True:
            event=await receive()
            if event['type']=='http.disconnect':return
            chunk=event.get('body',b'')
            if len(chunk)>MAX_BODY-len(body):return await respond(413,{'error':'image_request_too_large'})
            body.extend(chunk)
            if not event.get('more_body',False):break
        try:
            request=json.loads(body)
            if set(request)!={'image_base64','image_sha256','frame_index','timestamp_s'}:raise ValueError()
            index,time=request['frame_index'],request['timestamp_s']
            if type(index) is not int or index<0 or type(time) not in (int,float) or not math.isfinite(time) or time<0:raise ValueError()
            if abs(time-index/24)>0.0001:raise ValueError()
            raw=base64.b64decode(request['image_base64'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=request['image_sha256']:raise ValueError()
            from PIL import Image
            from .contracts import Frame
            with Image.open(io.BytesIO(raw)) as source:
                if source.width*source.height>16000000 or source.width<source.height:raise ValueError()
                if source.getexif().get(274,1)!=1:raise ValueError()
                image=source.convert('RGB')
            frame=Frame(request['image_sha256'],index,index,1,24,time,*image.size,SimpleNamespace(to_image=lambda:image))
        except Exception:return await respond(400,{'error':'invalid_image_request'})
        try:
            async with semaphore:
                detections=await asyncio.to_thread(detector.detect,frame)
            return await respond(200,{'schema':'touchline.detector-response.v1','frame_index':index,
                'timestamp_s':time,'width':image.width,'height':image.height,
                'checkpoint_sha256':detector.provenance()['checkpoint'],
                'detections':[{'class_id':1,'bbox_xyxy':d.box.values(),'score':d.score} for d in detections],
                'training_performed':False,'identity_inferred':False})
        except Exception:return await respond(503,{'error':'detector_execution_failed'})
    return app


import urllib.error
import urllib.request
from .contracts import Box, Detection, Frame


class DetectorApiClient:
    """Optional protected image endpoint; Sandbox recording analysis stays primary."""
    def __init__(self, endpoint_url, *, token_id=None, token_secret=None, bearer_token=None,
                 expected_checkpoint_sha256=None, timeout=30., opener=None):
        from urllib.parse import urlsplit
        parts = urlsplit(endpoint_url) if isinstance(endpoint_url, str) else None
        if parts is None or parts.scheme != 'https' or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('Explicit HTTPS endpoint without embedded credentials required')
        if bool(token_id) != bool(token_secret) or token_id and bearer_token:
            raise ValueError('Select one complete endpoint authentication method')
        if not math.isfinite(float(timeout)) or not 0 < float(timeout) <= 120:
            raise ValueError('Bounded endpoint timeout required')
        if expected_checkpoint_sha256 is not None and not self._valid_digest(expected_checkpoint_sha256):
            raise ValueError('Expected checkpoint must be a SHA256 digest')
        self.endpoint_url=endpoint_url.rstrip('/'); self.token_id=token_id; self.token_secret=token_secret
        self.bearer_token=bearer_token; self.expected_checkpoint_sha256=expected_checkpoint_sha256
        self.timeout=float(timeout); self.opener=opener; self._cached_provenance=None

    @staticmethod
    def _valid_digest(value):
        return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

    def _get_headers(self):
        headers={'Content-Type':'application/json','Accept':'application/json'}
        if self.token_id:
            headers.update({'Modal-Token-Id':self.token_id,'Modal-Token-Secret':self.token_secret})
        elif self.bearer_token:
            headers['Authorization']='Bearer '+self.bearer_token
        return headers

    def _request(self, path, payload=None):
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise RuntimeError('Detector redirects are prohibited')
        request=urllib.request.Request(self.endpoint_url+path, data=payload,
            method='POST' if payload is not None else 'GET', headers=self._get_headers())
        if self.opener is None:
            transport=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open
        else:
            transport=self.opener
        try:
            with transport(request,timeout=self.timeout) as response:
                body=response.read(MAX_BODY+1)
                if len(body)>MAX_BODY:raise ValueError('Detector response exceeds bound')
                data=json.loads(body)
                if not isinstance(data,dict):raise ValueError('Detector response must be an object')
                return data
        except (ValueError,RuntimeError):
            raise
        except Exception:
            raise RuntimeError('Detector request failed; inspect safe provider diagnostics') from None

    def health(self):
        data=self._request('/healthz'); digest=data.get('checkpoint_sha256')
        if data.get('status')!='ready' or not self._valid_digest(digest):
            raise ValueError('Detector is not ready with valid model provenance')
        if self.expected_checkpoint_sha256 is not None and digest!=self.expected_checkpoint_sha256:
            raise ValueError('Serving checkpoint hash mismatch')
        if self._cached_provenance is not None and digest!=self._cached_provenance['checkpoint']:
            raise ValueError('Serving checkpoint changed during run')
        self._cached_provenance={'checkpoint':digest,'endpoint':self.endpoint_url,'backend':'modal_asgi'}
        return data

    def provenance(self):
        if self._cached_provenance is None:self.health()
        return dict(self._cached_provenance)

    def detect(self, frame):
        if frame.image is None:raise ValueError('Frame image required')
        expected=self.provenance()['checkpoint']
        image=frame.image.to_image() if hasattr(frame.image,'to_image') else frame.image
        if image.size!=(frame.width,frame.height):raise ValueError('Frame geometry differs from image')
        if abs(frame.timestamp_s-frame.index/24)>.0001:raise ValueError('Exact 24FPS frame timestamps required')
        buffer=io.BytesIO();image.save(buffer,format='PNG');raw=buffer.getvalue()
        payload=json.dumps({'image_base64':base64.b64encode(raw).decode(),'image_sha256':hashlib.sha256(raw).hexdigest(),
            'frame_index':frame.index,'timestamp_s':frame.timestamp_s}).encode()
        if len(payload)>MAX_BODY:raise ValueError('Detector image request exceeds bound')
        data=self._request('/v1/detect',payload)
        if (data.get('schema')!='touchline.detector-response.v1' or data.get('frame_index')!=frame.index
            or data.get('width')!=frame.width or data.get('height')!=frame.height or data.get('checkpoint_sha256')!=expected
            or type(data.get('timestamp_s')) not in (int,float) or not math.isfinite(data['timestamp_s'])
            or abs(data['timestamp_s']-frame.timestamp_s)>.0001):
            raise ValueError('Remote detector frame/model lineage mismatch')
        rows=data.get('detections')
        if not isinstance(rows,list) or len(rows)>1000:raise ValueError('Bounded detector list required')
        result=[]
        for row in rows:
            if not isinstance(row,dict) or row.get('class_id')!=1:raise ValueError('Person detections required')
            box=row.get('bbox_xyxy');score=row.get('score')
            if (not isinstance(box,list) or len(box)!=4 or not all(type(x) in (int,float) and math.isfinite(x) for x in box)
                or not 0<=box[0]<box[2]<=frame.width or not 0<=box[1]<box[3]<=frame.height
                or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1):
                raise ValueError('Invalid source-pixel person detection')
            result.append(Detection(Box(*box),score))
        return result
