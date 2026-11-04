import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from PIL import Image
from touchline.audio import parse_meta_muse_response
from touchline.contracts import Frame
from touchline.detector_api import DetectorApiClient
from touchline.modal_trial import parse_sam_proposals
from touchline.voice_review import append_transcript, select_speech_provider


class Response:
    def __init__(self, data):self.data=json.dumps(data).encode()
    def read(self, size=-1):return self.data[:size] if size>=0 else self.data
    def __enter__(self):return self
    def __exit__(self, *args):pass


class SelectedIntegrationTests(unittest.TestCase):
    def test_speech_never_silently_switches_provider(self):
        self.assertEqual(select_speech_provider('sarvam', meta_key='meta-fixture', sarvam_key='sarvam-fixture'), ('sarvam','sarvam-fixture'))
        with self.assertRaises(ValueError):select_speech_provider('meta',sarvam_key='fixture')
        with self.assertRaises(ValueError):select_speech_provider('unknown',meta_key='fixture')

    def test_voice_clips_keep_timestamps_and_existing_ids(self):
        old={'segments':[{'segment_id':'seg_001','text':'old','video_start_s':2.,'video_end_s':3.}], 'manual_cues':[], 'transcript_full':'old'}
        new={'segments':[{'segment_id':'seg_001','text':'new','video_start_s':8.,'video_end_s':9.}], 'manual_cues':[], 'transcript_full':'new'}
        frozen=copy.deepcopy(old); result=append_transcript(old,new)
        self.assertEqual(old,frozen)
        self.assertEqual([r['segment_id'] for r in result['segments']],['seg_001','seg_002'])
        self.assertEqual(result['segments'][1]['video_start_s'],8.)
        self.assertEqual(result['segments'][1]['source_segment_id'],'seg_001')
        self.assertEqual(len(result['clips']),2)
        with self.assertRaises(ValueError):append_transcript({}, {'segments':[{'video_start_s':float('nan'),'video_end_s':1}]})

    def test_unstamped_speech_has_explicit_estimated_timing(self):
        result=parse_meta_muse_response({'text':'Press now. Recover.'},audio_duration_s=4.)
        self.assertTrue(all(r['alignment_review_required'] for r in result))
        self.assertEqual(result[0]['timing_provenance'],'estimated_equal_duration')

    def test_sam_polygon_rejects_negative_overflow_and_unpaired_vertices(self):
        base='<0f>0<|box;x1=1;y1=2;x2=4;y2=9;w=10;h=10|>'
        for points in ('-1,2,4,2,4,9','1,2,14,2,4,9','1,2,4,2,4'):
            with self.subTest(points=points), self.assertRaises(ValueError):
                parse_sam_proposals(base+'<|poly;points='+points+'|>',10,10)

    def test_remote_detector_rejects_wrong_frame_geometry_weights_and_boxes(self):
        frame=Frame('source',24,24,1,24,1.,64,32,SimpleNamespace(to_image=lambda:Image.new('RGB',(64,32))))
        base={'schema':'touchline.detector-response.v1','frame_index':24,'timestamp_s':1.,'width':64,'height':32,
              'checkpoint_sha256':'a'*64,'detections':[{'class_id':1,'bbox_xyxy':[1,2,10,20],'score':.9}]}
        changes=[{'frame_index':25},{'width':65},{'checkpoint_sha256':'b'*64},
                 {'detections':[{'class_id':1,'bbox_xyxy':[1,2,100,20],'score':.9}]},
                 {'detections':[{'class_id':1,'bbox_xyxy':[1,2,10,20],'score':float('nan')}]}]
        for change in changes:
            def opener(req,timeout=30):
                return Response({'status':'ready','checkpoint_sha256':'a'*64} if req.full_url.endswith('/healthz') else {**base,**change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                DetectorApiClient('https://fixture.modal.run',opener=opener).detect(frame)

    def test_remote_detector_rejects_credential_redirect_targets(self):
        for url in ('http://example.com','https://user:secret@example.com','https://example.com?token=fixture'):
            with self.subTest(url=url), self.assertRaises(ValueError):DetectorApiClient(url)
