"""Loopback-only Studio and synthetic audio regression checks."""
import io
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

from touchline.audio import convert_audio_to_wav16k, load_sarvam_key
from touchline import studio


class StudioFileSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.runs=cls.root/'runs';cls.runs.mkdir()
        cls.web=cls.root/'web';cls.web.mkdir()
        for name in ('index.html','styles.css','app.js'):(cls.web/name).write_text('public '+name)
        (cls.root/'app.js').write_text('outside web sentinel')
        (cls.root/'source_video.mp4').write_bytes(b'outside runs sentinel')
        (cls.runs/'.fixture_key').write_text('fake credential sentinel')
        cls.run_dir=cls.runs/'fixture-run';cls.run_dir.mkdir()
        (cls.run_dir/'manifest.json').write_text('{}')
        (cls.run_dir/'observations.jsonl').write_text('')
        (cls.run_dir/'source_video.mp4').write_bytes(b'0123456789')
        (cls.run_dir/'private.json').write_text('{"not_a_public_artifact":true}')
        cls.ready=threading.Event()
        def server_factory(address,handler):
            cls.server=ThreadingHTTPServer(('127.0.0.1',0),handler)
            cls.ready.set();return cls.server
        cls.patches=[patch.object(studio,'ThreadingHTTPServer',side_effect=server_factory),
                     patch.object(studio,'ensure_calibrated_demo_run',return_value=None),
                     patch.object(studio,'WEB_DIR',cls.web)]
        for p in cls.patches:p.start()
        cls.thread=threading.Thread(target=studio.run_studio_server,args=(cls.runs,),daemon=True)
        cls.thread.start()
        if not cls.ready.wait(5):raise RuntimeError('Loopback server did not start')

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(5)
        for p in reversed(cls.patches):p.stop()
        cls.tmp.cleanup()

    def fetch(self,path,method='GET',headers=None,body=None):
        conn=HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        try:
            conn.request(method,path,headers=headers or {},body=body)
            response=conn.getresponse();return response.status,dict(response.getheaders()),response.read()
        finally:conn.close()

    def test_traversal_hidden_and_unlisted_files_are_denied_before_read(self):
        for path in ['/web/../app.js','/web/%2e%2e/app.js','/web/%252e%252e/app.js',
                     '/web/..%5capp.js','/runs/../source_video.mp4','/runs/%2e%2e/source_video.mp4',
                     '/runs/.fixture_key','/runs/%2efixture_key','/runs/fixture-run/private.json']:
            for method in ('GET','HEAD'):
                with self.subTest(path=path,method=method):
                    status,headers,body=self.fetch(path,method)
                    self.assertEqual(status,404)
                    self.assertNotIn(b'sentinel',body)

    def test_static_assets_and_video_ranges_still_work(self):
        self.assertEqual(self.fetch('/app.js')[0],200)
        self.assertEqual(self.fetch('/web/styles.css')[0],200)
        status,headers,body=self.fetch('/runs/fixture-run/source_video.mp4',headers={'Range':'bytes=2-5'})
        self.assertEqual(status,206);self.assertEqual(body,b'2345')
        self.assertEqual(headers['Content-Range'],'bytes 2-5/10')
        status,headers,body=self.fetch('/runs/fixture-run/source_video.mp4','HEAD')
        self.assertEqual(status,200);self.assertEqual(headers['Content-Length'],'10');self.assertEqual(body,b'')

    def test_http_error_has_trace_id_without_request_contents(self):
        status,headers,body=self.fetch('/missing-fixture-private-value')
        self.assertEqual(status,404)
        payload=json.loads(body)
        self.assertRegex(payload['trace_id'],r'^[a-f0-9]{32}$')
        self.assertEqual(headers['X-Touchline-Trace-ID'],payload['trace_id'])
        self.assertNotIn('fixture-private-value',body.decode())

    def test_unhandled_json_failure_returns_safe_error_instead_of_traceback(self):
        status,headers,body=self.fetch('/api/run/fixture-run/transcript','POST',body=b'fixture-private-invalid-json')
        self.assertEqual(status,400)
        self.assertRegex(headers['X-Touchline-Trace-ID'],r'^[a-f0-9]{32}$')
        self.assertNotIn('fixture-private',body.decode())

    def test_out_of_bounds_ranges_do_not_create_negative_lengths(self):
        for value in ('bytes=100-200','bytes=8-2'):
            with self.subTest(value=value):
                self.assertEqual(self.fetch('/runs/fixture-run/source_video.mp4',headers={'Range':value})[0],416)

    def test_symlink_to_outside_artifact_is_denied(self):
        link=self.run_dir/'video_with_subtitles.mp4'
        try:link.symlink_to(self.root/'source_video.mp4')
        except OSError:self.skipTest('Symlink creation unavailable on this Windows runtime')
        try:self.assertEqual(self.fetch('/runs/fixture-run/video_with_subtitles.mp4')[0],404)
        finally:link.unlink()

    def test_assumed_demo_scale_is_not_a_validated_calibration(self):
        self.assertFalse(studio.make_broadcast_calibration().is_valid)

    def test_json_api_denies_symlinked_artifacts_without_reading_them(self):
        outside=self.root/'outside-fixture.json'
        outside.write_text('{"fake_marker":"outside sentinel"}')
        for filename in ('analytics.json','observations.jsonl','transcript.json'):
            target=self.run_dir/filename
            original=target.read_bytes() if target.exists() else None
            if target.exists():target.unlink()
            try:
                try:target.symlink_to(outside)
                except OSError:self.skipTest('Symlink creation unavailable')
                status,headers,body=self.fetch('/api/run/fixture-run')
                if filename=='transcript.json':
                    self.assertEqual(status,200)
                else:self.assertEqual(status,404)
                self.assertNotIn(b'outside sentinel',body)
            finally:
                if target.is_symlink():target.unlink()
                if original is not None:target.write_bytes(original)


class AudioSampleSafetyTests(unittest.TestCase):
    def wav(self,seconds=2,rate=44100,channels=2):
        buf=io.BytesIO()
        with wave.open(buf,'wb') as writer:
            writer.setnchannels(channels);writer.setsampwidth(2);writer.setframerate(rate)
            writer.writeframes(b'\0\0'*int(seconds*rate)*channels)
        return buf.getvalue()

    def test_resampling_writes_logical_samples_without_plane_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'fixture.wav'
            info=convert_audio_to_wav16k(self.wav(),target)
            self.assertEqual(info['transcoded_with'],'pyav')
            with wave.open(str(target),'rb') as reader:
                self.assertEqual((reader.getnchannels(),reader.getsampwidth(),reader.getframerate()),(1,2,16000))
                self.assertEqual(reader.getnframes(),32000)
                self.assertEqual(reader.getnframes()/reader.getframerate(),info['duration_s'])

    def test_failed_normalization_cannot_passthrough_nonconforming_wav(self):
        with tempfile.TemporaryDirectory() as tmp, patch('av.open',side_effect=RuntimeError('fixture decode failure')):
            target=Path(tmp)/'fixture.wav'
            with self.assertRaises(ValueError):convert_audio_to_wav16k(self.wav(),target)
            self.assertFalse(target.exists())

    def test_key_lookup_never_searches_parent_folders(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'SARVAM_API_KEY':''}):
            root=Path(tmp);runs=root/'runs';runs.mkdir()
            (root/'.sarvam_key').write_text('synthetic parent sentinel')
            self.assertIsNone(load_sarvam_key(runs))
