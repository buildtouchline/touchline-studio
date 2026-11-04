"""Offline selected-frame Meta contract tests; no network or credentials."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from touchline.meta_activity import (
    MODELS, MetaCallError, call_meta, call_voice, execute_activity, list_activities, meta_payload,
    new_activity, save_activity, verified_activity_frame,
)
import io
import wave
from urllib.error import HTTPError


class FakeResponse:
    status = 200
    headers = {"x-request-id": "test-request-id"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit):
        return json.dumps({"id": "response-from-meta", "model": MODELS["spark"],
                           "output": [{"content": [{"text": "A player is visible near the touchline."}]}]}).encode()


class MetaActivityTests(unittest.TestCase):
    def test_model_payload_uses_official_meta_endpoint_and_bounded_output(self):
        payload = meta_payload("spark", b"synthetic-jpeg-bytes")
        self.assertEqual(payload["model"], "muse-spark-1.3")
        self.assertEqual(payload["max_output_tokens"], 700)
        self.assertEqual(payload["reasoning"], {"effort": "minimal"})
        self.assertTrue(payload["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(meta_payload("sam", b"image")["model"], "sam-3.1")
        self.assertEqual(meta_payload("wonder", b"image", objective="Check shoulder before receiving")["model"], "muse-wonder-1.0")
        self.assertNotIn("max_output_tokens", meta_payload("sam", b"image"))
        seen = []
        call_meta(meta_payload("sam", b"image"), "test-only",
                  transport=lambda _req, timeout: (seen.append(timeout), FakeResponse())[1])
        self.assertEqual(seen, [35])

    def test_mock_transport_keeps_key_out_of_persisted_activity(self):
        sent = []

        def transport(req, timeout):
            sent.append((req.full_url, timeout, json.loads(req.data)))
            self.assertEqual(req.get_header("Authorization"), "Bearer synthetic-test-secret")
            return FakeResponse()

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "source_video.mp4").write_bytes(b"offline fixture placeholder")
            record = new_activity("test-run", "spark", 1.25)
            save_activity(run, record)
            with patch("touchline.meta_activity.source_frame", return_value=(b"sample-jpeg", 1.2)):
                result = execute_activity(run, record, "synthetic-test-secret", transport=transport)
            self.assertEqual(result["status"], "response_received")
            self.assertEqual(result["frame_time_s"], 1.2)
            self.assertEqual(result["request_id"], "response-from-meta")
            self.assertIn("player is visible", result["output_preview"])
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0][0], "https://api.meta.ai/v1/responses")
            self.assertLessEqual(sent[0][1], 60)
            self.assertNotIn("synthetic-test-secret", (run / "meta_activity" / (record["activity_id"] + ".json")).read_text())
            self.assertEqual(list_activities(run)[0]["status"], "response_received")
            frame = run / "meta_activity" / (record["activity_id"] + ".jpg")
            self.assertEqual(frame.read_bytes(), b"sample-jpeg")
            self.assertEqual(verified_activity_frame(run, result), b"sample-jpeg")
            frame.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "lineage"):
                verified_activity_frame(run, result)

    def test_legacy_frame_requires_exact_redecoded_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            record = new_activity("test-run", "sam", 1.0)
            record["frame_sha256"] = hashlib.sha256(b"original").hexdigest()
            with patch("touchline.meta_activity.source_frame", return_value=(b"original", 1.0)):
                self.assertEqual(verified_activity_frame(run, record), b"original")
            with patch("touchline.meta_activity.source_frame", return_value=(b"different", 1.0)):
                with self.assertRaisesRegex(ValueError, "lineage"):
                    verified_activity_frame(run, record)

    def test_missing_key_and_invalid_time_fail_closed(self):
        with self.assertRaises(ValueError):
            call_meta(meta_payload("sam", b"image"), "")
        for value in (-1, float("nan"), "1.0"):
            with self.assertRaises(ValueError):
                new_activity("run", "sam", value)
        with self.assertRaises(ValueError):
            new_activity("run", "unapproved-model", 0.0)

    def test_failed_transport_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            record = new_activity("test-run", "spark", 0.0)
            with patch("touchline.meta_activity.source_frame", return_value=(b"image", 0.0)):
                result = execute_activity(run, record, "synthetic-key", transport=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()))
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("output_preview", result)
            self.assertNotIn("synthetic-key", json.dumps(result))

    def test_empty_muse_response_is_failed_but_keeps_provider_id(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            record = new_activity("test-run", "wonder", 0.0, objective="Check shoulder before receiving")
            with patch("touchline.meta_activity.source_frame", return_value=(b"image", 0.0)):
                with patch("touchline.meta_activity.call_meta", return_value=({"id": "real-response-id", "model": MODELS["wonder"], "status": "incomplete", "output": []}, None)):
                    result = execute_activity(run, record, "synthetic-key")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_code"], "provider_response_not_completed")
            self.assertEqual(result["request_id"], "real-response-id")

    def test_voice_request_is_documented_multipart_and_retains_turns(self):
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 3200)
        seen = []

        class VoiceResponse(FakeResponse):
            def read(self, _limit):
                return json.dumps({"sessionId": "provider-session", "transcript": "Check your shoulder.",
                                   "turns": [{"turnId": 1, "startMs": 0, "endMs": 200,
                                              "transcript": "Check your shoulder.", "speaker": "A"}]}).encode()

        def transport(req, timeout):
            seen.append(req)
            self.assertLessEqual(timeout, 70)
            return VoiceResponse()

        result, request_id, duration = call_voice(audio_bytes.getvalue(), "test-only", "meta-test", transport=transport)
        self.assertEqual(result["transcript"], "Check your shoulder.")
        self.assertEqual(request_id, "test-request-id")
        self.assertAlmostEqual(duration, 0.2)
        self.assertEqual(seen[0].full_url, "https://api.meta.ai/v1/asr/transcribe?sessionId=meta-test")
        self.assertIn(b'name="request"', seen[0].data)
        self.assertIn(b'name="audio"', seen[0].data)
        self.assertNotIn(b"test-only", seen[0].data)

    def test_voice_rejects_unsupported_pcm_before_provider(self):
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0\0\0" * 3200)
        with self.assertRaises(ValueError):
            call_voice(audio_bytes.getvalue(), "test-only", "meta-test", transport=lambda *_args, **_kwargs: self.fail("sent invalid audio"))

    def test_provider_http_error_is_typed_and_secret_redacted(self):
        def transport(_req, timeout):
            body = b'{"error":{"code":"quota_exceeded","message":"Limit reached for Bearer test-only"}}'
            raise HTTPError("https://api.meta.ai/v1/responses", 429, "rate limited", {}, io.BytesIO(body))
        with self.assertRaises(MetaCallError) as caught:
            call_meta(meta_payload("spark", b"image"), "test-only", transport=transport)
        self.assertEqual(caught.exception.http_status, 429)
        self.assertEqual(caught.exception.code, "quota_exceeded")
        self.assertNotIn("test-only", caught.exception.detail)

    def test_malformed_provider_response_is_not_success(self):
        class BadResponse(FakeResponse):
            def read(self, _limit):
                return b"{invalid"
        with self.assertRaises(MetaCallError) as caught:
            call_meta(meta_payload("spark", b"image"), "test-only", transport=lambda *_args, **_kwargs: BadResponse())
        self.assertEqual(caught.exception.code, "malformed_provider_response")


if __name__ == "__main__":
    unittest.main()
