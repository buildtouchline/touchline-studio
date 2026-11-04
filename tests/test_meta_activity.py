"""Offline selected-frame Meta contract tests; no network or credentials."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from touchline.meta_activity import (
    MODELS, call_meta, execute_activity, list_activities, meta_payload,
    new_activity, save_activity,
)


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
        self.assertEqual(payload["max_output_tokens"], 220)
        self.assertTrue(payload["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(meta_payload("sam", b"image")["model"], "sam-3.1")
        self.assertEqual(meta_payload("wonder", b"image")["model"], "muse-wonder-1.0")
        self.assertNotIn("max_output_tokens", meta_payload("sam", b"image"))

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
            self.assertLessEqual(sent[0][1], 35)
            self.assertNotIn("synthetic-test-secret", (run / "meta_activity" / (record["activity_id"] + ".json")).read_text())
            self.assertEqual(list_activities(run)[0]["status"], "response_received")

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
            record = new_activity("test-run", "wonder", 0.0)
            with patch("touchline.meta_activity.source_frame", return_value=(b"image", 0.0)):
                with patch("touchline.meta_activity.call_meta", return_value=({"id": "real-response-id", "model": MODELS["wonder"], "status": "incomplete", "output": []}, None)):
                    result = execute_activity(run, record, "synthetic-key")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_code"], "no_reviewable_output")
            self.assertEqual(result["request_id"], "real-response-id")


if __name__ == "__main__":
    unittest.main()
