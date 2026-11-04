"""Loopback Meta activity routes: authentication, fail-closed budget and review."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib import error, request

from touchline.meta_activity import new_activity, save_activity, save_activity_frame


ROOT = Path(__file__).resolve().parents[1]


class MetaStudioApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.runs = Path(self.temporary.name) / "runs"
        self.run = self.runs / "existing-run"
        self.run.mkdir(parents=True)
        (self.run / "manifest.json").write_text(json.dumps({"run_id": "existing-run", "status": "completed"}))
        (self.run / "observations.jsonl").write_text("")
        (self.run / "source_video.mp4").write_bytes(b"not a real recording; request must fail before decode")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        env = {k: v for k, v in os.environ.items() if k not in ("META_API_KEY", "MODEL_API_KEY", "TOUCHLINE_META_MAX_CALLS")}
        env["PYTHONPATH"] = str(ROOT / "src")
        script = "from pathlib import Path; from touchline.studio import run_studio_server; import sys; run_studio_server(Path(sys.argv[1]),port=int(sys.argv[2]),auth_token='test-only-token')"
        self.proc = subprocess.Popen([sys.executable, "-B", "-c", script, str(self.runs), str(self.port)],
                                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=.1):
                    break
            except OSError:
                time.sleep(.03)
        else:
            self.fail("Studio failed to start")

    def tearDown(self):
        self.proc.terminate()
        self.proc.wait(timeout=3)
        self.temporary.cleanup()

    def url(self, suffix):
        return f"http://127.0.0.1:{self.port}/api/run/existing-run/{suffix}"

    def test_auth_budget_and_no_provider_dispatch(self):
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(self.url("meta-activity"), timeout=3)
        self.assertEqual(caught.exception.code, 401)
        req = request.Request(self.url("meta-frame"), json.dumps({"model": "sam", "timestamp_s": 0}).encode(),
                              {"Authorization": "Bearer test-only-token", "Content-Type": "application/json"})
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(req, timeout=3)
        self.assertEqual(caught.exception.code, 503)
        self.assertFalse((self.run / "meta_activity").exists())
        audio = request.Request(self.url("meta-audio?anchor_s=1"), b"x" * 200,
                                {"Authorization": "Bearer test-only-token", "Content-Type": "audio/wav"})
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(audio, timeout=3)
        self.assertEqual(caught.exception.code, 503)
        self.assertFalse((self.run / "meta_activity").exists())

    def test_browser_cannot_configure_provider_key(self):
        req = request.Request(f"http://127.0.0.1:{self.port}/api/meta-key", b'{"api_key":"synthetic-test-only"}',
                              {"Authorization": "Bearer test-only-token", "Content-Type": "application/json"})
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(req, timeout=3)
        self.assertEqual(caught.exception.code, 410)
        self.assertFalse((self.runs / ".meta_key").exists())

    def test_review_is_run_scoped_and_persisted_without_identity_claim(self):
        row = new_activity("existing-run", "sam", 0.0)
        row["status"] = "response_received"
        row["request_id"] = "mock-response-id"
        row["output_preview"] = "<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=4;h=4|>"
        save_activity(self.run, row)
        review_url = self.url(f"meta-activity/{row['activity_id']}/review")
        req = request.Request(review_url, json.dumps({"decision": "confirmed_observation"}).encode(),
                              {"Authorization": "Bearer test-only-token", "Content-Type": "application/json"})
        with request.urlopen(req, timeout=3) as response:
            saved = json.load(response)
        self.assertEqual(saved["review"]["decision"], "confirmed_observation")
        self.assertEqual(saved["review"]["reviewer"], "local_unauthenticated_coach")
        self.assertNotIn("player_id", saved)
        listing = request.Request(self.url("meta-activity"), headers={"Authorization": "Bearer test-only-token"})
        with request.urlopen(listing, timeout=3) as response:
            after = json.load(response)
        self.assertEqual(after["activities"][0]["review"], saved["review"])
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(req, timeout=3)
        self.assertEqual(caught.exception.code, 409)

    def test_approved_voice_turn_attaches_to_session_only_after_review(self):
        row = new_activity("existing-run", "voice", 12.5)
        row.update(status="response_received", output_preview="Check your shoulder.", audio_duration_s=2.0,
                   turns=[{"startMs": 100, "endMs": 1200, "transcript": "Check your shoulder.", "speaker": "A"}])
        save_activity(self.run, row)
        transcript_path = self.run / "transcript.json"
        self.assertFalse(transcript_path.exists())
        req = request.Request(self.url(f"meta-activity/{row['activity_id']}/review"),
                              b'{"decision":"confirmed_observation"}',
                              {"Authorization": "Bearer test-only-token", "Content-Type": "application/json"})
        with request.urlopen(req, timeout=3) as response:
            saved = json.load(response)
        self.assertEqual(saved["review"]["decision"], "confirmed_observation")
        self.assertEqual(saved["transcript_clip_id"], "clip_000001")
        transcript = json.loads(transcript_path.read_text())
        self.assertEqual(transcript["segments"][0]["video_start_s"], 12.6)
        self.assertEqual(transcript["segments"][0]["video_end_s"], 13.7)
        self.assertEqual(transcript["segments"][0]["speaker_label"], "A")
        self.assertFalse(transcript["segments"][0]["speaker_verified"])
        self.assertTrue(transcript["segments"][0]["alignment_review_required"])

    def test_edited_voice_approval_uses_clip_extent_not_original_turn_text(self):
        row = new_activity("existing-run", "voice", 4.0)
        row.update(status="response_received", output_preview="Original words", audio_duration_s=1.5,
                   turns=[{"startMs": 0, "endMs": 1000, "transcript": "Original words"}])
        save_activity(self.run, row)
        req = request.Request(self.url(f"meta-activity/{row['activity_id']}/review"),
                              json.dumps({"decision": "confirmed_observation", "edited_text": "Coach corrected words"}).encode(),
                              {"Authorization": "Bearer test-only-token", "Content-Type": "application/json"})
        with request.urlopen(req, timeout=3) as response:
            saved = json.load(response)
        transcript = json.loads((self.run / "transcript.json").read_text())
        self.assertEqual(saved["review"]["edited_text"], "Coach corrected words")
        self.assertEqual(transcript["segments"][0]["text"], "Coach corrected words")
        self.assertEqual(transcript["segments"][0]["timing_provenance"], "coach_edited_clip_extent")
        self.assertEqual(transcript["segments"][0]["video_end_s"], 5.5)

    def test_rejected_voice_does_not_publish_transcript(self):
        row = new_activity("existing-run", "voice", 2.0)
        row.update(status="response_received", output_preview="Unclear words", audio_duration_s=1.0)
        save_activity(self.run, row)
        req = request.Request(self.url(f"meta-activity/{row['activity_id']}/review"),
                              b'{"decision":"rejected_observation"}',
                              {"Authorization": "Bearer test-only-token", "Content-Type": "application/json"})
        with request.urlopen(req, timeout=3) as response:
            saved = json.load(response)
        self.assertEqual(saved["review"]["decision"], "rejected_observation")
        self.assertFalse((self.run / "transcript.json").exists())

    def test_truncated_voice_preview_cannot_be_approved_as_full_transcript(self):
        row = new_activity("existing-run", "voice", 2.0)
        row.update(status="response_received", output_preview="Incomplete preview",
                   output_truncated=True, audio_duration_s=1.0)
        save_activity(self.run, row)
        req = request.Request(self.url(f"meta-activity/{row['activity_id']}/review"),
                              b'{"decision":"confirmed_observation"}',
                              {"Authorization": "Bearer test-only-token", "Content-Type": "application/json"})
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(req, timeout=3)
        self.assertEqual(caught.exception.code, 409)
        self.assertFalse((self.run / "transcript.json").exists())

    def test_saved_meta_frame_is_served_only_when_hash_matches(self):
        import hashlib
        row = new_activity("existing-run", "sam", 0.0)
        jpeg = b"\xff\xd8offline-test-frame\xff\xd9"
        row.update(status="response_received", frame_sha256=hashlib.sha256(jpeg).hexdigest())
        save_activity(self.run, row)
        save_activity_frame(self.run, row, jpeg)
        url = self.url(f"meta-activity/{row['activity_id']}/frame.jpg")
        with request.urlopen(request.Request(url, headers={"Authorization": "Bearer test-only-token"}), timeout=3) as response:
            self.assertEqual(response.read(), jpeg)
        (self.run / "meta_activity" / f"{row['activity_id']}.jpg").write_bytes(b"corrupt")
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(request.Request(url, headers={"Authorization": "Bearer test-only-token"}), timeout=3)
        self.assertEqual(caught.exception.code, 409)


if __name__ == "__main__":
    unittest.main()
