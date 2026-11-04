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

from touchline.meta_activity import new_activity, save_activity


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


if __name__ == "__main__":
    unittest.main()
