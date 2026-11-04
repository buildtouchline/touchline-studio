"""Offline checks for the combined coach UI and deliberately closed upload gate."""

import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


class StudioIntegrationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.runs = Path(self.temporary.name) / "runs"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        code = (
            "from pathlib import Path; from touchline.studio import run_studio_server; "
            "run_studio_server(Path(__import__('sys').argv[1]), port=int(__import__('sys').argv[2]))"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        self.process = subprocess.Popen(
            [sys.executable, "-B", "-c", code, str(self.runs), str(self.port)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                if self.process.poll() is not None:
                    self.fail("Studio exited before serving")
                time.sleep(0.05)
        else:
            self.fail("Studio did not start")

    def tearDown(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.temporary.cleanup()

    def test_ported_modules_and_readiness_are_served(self):
        for path in ("/", "/app.js", "/web/api-client.js", "/web/vikzar-icon.png"):
            with self.subTest(path=path), urlopen(f"http://127.0.0.1:{self.port}{path}") as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
        with urlopen(f"http://127.0.0.1:{self.port}/api/analyze/readiness") as response:
            self.assertEqual(json.load(response)["ready"], False)

    def test_rejected_body_cannot_become_second_keepalive_request(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as sock:
            sock.settimeout(2)
            sock.sendall(
                b"POST /api/analyze HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4\r\n"
                b"Connection: keep-alive\r\n\r\nABCD"
                b"GET /api/runs HTTP/1.1\r\nHost: localhost\r\n\r\n"
            )
            chunks = []
            while True:
                part = sock.recv(4096)
                if not part:
                    break
                chunks.append(part)
        response = b"".join(chunks)
        self.assertEqual(response.count(b"HTTP/1.1 "), 1)
        self.assertIn(b"HTTP/1.1 503", response)
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_malformed_or_missing_upload_length_creates_no_job(self):
        for header in (b"", b"Content-Length: -1\r\n", b"Content-Length: nope\r\n"):
            with self.subTest(header=header), socket.create_connection(("127.0.0.1", self.port), timeout=2) as sock:
                sock.settimeout(2)
                sock.sendall(b"POST /api/analyze HTTP/1.1\r\nHost: localhost\r\n" + header + b"\r\n")
                response = sock.recv(4096)
                self.assertIn(b"HTTP/1.1 503", response)
            self.assertEqual(list(self.runs.iterdir()), [])

    def test_historical_failed_job_remains_failed_after_server_start(self):
        job_id = "upload-123456-abcdef"
        job = self.runs / job_id
        job.mkdir()
        (job / "job_status.json").write_text(json.dumps({"job_id": job_id, "status": "failed"}))
        with urlopen(f"http://127.0.0.1:{self.port}/api/job/{job_id}") as response:
            self.assertEqual(json.load(response)["status"], "failed")

    def test_imported_run_exposes_only_hash_verified_annotated_media(self):
        run = self.runs / "imported-one"
        normalized = run / "normalized"
        normalized.mkdir(parents=True)
        frame_map = (json.dumps({"output_frame_index": 0, "source_relative_time": [0, 1]}) + "\n").encode()
        (normalized / "frame-map.jsonl").write_bytes(frame_map)
        overlay = run / "overlay24.mp4"
        overlay.write_bytes(b"offline-annotated-fixture")
        (run / "source_video.mp4").write_bytes(b"offline-source-fixture")
        manifest = {"run_id": "imported-one", "status": "completed",
                    "frame_map_sha256": hashlib.sha256(frame_map).hexdigest(),
                    "artifacts": {"annotated_video": {"filename": "overlay24.mp4",
                        "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest()}}}
        (run / "manifest.json").write_text(json.dumps(manifest))
        (run / "analytics.json").write_text("{}")
        (run / "observations.jsonl").write_text("")
        url = f"http://127.0.0.1:{self.port}/api/run/imported-one"
        with urlopen(url) as response:
            payload = json.load(response)
        self.assertTrue(payload["annotated_video_verified"])
        self.assertEqual(payload["annotated_video_url"], "/runs/imported-one/overlay24.mp4")
        self.assertEqual(payload["frame_map"][0]["source_relative_time"], [0, 1])
        overlay.write_bytes(b"corrupted-after-import")
        with urlopen(url) as response:
            payload = json.load(response)
        self.assertFalse(payload["annotated_video_verified"])
        self.assertIsNone(payload["annotated_video_url"])


if __name__ == "__main__":
    unittest.main()
