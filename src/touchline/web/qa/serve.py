"""Loopback-only frontend fixture server; never serves real run data."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import subprocess


WEB = Path(__file__).resolve().parent.parent


def ensure_fixture():
    """Generate an ignored synthetic test pattern when a fresh checkout lacks it."""
    fixture = WEB / "qa" / "fixture.mp4"
    if fixture.is_file():
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to generate the synthetic QA video")
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=640x360:rate=12:duration=4", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(fixture),
    ], check=True)


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def translate_path(self, path):
        clean = path.split("?", 1)[0]
        if clean.startswith("/web/"):
            clean = clean[len("/web/"):]
        else:
            clean = clean.lstrip("/")
        target = (WEB / clean).resolve()
        if not target.is_relative_to(WEB):
            return str(WEB / "__unavailable__")
        return str(target)

    def do_HEAD(self):
        if self.path.split("?", 1)[0] == "/web/qa/fixture.mp4":
            self._fixture_video(head=True)
        else:
            super().do_HEAD()

    def do_GET(self):
        if self.path in ("/", "/demo", "/index.html", "/web/index.html"):
            self.send_response(302)
            self.send_header("Location", "/web/qa/preview.html?scenario=two_sessions&view=overview")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.split("?", 1)[0] == "/web/qa/fixture.mp4":
            self._fixture_video(head=False)
        else:
            super().do_GET()

    def _fixture_video(self, head):
        path = WEB / "qa" / "fixture.mp4"
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        header = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", header) if header else None
        if header and not match:
            self.send_error(416)
            return
        start = int(match.group(1)) if match else 0
        end = min(size - 1, int(match.group(2))) if match and match.group(2) else size - 1
        if start > end or start >= size:
            self.send_error(416)
            return
        self.send_response(206 if match else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        with path.open("rb") as stream:
            stream.seek(start)
            left = end - start + 1
            while left:
                chunk = stream.read(min(left, 1024 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    ensure_fixture()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
