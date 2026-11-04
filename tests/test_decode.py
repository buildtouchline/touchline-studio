import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np

from touchline.fixtures import SyntheticDetector
from touchline.pipeline import run_pipeline
from touchline.video.decode import decode_video, probe_video
from touchline.video.sampling import Anchor24Sampler


def create_test_mp4(
    path: Path,
    frame_count: int = 48,
    fps: int = 24,
    width: int = 64,
    height: int = 64,
) -> Path:
    """Encode a minimal deterministic MP4 file for testing video decode."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(frame_count):
        # Create distinct pattern so frames are valid image data
        img = np.full((height, width, 3), fill_value=(i % 256), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        packet = stream.encode(frame)
        if packet:
            container.mux(packet)

    # Flush remaining packets
    packet = stream.encode()
    if packet:
        container.mux(packet)

    container.close()
    return path


class DecodeTests(unittest.TestCase):
    def test_probe_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp4_path = Path(tmp) / "sample.mp4"
            create_test_mp4(mp4_path, frame_count=24, fps=24, width=128, height=72)

            info = probe_video(mp4_path)
            self.assertEqual(info["width"], 128)
            self.assertEqual(info["height"], 72)
            self.assertEqual(info["stream_index"], 0)
            self.assertFalse(info["is_rotated"])
            self.assertEqual(info["rotation"], "0")
            self.assertIsNotNone(info["duration_s"])
            self.assertAlmostEqual(info["duration_s"], 1.0, delta=0.1)

    def test_decode_video_preserves_pts_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp4_path = Path(tmp) / "sample_48.mp4"
            create_test_mp4(mp4_path, frame_count=48, fps=24, width=64, height=64)

            frames = list(decode_video(mp4_path, source_id="source_test"))
            self.assertEqual(len(frames), 48)

            for idx, frame in enumerate(frames):
                self.assertEqual(frame.source_id, "source_test")
                self.assertEqual(frame.index, idx)
                self.assertEqual(frame.width, 64)
                self.assertEqual(frame.height, 64)
                self.assertIsNotNone(frame.pts)
                self.assertGreater(frame.time_base_den, 0)
                # First frame must be exactly 0.0s elapsed
                if idx == 0:
                    self.assertAlmostEqual(frame.timestamp_s, 0.0)

            # Timestamps must be strictly monotonic
            timestamps = [f.timestamp_s for f in frames]
            self.assertTrue(all(b > a for a, b in zip(timestamps, timestamps[1:])))

    def test_decode_video_clip_navigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp4_path = Path(tmp) / "sample_clip.mp4"
            create_test_mp4(mp4_path, frame_count=48, fps=24, width=64, height=64)

            # Seek interval [0.5s, 1.0s]
            clip_frames = list(
                decode_video(
                    mp4_path,
                    source_id="clip_test",
                    start_time_s=0.5,
                    end_time_s=1.0,
                )
            )
            self.assertGreater(len(clip_frames), 0)
            for f in clip_frames:
                self.assertGreaterEqual(f.timestamp_s, 0.5 - 1e-4)
                self.assertLessEqual(f.timestamp_s, 1.0 + 1e-4)

    def test_decode_video_input_validations(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_file = Path(tmp) / "does_not_exist.mp4"
            with self.assertRaises(ValueError) as cm:
                list(decode_video(missing_file, source_id="test"))
            self.assertIn("video file does not exist", str(cm.exception))

            valid_mp4 = Path(tmp) / "valid.mp4"
            create_test_mp4(valid_mp4, frame_count=10)
            with self.assertRaises(ValueError) as cm:
                list(decode_video(valid_mp4, source_id="test", stream_index=99))
            self.assertIn("stream index out of range", str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                list(decode_video(valid_mp4, source_id="test", start_time_s=-1.0))
            self.assertIn("start_time_s must be non-negative", str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                list(decode_video(valid_mp4, source_id="test", start_time_s=2.0, end_time_s=1.0))
            self.assertIn("end_time_s must be greater than", str(cm.exception))

    def test_decode_integration_with_pipeline_and_anchor_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp4_path = Path(tmp) / "pipeline_real_sample.mp4"
            create_test_mp4(mp4_path, frame_count=48, fps=24, width=320, height=180)

            out_dir = Path(tmp) / "run_out"
            result = run_pipeline(
                decode_video(mp4_path, source_id="real_fixture_source"),
                SyntheticDetector(),
                Anchor24Sampler(),
                out_dir,
                {"id": "real_fixture_source", "path": str(mp4_path)},
                synthetic=False,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["decoded_frames"], 48)
            # 24 FPS with Anchor24Sampler (frames 5, 17, 29, 41) yields exactly 4 frames
            self.assertEqual(result["selected_frames"], 4)
            self.assertFalse(result["synthetic"])

            # Check coverage ledger records real frames
            coverage_records = [
                json.loads(line)
                for line in (out_dir / "coverage.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(coverage_records), 48)
            sampled_indices = [
                r["frame"]["index"]
                for r in coverage_records
                if r["state"] == "detections_present"
            ]
            self.assertEqual(sampled_indices, [5, 17, 29, 41])

            # Check that analytics and review dashboard were generated
            self.assertTrue((out_dir / "analytics.json").is_file())
            self.assertTrue((out_dir / "review.html").is_file())


if __name__ == "__main__":
    unittest.main()
