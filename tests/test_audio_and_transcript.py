import io
import json
from pathlib import Path
import tempfile
import unittest
import wave

from touchline.analytics import compute_track_analytics
from touchline.audio import (
    generate_tactical_coach_analysis_with_meta_muse,
    load_meta_key,
    parse_meta_muse_response,
    save_meta_key,
    transcribe_coach_audio,
    transcribe_with_meta_muse,
    align_audio_segment_to_video,
    convert_audio_to_wav16k,
    load_sarvam_key,
    mask_api_key,
    parse_sarvam_response,
    save_sarvam_key,
    transcribe_with_sarvam,
)
from touchline.metrics import Calibration
from review_fixtures import review_entry
from touchline.viewer import render_inspection_dashboard


def _wav_bytes(seconds=1.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


FRAMES = [{"index": i, "timestamp_s": i * 0.5, "pts": i * 1000} for i in range(20)]


class AudioAlignmentTests(unittest.TestCase):
    def test_alignment_maps_audio_to_video_and_nearest_pts(self):
        r = align_audio_segment_to_video(1.0, 2.0, video_anchor_start_s=3.0, frame_lookup=FRAMES)
        self.assertEqual(r["video_start_s"], 4.0)
        self.assertEqual(r["video_end_s"], 5.0)
        self.assertEqual(r["nearest_frame_pts"], 8000)
        self.assertEqual(r["audio_start_s"], 1.0)

    def test_word_timestamps_grouped_into_segments(self):
        raw = {
            "transcript": "Scan before receiving. Press now.",
            "timestamps": {
                "words": ["Scan", "before", "receiving.", "Press", "now."],
                "start_time_seconds": [0.0, 0.4, 0.8, 3.0, 3.4],
                "end_time_seconds": [0.3, 0.7, 1.2, 3.3, 3.7],
            },
        }
        segs = parse_sarvam_response(raw, audio_duration_s=4.0, video_anchor_start_s=2.0, frame_lookup=FRAMES)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["text"], "Scan before receiving.")
        self.assertEqual(segs[1]["video_start_s"], 5.0)
        self.assertFalse(segs[0]["speaker_verified"])

    def test_transcribe_with_fake_opener_writes_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            wav.write_bytes(_wav_bytes(2.0))
            seen = {}

            def opener(req, timeout=0):
                seen["key"] = req.get_header("Api-subscription-key")
                return _FakeResp({"transcript": "Good pass. Check shoulder."})

            out = transcribe_with_sarvam(
                wav, "secret-key", video_anchor_start_s=1.0, frame_lookup=FRAMES,
                manual_cues=[{"audio_offset_s": 0.5, "note": "cue"}], opener=opener,
            )
            self.assertEqual(seen["key"], "secret-key")
            self.assertEqual(out["schema_version"], "touchline.transcript.v1")
            self.assertEqual(len(out["segments"]), 2)
            self.assertEqual(out["manual_cues"][0]["video_start_s"], 1.5)

    def test_key_saved_outside_repo_and_masked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_sarvam_key(Path(tmp), "abcd1234wxyz")
            self.assertTrue(path.is_file())
            import os
            old = os.environ.pop("SARVAM_API_KEY", None)
            try:
                self.assertEqual(load_sarvam_key(Path(tmp)), "abcd1234wxyz")
            finally:
                if old is not None:
                    os.environ["SARVAM_API_KEY"] = old
            self.assertEqual(mask_api_key("abcd1234wxyz"), "abcd...wxyz")

    def test_wav_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = convert_audio_to_wav16k(_wav_bytes(1.0, 44100), Path(tmp) / "o.wav")
            self.assertEqual(info["sample_rate_hz"], 16000)
            self.assertAlmostEqual(info["duration_s"], 1.0, delta=0.05)


class IdentityReviewRefTests(unittest.TestCase):
    def _obs(self):
        return [
            {"local_track_id": 5, "bbox_xyxy": [10.0 + i, 20.0, 30.0 + i, 60.0], "detector_score": 0.9,
             "frame": {"index": i, "timestamp_s": i * 0.25, "pts": i, "width": 320, "height": 180}}
            for i in range(3)
        ]

    def _cal(self):
        return Calibration(
            matrix=(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0), independent_check_ref="x",
            valid_from_s=0.0, valid_until_s=10.0, image_bounds=(0, 0, 320, 180), source_geometry=(320, 180),
        )

    def test_unconfirmed_track_has_no_fabricated_review_ref(self):
        t = compute_track_analytics(5, self._obs(), 320, 180, calibration=self._cal())
        self.assertFalse(t["identity_reviewed"])
        self.assertIsNone(t["identity_review_ref"])

    def test_confirmed_association_sets_review_ref(self):
        t = compute_track_analytics(5, self._obs(), 320, 180, calibration=self._cal(),
                                    run_id="fixture-run",association=review_entry("fixture-run",5,revision=2))
        self.assertTrue(t["identity_reviewed"])
        self.assertTrue(t["identity_review_ref"].startswith("association-sha256:"))

    def test_geometry_mismatch_suppresses_physical_metrics(self):
        t = compute_track_analytics(5, self._obs(), 640, 360, calibration=self._cal())
        self.assertEqual(t["movement"]["status"], "unavailable")
        self.assertIn("geometry", t["movement"]["reason"])


class ViewerTranscriptTests(unittest.TestCase):
    def test_dashboard_embeds_mic_panel_and_transcript(self):
        from touchline.analytics import generate_match_analytics
        obs = IdentityReviewRefTests()._obs()
        analytics = generate_match_analytics({"run_id": "r"}, obs)
        html_out = render_inspection_dashboard({"run_id": "r"}, analytics, obs,
                                               transcript={"segments": [{"text": "hello", "video_start_s": 0.25}]})
        self.assertIn("micDeviceSelect", html_out)
        self.assertIn("PRELIMINARY / UNVALIDATED", html_out)
        self.assertIn('"hello"', html_out)
        render_body = html_out.split("function render() {", 1)[1].split("// Playback Controls", 1)[0]
        self.assertEqual(render_body.count("const frameIdx"), 1)




class MetaMuseAudioAndTacticsTests(unittest.TestCase):
    def test_meta_key_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            save_meta_key(p, "test_meta_key_12345")
            self.assertEqual(load_meta_key(p), "test_meta_key_12345")

    def test_parse_meta_muse_response_with_words(self):
        raw = {
            "text": "Marcus accelerate into the gap.",
            "words": [
                {"word": "Marcus", "start": 0.0, "end": 0.4},
                {"word": "accelerate", "start": 0.5, "end": 1.2},
                {"word": "into", "start": 1.3, "end": 1.5},
                {"word": "the", "start": 1.6, "end": 1.8},
                {"word": "gap.", "start": 1.9, "end": 2.3},
            ],
        }
        segs = parse_meta_muse_response(raw, audio_duration_s=3.0, video_anchor_start_s=10.0, frame_lookup=FRAMES)
        self.assertTrue(len(segs) >= 1)
        self.assertIn("Marcus", segs[0]["text"])
        self.assertEqual(segs[0]["video_start_s"], 10.0)

    def test_transcribe_with_meta_muse_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            w_file = Path(tmp) / "coach.wav"
            w_file.write_bytes(_wav_bytes(seconds=1.0))
            fake_payload = {
                "text": "Good positioning Bruno.",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Good positioning Bruno."}]
            }
            def fake_opener(req, timeout=45):
                return _FakeResp(fake_payload)

            res = transcribe_with_meta_muse(
                w_file,
                "test_meta_key",
                video_anchor_start_s=5.0,
                frame_lookup=FRAMES,
                opener=fake_opener,
            )
            self.assertEqual(res["provider"], "meta_muse")
            self.assertEqual(res["model"], "muse-voice-transcribe")
            self.assertEqual(len(res["segments"]), 1)
            self.assertEqual(res["segments"][0]["video_start_s"], 5.0)

    def test_transcribe_coach_audio_dispatcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            w_file = Path(tmp) / "coach.wav"
            w_file.write_bytes(_wav_bytes(seconds=1.0))
            fake_payload = {"text": "Press now.", "segments": [{"start": 0.0, "end": 0.8, "text": "Press now."}]}
            def fake_opener(req, timeout=45):
                return _FakeResp(fake_payload)

            # Meta dispatch
            res_meta = transcribe_coach_audio(
                w_file,
                api_key="meta_fake_key",
                provider="meta",
                video_anchor_start_s=2.0,
                frame_lookup=FRAMES,
                opener=fake_opener,
            )
            self.assertEqual(res_meta["provider"], "meta_muse")

    def test_generate_tactical_coach_analysis_with_meta_muse(self):
        fake_llm_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "tactical_summary": "High intensity pressing and excellent half-space exploitation.",
                            "key_strengths": ["Rapid transition speed", "Direct attacking mindset"],
                            "areas_for_growth": ["Sustained defensive shape", "Patience in possession"],
                            "training_drills": [
                                {"drill_name": "3v2 Counter-Press Transition", "objective": "Win back in 5s", "reps": "6 reps"}
                            ]
                        })
                    }
                }
            ]
        }
        def fake_opener(req, timeout=45):
            return _FakeResp(fake_llm_response)

        res = generate_tactical_coach_analysis_with_meta_muse(
            player_name="Marcus Rashford",
            physical_metrics={"distance_m": 8.43, "peak_speed_mps": 2.55},
            aligned_voice_notes=["Marcus accelerate into the half space"],
            api_key="fake_key",
            opener=fake_opener,
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["provider"], "meta_muse")
        self.assertIn("tactical_summary", res["tactical_assessment"])
        self.assertEqual(len(res["tactical_assessment"]["training_drills"]), 1)


if __name__ == "__main__":
    unittest.main()
