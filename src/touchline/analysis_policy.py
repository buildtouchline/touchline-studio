"""Analysis-first policy. No environment access, provider SDKs or network calls."""
from .diagnostics import diagnosed, diagnostic_progress
from fractions import Fraction
from .video.decode import decode_video, probe_video

# Maximum cumulative project charges, including prior attributable usage.
PROVIDER_CAPS_USD = {"modal": "50", "meta": "75", "baseten": "50", "fireworks": "50"}


def validate_analysis_frames(frames):
    """Validate every frame using exact source PTS, with half-tick quantization tolerance."""
    origin = previous = geometry = None
    count = 0
    for frame in frames:
        current_geometry = (frame.width, frame.height)
        if frame.width <= 0 or frame.height <= 0 or frame.width < frame.height:
            raise ValueError("Analysis requires positive landscape frame dimensions")
        if geometry is None:
            geometry = current_geometry
        if current_geometry != geometry:
            raise ValueError("Analysis requires consistent frame dimensions")
        if type(frame.pts) is not int or frame.time_base_num <= 0 or frame.time_base_den <= 0:
            raise ValueError("Valid source PTS and time base required")
        tick = Fraction(frame.time_base_num, frame.time_base_den)
        # Coarse clocks cannot reliably distinguish the requested capture cadence.
        if tick > Fraction(1, 24):
            raise ValueError("Source clock is too coarse to verify constant 24 FPS")
        pts = frame.pts * tick
        if previous is not None and pts <= previous:
            raise ValueError("Duplicate or reversed source timestamps")
        if origin is None:
            origin = pts
        if abs((pts-origin) - Fraction(count, 24)) > tick / 2:
            raise ValueError("Analysis requires constant 24/1 FPS; gaps/VFR are unsupported")
        previous = pts
        count += 1
        if count % 240 == 0:
            diagnostic_progress('media.preflight',decoded_frames=count)
    if count < 2:
        raise ValueError("At least two decoded frames are needed to verify cadence")
    return {"width": geometry[0], "height": geometry[1], "fps_num": 24, "fps_den": 1,
            "frames_verified": count, "timing_verified": True}


@diagnosed("media.preflight")
def preflight_analysis_video(path):
    """CPU decode the complete selected source before model construction or paid work.

    Low-level decoders retain general-format support for datasets/fixtures. No
    silent conversion occurs: use normalize-video for an explicit 24 FPS copy.
    """
    info = probe_video(path)
    if info["width"] <= 0 or info["height"] <= 0 or info["width"] < info["height"] or info["is_rotated"]:
        raise ValueError("Analysis requires unrotated landscape video; panoramic dimensions are supported")
    if info["average_fps"] != 24.0:
        raise ValueError("Analysis requires exactly 24/1 FPS; create an analysis copy with normalize-video first")
    result = validate_analysis_frames(decode_video(path, "analysis-preflight"))
    return {**result, "policy": "touchline.analysis.source-geometry24.v2",
            "accuracy_target_verified": False}
