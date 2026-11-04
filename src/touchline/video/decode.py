from ..diagnostics import diagnosed
from pathlib import Path
from typing import Any
from ..contracts import Frame


@diagnosed("media.probe")
def probe_video(path: str | Path, stream_index: int = 0) -> dict[str, Any]:
    """Probe video stream metadata, duration, geometry, rotation, and time base.

    Does not decode video frames into memory, allowing fast preflight inspection.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install the vision extra and a compatible PyAV runtime first") from exc
    path = Path(path)
    if not path.is_file():
        raise ValueError("video file does not exist")
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if not 0 <= stream_index < len(streams):
            raise ValueError("video stream index out of range")
        stream = streams[stream_index]
        rotation = str(stream.metadata.get("rotate", "0"))
        duration_s = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else (float(container.duration / av.time_base) if container.duration is not None else None)
        )
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        return {
            "path": str(path.resolve()),
            "format_name": container.format.name,
            "stream_index": stream_index,
            "codec_name": stream.codec_context.name if stream.codec_context else None,
            "width": stream.width,
            "height": stream.height,
            "duration_s": duration_s,
            "frame_count": stream.frames,
            "average_fps": fps,
            "time_base_num": stream.time_base.numerator if stream.time_base else None,
            "time_base_den": stream.time_base.denominator if stream.time_base else None,
            "rotation": rotation,
            "is_rotated": rotation not in ("0", "0.0", ""),
        }


def decode_video(
    path: str | Path,
    source_id: str,
    stream_index: int = 0,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
    decoder_threads: int | None = None,
):
    """Sequential PyAV decode, preserving original PTS; raises on corrupt timing/rotation.

    Args:
        path: Path to video file.
        source_id: Unique identifier for the video source.
        stream_index: Video stream index (default: 0).
        start_time_s: Optional presentation time lower bound for clip navigation.
        end_time_s: Optional presentation time upper bound for clip navigation.
    """
    if decoder_threads is not None and (type(decoder_threads) is not int or not 1<=decoder_threads<=8):
        raise ValueError('Explicit decoder threads must be an integer from1 to8')
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install the vision extra and a compatible PyAV runtime first") from exc
    path = Path(path)
    if not path.is_file():
        raise ValueError("video file does not exist")
    if start_time_s is not None and start_time_s < 0:
        raise ValueError("start_time_s must be non-negative")
    if start_time_s is not None and end_time_s is not None and end_time_s < start_time_s:
        raise ValueError("end_time_s must be greater than or equal to start_time_s")

    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if not 0 <= stream_index < len(streams):
            raise ValueError("video stream index out of range")
        stream = streams[stream_index]
        if decoder_threads is not None:
            stream.thread_type='AUTO'
            stream.codec_context.thread_count=decoder_threads
        rotation = str(stream.metadata.get("rotate", "0"))
        if rotation not in ("0", "0.0", ""):
            raise ValueError("rotated sources need an explicit geometry transform; normalize/review before using this starter")
        origin = None
        previous = None
        geometry = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise ValueError(f"missing PTS or corrupt frame at {index}; preserve the failed run for review")
            raw = float(frame.pts * frame.time_base)
            if previous is not None and raw <= previous:
                raise ValueError("non-increasing presentation timestamps")
            previous = raw
            if origin is None:
                origin = raw
                geometry = (frame.width, frame.height)
            if geometry != (frame.width, frame.height):
                raise ValueError("source geometry changed; explicit transform/calibration revision required")

            elapsed = raw - origin
            if start_time_s is not None and elapsed < start_time_s:
                continue
            if end_time_s is not None and elapsed > end_time_s:
                break

            yield Frame(
                source_id,
                index,
                frame.pts,
                frame.time_base.numerator,
                frame.time_base.denominator,
                elapsed,
                frame.width,
                frame.height,
                frame,
            )
