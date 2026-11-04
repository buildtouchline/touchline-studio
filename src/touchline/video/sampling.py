import math


class FixedRateSampler:
    """Select the first available frame at/after each due PTS; no invented frames."""
    def __init__(self, fps=6.0):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.fps, self.next_tick, self.last_time = fps, 0, None

    def select(self, index, timestamp_s):
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("invalid presentation time")
        if self.last_time is not None and timestamp_s <= self.last_time:
            raise ValueError("presentation times must increase")
        self.last_time = timestamp_s
        if timestamp_s + 1e-9 < self.next_tick / self.fps:
            return False
        self.next_tick = math.floor((timestamp_s + 1e-9) * self.fps) + 1
        return True


class Anchor24Sampler:
    """Sixth/eighteenth ordinal frames, checking actual decoded source timing."""
    def __init__(self, tolerance_s=0.0001):
        self.last_index = -1
        self.tolerance_s = tolerance_s

    def select(self, index, timestamp_s):
        if index != self.last_index + 1:
            raise ValueError("anchor sampler must inspect every decoded frame")
        if not math.isfinite(timestamp_s) or abs(timestamp_s - index/24) > self.tolerance_s:
            raise ValueError("source is not verified constant 24 FPS; use timestamp-based sampling")
        self.last_index = index
        return index % 24 in (5, 17)


def merge_intervals(intervals, padding_s=0.0, duration_s=None):
    if not math.isfinite(padding_s) or padding_s < 0:
        raise ValueError("padding must be finite/nonnegative")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s < 0):
        raise ValueError("invalid duration")
    prepared = []
    for start, end in intervals:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("invalid half-open source interval")
        start, end = max(0, start-padding_s), end+padding_s
        if duration_s is not None:
            end = min(end, duration_s)
        if start < end:
            prepared.append((start, end))
    result = []
    for start, end in sorted(prepared):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result
