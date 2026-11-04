from pathlib import Path
from .contracts import Frame, Box, Detection, CapabilityEnvelope
from .pipeline import run_pipeline, write_json
from .video.sampling import Anchor24Sampler, FixedRateSampler


def synthetic_frames(count=96):
    for i in range(count):
        yield Frame("synthetic-source", i, i, 1, 24, i / 24, 320, 180)


class SyntheticDetector:
    def provenance(self):
        return {"adapter": "synthetic-contract-fixture", "model": "none", "not_field_evidence": True}

    def detect(self, frame):
        x = 20 + frame.index * 0.5
        return [Detection(Box(x, 30, x + 30, 120), 0.9)]


class FailingSyntheticDetector:
    def __init__(self, fail_at_index=2):
        self.fail_at_index = fail_at_index

    def provenance(self):
        return {"adapter": "failing-synthetic-fixture", "model": "none"}

    def detect(self, frame):
        if frame.index >= self.fail_at_index:
            raise RuntimeError("simulated detector runtime failure")
        return [Detection(Box(10, 10, 40, 60), 0.8)]


def generate_complete_run_fixture(output_dir: str | Path, count: int = 48) -> dict:
    """Generate golden fixture for a complete synthetic run."""
    out = Path(output_dir)
    return run_pipeline(
        synthetic_frames(count),
        SyntheticDetector(),
        Anchor24Sampler(),
        out,
        {"id": "synthetic_source_complete", "description": "golden fixture for complete run"},
        synthetic=True,
    )


def generate_partial_run_fixture(output_dir: str | Path, max_selected: int = 2) -> dict:
    """Generate golden fixture for a partial synthetic run with explicit stop reason."""
    out = Path(output_dir)
    return run_pipeline(
        synthetic_frames(48),
        SyntheticDetector(),
        Anchor24Sampler(),
        out,
        {"id": "synthetic_source_partial", "description": "golden fixture for partial run"},
        synthetic=True,
        max_selected=max_selected,
    )


def generate_unsupported_capability_fixture(output_dir: str | Path) -> dict:
    """Generate golden fixture demonstrating explicit unsupported capabilities and machine-readable reasons."""
    out = Path(output_dir)
    manifest = generate_complete_run_fixture(out, count=24)
    manifest["capabilities"] = {
        "physical_metrics": CapabilityEnvelope(
            name="physical_metrics",
            supported=False,
            status="unsupported",
            reason="pitch calibration and confirmed player identity not supplied",
        ).__dict__,
        "3d_pose": CapabilityEnvelope(
            name="3d_pose",
            supported=False,
            status="unsupported",
            reason="3D pose estimation is post-demo research scope (TL-019)",
        ).__dict__,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def generate_failed_run_fixture(output_dir: str | Path) -> dict:
    """Generate golden fixture for an interrupted/failed run preserving partial artifacts and error envelope."""
    out = Path(output_dir)
    try:
        run_pipeline(
            synthetic_frames(24),
            FailingSyntheticDetector(fail_at_index=2),
            FixedRateSampler(2),
            out,
            {"id": "synthetic_source_failed", "description": "golden fixture for failed run"},
            synthetic=True,
        )
    except RuntimeError:
        pass
    import json
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
