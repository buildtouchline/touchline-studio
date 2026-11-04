"""Import already completed, hash-verified panorama output for local coach review.

Operator-only offline command. Never invokes Modal or reads credentials.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from touchline.panorama_import import import_completed_panorama


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export_dir", "source_video", "normalized_video", "frame_map",
                 "normalization_metadata", "runs_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--expected-original-sha256", required=True)
    args = parser.parse_args()
    run = import_completed_panorama(
        export_dir=args.export_dir, source_video=args.source_video,
        normalized_video=args.normalized_video, frame_map=args.frame_map,
        normalization_metadata=args.normalization_metadata,
        expected_original_sha256=args.expected_original_sha256,
        runs_root=args.runs_root,
    )
    print(f"Imported verified local run: {run.name}")


if __name__ == "__main__":
    main()
