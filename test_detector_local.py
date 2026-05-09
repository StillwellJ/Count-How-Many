"""
Test the crash detector on a single local image (no TripCheck API needed).
Useful to verify weights and severity mapping before running the full pipeline.

Usage:
  python test_detector_local.py path/to/image.jpg
  python test_detector_local.py path/to/image.jpg  # uses config.yaml
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from crash_detector import CrashDetector, load_config

load_dotenv()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_detector_local.py <image_path>", file=sys.stderr)
        print("  Example: python test_detector_local.py test_image.jpg", file=sys.stderr)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    weights_path = os.environ.get("YOLO_WEIGHTS_PATH") or config.get("weights_path")
    if not weights_path or not Path(weights_path).is_file():
        print(f"Error: Weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    detector = CrashDetector(weights_path, config)
    folder = detector.predict(str(image_path))
    print(f"Image: {image_path}")
    print(f"Result folder: {folder}")
    if folder != "no_crash":
        print(f"  -> Would save to: output/{folder}/")
    else:
        print("  -> No crash detected (would save to output/no_crash/)")


if __name__ == "__main__":
    main()
