"""
TripCheck crash detection pipeline.
Continuously polls all TripCheck cameras, runs YOLOv8 crash detection,
and saves images to folders by result: no_crash or by severity.

Test with one cycle and limited cameras:
  python main.py --test
  python main.py --test --cameras 3
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import yaml

from tripcheck_fetcher import get_camera_inventory_cached, fetch_camera_image
from crash_detector import CrashDetector, load_config

load_dotenv()

# Show per-region camera counts and any failed regions when fetching inventory
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_dirs(output_dir: str, severity_folders: dict) -> None:
    """Create output directory and all severity subfolders."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for folder in set(severity_folders.values()):
        Path(output_dir, folder).mkdir(parents=True, exist_ok=True)


def run_one_cycle(
    cameras: list,
    detector: CrashDetector,
    config: dict,
    output_dir: str,
    image_timeout: int,
    max_cameras: int | None,
) -> tuple[int, int]:
    """Process one cycle of cameras. Returns (processed_count, crash_count)."""
    processed = 0
    crash_count = 0
    for i, cam in enumerate(cameras):
        if max_cameras is not None and i >= max_cameras:
            break
        cctv_url = cam.get("cctv-url") or cam.get("cctv_url")
        if not cctv_url:
            continue

        device_id = cam.get("device-id") or cam.get("device_id") or i
        device_name = (cam.get("device-name") or cam.get("device_name") or f"camera_{device_id}").replace("/", "-")

        image_bytes = fetch_camera_image(cctv_url, timeout=image_timeout)
        if not image_bytes:
            continue

        tmp_path = Path(output_dir).parent / ".tmp_capture.jpg"
        try:
            tmp_path.write_bytes(image_bytes)
        except OSError:
            continue

        try:
            folder_name = detector.predict(str(tmp_path))
        except Exception as e:
            print(f"  {device_name}: inference error - {e}")
            folder_name = "no_crash"
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

        dest_dir = Path(output_dir) / folder_name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        include_severity = config.get("filename_include_severity", False)
        if include_severity and folder_name != "no_crash":
            safe_name = f"{device_name}_{timestamp}_{folder_name}.jpg"
        else:
            safe_name = f"{device_name}_{timestamp}.jpg"
        dest_path = dest_dir / safe_name

        if folder_name == "no_crash":
            continue

        try:
            dest_path.write_bytes(image_bytes)
        except OSError as e:
            print(f"  {device_name}: could not write {dest_path}: {e}")
            continue

        processed += 1
        if folder_name != "no_crash":
            crash_count += 1
            print(f"  Crash detected [{folder_name}]: {dest_path}")

    return processed, crash_count


def main() -> None:
    parser = argparse.ArgumentParser(description="TripCheck crash detection – poll cameras and run YOLOv8.")
    parser.add_argument("--test", action="store_true", help="Run one cycle then exit (for testing).")
    parser.add_argument("--cameras", type=int, default=None, metavar="N", help="Limit to first N cameras per cycle (for quick tests).")
    args = parser.parse_args()

    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)

    weights_path = os.environ.get("YOLO_WEIGHTS_PATH") or config.get("weights_path")
    if not weights_path or not Path(weights_path).is_file():
        print(
            f"Error: Weights file not found: {weights_path}. "
            "Place your YOLOv8 .pt weights and set weights_path in config.yaml or YOLO_WEIGHTS_PATH in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("TRIPCHECK_API_KEY")
    if not api_key:
        print(
            "Error: TRIPCHECK_API_KEY not set. Get a key from https://apiportal.odot.state.or.us/product#product=tripcheck-api-data",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = config.get("output_dir", "output")
    severity_folders = config.get("severity_folders", {})
    ensure_dirs(output_dir, severity_folders)

    poll_interval = config.get("poll_interval_seconds", 60)
    image_timeout = config.get("image_fetch_timeout", 15)
    inventory_refresh = config.get("inventory_refresh_seconds", 86400)
    cache_file = Path(output_dir).parent / ".camera_inventory_cache.json"
    max_cameras = args.cameras

    detector = CrashDetector(weights_path, config)

    if args.test:
        print("Test mode: one cycle then exit.")
        if max_cameras:
            print(f"Limiting to first {max_cameras} cameras.")
    else:
        print("TripCheck crash detection started. Polling all cameras continuously.")
    print(f"Output dir: {output_dir} | Weights: {weights_path}")

    cycle = 0
    while True:
        cycle += 1
        try:
            cameras = get_camera_inventory_cached(
                api_key,
                cache_ttl_seconds=inventory_refresh,
                cache_file=str(cache_file),
            )
        except Exception as e:
            print(f"Failed to fetch camera inventory: {e}")
            if args.test:
                sys.exit(1)
            time.sleep(poll_interval)
            continue

        if not cameras:
            print("No cameras in inventory. Retrying later.")
            if args.test:
                sys.exit(1)
            time.sleep(poll_interval)
            continue

        print(f"Cycle {cycle}: Processing {len(cameras)} cameras" + (f" (first {max_cameras})" if max_cameras else "") + ".")
        processed, crash_count = run_one_cycle(
            cameras, detector, config, output_dir, image_timeout, max_cameras
        )
        print(f"  Done: {processed} images saved, {crash_count} crash detections.")

        if args.test:
            print("Test complete. Exiting.")
            break
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
