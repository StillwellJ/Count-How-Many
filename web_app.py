"""
Live TripCheck crash-detection dashboard.

Polls cameras, runs inference, and streams each result to the browser via SSE.

  python web_app.py
  python web_app.py --cameras 5 --port 8080
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, render_template

from crash_detector import CrashDetector, PredictionResult, load_config
from tripcheck_fetcher import fetch_camera_image, get_camera_inventory_cached

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FOLDER_LABELS = {
    "no_crash": "No Crash",
    "severity_moderate": "Moderate Accident",
    "severity_severe": "Severe Accident",
    "severity_minor": "Minor Accident",
}

app = Flask(__name__)

_state_lock = threading.Lock()
_subscribers: list[queue.Queue] = []
_worker_status: dict = {
    "running": False,
    "cycle": 0,
    "processed": 0,
    "crash_count": 0,
    "total_cameras": 0,
    "message": "Starting…",
}


def _broadcast(event: dict) -> None:
    with _state_lock:
        dead: list[queue.Queue] = []
        for q in _subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def _set_status(**kwargs) -> None:
    with _state_lock:
        _worker_status.update(kwargs)
    _broadcast({"type": "status", **_worker_status})


def _guess_mime(image_bytes: bytes) -> str:
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


def _run_worker(
    detector: CrashDetector,
    api_key: str,
    config: dict,
    max_cameras: int | None,
    delay_seconds: float,
) -> None:
    output_dir = config.get("output_dir", "output")
    image_timeout = config.get("image_fetch_timeout", 15)
    inventory_refresh = config.get("inventory_refresh_seconds", 86400)
    cache_file = str(Path(output_dir).parent / ".camera_inventory_cache.json")

    _set_status(running=True, message="Loading camera inventory…")

    while True:
        try:
            cameras = get_camera_inventory_cached(
                api_key,
                cache_ttl_seconds=inventory_refresh,
                cache_file=cache_file,
            )
        except Exception as e:
            log.exception("Failed to fetch camera inventory")
            _set_status(message=f"Inventory error: {e}")
            time.sleep(10)
            continue

        if not cameras:
            _set_status(message="No cameras in inventory. Retrying…")
            time.sleep(10)
            continue

        cycle = _worker_status["cycle"] + 1
        limit = min(len(cameras), max_cameras) if max_cameras else len(cameras)
        processed = 0
        crash_count = 0

        _set_status(
            cycle=cycle,
            processed=0,
            crash_count=0,
            total_cameras=limit,
            message=f"Cycle {cycle}: processing {limit} cameras",
        )

        for i, cam in enumerate(cameras):
            if max_cameras is not None and i >= max_cameras:
                break

            cctv_url = cam.get("cctv-url") or cam.get("cctv_url")
            if not cctv_url:
                continue

            device_name = (
                cam.get("device-name") or cam.get("device_name") or f"camera_{i}"
            ).replace("/", "-")

            _set_status(message=f"Fetching {device_name}…")

            image_bytes = fetch_camera_image(cctv_url, timeout=image_timeout)
            if not image_bytes:
                _broadcast(
                    {
                        "type": "skip",
                        "camera": device_name,
                        "message": "Could not download image",
                    }
                )
                continue

            try:
                result = detector.predict_with_details(image_bytes)
            except Exception as e:
                log.exception("Inference failed for %s", device_name)
                result = PredictionResult(folder="no_crash", error=str(e))

            processed += 1
            if result.folder != "no_crash":
                crash_count += 1

            mime = _guess_mime(image_bytes)
            b64 = base64.b64encode(image_bytes).decode("ascii")

            event = {
                "type": "inference",
                "camera": device_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "folder": result.folder,
                "folder_label": FOLDER_LABELS.get(result.folder, result.folder),
                "output_path": f"{output_dir}/{result.folder}/",
                "class_name": result.class_name,
                "confidence": round(result.confidence, 4),
                "probabilities": result.probabilities,
                "error": result.error,
                "image_data_url": f"data:{mime};base64,{b64}",
                "is_crash": result.folder != "no_crash",
            }
            _broadcast(event)

            _set_status(
                processed=processed,
                crash_count=crash_count,
                message=f"Processed {device_name} → {event['folder_label']}",
            )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        _set_status(
            processed=processed,
            crash_count=crash_count,
            message=f"Cycle {cycle} complete ({crash_count} crash detections)",
        )
        time.sleep(config.get("poll_interval_seconds", 60))


def _start_worker(
    detector: CrashDetector,
    api_key: str,
    config: dict,
    max_cameras: int | None,
    delay_seconds: float,
) -> None:
    thread = threading.Thread(
        target=_run_worker,
        args=(detector, api_key, config, max_cameras, delay_seconds),
        daemon=True,
        name="inference-worker",
    )
    thread.start()


@app.route("/")
def index():
    return render_template("live.html")


@app.route("/api/status")
def api_status():
    with _state_lock:
        return dict(_worker_status)


@app.route("/stream")
def stream():
    q: queue.Queue = queue.Queue(maxsize=32)

    with _state_lock:
        _subscribers.append(q)
        q.put({"type": "status", **_worker_status})

    def generate():
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            with _state_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return Response(generate(), mimetype="text/event-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live TripCheck crash detection dashboard.")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000).")
    parser.add_argument(
        "--cameras",
        type=int,
        default=10,
        metavar="N",
        help="Cameras per cycle (default: 10). Use 0 for all.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        metavar="SEC",
        help="Seconds between each camera during a cycle (default: 1.5).",
    )
    args = parser.parse_args()

    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)

    weights_path = os.environ.get("YOLO_WEIGHTS_PATH") or config.get("weights_path")
    if not weights_path or not Path(weights_path).is_file():
        print(f"Error: Weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("TRIPCHECK_API_KEY")
    if not api_key:
        print("Error: TRIPCHECK_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    max_cameras = None if args.cameras == 0 else args.cameras

    log.info("Loading model from %s…", weights_path)
    detector = CrashDetector(weights_path, config)

    _start_worker(detector, api_key, config, max_cameras, args.delay)

    print(f"Live dashboard: http://127.0.0.1:{args.port}")
    if max_cameras:
        print(f"Processing {max_cameras} cameras per cycle ({args.delay}s between each).")
    else:
        print(f"Processing all cameras per cycle ({args.delay}s between each).")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
