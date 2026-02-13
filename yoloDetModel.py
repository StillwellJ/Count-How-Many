#!/usr/bin/env python3
import argparse
import time
import hashlib
import requests
import cv2
import numpy as np
from ultralytics import YOLO

import re
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO-12 real time detection on webcam"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO-12 .pt weights file (pretrained or your custom model).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture (default: 0 for built-in/webcam).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square). Lower this if you need more FPS.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--show_fps",
        action="store_true",
        help="Overlay FPS on the output window.",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="TripCheck camera image URL",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval for URL mode (seconds)",
    )
    parser.add_argument(
        "--timelapse",
        type=str,
        help="Run inference on TripCheck camera URL everytime it updates and saves the result",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Folder path to test"
    )
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. "
            "Try a different index like 1 or 2, or check that your webcam is plugged in."
        )

    # Optional: tweak resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)

    return cap

def fetch_tripcheck_image(url: str) -> tuple:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.content
    h = hashlib.md5(data).hexdigest()
    
    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    return image, h

def main():
    args = parse_args()

    print(f"[INFO] Loading YOLO-12 model from: {args.weights}")
    model = YOLO(args.weights)

    if args.test:
        test()

    elif args.url:
        print(f"[INFO] Running in TripCheck mode")
        print(f"[INFO] URL: {args.url}")

        window_name = "YOLO-12 Object Detection (press 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_hash = None
        while True:
            try:
                frame, h = fetch_tripcheck_image(args.url)
                if frame is None:
                    print("[WARN] Failed to decode image")
                    time.sleep(args.interval)
                    continue
                
                if h == last_hash:
                    print("[INFO] Image unchanged, skipping")
                    time.sleep(args.interval)
                    continue

                last_hash = h
                results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False,)

                annotated = results[0].plot()
                cv2.imshow(window_name, annotated)
            
            except Exception as e:
                print(f"[WARN] {e}")
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(args.interval)

    elif args.timelapse:
        time_lapse()

    else:
        print(f"[INFO] Opening camera index {args.camera}")
        cap = open_camera(args.camera)

        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Failed to grab frame from camera")
                    break

                results = model(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    verbose=False,
                )

                annotated_frame = results[0].plot()

                if args.show_fps:
                    current_time = time.time()
                    fps = 1.0 / (current_time - prev_time)
                    prev_time = current_time

                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera and windows closed, exiting.")


def time_lapse():
    args = parse_args()

    print(f"[INFO] Loading YOLO-12 model from: {args.weights}")
    model = YOLO(args.weights)

    # window_name = "YOLO-12 Object Detection (press 'q' to quit)"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.timelapse:
        print(f"[INFO] Running in TripCheck mode with timelapse")
        print(f"[INFO] Path: {args.timelapse}")

        urls = []

        with open(args.timelapse) as file:
            urls = file.readlines()

        hash_dict = {url : None for url in urls}


        while True:
            try:
                for url in urls:
                    frame, h = fetch_tripcheck_image(url)

                    folder_name = re.sub(r".*cams/|_.*|\n", "", url)
                    folder_name = re.sub(r"%20", " ", folder_name)

                    if not os.path.isdir("./timelapse/"+folder_name):
                        os.makedirs("./timelapse/"+folder_name)

                    if frame is None:
                        print("[WARN] Failed to decode image")
                        time.sleep(20)
                        continue
                    
                    if h == hash_dict[url]:
                        print("[INFO] Image unchanged, skipping")
                        time.sleep(20)
                        continue
                        
                    hash_dict[url] = h
                    results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False,)
                    annotated = results[0].plot(labels=True, conf=True)

                    t = time.localtime()
                    time_dir = f"{t.tm_mon}-{t.tm_mday}_{t.tm_hour}-{t.tm_min}"
                    save_location = f"./timelapse/{folder_name}/{time_dir}.jpg"
                    cv2.imwrite(save_location, annotated)


            except Exception as e:
                print(e)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(150)


def test():
    args = parse_args()

    input_path = Path(args.test)
    print(input_path)
    output_path = Path(r"C:\Users\fiery\Desktop\Ares\School\Capstone\Count-How-Many\results")

    output_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    image_files = list(input_path.glob("*.png"))
    print(image_files)

    for image in image_files:
        result = model.predict(
            source=str(image),
            save=True,
            project=output_path,
            name="predictions",
            exist_ok=True
        )
        


if __name__ == "__main__":
    main()
