#!/usr/bin/env python3
import argparse
import time

import cv2
from ultralytics import YOLO


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


def main():
    args = parse_args()

    print(f"[INFO] Loading YOLO-12 model from: {args.weights}")
    model = YOLO(args.weights)

    print(f"[INFO] Opening camera index {args.camera}")
    cap = open_camera(args.camera)

    window_name = "YOLO-12 Object Detection (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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


if __name__ == "__main__":
    main()
