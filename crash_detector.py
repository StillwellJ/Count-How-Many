"""
Crash detection using Ultralytics YOLO (e.g. trained YOLO11 weights).
TripCheck API frames include headers/footers; optional inference_crop strips them
so predict() runs only on the camera image region.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


@dataclass
class PredictionResult:
    folder: str
    class_id: int | None = None
    class_name: str | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "folder": self.folder,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "error": self.error,
        }


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _apply_inference_crop(img: np.ndarray, crop: dict | None) -> np.ndarray:
    """
    Crop BGR image by fractional margins (0-1 each).
    top/bottom/left/right are fractions of height/width removed from that edge.
    """
    if not crop or not crop.get("enabled", True):
        return img
    h, w = img.shape[:2]
    top = int(h * float(crop.get("top", 0)))
    bottom_rm = int(h * float(crop.get("bottom", 0)))
    left = int(w * float(crop.get("left", 0)))
    right_rm = int(w * float(crop.get("right", 0)))
    y1, y2 = top, h - bottom_rm
    x1, x2 = left, w - right_rm
    if y2 <= y1 or x2 <= x1:
        return img
    return img[y1:y2, x1:x2]


def _load_bgr(source: Union[str, Path, bytes]) -> np.ndarray | None:
    """Load BGR uint8 image from file path or JPEG/PNG bytes."""
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    path = str(source)
    return cv2.imread(path, cv2.IMREAD_COLOR)


class CrashDetector:
    def __init__(self, weights_path: str, config: dict):
        self.config = config
        self.model = YOLO(weights_path)
        self.task = getattr(self.model, "task", "detect")
        self.conf_thres = config.get("confidence_threshold", 0.25)
        self.predict_conf_min = config.get("predict_conf_min", 0.01)
        self.severity_map = config.get("severity_folders", {})
        self.use_highest = config.get("use_highest_severity", True)
        self.log_detections = config.get("log_detections", False)
        self.inference_crop = config.get("inference_crop")
        self._log = logging.getLogger(__name__)

        names = self.model.names or {}
        self._log.info(
            "Loaded weights %s: task=%s, classes=%s",
            weights_path,
            self.task,
            names if len(names) <= 10 else f"{len(names)} classes",
        )
        if self.task == "classify" and len(names) >= 100:
            raise ValueError(
                f"Weights at {weights_path!r} look like a generic ImageNet classifier "
                f"({len(names)} classes), not your crash model. "
                "Point weights_path in config.yaml to your trained crash weights "
                "(e.g. best.pt with moderate-accident / no-accident / severe-accident)."
            )

    def _detection_to_folder(self, class_id: int, class_name: str | None) -> str:
        """Map a detected class to an output folder name."""
        folder = self.severity_map.get(class_id)
        if folder is not None:
            return folder
        if class_name is not None:
            folder = self.severity_map.get(class_name)
            if folder is not None:
                return folder
        # Default: treat unknown crash class as severity_moderate
        return "severity_moderate"

    def _prepare_bgr(self, source: Union[str, Path, bytes]) -> np.ndarray | None:
        img = _load_bgr(source)
        if img is None:
            return None
        return _apply_inference_crop(img, self.inference_crop)

    def predict(self, source: Union[str, Path, bytes]) -> str:
        """Return output folder name (no_crash or severity_*)."""
        return self.predict_with_details(source).folder

    def predict_with_details(self, source: Union[str, Path, bytes]) -> PredictionResult:
        """
        Run inference and return folder plus class name, confidence, and probabilities.
        """
        img = self._prepare_bgr(source)
        if img is None:
            return PredictionResult(folder="no_crash", error="Could not decode image")

        results = self.model.predict(
            source=img,
            conf=self.predict_conf_min,
            verbose=False,
        )
        if not results:
            return PredictionResult(folder="no_crash", error="No model output")

        if self.task == "classify":
            return self._predict_classify_details(results[0])
        return self._predict_detect_details(results)

    def _class_probabilities(self, result) -> dict[str, float]:
        if result.probs is None:
            return {}
        names = self.model.names or {}
        return {
            names.get(i, str(i)): float(result.probs.data[i])
            for i in range(len(result.probs.data))
        }

    def _predict_classify_details(self, result) -> PredictionResult:
        if result.probs is None:
            return PredictionResult(folder="no_crash", error="No classification probabilities")

        names = self.model.names or {}
        top1 = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        name = names.get(top1, "")
        probs = self._class_probabilities(result)
        folder = self._detection_to_folder(top1, name)

        if self.log_detections:
            self._log.info(
                "Classification: class_id=%s name=%s conf=%.3f -> folder=%s",
                top1,
                name,
                conf,
                folder,
            )

        if conf < self.conf_thres:
            return PredictionResult(
                folder="no_crash",
                class_id=top1,
                class_name=name,
                confidence=conf,
                probabilities=probs,
            )

        if folder == "no_crash":
            return PredictionResult(
                folder="no_crash",
                class_id=top1,
                class_name=name,
                confidence=conf,
                probabilities=probs,
            )

        return PredictionResult(
            folder=folder,
            class_id=top1,
            class_name=name,
            confidence=conf,
            probabilities=probs,
        )

    def _predict_detect_details(self, results) -> PredictionResult:
        detections = []
        names = self.model.names or {}

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_thres:
                    continue
                cls_id = int(box.cls[0])
                name = names.get(cls_id, "")
                detections.append((cls_id, name, conf))

        if self.log_detections and detections:
            for cls_id, name, conf in detections:
                folder = self._detection_to_folder(cls_id, name)
                self._log.info("Detection: class_id=%s name=%s conf=%.3f -> folder=%s", cls_id, name, conf, folder)

        if not detections:
            return PredictionResult(folder="no_crash")

        crash_detections = []
        for cls_id, name, conf in detections:
            folder = self._detection_to_folder(cls_id, name)
            if folder == "no_crash":
                continue
            crash_detections.append((cls_id, name, conf, folder))

        if not crash_detections:
            best = max(detections, key=lambda x: x[2])
            return PredictionResult(
                folder="no_crash",
                class_id=best[0],
                class_name=best[1],
                confidence=best[2],
            )

        if self.use_highest:
            worst = max(crash_detections, key=lambda x: x[0])
        else:
            worst = crash_detections[0]

        return PredictionResult(
            folder=worst[3],
            class_id=worst[0],
            class_name=worst[1],
            confidence=worst[2],
        )

    def _predict_classify(self, result) -> str:
        """Whole-image classification (YOLO-cls): top-1 class above confidence threshold."""
        return self._predict_classify_details(result).folder

    def _predict_detect(self, results) -> str:
        """Object detection (YOLO detect): bounding boxes above confidence threshold."""
        return self._predict_detect_details(results).folder
