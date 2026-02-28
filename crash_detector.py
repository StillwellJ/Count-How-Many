"""
Crash detection using YOLOv8 with custom weights.
Maps model outputs to severity folders from config.
"""
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class CrashDetector:
    def __init__(self, weights_path: str, config: dict):
        self.config = config
        self.model = YOLO(weights_path)
        self.conf_thres = config.get("confidence_threshold", 0.25)
        self.predict_conf_min = config.get("predict_conf_min", 0.01)
        self.severity_map = config.get("severity_folders", {})
        self.use_highest = config.get("use_highest_severity", True)
        self.log_detections = config.get("log_detections", False)
        self._log = logging.getLogger(__name__)

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

    def predict(self, image_path: str | Path) -> str:
        """
        Run crash detection on an image.
        Returns the folder name where the image should be saved:
        - "no_crash" if no crash detected above confidence threshold
        - Otherwise the severity folder (e.g. severity_minor, severity_severe).
        """
        # Use low predict_conf_min so model returns all detections; we filter by conf_thres below
        results = self.model.predict(
            source=str(image_path),
            conf=self.predict_conf_min,
            verbose=False,
        )
        if not results:
            return "no_crash"

        detections = []
        names = self.model.names or {}

        for r in results:
            if r.boxes is None:
                continue
            for i, box in enumerate(r.boxes):
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
            return "no_crash"

        # Filter to crash classes only (exclude "no_crash" if it's a class)
        crash_detections = []
        for cls_id, name, conf in detections:
            folder = self._detection_to_folder(cls_id, name)
            if folder == "no_crash":
                continue
            crash_detections.append((cls_id, name, conf, folder))

        if not crash_detections:
            return "no_crash"

        if self.use_highest_severity:
            # Use highest class index as worst severity (assuming indices are ordered)
            worst = max(crash_detections, key=lambda x: x[0])
            return worst[3]
        return crash_detections[0][3]
