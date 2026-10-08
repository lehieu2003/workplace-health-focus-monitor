"""Smartphone distraction detector using YOLOv8n."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.config import AppConfig

# Point configs to safe temp/cache paths to avoid macOS permission warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

@dataclass
class DetectedObject:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    label: str

class PhoneDetector:
    # COCO dataset class index for cell phone is 67
    CELL_PHONE_CLASS_ID = 67

    def __init__(self, config: Optional[AppConfig] = None, load_model: bool = True):
        self.config = config or AppConfig()
        self.model = None
        self.load_model = load_model
        self.cached_detections: List[DetectedObject] = []
        if load_model:
            self._init_model()

    def _init_model(self) -> None:
        if self.model is None and self.load_model:
            from ultralytics import YOLO
            # Lightweight nano model (~6MB) optimized for high-FPS inference
            self.model = YOLO("yolov8n.pt")

    def should_run_inference(self, frame_count: int) -> bool:
        """Determine if YOLO inference should run on this frame or reuse cached."""
        return (frame_count % self.config.phone_skip_frames) == 0

    def filter_detections(self, raw_boxes: List[Tuple[list, float, int]]) -> List[DetectedObject]:
        """Filter raw bounding boxes for cell phone with confidence >= threshold."""
        results = []
        for box, conf, cls_id in raw_boxes:
            if cls_id == self.CELL_PHONE_CLASS_ID and conf >= self.config.phone_conf_threshold:
                x1, y1, x2, y2 = [int(v) for v in box]
                results.append(DetectedObject(
                    box=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    label="cell phone"
                ))
        return results

    def detect(self, frame, frame_count: int) -> Tuple[bool, List[DetectedObject]]:
        """
        Run inference on frame with frame skipping.
        Returns (is_phone_detected, list_of_detected_phones).
        """
        if not self.load_model:
            return False, []

        if not self.should_run_inference(frame_count) and self.cached_detections:
            return len(self.cached_detections) > 0, self.cached_detections

        self._init_model()
        results = self.model(frame, verbose=False, conf=self.config.phone_conf_threshold)

        raw_boxes = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                raw_boxes.append((xyxy, conf, cls_id))

        self.cached_detections = self.filter_detections(raw_boxes)
        return len(self.cached_detections) > 0, self.cached_detections

