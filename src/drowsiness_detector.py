"""Drowsiness and fatigue detection using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)."""

import math
from enum import Enum
from typing import List, Optional, Tuple
from src.config import AppConfig

class DrowsinessStatus(Enum):
    ALERT = "ALERT"
    EYES_CLOSED = "EYES_CLOSED"
    YAWNING = "YAWNING"

def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) from 6 landmark points.
    eye_landmarks = [p1, p2, p3, p4, p5, p6]
    where p1, p4 are horizontal eye corners; p2,p6 and p3,p5 are vertical eyelid points.
    """
    if len(eye_landmarks) < 6:
        return 0.30

    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_mar(mouth_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) from 8 inner/outer mouth landmarks.
    m1, m5: left and right corners.
    m2-m8, m3-m7, m4-m6: upper/lower lip points.
    """
    if len(mouth_landmarks) < 8:
        return 0.0

    m1, m2, m3, m4, m5, m6, m7, m8 = mouth_landmarks[:8]
    v1 = euclidean_dist(m2, m8)
    v2 = euclidean_dist(m3, m7)
    v3 = euclidean_dist(m4, m6)
    h = euclidean_dist(m1, m5)

    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (2.0 * h)

class DrowsinessDetector:
    # MediaPipe Face Mesh landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Inner mouth landmarks: 78, 81, 13, 311, 308, 402, 14, 178 (aperture is 0 when mouth is closed)
    MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.eye_closed_start_time: Optional[float] = None
        self.yawn_start_time: Optional[float] = None
        self._mp_face_mesh = None
        self._face_mesh = None

    def update_metrics(
        self,
        avg_ear: float,
        mar: float,
        current_time: float,
        jaw_open: float = 0.0,
        avg_blink: float = 0.0,
    ) -> DrowsinessStatus:
        """Evaluate EAR/MAR and blendshapes against temporal duration thresholds."""
        # Check eyes closed:
        # Either blendshape detects eye closed (avg_blink >= blink_threshold)
        # OR geometric EAR drops below ear_threshold (micro-sleep)
        is_eye_closed = (avg_blink >= self.config.blink_threshold) or (avg_ear < self.config.ear_threshold)
        if is_eye_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
            elif (current_time - self.eye_closed_start_time) >= self.config.ear_time_threshold_sec:
                return DrowsinessStatus.EYES_CLOSED
        else:
            self.eye_closed_start_time = None

        # Check yawning:
        # Requires either genuine jaw drop (jaw_open >= jaw_open_threshold)
        # OR very large mouth aspect ratio (mar >= mar_threshold)
        is_yawning = (jaw_open >= self.config.jaw_open_threshold) or (mar >= self.config.mar_threshold)
        if is_yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
            elif (current_time - self.yawn_start_time) >= self.config.mar_time_threshold_sec:
                return DrowsinessStatus.YAWNING
        else:
            self.yawn_start_time = None

        return DrowsinessStatus.ALERT

    def init_mediapipe(self) -> None:
        """Lazy load MediaPipe FaceLandmarker task with Blendshapes enabled."""
        if self._face_mesh is None:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python

            base_options = mp_python.BaseOptions(model_asset_path=self.config.face_model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_mesh = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_rgb, img_w: int, img_h: int, current_time: float) -> Tuple[DrowsinessStatus, float, float]:
        """Process RGB frame, calculate EAR & MAR/Blendshapes, and return current fatigue status."""
        self.init_mediapipe()
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._face_mesh.detect(mp_image)
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return DrowsinessStatus.ALERT, 0.30, 0.0

        landmarks = results.face_landmarks[0]

        def get_coords(indices):
            return [(landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in indices]

        left_eye_pts = get_coords(self.LEFT_EYE_INDICES)
        right_eye_pts = get_coords(self.RIGHT_EYE_INDICES)
        mouth_pts = get_coords(self.MOUTH_INDICES)

        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(mouth_pts)

        # Extract MediaPipe blendshapes for jawOpen and eyeBlink
        jaw_open = 0.0
        avg_blink = 0.0
        if results.face_blendshapes and len(results.face_blendshapes) > 0:
            scores = {cat.category_name: cat.score for cat in results.face_blendshapes[0]}
            jaw_open = scores.get("jawOpen", 0.0)
            blink_l = scores.get("eyeBlinkLeft", 0.0)
            blink_r = scores.get("eyeBlinkRight", 0.0)
            avg_blink = (blink_l + blink_r) / 2.0

        effective_mar = max(mar, jaw_open)
        status = self.update_metrics(
            avg_ear=avg_ear,
            mar=mar,
            current_time=current_time,
            jaw_open=jaw_open,
            avg_blink=avg_blink,
        )
        return status, avg_ear, effective_mar
