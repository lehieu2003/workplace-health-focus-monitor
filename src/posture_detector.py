"""Posture detection and spine/neck angle calculation module."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from src.config import AppConfig

@dataclass
class PostureAngles:
    neck_angle: float
    torso_angle: float
    shoulder_tilt: float
    hip_visible: bool = True
    head_drop_ratio: float = 0.0

@dataclass
class PostureStatusResult:
    is_good: bool
    reason: str
    delta_neck: float
    delta_torso: float
    delta_shoulder: float
    angles: PostureAngles

def calculate_angle_from_vertical(point_top: Tuple[float, float], point_bottom: Tuple[float, float]) -> float:
    """Calculate absolute angle (degrees) between line (point_top -> point_bottom) and vertical axis."""
    dx = abs(point_top[0] - point_bottom[0])
    dy = abs(point_bottom[1] - point_top[1])
    if dy == 0:
        return 90.0
    rad = math.atan2(dx, dy)
    return math.degrees(rad)

def calculate_tilt_angle(point_left: Tuple[float, float], point_right: Tuple[float, float]) -> float:
    """Calculate inclination angle (degrees) between horizontal axis and line connecting left and right points."""
    dx = abs(point_right[0] - point_left[0])
    dy = abs(point_right[1] - point_left[1])
    if dx == 0:
        return 90.0
    rad = math.atan2(dy, dx)
    return math.degrees(rad)

class PostureDetector:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.baseline_neck: Optional[float] = None
        self.baseline_torso: Optional[float] = None
        self.baseline_shoulder: Optional[float] = None
        self.baseline_head_drop: Optional[float] = None
        self.smoothed_angles: Optional[PostureAngles] = None
        self._mp_pose = None
        self._pose = None

    @property
    def is_calibrated(self) -> bool:
        return self.baseline_neck is not None

    def calibrate(self, angles: PostureAngles) -> None:
        """Lock in user's sitting posture as the baseline."""
        self.baseline_neck = angles.neck_angle
        self.baseline_torso = angles.torso_angle
        self.baseline_shoulder = angles.shoulder_tilt
        self.baseline_head_drop = angles.head_drop_ratio

    def reset_calibration(self) -> None:
        self.baseline_neck = None
        self.baseline_torso = None
        self.baseline_shoulder = None
        self.baseline_head_drop = None
        self.smoothed_angles = None

    def compute_angles_from_points(
        self,
        ear_pt: Tuple[float, float],
        shoulder_pt: Tuple[float, float],
        hip_pt: Tuple[float, float],
        left_shoulder: Tuple[float, float],
        right_shoulder: Tuple[float, float],
        hip_visible: bool = True,
        head_drop_ratio: float = 0.0,
    ) -> PostureAngles:
        """Compute neck, torso, and shoulder tilt angles with temporal smoothing."""
        neck_ang = calculate_angle_from_vertical(ear_pt, shoulder_pt)
        torso_ang = calculate_angle_from_vertical(shoulder_pt, hip_pt) if hip_visible else 0.0
        shoulder_tilt = calculate_tilt_angle(left_shoulder, right_shoulder)

        # Exponential Moving Average (EMA) smoothing to eliminate single-frame coordinate jitter
        alpha = 0.35
        if self.smoothed_angles is not None:
            neck_ang = alpha * neck_ang + (1.0 - alpha) * self.smoothed_angles.neck_angle
            torso_ang = alpha * torso_ang + (1.0 - alpha) * self.smoothed_angles.torso_angle
            shoulder_tilt = alpha * shoulder_tilt + (1.0 - alpha) * self.smoothed_angles.shoulder_tilt
            head_drop_ratio = alpha * head_drop_ratio + (1.0 - alpha) * self.smoothed_angles.head_drop_ratio

        angles = PostureAngles(
            neck_angle=neck_ang,
            torso_angle=torso_ang,
            shoulder_tilt=shoulder_tilt,
            hip_visible=hip_visible,
            head_drop_ratio=head_drop_ratio,
        )
        self.smoothed_angles = angles
        return angles

    def evaluate(self, angles: PostureAngles) -> PostureStatusResult:
        """Evaluate if current posture deviates beyond acceptable thresholds."""
        if not self.is_calibrated:
            # If not yet calibrated, use reasonable defaults
            base_neck = 12.0
            base_torso = 8.0
            base_shoulder = 0.0
        else:
            base_neck = self.baseline_neck
            base_torso = self.baseline_torso if self.baseline_torso is not None else 8.0
            base_shoulder = self.baseline_shoulder if self.baseline_shoulder is not None else 0.0

        delta_neck = abs(angles.neck_angle - base_neck)
        delta_torso = abs(angles.torso_angle - base_torso) if angles.hip_visible else 0.0
        delta_shoulder = abs(angles.shoulder_tilt - base_shoulder)

        reasons = []
        if delta_neck > self.config.neck_angle_thresh_deg:
            reasons.append("SLOUCH / FORWARD HEAD")
        # Only evaluate torso leaning if hips are actually visible (not blocked by desk)
        if angles.hip_visible and delta_torso > self.config.torso_angle_thresh_deg:
            reasons.append("TORSO LEANING")
        if delta_shoulder > self.config.shoulder_tilt_thresh_deg:
            reasons.append("SHOULDER ASYMMETRY")
        if self.baseline_head_drop is not None and self.baseline_head_drop > 0.0:
            if angles.head_drop_ratio < (self.baseline_head_drop * 0.70):
                reasons.append("SLOUCH / HEAD DROPPED")

        is_good = len(reasons) == 0
        reason_str = "GOOD" if is_good else " & ".join(reasons)

        return PostureStatusResult(
            is_good=is_good,
            reason=reason_str,
            delta_neck=delta_neck,
            delta_torso=delta_torso,
            delta_shoulder=delta_shoulder,
            angles=angles,
        )

    def init_mediapipe(self) -> None:
        """Lazy load MediaPipe PoseLandmarker task."""
        if self._pose is None:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python

            base_options = mp_python.BaseOptions(model_asset_path=self.config.pose_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_rgb, img_w: int, img_h: int) -> Tuple[Optional[PostureAngles], Optional[dict]]:
        """Run pose detection on RGB image and extract key coordinate points."""
        self.init_mediapipe()
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._pose.detect(mp_image)
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None, None

        landmarks = results.pose_landmarks[0]

        # Landmark indices in MediaPipe Pose
        LEFT_EAR = 7
        RIGHT_EAR = 8
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24

        def to_px(lm):
            return (lm.x * img_w, lm.y * img_h)

        l_ear = to_px(landmarks[LEFT_EAR])
        r_ear = to_px(landmarks[RIGHT_EAR])
        l_sh = to_px(landmarks[LEFT_SHOULDER])
        r_sh = to_px(landmarks[RIGHT_SHOULDER])
        l_hip = to_px(landmarks[LEFT_HIP])
        r_hip = to_px(landmarks[RIGHT_HIP])

        # Check if hips are actually visible or hidden by desk
        l_hip_vis = getattr(landmarks[LEFT_HIP], "visibility", 1.0)
        r_hip_vis = getattr(landmarks[RIGHT_HIP], "visibility", 1.0)
        hip_visible = (l_hip_vis > 0.5) and (r_hip_vis > 0.5)

        mid_ear = ((l_ear[0] + r_ear[0]) / 2.0, (l_ear[1] + r_ear[1]) / 2.0)
        mid_sh = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)

        sh_w = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
        head_drop_ratio = (mid_sh[1] - mid_ear[1]) / (sh_w + 1e-5)

        angles = self.compute_angles_from_points(
            ear_pt=mid_ear,
            shoulder_pt=mid_sh,
            hip_pt=mid_hip,
            left_shoulder=l_sh,
            right_shoulder=r_sh,
            hip_visible=hip_visible,
            head_drop_ratio=head_drop_ratio,
        )

        landmark_points = {
            "mid_ear": mid_ear,
            "mid_shoulder": mid_sh,
            "mid_hip": mid_hip,
            "left_shoulder": l_sh,
            "right_shoulder": r_sh,
            "left_hip": l_hip,
            "right_hip": r_hip,
        }
        return angles, landmark_points
