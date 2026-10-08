"""Configuration and constants for Workplace Health & Focus Monitor."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

class SystemStatus(Enum):
    GOOD_POSTURE = "GOOD POSTURE & FOCUSED"
    BAD_POSTURE = "BAD POSTURE: SLOUCH DETECTED"
    DROWSY_EYES = "DROWSY: EYES CLOSED"
    DROWSY_YAWN = "DROWSY: YAWNING DETECTED"
    PHONE_DISTRACTION = "DISTRACTED: PHONE IN USE"
    NO_PERSON = "NO PERSON DETECTED"
    CALIBRATING = "CALIBRATING BASELINE..."

@dataclass
class ColorPalette:
    # Colors in BGR format for OpenCV
    GOOD_GREEN: Tuple[int, int, int] = (46, 204, 113)      # #2ecc71
    BAD_RED: Tuple[int, int, int] = (60, 60, 231)          # #e74c3c
    ALERT_AMBER: Tuple[int, int, int] = (22, 160, 243)     # #f39c12
    PHONE_PURPLE: Tuple[int, int, int] = (189, 75, 142)    # #8e44ad
    DARK_BG: Tuple[int, int, int] = (25, 25, 25)           # Overlay background
    PANEL_BG: Tuple[int, int, int] = (40, 40, 40)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    GRAY: Tuple[int, int, int] = (170, 170, 170)
    CYAN: Tuple[int, int, int] = (240, 200, 0)

@dataclass
class AppConfig:
    camera_source: Union[int, str] = 0
    camera_index: int = 0  # Backward-compatible alias
    frame_width: int = 1280
    frame_height: int = 720
    
    # Posture angular thresholds (degrees difference from calibrated baseline)
    neck_angle_thresh_deg: float = 18.0
    torso_angle_thresh_deg: float = 15.0
    shoulder_tilt_thresh_deg: float = 12.0
    posture_debounce_sec: float = 2.0        # Sustained seconds of bad posture before alarming
    
    # Drowsiness thresholds (geometric + AI blendshapes)
    ear_threshold: float = 0.16              # Eye Aspect Ratio threshold (below this is closed)
    blink_threshold: float = 0.65            # MediaPipe Blendshapes eyeBlink score (0..1)
    ear_time_threshold_sec: float = 2.5      # Sustained seconds to trigger micro-sleep drowsiness
    
    mar_threshold: float = 0.75              # Mouth Aspect Ratio threshold for yawning
    jaw_open_threshold: float = 0.55         # MediaPipe Blendshapes jawOpen score (0..1)
    mar_time_threshold_sec: float = 2.5      # Sustained seconds of yawning
    
    # Distraction / Phone detection
    phone_conf_threshold: float = 0.60       # YOLO confidence threshold for phone
    phone_skip_frames: int = 3               # Run YOLO every N frames to maintain high FPS
    
    # Model paths
    pose_model_path: str = "assets/models/pose_landmarker.task"
    face_model_path: str = "assets/models/face_landmarker.task"
    yolo_model_path: str = "yolov8n.pt"

    # Alert debouncing
    alert_cooldown_sec: float = 2.0          # Minimum seconds between consecutive beeps
    mac_sound_path: str = "/System/Library/Sounds/Ping.aiff"
    mac_sound_warning: str = "/System/Library/Sounds/Tink.aiff"

    # Default video transformations from env
    rotate_deg: int = 0
    no_mirror: bool = False
    enable_sound: bool = True

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "AppConfig":
        """Load configurations from .env file with fallbacks to defaults."""
        load_dotenv(env_path)
        
        # Camera source: int or string URL
        raw_cam = os.getenv("CAMERA_SOURCE", "0").strip()
        cam_source: Union[int, str] = int(raw_cam) if raw_cam.isdigit() else raw_cam

        return cls(
            camera_source=cam_source,
            camera_index=cam_source if isinstance(cam_source, int) else 0,
            frame_width=int(os.getenv("FRAME_WIDTH", "1280")),
            frame_height=int(os.getenv("FRAME_HEIGHT", "720")),
            rotate_deg=int(os.getenv("ROTATE", "0")),
            no_mirror=os.getenv("NO_MIRROR", "false").lower() in ("true", "1", "yes"),
            enable_sound=os.getenv("ALERT_SOUND", "true").lower() in ("true", "1", "yes"),
            alert_cooldown_sec=float(os.getenv("ALERT_COOLDOWN_SEC", "2.0")),
            neck_angle_thresh_deg=float(os.getenv("NECK_ANGLE_THRESH_DEG", "15.0")),
            torso_angle_thresh_deg=float(os.getenv("TORSO_ANGLE_THRESH_DEG", "12.0")),
            shoulder_tilt_thresh_deg=float(os.getenv("SHOULDER_TILT_THRESH_DEG", "10.0")),
            ear_threshold=float(os.getenv("EAR_THRESH", "0.21")),
            ear_time_threshold_sec=float(os.getenv("EAR_TIME_SEC", "1.5")),
            mar_threshold=float(os.getenv("MAR_THRESH", "0.65")),
            mar_time_threshold_sec=float(os.getenv("MAR_TIME_SEC", "2.0")),
            phone_conf_threshold=float(os.getenv("PHONE_CONF_THRESH", "0.45")),
        )

def load_dotenv(env_path: str = ".env") -> None:
    """Lightweight zero-dependency .env parser supporting inline comments."""
    import os
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Strip inline comments if value is not quoted
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                else:
                    v = v.split("#", 1)[0].strip().strip("\"'")
                os.environ[k] = v
    except Exception:
        pass
