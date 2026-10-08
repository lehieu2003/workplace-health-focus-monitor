"""Background Vision Worker Thread managing computer vision inference and CPU throttling."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import cv2
import numpy as np

from src.config import AppConfig, SystemStatus
from src.posture_detector import PostureDetector, PostureStatusResult, PostureAngles
from src.drowsiness_detector import DrowsinessDetector, DrowsinessStatus
from src.phone_detector import PhoneDetector
from src.audio_alerter import AudioAlerter
from src.notification_manager import NotificationManager
from src.hud_renderer import HUDRenderer

@dataclass
class WorkerState:
    current_status: SystemStatus = SystemStatus.NO_PERSON
    posture_result: Optional[PostureStatusResult] = None
    angles: Optional[PostureAngles] = None
    ear: float = 0.30
    mar: float = 0.20
    is_calibrated: bool = False
    is_preview_visible: bool = False
    is_muted: bool = False
    notifications_enabled: bool = True
    total_active_seconds: float = 0.0
    good_posture_seconds: float = 0.0
    posture_score: float = 100.0
    violation_count: int = 0
    violation_breakdown: Dict[str, int] = field(default_factory=lambda: {"slouch": 0, "drowsy": 0, "phone": 0})
    fps: float = 30.0
    status_emoji: str = "⚪"
    status_label: str = "Đang khởi động..."

class VisionWorker(threading.Thread):
    PREVIEW_WINDOW_NAME = "Workplace Health Monitor - Camera Preview"

    def __init__(
        self,
        config: AppConfig,
        posture_detector: PostureDetector,
        drowsiness_detector: DrowsinessDetector,
        phone_detector: PhoneDetector,
        audio_alerter: AudioAlerter,
        notification_manager: NotificationManager,
        hud_renderer: HUDRenderer,
        synthetic_mode: bool = False,
    ):
        super().__init__(daemon=True, name="VisionWorkerThread")
        self.config = config
        self.posture_detector = posture_detector
        self.drowsiness_detector = drowsiness_detector
        self.phone_detector = phone_detector
        self.audio_alerter = audio_alerter
        self.notification_manager = notification_manager
        self.hud_renderer = hud_renderer
        self.synthetic_mode = synthetic_mode

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._is_preview_visible = False
        self._pending_calibrate = False
        self._latest_rendered_frame: Optional[np.ndarray] = None

        # Internal state & stats
        self.state = WorkerState(
            is_muted=audio_alerter.is_muted,
            notifications_enabled=notification_manager.enabled,
        )
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.in_violation = False
        self.bad_posture_start_time: Optional[float] = None
        self.phone_start_time: Optional[float] = None
        self.cap: Optional[cv2.VideoCapture] = None

    def get_state(self) -> WorkerState:
        with self._lock:
            # Return copy of state
            return WorkerState(
                current_status=self.state.current_status,
                posture_result=self.state.posture_result,
                angles=self.state.angles,
                ear=self.state.ear,
                mar=self.state.mar,
                is_calibrated=self.state.is_calibrated,
                is_preview_visible=self._is_preview_visible,
                is_muted=self.audio_alerter.is_muted,
                notifications_enabled=self.notification_manager.enabled,
                total_active_seconds=self.state.total_active_seconds,
                good_posture_seconds=self.state.good_posture_seconds,
                posture_score=self.state.posture_score,
                violation_count=self.state.violation_count,
                violation_breakdown=dict(self.state.violation_breakdown),
                fps=self.state.fps,
                status_emoji=self.state.status_emoji,
                status_label=self.state.status_label,
            )

    def get_latest_rendered_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_rendered_frame

    def toggle_preview(self) -> bool:
        with self._lock:
            self._is_preview_visible = not self._is_preview_visible
            if not self._is_preview_visible:
                self._latest_rendered_frame = None
            return self._is_preview_visible

    def set_preview(self, visible: bool) -> None:
        with self._lock:
            self._is_preview_visible = visible
            if not visible:
                self._latest_rendered_frame = None

    def calibrate(self) -> bool:
        """Request baseline calibration on next valid frame."""
        with self._lock:
            self._pending_calibrate = True
            return True

    def toggle_mute(self) -> bool:
        muted = self.audio_alerter.toggle_mute()
        with self._lock:
            self.state.is_muted = muted
        return muted

    def toggle_notifications(self) -> bool:
        enabled = self.notification_manager.toggle()
        with self._lock:
            self.state.notifications_enabled = enabled
        return enabled

    def reset_stats(self) -> None:
        with self._lock:
            self.start_time = time.time()
            self.state.total_active_seconds = 0.0
            self.state.good_posture_seconds = 0.0
            self.state.posture_score = 100.0
            self.state.violation_count = 0
            self.state.violation_breakdown = {"slouch": 0, "drowsy": 0, "phone": 0}

    def rotate_camera(self) -> int:
        with self._lock:
            self.config.rotate_deg = (self.config.rotate_deg + 90) % 360
            return self.config.rotate_deg

    def stop(self) -> None:
        self._stop_event.set()

    def _init_camera(self) -> bool:
        if self.synthetic_mode:
            return True

        cam_source = self.config.camera_source
        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            # Fallback
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        return True

    def run(self) -> None:
        if not self._init_camera() and not self.synthetic_mode:
            with self._lock:
                self.state.status_emoji = "❌"
                self.state.status_label = "Lỗi kết nối Camera"
            return

        frame_count = 0
        fps = 30.0
        self.start_time = time.time()
        self.last_frame_time = time.time()

        try:
            while not self._stop_event.is_set():
                t0 = time.time()
                dt = t0 - self.last_frame_time
                self.last_frame_time = t0
                frame_count += 1

                # Read frame
                if self.synthetic_mode:
                    frame = np.zeros((self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)
                    time.sleep(0.04)
                else:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        time.sleep(0.1)
                        continue

                # Apply video rotation & mirror
                if self.config.rotate_deg == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.config.rotate_deg == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.config.rotate_deg == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if not self.config.no_mirror:
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Inference
                angles, landmark_points = self.posture_detector.process_frame(frame_rgb, w, h)
                posture_result = None
                if angles is not None:
                    # Check pending calibration
                    with self._lock:
                        if self._pending_calibrate:
                            self.posture_detector.calibrate(angles)
                            self._pending_calibrate = False
                    posture_result = self.posture_detector.evaluate(angles)

                fatigue_status, ear, mar = self.drowsiness_detector.process_frame(frame_rgb, w, h, current_time=t0)
                has_phone, phone_boxes = self.phone_detector.detect(frame, frame_count=frame_count)

                # Evaluate system status
                is_violating = False
                emoji = "🟢"
                label = "Tư thế chuẩn"
                violation_type = None

                # Debounce phone detection: must be sustained >= 1.5s
                is_phone_violating = False
                if has_phone:
                    if self.phone_start_time is None:
                        self.phone_start_time = t0
                    elif (t0 - self.phone_start_time) >= 1.5:
                        is_phone_violating = True
                else:
                    self.phone_start_time = None

                # Debounce posture: must be sustained >= posture_debounce_sec (2.0s)
                is_posture_violating = False
                if posture_result and not posture_result.is_good:
                    if self.bad_posture_start_time is None:
                        self.bad_posture_start_time = t0
                    elif (t0 - self.bad_posture_start_time) >= self.config.posture_debounce_sec:
                        is_posture_violating = True
                else:
                    self.bad_posture_start_time = None

                if landmark_points is None:
                    current_status = SystemStatus.NO_PERSON
                    emoji = "⚪"
                    label = "Không có người"
                elif not self.posture_detector.is_calibrated:
                    current_status = SystemStatus.CALIBRATING
                    emoji = "🔵"
                    label = "Chưa Calibrate"
                elif is_phone_violating:
                    current_status = SystemStatus.PHONE_DISTRACTION
                    is_violating = True
                    emoji = "🟣"
                    label = "Dùng điện thoại"
                    violation_type = "phone"
                elif fatigue_status == DrowsinessStatus.EYES_CLOSED:
                    current_status = SystemStatus.DROWSY_EYES
                    is_violating = True
                    emoji = "🟠"
                    label = "Buồn ngủ (Nhắm mắt)"
                    violation_type = "drowsy"
                elif fatigue_status == DrowsinessStatus.YAWNING:
                    current_status = SystemStatus.DROWSY_YAWN
                    is_violating = True
                    emoji = "🟠"
                    label = "Đang ngáp"
                    violation_type = "drowsy"
                elif is_posture_violating:
                    current_status = SystemStatus.BAD_POSTURE
                    is_violating = True
                    emoji = "🔴"
                    label = "Gù lưng / Sai tư thế"
                    violation_type = "slouch"
                elif posture_result and not posture_result.is_good:
                    # Momentary shift (<2s): Show warning yellow on menu bar but do NOT beep or increment violation
                    current_status = SystemStatus.BAD_POSTURE
                    emoji = "🟡"
                    label = "Hơi lệch tư thế"
                else:
                    current_status = SystemStatus.GOOD_POSTURE
                    emoji = "🟢"
                    label = "Tư thế chuẩn"

                # Update stats & durations
                with self._lock:
                    self.state.total_active_seconds += dt
                    if not is_violating and current_status == SystemStatus.GOOD_POSTURE:
                        self.state.good_posture_seconds += dt

                    if self.state.total_active_seconds > 0:
                        self.state.posture_score = min(
                            100.0,
                            max(0.0, (self.state.good_posture_seconds / self.state.total_active_seconds) * 100.0),
                        )

                    # Trigger state transition for violations
                    if is_violating and not self.in_violation:
                        self.state.violation_count += 1
                        if violation_type:
                            self.state.violation_breakdown[violation_type] = (
                                self.state.violation_breakdown.get(violation_type, 0) + 1
                            )
                        self.in_violation = True
                    elif not is_violating:
                        self.in_violation = False

                    # Audio alerts (immediate debounce 2s)
                    if is_violating:
                        self.audio_alerter.trigger(t0)

                    # Update worker state
                    self.state.current_status = current_status
                    self.state.posture_result = posture_result
                    self.state.angles = angles
                    self.state.ear = ear
                    self.state.mar = mar
                    self.state.is_calibrated = self.posture_detector.is_calibrated
                    self.state.status_emoji = emoji
                    self.state.status_label = label

                # macOS Native Banner notification (only when sustained >= 5.0s)
                if is_violating:
                    self.notification_manager.check_and_notify(
                        is_violating=True,
                        title="Workplace Health Monitor",
                        message=f"{label}! Hãy điều chỉnh lại tư thế nhé.",
                        current_time=t0,
                    )
                else:
                    self.notification_manager.check_and_notify(
                        is_violating=False,
                        title="",
                        message="",
                        current_time=t0,
                    )

                # FPS calculation
                elapsed = time.time() - t0
                if elapsed > 0:
                    current_fps = 1.0 / elapsed
                    fps = 0.9 * fps + 0.1 * current_fps
                with self._lock:
                    self.state.fps = fps

                # Handle Preview Frame Rendering
                with self._lock:
                    preview_active = self._is_preview_visible

                if preview_active:
                    rendered = self.hud_renderer.render(
                        frame=frame,
                        status=current_status,
                        posture_result=posture_result,
                        landmark_points=landmark_points,
                        phone_boxes=phone_boxes,
                        ear=ear,
                        mar=mar,
                        fps=fps,
                        is_muted=self.audio_alerter.is_muted,
                        violation_count=self.state.violation_count,
                    )
                    with self._lock:
                        self._latest_rendered_frame = rendered
                else:
                    with self._lock:
                        self._latest_rendered_frame = None
                    # CPU THROTTLING: when running in background without window, sleep slightly
                    # Keeps Apple Silicon M4 CPU under ~3%
                    time.sleep(0.06)

        finally:
            if self.cap is not None:
                self.cap.release()


