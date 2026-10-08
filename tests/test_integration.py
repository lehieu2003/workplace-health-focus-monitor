import numpy as np
import pytest
from src.config import AppConfig, SystemStatus
from src.posture_detector import PostureDetector
from src.drowsiness_detector import DrowsinessDetector
from src.phone_detector import PhoneDetector
from src.audio_alerter import AudioAlerter
from src.hud_renderer import HUDRenderer

def test_full_pipeline_synthetic_frame():
    config = AppConfig()
    posture_detector = PostureDetector(config=config)
    drowsiness_detector = DrowsinessDetector(config=config)
    phone_detector = PhoneDetector(config=config, load_model=False) # skip downloading model during CI unit test
    audio_alerter = AudioAlerter(config=config, enable_sound=False)
    hud_renderer = HUDRenderer(config=config)

    # Synthetic 720p frame
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Posture inference (on blank frame returns None)
    angles, landmarks = posture_detector.process_frame(dummy_frame, 1280, 720)
    assert angles is None
    assert landmarks is None

    # 2. Drowsiness inference
    fatigue_status, ear, mar = drowsiness_detector.process_frame(dummy_frame, 1280, 720, current_time=1.0)
    assert ear > 0.0

    # 3. Phone inference (with mock)
    has_phone, phone_boxes = phone_detector.detect(dummy_frame, frame_count=0)
    assert has_phone is False

    # 4. State evaluation when no person is visible
    status = SystemStatus.NO_PERSON if landmarks is None else SystemStatus.GOOD_POSTURE
    assert status == SystemStatus.NO_PERSON

    # 5. Render HUD
    rendered = hud_renderer.render(
        frame=dummy_frame,
        status=status,
        posture_result=None,
        landmark_points=None,
        phone_boxes=[],
        ear=ear,
        mar=mar,
        fps=60.0,
        is_muted=False,
        violation_count=0,
        notification_text="Test notification"
    )
    assert rendered.shape == (720, 1280, 3)

