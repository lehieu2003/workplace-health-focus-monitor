import numpy as np
import pytest
from src.config import AppConfig, ColorPalette, SystemStatus
from src.posture_detector import PostureAngles, PostureStatusResult
from src.hud_renderer import HUDRenderer

def test_hud_renderer_creates_valid_frame():
    config = AppConfig()
    renderer = HUDRenderer(config=config)

    # Create dummy 1280x720 3-channel black image
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    angles = PostureAngles(neck_angle=12.0, torso_angle=8.0, shoulder_tilt=1.0)
    posture_result = PostureStatusResult(
        is_good=True,
        reason="GOOD",
        delta_neck=0.0,
        delta_torso=0.0,
        delta_shoulder=0.0,
        angles=angles
    )

    rendered = renderer.render(
        frame=dummy_frame,
        status=SystemStatus.GOOD_POSTURE,
        posture_result=posture_result,
        landmark_points=None,
        phone_boxes=[],
        ear=0.30,
        mar=0.20,
        fps=30.0,
        is_muted=False,
        violation_count=0,
        notification_text=None
    )

    assert rendered.shape == (720, 1280, 3)
    assert rendered.dtype == np.uint8
    # Frame should have been modified (not entirely black anymore)
    assert np.any(rendered > 0)

