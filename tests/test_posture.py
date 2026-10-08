import pytest
import math
from src.config import AppConfig
from src.posture_detector import (
    PostureDetector,
    PostureAngles,
    PostureStatusResult,
    calculate_angle_from_vertical,
    calculate_tilt_angle
)

def test_calculate_angle_from_vertical():
    # Straight vertical line: point1=(100, 100), point2=(100, 200) -> 0 degrees
    angle = calculate_angle_from_vertical((100, 100), (100, 200))
    assert math.isclose(angle, 0.0, abs_tol=1e-2)

    # 45 degree tilt: dx = dy
    angle_45 = calculate_angle_from_vertical((100, 100), (200, 200))
    assert math.isclose(angle_45, 45.0, abs_tol=1e-2)

def test_calculate_tilt_angle():
    # Horizontal line: y1 = y2 -> 0 degrees
    tilt = calculate_tilt_angle((50, 100), (150, 100))
    assert math.isclose(tilt, 0.0, abs_tol=1e-2)

    # Slanted line: dx = 100, dy = 100 -> 45 degrees
    tilt_45 = calculate_tilt_angle((50, 100), (150, 200))
    assert math.isclose(tilt_45, 45.0, abs_tol=1e-2)

def test_posture_detector_calibration_and_evaluation():
    config = AppConfig(neck_angle_thresh_deg=15.0, torso_angle_thresh_deg=12.0)
    detector = PostureDetector(config=config)
    
    assert not detector.is_calibrated
    
    # Baseline upright posture angles
    upright = PostureAngles(neck_angle=10.0, torso_angle=5.0, shoulder_tilt=1.0)
    detector.calibrate(upright)
    assert detector.is_calibrated
    assert detector.baseline_neck == 10.0
    assert detector.baseline_torso == 5.0

    # Normal slight shift (within thresholds: neck +5 deg, torso +3 deg)
    normal = PostureAngles(neck_angle=15.0, torso_angle=8.0, shoulder_tilt=2.0)
    result = detector.evaluate(normal)
    assert result.is_good is True
    assert result.reason == "GOOD"

    # Forward head / slouching (neck +20 deg > threshold 15 deg)
    slouched = PostureAngles(neck_angle=32.0, torso_angle=7.0, shoulder_tilt=2.0)
    result_slouch = detector.evaluate(slouched)
    assert result_slouch.is_good is False
    assert "NECK" in result_slouch.reason or "SLOUCH" in result_slouch.reason

    # Severe torso leaning back or leaning forward (torso +18 deg > threshold 12 deg)
    leaned = PostureAngles(neck_angle=12.0, torso_angle=22.0, shoulder_tilt=2.0)
    result_lean = detector.evaluate(leaned)
    assert result_lean.is_good is False
    assert "TORSO" in result_lean.reason or "LEAN" in result_lean.reason

