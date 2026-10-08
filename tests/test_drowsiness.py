import pytest
import math
from src.config import AppConfig
from src.drowsiness_detector import (
    calculate_ear,
    calculate_mar,
    DrowsinessDetector,
    DrowsinessStatus
)

def test_calculate_ear_open_vs_closed():
    # Synthetic wide open eye coordinates: width = 20, height = 10
    # p1=(0,0), p4=(20,0), p2=(5, 5), p6=(5, -5), p3=(15, 5), p5=(15, -5)
    open_eye = [(0.0, 0.0), (5.0, 5.0), (15.0, 5.0), (20.0, 0.0), (15.0, -5.0), (5.0, -5.0)]
    ear_open = calculate_ear(open_eye)
    # dist(p2, p6) = 10, dist(p3, p5) = 10, dist(p1, p4) = 20 -> (10 + 10) / (2 * 20) = 0.5
    assert math.isclose(ear_open, 0.5, abs_tol=1e-2)

    # Closed eye: vertical distance ~ 1
    closed_eye = [(0.0, 0.0), (5.0, 0.5), (15.0, 0.5), (20.0, 0.0), (15.0, -0.5), (5.0, -0.5)]
    ear_closed = calculate_ear(closed_eye)
    # (1 + 1) / (2 * 20) = 0.05
    assert ear_closed < 0.15

def test_calculate_mar_neutral_vs_yawning():
    # Closed mouth / neutral: width = 40, height = 4
    # m1=(0,0), m5=(40,0), vertical points with height 4
    closed_mouth = [
        (0.0, 0.0),
        (10.0, 2.0), (20.0, 2.0), (30.0, 2.0),
        (40.0, 0.0),
        (30.0, -2.0), (20.0, -2.0), (10.0, -2.0)
    ]
    mar_neutral = calculate_mar(closed_mouth)
    assert mar_neutral < 0.4

    # Yawning mouth: width = 40, height = 30
    yawn_mouth = [
        (0.0, 0.0),
        (10.0, 15.0), (20.0, 15.0), (30.0, 15.0),
        (40.0, 0.0),
        (30.0, -15.0), (20.0, -15.0), (10.0, -15.0)
    ]
    mar_yawn = calculate_mar(yawn_mouth)
    assert mar_yawn > 0.65

def test_drowsiness_detector_temporal_tracker():
    config = AppConfig(
        ear_threshold=0.21,
        ear_time_threshold_sec=1.5,
        mar_threshold=0.65,
        mar_time_threshold_sec=2.0
    )
    detector = DrowsinessDetector(config=config)

    # T=0: Normal open eyes
    status = detector.update_metrics(avg_ear=0.32, mar=0.30, current_time=0.0)
    assert status == DrowsinessStatus.ALERT

    # T=1.0: First closed eye detected (start_time recorded at 1.0s)
    status_1 = detector.update_metrics(avg_ear=0.15, mar=0.30, current_time=1.0)
    assert status_1 == DrowsinessStatus.ALERT

    # T=2.0: Closed for 1.0s (less than 1.5s threshold, normal blink or brief close)
    status_2 = detector.update_metrics(avg_ear=0.15, mar=0.30, current_time=2.0)
    assert status_2 == DrowsinessStatus.ALERT

    # T=2.6: Closed eyes sustained for 1.6s (2.6 - 1.0 >= 1.5s) -> triggers EYES_CLOSED
    status_alert = detector.update_metrics(avg_ear=0.15, mar=0.30, current_time=2.6)
    assert status_alert == DrowsinessStatus.EYES_CLOSED

    # T=3.0: Eyes opened back -> resets to ALERT
    status_reset = detector.update_metrics(avg_ear=0.30, mar=0.30, current_time=3.0)
    assert status_reset == DrowsinessStatus.ALERT

    # Yawn sustained: start at T=4.0, sustained past T=6.1 (2.1s > 2.0s threshold)
    detector.update_metrics(avg_ear=0.30, mar=0.75, current_time=4.0)
    status_yawn = detector.update_metrics(avg_ear=0.30, mar=0.75, current_time=6.1)
    assert status_yawn == DrowsinessStatus.YAWNING
