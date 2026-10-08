import pytest
from src.config import AppConfig
from src.phone_detector import PhoneDetector, DetectedObject

def test_phone_detector_filter():
    config = AppConfig(phone_conf_threshold=0.50)
    detector = PhoneDetector(config=config, load_model=False)

    # Mock raw detections: [box, conf, class_id]
    mock_boxes = [
        ([10, 20, 50, 80], 0.85, 67),  # cell phone, conf 0.85 -> match
        ([100, 100, 200, 300], 0.90, 0), # person, class 0 -> ignore
        ([50, 50, 90, 90], 0.35, 67),    # cell phone, low conf 0.35 -> ignore
    ]

    detected_phones = detector.filter_detections(mock_boxes)
    assert len(detected_phones) == 1
    assert detected_phones[0].class_id == 67
    assert detected_phones[0].confidence == 0.85
    assert detected_phones[0].box == (10, 20, 50, 80)
    assert detected_phones[0].label == "cell phone"

def test_phone_detector_frame_skipping():
    config = AppConfig(phone_skip_frames=3)
    detector = PhoneDetector(config=config, load_model=False)

    # Initial state
    assert detector.should_run_inference(frame_count=0) is True
    assert detector.should_run_inference(frame_count=1) is False
    assert detector.should_run_inference(frame_count=2) is False
    assert detector.should_run_inference(frame_count=3) is True

