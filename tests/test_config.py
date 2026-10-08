import pytest
from src.config import AppConfig, ColorPalette, SystemStatus

def test_config_defaults():
    config = AppConfig()
    assert config.camera_index == 0
    assert config.neck_angle_thresh_deg == 18.0
    assert config.torso_angle_thresh_deg == 15.0
    assert config.shoulder_tilt_thresh_deg == 12.0
    assert config.ear_threshold == 0.16
    assert config.ear_time_threshold_sec == 2.5
    assert config.mar_threshold == 0.75
    assert config.mar_time_threshold_sec == 2.5
    assert config.phone_conf_threshold == 0.60
    assert config.alert_cooldown_sec == 2.0

def test_color_palette():
    palette = ColorPalette()
    # Check OpenCV BGR formats (B, G, R)
    assert len(palette.GOOD_GREEN) == 3
    assert len(palette.BAD_RED) == 3
    assert len(palette.ALERT_AMBER) == 3
    assert len(palette.PHONE_PURPLE) == 3

def test_system_status_enum():
    assert SystemStatus.GOOD_POSTURE.value == "GOOD POSTURE & FOCUSED"
    assert SystemStatus.BAD_POSTURE.value == "BAD POSTURE: SLOUCH DETECTED"
    assert SystemStatus.DROWSY_EYES.value == "DROWSY: EYES CLOSED"
    assert SystemStatus.DROWSY_YAWN.value == "DROWSY: YAWNING DETECTED"
    assert SystemStatus.PHONE_DISTRACTION.value == "DISTRACTED: PHONE IN USE"
    assert SystemStatus.NO_PERSON.value == "NO PERSON DETECTED"

def test_parse_camera_source():
    from src.main import parse_camera_source
    assert parse_camera_source("0") == 0
    assert parse_camera_source("1") == 1
    assert parse_camera_source("http://192.168.1.23:8080") == "http://192.168.1.23:8080/video"
    assert parse_camera_source("http://192.168.1.23:8080/video") == "http://192.168.1.23:8080/video"
    assert parse_camera_source("http://192.168.1.23:8080/") == "http://192.168.1.23:8080/video"

def test_app_config_from_env(tmp_path):
    env_file = tmp_path / ".env.test"
    env_file.write_text("""
CAMERA_SOURCE=http://192.168.1.99:8080
FRAME_WIDTH=1920
FRAME_HEIGHT=1080
ROTATE=90
NO_MIRROR=true
ALERT_SOUND=false
NECK_ANGLE_THRESH_DEG=18.5
""")
    cfg = AppConfig.from_env(str(env_file))
    assert cfg.camera_source == "http://192.168.1.99:8080"
    assert cfg.frame_width == 1920
    assert cfg.frame_height == 1080
    assert cfg.rotate_deg == 90
    assert cfg.no_mirror is True
    assert cfg.enable_sound is False
    assert cfg.neck_angle_thresh_deg == 18.5

