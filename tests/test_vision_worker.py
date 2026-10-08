import pytest
import time
from src.config import AppConfig, SystemStatus
from src.posture_detector import PostureDetector
from src.drowsiness_detector import DrowsinessDetector
from src.phone_detector import PhoneDetector
from src.audio_alerter import AudioAlerter
from src.notification_manager import NotificationManager
from src.hud_renderer import HUDRenderer
from src.vision_worker import VisionWorker, WorkerState

def test_vision_worker_initial_state_and_stats():
    config = AppConfig()
    posture_detector = PostureDetector(config=config)
    drowsiness_detector = DrowsinessDetector(config=config)
    phone_detector = PhoneDetector(config=config, load_model=False)
    audio_alerter = AudioAlerter(config=config, enable_sound=False)
    notif_mgr = NotificationManager(enabled=True, send_system_banner=False)
    hud_renderer = HUDRenderer(config=config)

    worker = VisionWorker(
        config=config,
        posture_detector=posture_detector,
        drowsiness_detector=drowsiness_detector,
        phone_detector=phone_detector,
        audio_alerter=audio_alerter,
        notification_manager=notif_mgr,
        hud_renderer=hud_renderer,
        synthetic_mode=True
    )

    state = worker.get_state()
    assert state.is_preview_visible is False
    assert state.is_muted is False
    assert state.notifications_enabled is True
    assert state.posture_score == 100.0
    assert state.violation_count == 0

    # Toggle preview
    new_preview = worker.toggle_preview()
    assert new_preview is True
    assert worker.get_state().is_preview_visible is True

    # Toggle mute
    new_mute = worker.toggle_mute()
    assert new_mute is True
    assert worker.get_state().is_muted is True

    # Toggle notifications
    new_notif = worker.toggle_notifications()
    assert new_notif is False
    assert worker.get_state().notifications_enabled is False

def test_vision_worker_synthetic_start_and_stop():
    config = AppConfig()
    posture_detector = PostureDetector(config=config)
    drowsiness_detector = DrowsinessDetector(config=config)
    phone_detector = PhoneDetector(config=config, load_model=False)
    audio_alerter = AudioAlerter(config=config, enable_sound=False)
    notif_mgr = NotificationManager(enabled=True, send_system_banner=False)
    hud_renderer = HUDRenderer(config=config)

    worker = VisionWorker(
        config=config,
        posture_detector=posture_detector,
        drowsiness_detector=drowsiness_detector,
        phone_detector=phone_detector,
        audio_alerter=audio_alerter,
        notification_manager=notif_mgr,
        hud_renderer=hud_renderer,
        synthetic_mode=True
    )

    worker.start()
    assert worker.is_alive()
    time.sleep(0.8)
    worker.stop()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    state = worker.get_state()
    assert state.total_active_seconds > 0.0
