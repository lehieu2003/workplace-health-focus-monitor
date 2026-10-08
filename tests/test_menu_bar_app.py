import pytest
from src.config import AppConfig
from src.posture_detector import PostureDetector
from src.drowsiness_detector import DrowsinessDetector
from src.phone_detector import PhoneDetector
from src.audio_alerter import AudioAlerter
from src.notification_manager import NotificationManager
from src.hud_renderer import HUDRenderer
from src.vision_worker import VisionWorker
from src.menu_bar_app import HealthMonitorMenuBarApp

def test_menu_bar_app_init():
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

    app = HealthMonitorMenuBarApp(worker=worker)
    assert app.name == "Workplace Health"
    assert "📸" in app.preview_item.title
    assert "🎯" in app.calibrate_item.title

    # Test update tick
    app.update_status(None)
    assert len(app.title) > 0

def test_menu_bar_app_actions():
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

    app = HealthMonitorMenuBarApp(worker=worker)
    
    # Toggle preview on
    app.on_toggle_preview(None)
    assert worker.get_state().is_preview_visible is True
    assert app.preview_timer.is_alive() is True

    # Render preview tick with no frame should not fail
    app._render_preview(None)

    # Toggle preview off
    app.on_toggle_preview(None)
    assert worker.get_state().is_preview_visible is False
    assert app.preview_timer.is_alive() is False

    # Toggle mute
    app.on_toggle_mute(None)
    assert worker.get_state().is_muted is True
    assert "TẮT" in app.mute_item.title

