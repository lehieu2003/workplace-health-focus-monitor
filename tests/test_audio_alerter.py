import pytest
from unittest.mock import patch
from src.config import AppConfig
from src.audio_alerter import AudioAlerter

def test_audio_alerter_cooldown():
    config = AppConfig(alert_cooldown_sec=2.0)
    alerter = AudioAlerter(config=config, enable_sound=False) # mock mode without launching afplay

    # First trigger at T=0.0 -> Allowed
    assert alerter.trigger(current_time=0.0) is True

    # Immediate second trigger at T=0.5 -> Suppressed by cooldown
    assert alerter.trigger(current_time=0.5) is False

    # Trigger at T=1.9 -> Suppressed (< 2.0s)
    assert alerter.trigger(current_time=1.9) is False

    # Trigger at T=2.1 -> Allowed (>= 2.0s)
    assert alerter.trigger(current_time=2.1) is True

def test_audio_alerter_mute_toggle():
    config = AppConfig()
    alerter = AudioAlerter(config=config, enable_sound=False)

    assert alerter.is_muted is False
    assert alerter.trigger(current_time=10.0) is True

    # Toggle mute ON
    alerter.toggle_mute()
    assert alerter.is_muted is True
    # When muted, trigger returns False
    assert alerter.trigger(current_time=15.0) is False

    # Toggle mute OFF
    alerter.toggle_mute()
    assert alerter.is_muted is False
    assert alerter.trigger(current_time=15.0) is True

import time

@patch("src.audio_alerter.subprocess.Popen")
def test_audio_alerter_plays_afplay(mock_popen):
    config = AppConfig(alert_cooldown_sec=1.0)
    alerter = AudioAlerter(config=config, enable_sound=True)
    
    played = alerter.trigger(current_time=1.0)
    assert played is True
    # Give thread a small moment to execute worker
    time.sleep(0.05)
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert "afplay" in args[0]
