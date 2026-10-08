import pytest
from unittest.mock import patch
from src.notification_manager import NotificationManager

def test_notification_manager_sustained_threshold():
    mgr = NotificationManager(sustained_seconds=5.0, cooldown_seconds=30.0, enabled=True, send_system_banner=False)

    # T=0: Normal posture
    assert mgr.check_and_notify(is_violating=False, title="Alert", message="Bad posture", current_time=0.0) is False

    # T=1.0: First violation detected
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad posture", current_time=1.0) is False

    # T=4.0: Violated for 3.0s (< 5.0s threshold) -> Still False (no false alarm)
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad posture", current_time=4.0) is False

    # T=6.1: Violated for 5.1s (>= 5.0s threshold) -> True! Notification dispatched!
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad posture", current_time=6.1) is True

    # T=7.0: Immediately next second -> Suppressed by 30s cooldown
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad posture", current_time=7.0) is False

    # T=20.0: Posture recovered to normal
    assert mgr.check_and_notify(is_violating=False, title="Alert", message="Bad posture", current_time=20.0) is False

    # T=25.0: New violation started
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad posture", current_time=25.0) is False

    # T=30.0: Sustained for 5.0s, but only 23.9s elapsed since last notif (T=6.1) -> Suppressed by 30s cooldown
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad posture", current_time=30.0) is False

    # T=37.0: Sustained for 12.0s and 30.9s elapsed since last notif -> Dispatched!
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad posture", current_time=37.0) is True

def test_notification_manager_disabled():
    mgr = NotificationManager(sustained_seconds=1.0, cooldown_seconds=5.0, enabled=False, send_system_banner=False)
    # Even if violated for 10s, disabled manager returns False
    mgr.check_and_notify(is_violating=True, title="Alert", message="Bad", current_time=0.0)
    assert mgr.check_and_notify(is_violating=True, title="Alert", message="Bad", current_time=10.0) is False

@patch("src.notification_manager.subprocess.Popen")
def test_notification_manager_dispatches_osascript(mock_popen):
    mgr = NotificationManager(sustained_seconds=1.0, cooldown_seconds=5.0, enabled=True, send_system_banner=True)
    mgr.check_and_notify(is_violating=True, title="Postures", message="Sit straight", current_time=0.0)
    sent = mgr.check_and_notify(is_violating=True, title="Postures", message="Sit straight", current_time=2.0)
    assert sent is True
    import time
    time.sleep(0.05)
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert "osascript" in args[0]

