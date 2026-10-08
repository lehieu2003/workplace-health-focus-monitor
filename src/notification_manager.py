"""macOS Native Notification Manager with sustained duration debouncing."""

import subprocess
import threading
from typing import Optional

class NotificationManager:
    def __init__(
        self,
        sustained_seconds: float = 5.0,
        cooldown_seconds: float = 30.0,
        enabled: bool = True,
        send_system_banner: bool = True,
    ):
        self.sustained_seconds = sustained_seconds
        self.cooldown_seconds = cooldown_seconds
        self._enabled = enabled
        self.send_system_banner = send_system_banner

        self.violation_start_time: Optional[float] = None
        self.last_notification_time: float = -999.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = value

    def toggle(self) -> bool:
        """Toggle notifications enabled state and return new state."""
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled

    def check_and_notify(
        self, is_violating: bool, title: str, message: str, current_time: float, subtitle: str = "Cảnh báo sức khỏe"
    ) -> bool:
        """
        Check if violation is sustained long enough to trigger a notification.
        Returns True if a notification was dispatched, False otherwise.
        """
        with self._lock:
            if not self._enabled:
                self.violation_start_time = None
                return False

            if not is_violating:
                self.violation_start_time = None
                return False

            # First moment of violation
            if self.violation_start_time is None:
                self.violation_start_time = current_time
                return False

            # Check if violation has lasted for >= sustained_seconds
            duration = current_time - self.violation_start_time
            if duration < self.sustained_seconds:
                return False

            # Check cooldown since last sent notification
            if (current_time - self.last_notification_time) < self.cooldown_seconds:
                return False

            self.last_notification_time = current_time

        if not self.send_system_banner:
            return True

        # Dispatch macOS native notification asynchronously
        def _dispatch():
            try:
                # Escape double quotes
                safe_msg = message.replace('"', '\\"')
                safe_title = title.replace('"', '\\"')
                safe_sub = subtitle.replace('"', '\\"')
                apple_script = (
                    f'display notification "{safe_msg}" '
                    f'with title "{safe_title}" '
                    f'subtitle "{safe_sub}"'
                )
                subprocess.Popen(
                    ["osascript", "-e", apple_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        t = threading.Thread(target=_dispatch, daemon=True)
        t.start()
        return True

