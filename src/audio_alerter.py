"""Non-blocking audio alert manager using native macOS sound (afplay)."""

import os
import subprocess
import threading
from typing import Optional
from src.config import AppConfig

class AudioAlerter:
    def __init__(self, config: Optional[AppConfig] = None, enable_sound: bool = True):
        self.config = config or AppConfig()
        self.enable_sound = enable_sound
        self.last_alert_time: float = -999.0
        self._is_muted: bool = False
        self._lock = threading.Lock()

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    def toggle_mute(self) -> bool:
        """Toggle mute state and return the new state."""
        with self._lock:
            self._is_muted = not self._is_muted
            return self._is_muted

    def set_mute(self, muted: bool) -> None:
        with self._lock:
            self._is_muted = muted

    def trigger(self, current_time: float, sound_path: Optional[str] = None) -> bool:
        """
        Trigger an alert sound asynchronously if not muted and cooldown has elapsed.
        Returns True if sound was played, False if suppressed.
        """
        with self._lock:
            if self._is_muted:
                return False

            if (current_time - self.last_alert_time) < self.config.alert_cooldown_sec:
                return False

            self.last_alert_time = current_time

        if not self.enable_sound:
            return True

        target_sound = sound_path or self.config.mac_sound_path

        def _play_worker():
            try:
                # macOS native sound player - non-blocking process
                if os.path.exists(target_sound):
                    subprocess.Popen(
                        ["afplay", target_sound],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    # Fallback to system bell
                    subprocess.Popen(
                        ["osascript", "-e", "beep 1"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except Exception:
                pass

        # Dispatch playback in daemon thread
        thread = threading.Thread(target=_play_worker, daemon=True)
        thread.start()
        return True

