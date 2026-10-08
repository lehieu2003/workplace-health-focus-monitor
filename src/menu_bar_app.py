import os
import cv2
import rumps
from src.vision_worker import VisionWorker

# Safe fallback for application_support if sandboxed
_original_app_support = rumps.rumps.application_support
def _safe_app_support(name):
    try:
        return _original_app_support(name)
    except (PermissionError, OSError):
        import tempfile
        path = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(path, exist_ok=True)
        return path
rumps.rumps.application_support = _safe_app_support

class HealthMonitorMenuBarApp(rumps.App):
    def __init__(self, worker: VisionWorker):
        super().__init__("Workplace Health", title="🧘 100%", quit_button=None)
        self.worker = worker

        # Build menu items
        self.status_item = rumps.MenuItem("Trạng thái: 🟢 Đang kiểm tra...")
        self.score_item = rumps.MenuItem("Điểm tư thế: 100% (0 vi phạm)")
        self.preview_item = rumps.MenuItem("📸 Bật Cửa sổ Camera", callback=self.on_toggle_preview)
        self.calibrate_item = rumps.MenuItem("🎯 Căn chỉnh tư thế (Calibrate) [C]", callback=self.on_calibrate)
        self.rotate_item = rumps.MenuItem("🔄 Xoay Camera 90° [O]", callback=self.on_rotate_camera)
        self.mute_item = rumps.MenuItem("🔊 Âm thanh cảnh báo: BẬT [M]", callback=self.on_toggle_mute)
        self.notif_item = rumps.MenuItem("🔔 Thông báo macOS: BẬT", callback=self.on_toggle_notif)
        self.stats_item = rumps.MenuItem("📊 Xem thống kê chi tiết...", callback=self.on_show_stats)
        self.reset_item = rumps.MenuItem("🔄 Đặt lại thống kê [R]", callback=self.on_reset_stats)
        self.quit_item = rumps.MenuItem("🚪 Thoát ứng dụng [Q]", callback=self.on_quit)

        self.menu = [
            self.status_item,
            self.score_item,
            None,  # Separator
            self.preview_item,
            self.calibrate_item,
            self.rotate_item,
            None,  # Separator
            self.mute_item,
            self.notif_item,
            None,  # Separator
            self.stats_item,
            self.reset_item,
            None,  # Separator
            self.quit_item,
        ]

        # Timer updating status bar every second
        self.timer = rumps.Timer(self.update_status, 1.0)
        self.timer.start()

        # Timer rendering OpenCV preview on Cocoa main thread (~30 FPS)
        self.preview_timer = rumps.Timer(self._render_preview, 0.03)

    def update_status(self, _):
        """Called every second to synchronize Menu Bar icon with worker state."""
        state = self.worker.get_state()

        # Update title in status bar
        if state.status_emoji == "🟢":
            self.title = f"🧘 {state.posture_score:.0f}%"
        elif state.status_emoji in ("🔴", "🟠", "🟣"):
            self.title = f"{state.status_emoji} {state.status_label}"
        else:
            self.title = f"{state.status_emoji} {state.posture_score:.0f}%"

        # Update menu labels
        self.status_item.title = f"Trạng thái: {state.status_emoji} {state.status_label}"
        self.score_item.title = f"Điểm hôm nay: {state.posture_score:.1f}% ({state.violation_count} vi phạm)"

        self.preview_item.title = (
            "📸 Tắt Cửa sổ Camera" if state.is_preview_visible else "📸 Bật Cửa sổ Camera"
        )
        self.mute_item.title = (
            "🔇 Âm thanh cảnh báo: TẮT [M]" if state.is_muted else "🔊 Âm thanh cảnh báo: BẬT [M]"
        )
        self.notif_item.title = (
            "🔕 Thông báo macOS: TẮT" if not state.notifications_enabled else "🔔 Thông báo macOS: BẬT"
        )

    def _render_preview(self, _):
        """Renders camera frame with HUD on Cocoa main thread to avoid GUI thread exceptions."""
        frame = self.worker.get_latest_rendered_frame()
        if frame is not None:
            cv2.imshow(VisionWorker.PREVIEW_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q', 'Q' or ESC
                self.on_toggle_preview(None)
                return
            elif key in (ord("c"), ord("C")):
                self.on_calibrate(None)
            elif key in (ord("m"), ord("M")):
                self.on_toggle_mute(None)
            elif key in (ord("o"), ord("O")):
                self.on_rotate_camera(None)
            elif key in (ord("r"), ord("R")):
                self.on_reset_stats(None)

            try:
                if cv2.getWindowProperty(VisionWorker.PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.on_toggle_preview(None)
            except cv2.error:
                self.on_toggle_preview(None)

    def on_rotate_camera(self, _):
        deg = self.worker.rotate_camera()
        rumps.notification(
            title="Workplace Health Monitor",
            subtitle="Xoay Camera",
            message=f"Đã xoay khung hình camera: {deg}°",
        )

    def on_toggle_preview(self, _):
        is_visible = self.worker.toggle_preview()
        self.preview_item.title = "📸 Tắt Cửa sổ Camera" if is_visible else "📸 Bật Cửa sổ Camera"
        if is_visible:
            if not self.preview_timer.is_alive():
                self.preview_timer.start()
        else:
            if self.preview_timer.is_alive():
                self.preview_timer.stop()
            try:
                cv2.destroyWindow(VisionWorker.PREVIEW_WINDOW_NAME)
            except Exception:
                pass

    def on_calibrate(self, _):
        self.worker.calibrate()
        rumps.notification(
            title="Workplace Health Monitor",
            subtitle="Căn chỉnh tư thế",
            message="Đã ghi nhận tư thế ngồi hiện tại của bạn làm chuẩn!",
        )

    def on_toggle_mute(self, _):
        is_muted = self.worker.toggle_mute()
        self.mute_item.title = "🔇 Âm thanh cảnh báo: TẮT" if is_muted else "🔊 Âm thanh cảnh báo: BẬT"

    def on_toggle_notif(self, _):
        enabled = self.worker.toggle_notifications()
        self.notif_item.title = "🔕 Thông báo macOS: TẮT" if not enabled else "🔔 Thông báo macOS: BẬT"

    def on_show_stats(self, _):
        state = self.worker.get_state()
        mins = int(state.total_active_seconds // 60)
        secs = int(state.total_active_seconds % 60)
        good_mins = int(state.good_posture_seconds // 60)
        good_secs = int(state.good_posture_seconds % 60)

        breakdown = state.violation_breakdown
        msg = (
            f"⏱️ Tổng thời gian làm việc: {mins}p {secs}s\n"
            f"🧘 Thời gian ngồi thẳng chuẩn: {good_mins}p {good_secs}s\n"
            f"⭐ Điểm tư thế (Posture Score): {state.posture_score:.1f}%\n\n"
            f"⚠️ Chi tiết số lần vi phạm ({state.violation_count} lần):\n"
            f"  - Gù lưng / Cúi đầu: {breakdown.get('slouch', 0)} lần\n"
            f"  - Buồn ngủ / Ngáp: {breakdown.get('drowsy', 0)} lần\n"
            f"  - Dùng điện thoại: {breakdown.get('phone', 0)} lần"
        )
        rumps.alert(title="Báo Cáo Sức Khỏe & Tập Trung", message=msg)

    def on_reset_stats(self, _):
        self.worker.reset_stats()
        rumps.notification(
            title="Workplace Health Monitor",
            subtitle="Dữ liệu làm việc",
            message="Đã đặt lại thống kê phiên làm việc về 0.",
        )

    def on_quit(self, _):
        if self.preview_timer.is_alive():
            self.preview_timer.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.worker.stop()
        rumps.quit_application()
