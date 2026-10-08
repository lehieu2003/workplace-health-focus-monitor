"""HUD overlay renderer for live camera visualization and telemetry dashboard."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from src.config import AppConfig, ColorPalette, SystemStatus
from src.posture_detector import PostureStatusResult
from src.phone_detector import DetectedObject

class HUDRenderer:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.colors = ColorPalette()

    def _draw_transparent_rect(
        self, img: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], alpha: float
    ) -> None:
        """Draw an alpha-blended rectangle for sleek glassmorphic effect."""
        sub_img = img[y : y + h, x : x + w]
        white_rect = np.full(sub_img.shape, color, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 1.0 - alpha, white_rect, alpha, 1.0)
        img[y : y + h, x : x + w] = res

    def _get_status_color(self, status: SystemStatus) -> Tuple[int, int, int]:
        if status == SystemStatus.GOOD_POSTURE:
            return self.colors.GOOD_GREEN
        elif status == SystemStatus.BAD_POSTURE:
            return self.colors.BAD_RED
        elif status in (SystemStatus.DROWSY_EYES, SystemStatus.DROWSY_YAWN):
            return self.colors.ALERT_AMBER
        elif status == SystemStatus.PHONE_DISTRACTION:
            return self.colors.PHONE_PURPLE
        elif status == SystemStatus.CALIBRATING:
            return self.colors.CYAN
        return self.colors.GRAY

    def render(
        self,
        frame: np.ndarray,
        status: SystemStatus,
        posture_result: Optional[PostureStatusResult],
        landmark_points: Optional[dict],
        phone_boxes: List[DetectedObject],
        ear: float,
        mar: float,
        fps: float,
        is_muted: bool,
        violation_count: int,
        notification_text: Optional[str] = None,
    ) -> np.ndarray:
        """Render complete HUD overlay onto frame."""
        output = frame.copy()
        h, w, _ = output.shape

        # 1. Top Status Banner
        banner_h = 60
        status_color = self._get_status_color(status)
        self._draw_transparent_rect(output, 0, 0, w, banner_h, self.colors.DARK_BG, alpha=0.75)
        # Colored indicator strip at the top
        cv2.rectangle(output, (0, 0), (w, 5), status_color, -1)

        # Status text
        status_text = status.value
        cv2.circle(output, (30, 32), 10, status_color, -1)
        cv2.putText(
            output,
            status_text,
            (55, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # FPS & Mute badge on top-right
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            output,
            fps_text,
            (w - 180, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.colors.WHITE,
            1,
            cv2.LINE_AA,
        )

        mute_text = "SOUND: MUTED" if is_muted else "SOUND: ON"
        mute_color = self.colors.BAD_RED if is_muted else self.colors.GOOD_GREEN
        cv2.putText(
            output,
            mute_text,
            (w - 180, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            mute_color,
            1,
            cv2.LINE_AA,
        )

        # 2. Telemetry Card (Upper-Left)
        card_w, card_h = 290, 200
        card_x, card_y = 20, 80
        self._draw_transparent_rect(output, card_x, card_y, card_w, card_h, self.colors.PANEL_BG, alpha=0.70)
        cv2.rectangle(output, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

        cv2.putText(output, "TELEMETRY & VITALS", (card_x + 15, card_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, self.colors.WHITE, 1, cv2.LINE_AA)
        cv2.line(output, (card_x + 15, card_y + 32), (card_x + card_w - 15, card_y + 32), (100, 100, 100), 1)

        line_y = card_y + 55
        if posture_result is not None:
            angles = posture_result.angles
            neck_delta_str = f"(diff: {posture_result.delta_neck:+.1f} deg)"
            neck_color = self.colors.BAD_RED if posture_result.delta_neck > self.config.neck_angle_thresh_deg else self.colors.WHITE
            cv2.putText(output, f"Neck Angle: {angles.neck_angle:.1f} deg", (card_x + 15, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, neck_color, 1, cv2.LINE_AA)
            cv2.putText(output, neck_delta_str, (card_x + 175, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.colors.GRAY, 1, cv2.LINE_AA)

            line_y += 25
            torso_delta_str = f"(diff: {posture_result.delta_torso:+.1f} deg)"
            torso_color = self.colors.BAD_RED if posture_result.delta_torso > self.config.torso_angle_thresh_deg else self.colors.WHITE
            cv2.putText(output, f"Torso Tilt: {angles.torso_angle:.1f} deg", (card_x + 15, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, torso_color, 1, cv2.LINE_AA)
            cv2.putText(output, torso_delta_str, (card_x + 175, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.colors.GRAY, 1, cv2.LINE_AA)

            line_y += 25
            cv2.putText(output, f"Shoulder Tilt: {angles.shoulder_tilt:.1f} deg", (card_x + 15, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.colors.WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "Posture: Calibrating / No Person", (card_x + 15, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.colors.GRAY, 1, cv2.LINE_AA)
            line_y += 50

        line_y += 25
        ear_color = self.colors.ALERT_AMBER if ear < self.config.ear_threshold else self.colors.WHITE
        cv2.putText(output, f"Eye Aspect Ratio (EAR): {ear:.2f}", (card_x + 15, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ear_color, 1, cv2.LINE_AA)

        line_y += 25
        cv2.putText(output, f"Violations Count: {violation_count}", (card_x + 15, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.colors.ALERT_AMBER, 1, cv2.LINE_AA)

        # 3. Skeleton / Posture Lines
        if landmark_points is not None:
            m_ear = (int(landmark_points["mid_ear"][0]), int(landmark_points["mid_ear"][1]))
            m_sh = (int(landmark_points["mid_shoulder"][0]), int(landmark_points["mid_shoulder"][1]))
            m_hip = (int(landmark_points["mid_hip"][0]), int(landmark_points["mid_hip"][1]))
            l_sh = (int(landmark_points["left_shoulder"][0]), int(landmark_points["left_shoulder"][1]))
            r_sh = (int(landmark_points["right_shoulder"][0]), int(landmark_points["right_shoulder"][1]))

            skel_color = self.colors.GOOD_GREEN if (posture_result and posture_result.is_good) else self.colors.BAD_RED
            # Draw neck line (ear -> shoulder)
            cv2.line(output, m_ear, m_sh, skel_color, 3, cv2.LINE_AA)
            # Draw spine line (shoulder -> hip)
            cv2.line(output, m_sh, m_hip, skel_color, 3, cv2.LINE_AA)
            # Draw shoulder bar
            cv2.line(output, l_sh, r_sh, (255, 255, 255), 2, cv2.LINE_AA)
            # Joint points
            for pt in [m_ear, m_sh, m_hip, l_sh, r_sh]:
                cv2.circle(output, pt, 5, self.colors.CYAN, -1, cv2.LINE_AA)

        # 4. Phone Bounding Boxes
        for detected in phone_boxes:
            x1, y1, x2, y2 = detected.box
            cv2.rectangle(output, (x1, y1), (x2, y2), self.colors.PHONE_PURPLE, 3)
            label = f"PHONE ({detected.confidence*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - 25), (x1 + tw + 10, y1), self.colors.PHONE_PURPLE, -1)
            cv2.putText(output, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors.WHITE, 2, cv2.LINE_AA)

        # 5. Notification Toast / Flash message
        if notification_text:
            notif_w = 460
            notif_x = (w - notif_w) // 2
            notif_y = h - 90
            self._draw_transparent_rect(output, notif_x, notif_y, notif_w, 35, self.colors.DARK_BG, alpha=0.85)
            cv2.rectangle(output, (notif_x, notif_y), (notif_x + notif_w, notif_y + 35), self.colors.CYAN, 1)
            cv2.putText(output, notification_text, (notif_x + 20, notif_y + 23), cv2.FONT_HERSHEY_DUPLEX, 0.55, self.colors.CYAN, 1, cv2.LINE_AA)

        # 6. Bottom Legend Bar
        bar_h = 35
        self._draw_transparent_rect(output, 0, h - bar_h, w, bar_h, self.colors.DARK_BG, alpha=0.85)
        legend_str = "[C] Calibrate   |   [M] Mute   |   [O] Rotate   |   [R] Reset   |   [Q] Quit"
        cv2.putText(
            output,
            legend_str,
            (w // 2 - 340, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            self.colors.WHITE,
            1,
            cv2.LINE_AA,
        )

        return output

