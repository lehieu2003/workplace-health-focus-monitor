"""Workplace Health & Focus Monitor - Main Pipeline & Real-Time Loop."""

import argparse
import sys
import time
import cv2
from src.config import AppConfig, SystemStatus
from src.posture_detector import PostureDetector
from src.drowsiness_detector import DrowsinessDetector, DrowsinessStatus
from src.phone_detector import PhoneDetector
from src.audio_alerter import AudioAlerter
from src.hud_renderer import HUDRenderer

def parse_args():
    parser = argparse.ArgumentParser(description="Workplace Health & Focus Monitor")
    parser.add_argument("--camera", type=str, default=None, help="Camera index (e.g. 0) or IP camera stream URL (overrides .env)")
    parser.add_argument("--cli", action="store_true", help="Run in legacy standalone OpenCV window mode instead of Menu Bar")
    parser.add_argument("--no-sound", action="store_true", help="Disable audio alerts")
    parser.add_argument("--no-mirror", action="store_true", help="Do not flip frame horizontally")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None, help="Rotate video stream by degrees (0, 90, 180, 270)")
    parser.add_argument("--width", type=int, default=None, help="Window width (default from .env)")
    parser.add_argument("--height", type=int, default=None, help="Window height (default from .env)")
    parser.add_argument("--synthetic", action="store_true", help="Run in synthetic video mode for testing")
    return parser.parse_args()

def parse_camera_source(cam_arg: str):
    """Parse camera argument into either int index or string stream URL."""
    cam_arg = cam_arg.strip()
    if cam_arg.isdigit():
        return int(cam_arg)
    # If user passed IP camera root like http://192.168.1.23:8080 without /video, auto-append /video
    if cam_arg.startswith("http://") or cam_arg.startswith("https://"):
        if not cam_arg.endswith("/video") and not cam_arg.endswith(".mjpg") and not cam_arg.endswith("/mjpeg"):
            cam_arg = cam_arg.rstrip("/") + "/video"
    return cam_arg

def run_menubar_app(config: AppConfig, synthetic: bool = False):
    """Run in background macOS Menu Bar application mode (Default)."""
    from src.notification_manager import NotificationManager
    from src.vision_worker import VisionWorker
    from src.menu_bar_app import HealthMonitorMenuBarApp

    print("=" * 60)
    print(" 🍏 Workplace Health & Focus Monitor (macOS Menu Bar Mode)")
    print("=" * 60)
    print(" Khởi động Vision Worker chạy ngầm và tối ưu CPU...")

    posture_detector = PostureDetector(config=config)
    drowsiness_detector = DrowsinessDetector(config=config)
    phone_detector = PhoneDetector(config=config, load_model=True)
    audio_alerter = AudioAlerter(config=config, enable_sound=config.enable_sound)
    notif_manager = NotificationManager(enabled=True, send_system_banner=True)
    hud_renderer = HUDRenderer(config=config)

    worker = VisionWorker(
        config=config,
        posture_detector=posture_detector,
        drowsiness_detector=drowsiness_detector,
        phone_detector=phone_detector,
        audio_alerter=audio_alerter,
        notification_manager=notif_manager,
        hud_renderer=hud_renderer,
        synthetic_mode=synthetic,
    )
    worker.start()

    print("\n🚀 Ứng dụng đã chạy thành công trên thanh Menu Bar của macOS!")
    print("👉 Hãy nhìn vào góc trên bên phải màn hình Mac của bạn (biểu tượng 🧘).")
    print("💡 Click vào biểu tượng để: Căn chỉnh tư thế, Xem thống kê hoặc Bật cửa sổ Camera.\n")

    app = HealthMonitorMenuBarApp(worker=worker)
    try:
        app.run()
    finally:
        worker.stop()

def run_cli_monitor(config: AppConfig, synthetic: bool = False):
    """Run in standalone OpenCV window mode (Legacy/Demo)."""
    print("=" * 60)
    print(" Workplace Health & Focus Monitor (Standalone OpenCV Window)")
    print("=" * 60)
    print(" Initializing AI models (MediaPipe Pose, Face Mesh, YOLOv8)...")

    posture_detector = PostureDetector(config=config)
    drowsiness_detector = DrowsinessDetector(config=config)
    phone_detector = PhoneDetector(config=config, load_model=True)
    audio_alerter = AudioAlerter(config=config, enable_sound=config.enable_sound)
    hud_renderer = HUDRenderer(config=config)

    print(" Models loaded successfully.")
    print("\n Keyboard Shortcuts:")
    print("   [C] - Calibrate sitting posture baseline")
    print("   [M] - Toggle audio alerts (Mute / Unmute)")
    print("   [O] - Rotate camera feed (0 -> 90 -> 180 -> 270)")
    print("   [R] - Reset session violation stats")
    print("   [Q] - Quit application\n")

    cam_source = config.camera_source

    if synthetic:
        # Synthetic frame generator for CI / tests
        import numpy as np
        cap = None
        print(" Running in SYNTHETIC frame mode...")
    else:
        if isinstance(cam_source, str):
            print(f"📡 Connecting to IP camera stream: {cam_source} ...")
            cap = cv2.VideoCapture(cam_source)
            if not cap.isOpened():
                print(f"❌ Could not connect to IP camera at: {cam_source}")
                print("👉 Please check that your phone and Mac are on the same Wi-Fi network")
                print("   and the 'IP Webcam' app server is active on your phone.")
                return
            print("✅ Successfully connected to IP camera stream!")
        else:
            cap = cv2.VideoCapture(cam_source)
            if not cap.isOpened():
                print(f"⚠️  Could not open camera {cam_source}.")
                print("👉 On macOS, ensure Terminal / IDE has Camera Permissions enabled in:")
                print("   System Settings > Privacy & Security > Camera")
                print("Trying fallback camera index 1...")
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    print("❌ No webcam available. Exiting.")
                    return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    window_name = "Workplace Health & Focus Monitor - Live Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.frame_width, config.frame_height)

    # Telemetry and state variables
    frame_count = 0
    fps = 30.0
    fps_start_time = time.time()
    notification_msg = "Press [C] when sitting upright to Calibrate!"
    notification_until = time.time() + 5.0
    violation_count = 0
    in_violation_state = False
    session_start_time = time.time()
    current_rotation = config.rotate_deg

    bad_posture_start_time = None
    phone_start_time = None

    try:
        while True:
            t0 = time.time()
            frame_count += 1

            if synthetic:
                import numpy as np
                frame = np.zeros((config.frame_height, config.frame_width, 3), dtype=np.uint8)
                time.sleep(0.03)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("⚠️ Failed to read webcam frame.")
                    break

            # Apply rotation if specified (or toggled via 'O' key)
            if current_rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif current_rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif current_rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Mirror view for intuitive user experience (can be disabled via --no-mirror)
            if not config.no_mirror:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Posture Inference
            angles, landmark_points = posture_detector.process_frame(frame_rgb, w, h)
            posture_result = None
            if angles is not None:
                posture_result = posture_detector.evaluate(angles)

            # 2. Drowsiness Inference
            fatigue_status, ear, mar = drowsiness_detector.process_frame(frame_rgb, w, h, current_time=t0)

            # 3. Phone Distraction Inference
            has_phone, phone_boxes = phone_detector.detect(frame, frame_count=frame_count)

            # 4. Debounce Phone & Posture
            is_phone_violating = False
            if has_phone:
                if phone_start_time is None:
                    phone_start_time = t0
                elif (t0 - phone_start_time) >= 1.5:
                    is_phone_violating = True
            else:
                phone_start_time = None

            is_posture_violating = False
            if posture_result and not posture_result.is_good:
                if bad_posture_start_time is None:
                    bad_posture_start_time = t0
                elif (t0 - bad_posture_start_time) >= config.posture_debounce_sec:
                    is_posture_violating = True
            else:
                bad_posture_start_time = None

            # 5. Global State Aggregation & Priority Decision
            current_status = SystemStatus.GOOD_POSTURE
            is_violating = False

            if landmark_points is None:
                current_status = SystemStatus.NO_PERSON
            elif not posture_detector.is_calibrated:
                current_status = SystemStatus.CALIBRATING
            elif is_phone_violating:
                current_status = SystemStatus.PHONE_DISTRACTION
                is_violating = True
                audio_alerter.trigger(t0)
            elif fatigue_status == DrowsinessStatus.EYES_CLOSED:
                current_status = SystemStatus.DROWSY_EYES
                is_violating = True
                audio_alerter.trigger(t0)
            elif fatigue_status == DrowsinessStatus.YAWNING:
                current_status = SystemStatus.DROWSY_YAWN
                is_violating = True
                audio_alerter.trigger(t0)
            elif is_posture_violating:
                current_status = SystemStatus.BAD_POSTURE
                is_violating = True
                audio_alerter.trigger(t0)
            elif posture_result and not posture_result.is_good:
                current_status = SystemStatus.BAD_POSTURE
            else:
                current_status = SystemStatus.GOOD_POSTURE

            # Track violation transitions
            if is_violating and not in_violation_state:
                violation_count += 1
                in_violation_state = True
            elif not is_violating:
                in_violation_state = False

            # Calculate FPS moving average
            elapsed = time.time() - t0
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = 0.9 * fps + 0.1 * current_fps

            # Render HUD Overlay
            active_notification = notification_msg if time.time() < notification_until else None
            rendered_frame = hud_renderer.render(
                frame=frame,
                status=current_status,
                posture_result=posture_result,
                landmark_points=landmark_points,
                phone_boxes=phone_boxes,
                ear=ear,
                mar=mar,
                fps=fps,
                is_muted=audio_alerter.is_muted,
                violation_count=violation_count,
                notification_text=active_notification,
            )

            cv2.imshow(window_name, rendered_frame)

            # Key Listener
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC to quit
                break
            elif key in (ord('c'), ord('C')):    # C to calibrate
                if angles is not None:
                    posture_detector.calibrate(angles)
                    notification_msg = f"Calibrated! Baseline: Neck={angles.neck_angle:.1f}deg, Torso={angles.torso_angle:.1f}deg"
                else:
                    notification_msg = "Cannot calibrate: Please sit in front of camera!"
                notification_until = time.time() + 3.0
            elif key in (ord('m'), ord('M')):    # M to toggle mute
                muted = audio_alerter.toggle_mute()
                notification_msg = "Audio alerts MUTED" if muted else "Audio alerts ENABLED"
                notification_until = time.time() + 2.0
            elif key in (ord('r'), ord('R')):    # R to reset stats
                violation_count = 0
                notification_msg = "Session stats reset to 0."
                notification_until = time.time() + 2.0
            elif key in (ord('o'), ord('O')):    # O to rotate 90 degrees
                current_rotation = (current_rotation + 90) % 360
                notification_msg = f"Camera rotation: {current_rotation} deg"
                notification_until = time.time() + 2.0

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        total_time = time.time() - session_start_time
        print("\n" + "=" * 60)
        print(" Session Summary:")
        print(f"   Total Duration: {total_time:.1f} seconds")
        print(f"   Total Violations Recorded: {violation_count}")
        print("=" * 60)

def run_monitor():
    args = parse_args()
    config = AppConfig.from_env()

    # Apply command-line overrides if supplied
    if args.camera is not None:
        cam_source = parse_camera_source(args.camera)
        config.camera_source = cam_source
        config.camera_index = cam_source if isinstance(cam_source, int) else 0
    else:
        if isinstance(config.camera_source, str):
            config.camera_source = parse_camera_source(config.camera_source)

    if args.rotate is not None:
        config.rotate_deg = args.rotate
    if args.no_mirror:
        config.no_mirror = True
    if args.no_sound:
        config.enable_sound = False
    if args.width is not None:
        config.frame_width = args.width
    if args.height is not None:
        config.frame_height = args.height

    if args.cli:
        run_cli_monitor(config, synthetic=args.synthetic)
    else:
        run_menubar_app(config, synthetic=args.synthetic)

if __name__ == "__main__":
    run_monitor()

