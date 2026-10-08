# Workplace Health & Focus Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time computer vision application on macOS (Mac Mini M4) to monitor posture, detect drowsiness (eye closure/yawning), detect smartphone usage, and provide instant visual HUD status and audio alerts.

**Architecture:** A modular pipeline utilizing OpenCV for camera capture, MediaPipe Pose for spinal/neck angles, MediaPipe Face Mesh for EAR/MAR fatigue detection, asynchronous YOLOv8n for phone detection, macOS native non-blocking audio alerts (`afplay`), and a real-time OpenCV HUD overlay with keyboard shortcuts (`C`, `M`, `R`, `Q`).

**Tech Stack:** Python 3.9+, OpenCV (`opencv-python`), MediaPipe (`mediapipe`), Ultralytics YOLOv8 (`ultralytics`), NumPy (`numpy`), Pytest (`pytest`).

**Spec:** `docs/superpowers/specs/2026-10-08-workplace-focus-monitor-design.md`

## Global Constraints
- Target platform: macOS on Apple Silicon (M4).
- Audio system: non-blocking background sound trigger (`afplay /System/Library/Sounds/Ping.aiff` or `/System/Library/Sounds/Tink.aiff`).
- Latency goal: maintain >= 30 FPS webcam stream without freezing.
- Interactive controls: `C` (Calibrate baseline), `M` (Mute/Unmute audio), `R` (Reset statistics), `Q` (Quit).

## Review Focus
- Camera feed unavailable or permission denied -> graceful error message instead of crash.
- No person detected in frame -> display "NO PERSON DETECTED" instead of throwing landmark index errors.
- Extreme head turning (landmark occlusion) -> maintain last valid state or flag status gracefully without jumping angles.
- Rapid blinks (< 0.3s) -> do not trigger drowsiness alarm (requires debounce >= 1.5s).
- Audio spam -> debounce cooldown of >= 2.0s between consecutive alerts.

---

### Task 1: Environment & Project Scaffolding
**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: Project root structure and verified virtual environment.

- [ ] **Step 1: Create `requirements.txt`**
  Add dependencies: `opencv-python>=4.8.0`, `mediapipe>=0.10.0`, `ultralytics>=8.0.0`, `numpy>=1.23.0`, `pytest>=7.0.0`.
- [ ] **Step 2: Create `src/__init__.py` and `tests/__init__.py`**
- [ ] **Step 3: Verify Python environment & package imports**
  Run quick check command.

---

### Task 2: Configuration Module
**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AppConfig` class with all geometric thresholds, time debounce parameters, and color constants (BGR).

- [ ] **Step 1: Write failing test in `tests/test_config.py`**
  Test default configuration parameters: `CAMERA_INDEX`, `NECK_ANGLE_THRESH_DEG = 15.0`, `TORSO_ANGLE_THRESH_DEG = 12.0`, `EAR_THRESH = 0.21`, `MAR_THRESH = 0.65`, `ALERT_COOLDOWN_SEC = 2.0`.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python3 -m pytest tests/test_config.py`
- [ ] **Step 3: Implement `src/config.py`**
- [ ] **Step 4: Run test to verify it passes**

---

### Task 3: Posture Detection & Calibration Engine
**Files:**
- Create: `src/posture_detector.py`
- Test: `tests/test_posture.py`

**Interfaces:**
- Consumes: `AppConfig` from `src/config.py`
- Produces: `PostureDetector` class:
  - `calibrate(landmarks)`: sets baseline angles $\theta_{neck}^{base}$ and $\theta_{torso}^{base}$.
  - `calculate_angles(landmarks) -> PostureAngles`: returns current neck angle, torso angle, shoulder tilt angle.
  - `evaluate_posture(angles) -> PostureStatus`: returns boolean `is_good`, reason string, and angle deviations.

- [ ] **Step 1: Write test for geometric angle computation and calibration in `tests/test_posture.py`**
  Mock landmark coordinates (ear, shoulder, hip) for straight posture and forward slouching.
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement `src/posture_detector.py`**
  Include mathematical angle calculation using `math.atan2` and `math.degrees`, baseline difference comparison, and MediaPipe Pose wrapper.
- [ ] **Step 4: Run test to verify it passes**

---

### Task 4: Drowsiness & Fatigue Detection Engine
**Files:**
- Create: `src/drowsiness_detector.py`
- Test: `tests/test_drowsiness.py`

**Interfaces:**
- Consumes: `AppConfig` from `src/config.py`
- Produces: `DrowsinessDetector` class:
  - `calculate_ear(eye_landmarks) -> float`: computes Eye Aspect Ratio.
  - `calculate_mar(mouth_landmarks) -> float`: computes Mouth Aspect Ratio.
  - `update(face_landmarks, current_time) -> FatigueStatus`: tracks consecutive closure/yawn duration and returns status (`NORMAL`, `EYES_CLOSED`, `YAWNING`).

- [ ] **Step 1: Write test for EAR and MAR formulas with mock eye/mouth coordinates in `tests/test_drowsiness.py`**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement `src/drowsiness_detector.py`**
  Implement Euclidean distance calculations and temporal sliding window duration checks.
- [ ] **Step 4: Run test to verify it passes**

---

### Task 5: Non-blocking Audio Alerter
**Files:**
- Create: `src/audio_alerter.py`
- Test: `tests/test_audio_alerter.py`

**Interfaces:**
- Consumes: `ALERT_COOLDOWN_SEC` from `src/config.py`
- Produces: `AudioAlerter` class:
  - `play_beep()`: runs macOS `afplay` in a non-blocking daemon thread with debounce cooldown check.
  - `toggle_mute() -> bool`: toggles mute state on/off.
  - `is_muted: bool` property.

- [ ] **Step 1: Write unit test in `tests/test_audio_alerter.py`**
  Test mute toggle and cooldown debouncing logic without blocking.
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement `src/audio_alerter.py`**
- [ ] **Step 4: Run test to verify it passes**

---

### Task 6: Phone Detection Engine (YOLOv8n Async)
**Files:**
- Create: `src/phone_detector.py`
- Test: `tests/test_phone_detector.py`

**Interfaces:**
- Produces: `PhoneDetector` class:
  - `detect(frame) -> List[BoundingBox]`: runs YOLOv8n inference (class 67 `cell phone`) every N frames or asynchronously, returning detected phone boxes.
  - `is_phone_detected: bool` property.

- [ ] **Step 1: Write test in `tests/test_phone_detector.py`**
  Test filtering logic and detection state tracking.
- [ ] **Step 2: Implement `src/phone_detector.py`**
- [ ] **Step 3: Run test to verify passes**

---

### Task 7: HUD Visual Overlay Renderer
**Files:**
- Create: `src/hud_renderer.py`

**Interfaces:**
- Consumes: Status from Posture, Fatigue, and Phone detectors.
- Produces: `HUDRenderer` class:
  - `render(frame, state, angles, fatigue, phone_boxes, fps) -> np.ndarray`: draws top status banner (Green/Red/Amber), left telemetry card, skeleton lines, and bottom shortcut bar.

- [ ] **Step 1: Implement `src/hud_renderer.py`**
  Create clean alpha-blended transparent overlay banner with OpenCV, sharp text rendering, indicator dots, and angle difference displays.

---

### Task 8: Main Application Pipeline & Integration
**Files:**
- Create: `src/main.py`
- Create: `README.md`

**Interfaces:**
- Combines VideoCapture, all detectors, audio alerter, and HUD renderer into a 30+ FPS main loop with key listeners (`C`, `M`, `R`, `Q`).

- [ ] **Step 1: Implement `src/main.py`**
- [ ] **Step 2: Implement `README.md` with setup guide, live demo script, and shortcuts**
- [ ] **Step 3: Run synthetic end-to-end pipeline test on video stream**

---

### Task 9: Live Demo & Presentation Verification
- [ ] **Step 1: Test webcam feed and ensure proper Mac camera permissions**
- [ ] **Step 2: Calibrate posture via `C` key**
- [ ] **Step 3: Verify all 3 behaviors: Good Posture (Green), Slouching (Red + Bip), Eyes closed / Phone use (Alert)**

