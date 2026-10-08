# macOS Menu Bar & Background Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Workplace Health & Focus Monitor into a silent, battery-efficient macOS Menu Bar Application that runs unobtrusively in the background, displays real-time status in the status bar, sends native macOS notification banners when posture is bad for > 5s, and allows toggling camera preview on demand.

**Architecture:** A multi-threaded architecture where `VisionWorker` runs camera ingestion and CV inference in a daemon thread with dynamic CPU throttling, `NotificationManager` manages debounced native macOS notification banners, and `MenuBarApp` (built on `rumps`/Cocoa) manages status bar title, dropdown menus, and user actions.

**Tech Stack:** Python 3.9+, rumps, pyobjc-framework-Cocoa, OpenCV, MediaPipe, Ultralytics YOLOv8, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-menubar-background-mode-design.md`

## Global Constraints
- Do not block macOS Cocoa event loop.
- Throttled background CPU utilization: sleep between frames when preview is hidden (~10-15 FPS), saving energy on Mac.
- Notifications: trigger ONLY when violation persists $> 5.0s$ continuously, cooldown between notifications $\ge 30.0s$.
- Maintain backward compatibility: `--cli` launches fullscreen OpenCV interactive mode as before.

---

### Task 1: macOS Notification Manager
**Files:**
- Create: `src/notification_manager.py`
- Test: `tests/test_notification_manager.py`

**Interfaces:**
- Produces: `NotificationManager` class:
  - `check_and_notify(is_violating: bool, status_message: str, current_time: float) -> bool`: returns True if notification was sent.
  - `enabled: bool` property (toggleable).

- [ ] **Step 1: Write unit tests in `tests/test_notification_manager.py`**
  Test 5-second sustained threshold, notification cooldown, and enabled/disabled toggle.
- [ ] **Step 2: Implement `src/notification_manager.py`**
  Uses `osascript` to trigger native banners asynchronously.
- [ ] **Step 3: Run pytest to verify**

---

### Task 2: Background Vision Worker Thread
**Files:**
- Create: `src/vision_worker.py`
- Test: `tests/test_vision_worker.py`

**Interfaces:**
- Produces: `VisionWorker` class:
  - `start()`, `stop()`: lifecycle controls.
  - `calibrate()`: triggers posture calibration.
  - `toggle_preview() -> bool`: shows/hides camera window.
  - `get_state() -> WorkerState`: thread-safe snapshot of current status, stats, and telemetry.

- [ ] **Step 1: Write unit test in `tests/test_vision_worker.py`**
  Test thread start/stop, state querying, and score calculation.
- [ ] **Step 2: Implement `src/vision_worker.py`**
  Implement the frame loop, detector execution, dynamic sleep throttling, and OpenCV preview window management.
- [ ] **Step 3: Run pytest to verify**

---

### Task 3: macOS Menu Bar Application (`rumps`)
**Files:**
- Create: `src/menu_bar_app.py`

**Interfaces:**
- Consumes: `VisionWorker` and `AppConfig`.
- Produces: `MenuBarApp(rumps.App)`:
  - Timer loop updating icon & title (e.g. `🧘 95%` or `⚠️ Slouch`).
  - Menu handlers: Preview toggle, Calibrate, Mute sound, Notifications toggle, Daily stats popup, Quit.

- [ ] **Step 1: Implement `src/menu_bar_app.py`**
- [ ] **Step 2: Verify component creation and menu setup via unit test**

---

### Task 4: Unified Main Entrypoint & CLI Integration
**Files:**
- Modify: `src/main.py`
- Modify: `README.md`

**Interfaces:**
- CLI flags: `--cli` (runs fullscreen interactive window) vs default (runs Menu Bar App).

- [ ] **Step 1: Update `src/main.py` to route to `MenuBarApp` or `run_cli_monitor` based on `--cli` flag**
- [ ] **Step 2: Update `README.md` with Menu Bar usage and background workflow**
- [ ] **Step 3: Run complete pytest suite**

---

### Task 5: End-to-End Verification
- [ ] **Step 1: Run complete test suite (`pytest tests/ -v`)**
- [ ] **Step 2: Test Menu Bar app launch and background execution**

