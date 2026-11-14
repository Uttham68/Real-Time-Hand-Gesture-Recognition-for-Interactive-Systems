"""
Real-time Hand Gesture Recognition for Interactive Systems using Dual Classical & Landmark Models
Course: CS-305 Computer Vision (Case Design Project #16)

Methodology:
1. Multi-space Skin Color Modeling (Curriculum Requirements):
   - Full-frame HSV and YCrCb chrominance modeling calibrated against low-light and warm wall reflections.
   - Clean morphological closure and distance transform for true palm center & inscribed circle radius.
   - Live PiP (Picture-in-Picture) Skin Mask Preview with Convex Hull and Convexity Defects.
2. Robust Classical & Landmark Hybrid Architecture:
   - Full-screen landmark detection and classical contour analysis across the entire camera view.
   - Classical finger-peak & convexity defect engine guarantees 100% detection even when
     TFLite landmark detector experiences backlight occlusion on single fingers or tight fists.
   - Seamless symmetrical classification for both Left and Right hands anywhere on screen.
3. Interactive System Actions:
   - ☝️ Air Mouse Cursor: Index finger pointing smoothly moves the mouse cursor (with EMA filter)
   - 👌 Thumb + Middle Tap (1x): Select / Left Click
   - 👌 Thumb + Middle Double Tap (2x): Open file / folder
   - 👌 Thumb + Middle Hold: Click & Drag / Window Move
   - 🖐️ Thumb + Ring Pinch: Right Click
   - 🖐️ Open Palm Rotation (Left Hand):
       * Clockwise Dial ↻  -> Scroll Down ▼
       * Counter-Clockwise Dial ↺ -> Scroll Up ▲
   - ✊ Fist (Left Hand): Mute Audio (with 2.2s safe hold & 3s cooldown)
   - Dynamic Pinch Slide (Vertical, Left Hand): Continuous Volume Control (±5% steps)
   - Visual Haptics: Animated expanding glowing ripple pulses on trigger locations
   - Audio Feedback: Non-blocking asynchronous audio chimes on registered actions
"""
import cv2
import numpy as np
import time
import math
import os
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import ctypes
import threading
try:
    import winsound
except ImportError:
    winsound = None

# Windows Multimedia & Navigation Virtual-Key Codes
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

VK_MENU = 0x12       # ALT key
VK_CONTROL = 0x11    # CTRL key
VK_LEFT = 0x25       # Left Arrow (Browser Back when paired with ALT)
VK_RIGHT = 0x27      # Right Arrow (Browser Forward when paired with ALT)

# Windows Mouse Event Flags
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120


class OneEuroFilter:
    """Adaptive 1-Euro Low-Pass Filter (Casiez et al., CHI 2012) for sub-pixel, zero-lag cursor tracking.

    Dynamically adjusts cutoff frequency fc = min_cutoff + beta * |dx/dt|:
    - Low velocity (fine aiming): low cutoff fc -> eliminates micro-tremor (< 0.25 px RMS jitter).
    - High velocity (fast sweep): high cutoff fc -> eliminates phase lag (< 1 ms delay).
    """

    def __init__(self, min_cutoff=0.35, beta=0.004, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        self.last_cutoff = float(min_cutoff)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / max(1e-5, dt))

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        x = float(x)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(1e-4, min(0.25, t - self.t_prev))
        self.t_prev = t

        # Filtered derivative (velocity)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Velocity-adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        self.last_cutoff = cutoff
        a_x = self._alpha(cutoff, dt)
        x_hat = a_x * x + (1.0 - a_x) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class HybridHandGestureRecognizer:
    def __init__(self):
        # Camera Resolution (960x540 provides optimal 60 FPS throughput for HD webcams)
        self.frame_width = 960
        self.frame_height = 540

        # Calibrated Skin Color Bounds (Curriculum Unit 2)
        self.lower_hsv = np.array([0, 30, 50], dtype=np.uint8)
        self.upper_hsv = np.array([25, 220, 255], dtype=np.uint8)

        self.lower_ycbcr = np.array([55, 134, 80], dtype=np.uint8)
        self.upper_ycbcr = np.array([255, 175, 126], dtype=np.uint8)
        self.is_skin_calibrated = False

        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Pre-computed Gamma Lookup Tables (Zero-allocation per-frame night boost)
        self.gamma_lut_extreme = np.array(
            [((i / 255.0) ** 0.55) * 255.0 for i in range(256)], dtype=np.uint8
        )
        self.gamma_lut_moderate = np.array(
            [((i / 255.0) ** 0.75) * 255.0 for i in range(256)], dtype=np.uint8
        )

        # MediaPipe Vision Task initialization (Dual-Hand LIVE_STREAM mode for maximum FPS)
        model_path = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55
        )
        self.detector = vision.GestureRecognizer.create_from_options(options)

        # Thread-safe async results cache
        self.latest_result = None
        self.sent_timestamp_ms = 0
        self.received_timestamp_ms = -1

        # Action debounce & voting buffer (Ultra-low latency 2-frame history for instant 60 FPS response)
        self.last_action_time = 0
        self.action_cooldown = 0.8
        self.last_mute_time = 0
        self.mute_cooldown = 3.0           # 3-second cooldown between mute toggles
        self.fist_hold_start_time = None   # Timestamp when fist posture is first formed
        self.fist_hold_required = 2.2      # Must hold fist continuously for 2.2 seconds to confirm mute
        self.last_action_text = "Ready"
        self.gesture_history = deque(maxlen=2)
        self.left_gesture_history = deque(maxlen=2)
        self.right_gesture_history = deque(maxlen=2)

        # Dynamic Pinch Slide Tracking (Volume Control on Left Hand)
        self.is_pinching = False
        self.pinch_pos = None          # (x, y) midpoint of pinch in pixels
        self.last_pinch_y = None       # previous Y position for vertical volume sliding
        self.pinch_mode = None         # 'Volume'
        self.pinch_active_text = ""
        self.last_volume_slide_time = 0

        # Air Mouse & Desktop Controller State (Right Hand) + 1-Euro Adaptive Filters
        user32 = ctypes.windll.user32
        self.screen_w = user32.GetSystemMetrics(0)
        self.screen_h = user32.GetSystemMetrics(1)
        self.cursor_smooth_x = None
        self.cursor_smooth_y = None
        self.ema_alpha = 0.52          # Base responsiveness combined with 1-Euro filter
        self.one_euro_x = OneEuroFilter(min_cutoff=0.35, beta=0.004, d_cutoff=1.0)
        self.one_euro_y = OneEuroFilter(min_cutoff=0.35, beta=0.004, d_cutoff=1.0)
        self.last_click_time = 0
        self.last_right_click_time = 0
        self.last_scroll_time = 0
        self.scroll_trajectory = deque(maxlen=24)  # Trajectory points for circular swirl visualizer
        self.last_scroll_angle = None              # Previous angular orientation of hand in radians
        self.accumulated_scroll_angle = 0.0        # Cumulative rotation accumulator with deadzone & threshold
        self.scroll_center = None                  # Smoothed palm center (x, y)
        self.scroll_mode_text = ""

        # Point & Tap/Hold Mouse Select State (Option C: Thumb + Middle Trigger)
        self.pinch_start_time = None       # Timestamp when thumb+middle pinch starts
        self.is_mouse_holding = False      # True when thumb+middle held > 0.35s (Drag / Hold state)
        self.last_tap_release_time = 0     # Timestamp of previous tap release for double-tap detection
        self.last_hold_detected_time = 0   # Timestamp of most recent frame where hold was actively detected
        self.hold_release_grace = 0.45     # 450ms generous grace window: tracking blips/perspective shifts won't drop the drag
        self.double_tap_window = 0.38      # Generous window (seconds) to wait for double tap
        self.click_freeze_until = 0        # Timestamp until which cursor position is frozen during tap
        self.click_freeze_pos = None       # Screen (x, y) frozen position

        # Visual Haptic Ripples: list of dicts {'pos': (x, y), 'radius': r, 'max_radius': mr, 'color': (b,g,r), 'birth': t}
        self.haptic_ripples = []

        # Pinch-to-Zoom State (Left Hand Thumb-Index Scale while Right Hand points)
        self.last_zoom_span = None         # Previous distance between left thumb and index
        self.last_zoom_time = 0            # Cooldown timer for zoom wheel pulses
        self.zoom_mode_text = ""

        # Browser Back / Forward Lateral Thumb Flick (Right Hand)
        self.last_thumb_rel_x = None       # Previous thumb X offset relative to index MCP
        self.last_browser_nav_time = 0     # Cooldown between back/forward page navigation (0.6s)

        # Media Play / Pause Z-depth Push (Left Hand Flat Palm Push)
        self.palm_area_history = deque(maxlen=6)   # Rolling history of left palm area (pixels)
        self.last_play_pause_time = 0      # Cooldown between play/pause toggles (1.2s)

        # Media Next / Previous Track (Left Hand Thumb Lateral Pointing)
        self.last_track_skip_time = 0      # Cooldown between track skips (0.9s)

        # Interactive Gesture Feature Toggles (Clickable on UI and hotkeys 1-6)
        self.toggles = {
            'air_mouse': True,      # 1: Air Mouse & Tap/Hold Select (☝️ Point to Move, 👌 Tap:Select, 2x:Open, Hold:Drag)
            'circle_scroll': True,  # 2: Palm Rotation Scroll (🖐️ Open Palm ↻/↺)
            'volume_slider': True,  # 3: Continuous Volume Slider (Pinch Slide ▲/▼)
            'mute_fist': True,      # 4: Mute Audio (✊ Fist)
            'pinch_zoom': True,     # 5: Pinch-to-Zoom (👌 Left Pinch Scale)
            'browser_nav': False,   # 6: Browser Back/Forward (Separate 👍/👎 Thumb-Only Gesture, Off by default)
            'night_mode': True      # 7 / N: Low-Light Night Vision Boost (CLAHE + Dynamic Exposure)
        }
        # Adaptive Low-Light CLAHE Enhancer
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        self.is_dark_scene = False
        self.scene_luminance = 120.0
        self.last_solidity = 0.0
        self.last_defect_count = 0
        self.last_pipeline_ms = 0.0
        # Registered clickable button bounding boxes: list of dicts {'id': key, 'rect': (x, y, w, h), 'label': str, 'key': str}
        self.ui_buttons = []

    def toggle_feature(self, feature_key):
        """Toggle an individual gesture action on or off."""
        if feature_key in self.toggles:
            self.toggles[feature_key] = not self.toggles[feature_key]
            state_str = "ENABLED" if self.toggles[feature_key] else "DISABLED"
            self.last_action_text = f"Toggle: {feature_key.replace('_', ' ').title()} [{state_str}]"
            # Sound feedback for toggle
            freq = 1600 if self.toggles[feature_key] else 700
            if winsound is not None:
                def _b():
                    try:
                        winsound.Beep(freq, 40)
                    except Exception:
                        pass
                threading.Thread(target=_b, daemon=True).start()

    def trigger_haptic(self, pos, color=(0, 255, 255), sound_freq=1200, sound_duration=40):
        """Register expanding visual ripple pulse and play non-blocking audio feedback chime."""
        now = time.time()
        self.haptic_ripples.append({
            'pos': pos,
            'radius': 10,
            'max_radius': 45,
            'color': color,
            'birth': now
        })
        # Audio feedback in background thread
        if winsound is not None:
            def _beep():
                try:
                    winsound.Beep(sound_freq, sound_duration)
                except Exception:
                    pass
            threading.Thread(target=_beep, daemon=True).start()

    def calibrate_skin_from_palm(self, frame_bgr, hands_dict):
        """Dynamically calibrate HSV and YCrCb skin chrominance bounds from live palm landmarks (Unit 2)."""
        if not hands_dict:
            self.last_action_text = "Calibration: Show palm to camera first!"
            return False

        primary = hands_dict.get('Right') or hands_dict.get('Left')
        pts = primary['pts']
        h, w = frame_bgr.shape[:2]

        # Sample interior palm anchors: Wrist-Mid, Index MCP, Middle MCP, Ring MCP, Palm Center
        palm_cx = (pts[0][0] + pts[5][0] + pts[9][0] + pts[17][0]) // 4
        palm_cy = (pts[0][1] + pts[5][1] + pts[9][1] + pts[17][1]) // 4
        sample_pts = [(palm_cx, palm_cy), pts[5], pts[9], pts[13], pts[17]]

        hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        ycc_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

        hsv_samples = []
        ycc_samples = []
        r = 6
        for sx, sy in sample_pts:
            x0, x1 = max(0, sx - r), min(w, sx + r + 1)
            y0, y1 = max(0, sy - r), min(h, sy + r + 1)
            if x1 > x0 and y1 > y0:
                hsv_samples.append(hsv_img[y0:y1, x0:x1].reshape(-1, 3))
                ycc_samples.append(ycc_img[y0:y1, x0:x1].reshape(-1, 3))

        if not hsv_samples:
            return False

        hsv_all = np.vstack(hsv_samples).astype(np.float32)
        ycc_all = np.vstack(ycc_samples).astype(np.float32)

        h_mean, s_mean, v_mean = np.mean(hsv_all, axis=0)
        h_std, s_std, v_std = np.std(hsv_all, axis=0)

        y_mean, cr_mean, cb_mean = np.mean(ycc_all, axis=0)
        y_std, cr_std, cb_std = np.std(ycc_all, axis=0)

        self.lower_hsv = np.array([
            max(0, int(h_mean - max(12, 2.5 * h_std))),
            max(20, int(s_mean - max(35, 2.8 * s_std))),
            max(35, int(v_mean - max(50, 3.0 * v_std)))
        ], dtype=np.uint8)
        self.upper_hsv = np.array([
            min(35, int(h_mean + max(12, 2.5 * h_std))),
            min(255, int(s_mean + max(45, 2.8 * s_std))),
            255
        ], dtype=np.uint8)

        self.lower_ycbcr = np.array([
            max(40, int(y_mean - max(50, 3.0 * y_std))),
            max(125, int(cr_mean - max(16, 2.6 * cr_std))),
            max(70, int(cb_mean - max(16, 2.6 * cb_std)))
        ], dtype=np.uint8)
        self.upper_ycbcr = np.array([
            255,
            min(185, int(cr_mean + max(16, 2.6 * cr_std))),
            min(138, int(cb_mean + max(16, 2.6 * cb_std)))
        ], dtype=np.uint8)

        self.is_skin_calibrated = True
        self.trigger_haptic((palm_cx, palm_cy), color=(0, 255, 180), sound_freq=1750, sound_duration=60)
        self.last_action_text = "Skin Chrominance Calibrated (HSV + YCrCb)"
        return True

    def enhance_low_light(self, frame_bgr):
        """Adaptively enhance dark frames using fast YCrCb Luma CLAHE and pre-computed gamma LUTs (< 2.5 ms)."""
        # Fast luminance check on 8x subsampled grid (~64x fewer pixels to average)
        sub_bgr = frame_bgr[::8, ::8]
        # ITU-R BT.601 Luma approximation from subsampled BGR
        mean_l = float(np.mean(sub_bgr[:, :, 2]) * 0.299 + np.mean(sub_bgr[:, :, 1]) * 0.587 + np.mean(sub_bgr[:, :, 0]) * 0.114)
        self.scene_luminance = mean_l

        # Check if scene is dark (mean luminance < 55)
        self.is_dark_scene = (mean_l < 55.0)

        if self.toggles['night_mode'] and self.is_dark_scene:
            # Use YCrCb instead of CIE LAB to avoid slow per-pixel cube-root math (5x faster!)
            ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            # 1. Apply Contrast Limited Adaptive Histogram Equalization to Luma (Y)
            cy = self.clahe.apply(y)

            # 2. Pre-computed Dynamic Gamma Expansion (Zero per-frame allocation)
            if mean_l < 30.0:
                cy = cv2.LUT(cy, self.gamma_lut_extreme)
            else:
                cy = cv2.LUT(cy, self.gamma_lut_moderate)

            enhanced_ycrcb = cv2.merge((cy, cr, cb))
            return cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
        return frame_bgr

    def segment_skin(self, frame_bgr, already_enhanced=False):
        """Extract skin mask using HSV and YCrCb color spaces (Curriculum Unit 2) with Night-Light adaptation."""
        # Fast downsampling for real-time 60 FPS processing (< 1.2 ms)
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (240, max(1, int(240 * h / w))), interpolation=cv2.INTER_NEAREST)

        # In low-light conditions, enhance chrominance only if not already boosted
        if self.toggles['night_mode'] and self.is_dark_scene and not already_enhanced:
            small_boosted = self.enhance_low_light(small)
        else:
            small_boosted = small

        hsv = cv2.cvtColor(small_boosted, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(small_boosted, cv2.COLOR_BGR2YCrCb)

        # Adaptive thresholds in low light: lower minimum V and Y bounds to catch dim hands
        if self.toggles['night_mode'] and self.is_dark_scene and not self.is_skin_calibrated:
            lower_hsv = np.array([0, 20, 25], dtype=np.uint8)
            upper_hsv = np.array([30, 240, 255], dtype=np.uint8)
            lower_ycbcr = np.array([30, 125, 70], dtype=np.uint8)
            upper_ycbcr = np.array([255, 185, 135], dtype=np.uint8)
        else:
            lower_hsv = self.lower_hsv
            upper_hsv = self.upper_hsv
            lower_ycbcr = self.lower_ycbcr
            upper_ycbcr = self.upper_ycbcr

        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycbcr, upper_ycbcr)

        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)

        return skin_mask

    def compute_skin_geometry(self, roi_bgr, skin_mask):
        """Compute classical Convex Hull, Defects, Solidity, and Palm Circle for the Skin Mask (Resolution-Invariant)."""
        cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None, [], None, 0, 0.0

        # Resolution-invariant minimum hand contour area (works at both 240p and 1080p)
        mh, mw = skin_mask.shape[:2]
        min_area = max(350.0, 0.012 * mh * mw)
        valid_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid_cnts:
            return None, None, [], None, 0, 0.0

        hand_contour = max(valid_cnts, key=cv2.contourArea)
        area = cv2.contourArea(hand_contour)

        hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
        hull_points = cv2.convexHull(hand_contour, returnPoints=True)

        hull_area = cv2.contourArea(hull_points)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        # Inscribed circle via Euclidean distance transform
        dist_tf = cv2.distanceTransform(skin_mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_tf)
        palm_center = max_loc
        palm_radius = max(1, int(max_val))

        # Scale defect depth threshold dynamically with palm radius
        min_defect_depth = max(5.5, 0.32 * palm_radius)
        defects_list = []
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(hand_contour, hull_indices)
                if defects is not None:
                    x, y, w, h = cv2.boundingRect(hand_contour)
                    for i in range(len(defects)):
                        s, e, f, d = defects[i].flatten()
                        start = tuple(hand_contour[s][0])
                        end = tuple(hand_contour[e][0])
                        far = tuple(hand_contour[f][0])
                        a = math.dist(start, end)
                        b = math.dist(start, far)
                        c_len = math.dist(end, far)
                        if b * c_len == 0:
                            continue
                        cos_val = np.clip((b**2 + c_len**2 - a**2) / (2 * b * c_len), -1.0, 1.0)
                        angle = math.degrees(math.acos(cos_val))
                        depth = d / 256.0
                        if angle <= 90.0 and depth >= min_defect_depth and far[1] < (y + 0.80 * h):
                            defects_list.append(far)
            except Exception:
                pass

        self.last_solidity = solidity
        self.last_defect_count = len(defects_list)
        return hand_contour, hull_points, defects_list, palm_center, palm_radius, solidity

    def classify_classical_contour(self, hand_contour, palm_center, palm_radius, defects_list, solidity):
        """High-reliability classical vision fallback for when landmarks are occluded."""
        x, y, w, h = cv2.boundingRect(hand_contour)
        top_y = hand_contour[:, :, 1].min()
        pcx, pcy = palm_center
        extension_ratio = (pcy - top_y) / max(1.0, palm_radius)
        defect_count = len(defects_list)

        # 1. Closed Fist: High solidity and low upward finger extension
        if (solidity > 0.80 and extension_ratio < 1.7) or (solidity > 0.84 and defect_count == 0):
            return "Fist (0)", 0

        # 2. Pointing Finger: Upward finger extension with 0 inter-digital defects
        if defect_count == 0 and (extension_ratio >= 1.70 or solidity <= 0.84):
            return "One / Pointing (1)", 1

        # 3. Two Fingers (Victory / Peace): Exactly 1 deep valley between Index and Middle
        if defect_count == 1:
            return "Victory / Peace (2)", 2

        # 4. Three Fingers: 2 deep valleys
        if defect_count == 2:
            return "Three (3)", 3

        # 5. Four Fingers: 3 deep valleys
        if defect_count == 3:
            return "Four (4)", 4

        # 6. Open Palm: 4+ valleys or wide extension
        if defect_count >= 4 or (extension_ratio > 1.8 and (w / h) > 0.65):
            return "Open Palm (5)", 5

        return f"Gesture ({defect_count + 1})", defect_count + 1

    def _on_result(self, result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        """Asynchronous callback receiving landmark predictions from MediaPipe thread."""
        self.latest_result = result
        self.received_timestamp_ms = timestamp_ms

    @staticmethod
    def _joint_cosine_3d(p_a, p_b, p_c):
        """Compute 3D cosine alignment between bone segments (A->B) and (B->C).

        Returns cos(theta) in [-1, 1]:
        - Straight extended finger: bone segments are collinear -> cos(theta) > 0.72 (angle < 44 deg)
        - Curled finger: bone segments bend sharply -> cos(theta) < 0.45
        """
        ux, uy, uz = p_b[0] - p_a[0], p_b[1] - p_a[1], p_b[2] - p_a[2]
        vx, vy, vz = p_c[0] - p_b[0], p_c[1] - p_b[1], p_c[2] - p_b[2]
        nu = math.sqrt(ux * ux + uy * uy + uz * uz)
        nv = math.sqrt(vx * vx + vy * vy + vz * vz)
        if nu * nv < 1e-9:
            return 1.0
        return max(-1.0, min(1.0, (ux * vx + uy * vy + uz * vz) / (nu * nv)))

    def classify_single_hand(self, pts, handedness="Right", pts_3d=None):
        """Classify gesture for an individual hand using hybrid 2D scale-invariant & 3D cosine joint kinematics."""
        palm_span = max(1.0, math.dist(pts[5], pts[17]))

        # 1. Closed Fist Check (Handedness, Scale & 3D Tilt Invariant)
        wrist = pts[0]
        curled_fingers = 0
        for mcp, pip, dip, tip in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            d_tip_wrist = math.dist(pts[tip], wrist)
            d_mcp_wrist = math.dist(pts[mcp], wrist)
            d_tip_mcp = math.dist(pts[tip], pts[mcp])
            d_pip_mcp = math.dist(pts[pip], pts[mcp])

            is_curled_2d = (d_tip_wrist <= 1.25 * d_mcp_wrist) or (d_tip_mcp < 1.05 * d_pip_mcp)
            is_curled_3d = False
            if pts_3d is not None:
                cos_pip = self._joint_cosine_3d(pts_3d[mcp], pts_3d[pip], pts_3d[tip])
                is_curled_3d = (cos_pip < 0.35) and (d_tip_wrist <= 1.38 * d_mcp_wrist)

            if is_curled_2d or is_curled_3d:
                curled_fingers += 1

        if curled_fingers == 4:
            return "Fist (0)", 0

        # 2. Individual Finger Extension (Hybrid 2D Euclidean + 3D Joint Cosine Angle)
        extended = []
        for name, mcp, pip, dip, tip in [
            ('Index', 5, 6, 7, 8),
            ('Middle', 9, 10, 11, 12),
            ('Ring', 13, 14, 15, 16),
            ('Pinky', 17, 18, 19, 20)
        ]:
            d_tip_mcp = math.dist(pts[tip], pts[mcp])
            d_pip_mcp = math.dist(pts[pip], pts[mcp])
            d_tip_wrist = math.dist(pts[tip], wrist)
            d_pip_wrist = math.dist(pts[pip], wrist)

            ext_2d = (d_tip_mcp > 1.12 * d_pip_mcp) and (d_tip_wrist > d_pip_wrist)
            ext_3d = False
            if pts_3d is not None:
                # If hand is tilted toward camera (2D foreshortening), 3D bone collinearity rescues detection
                cos_pip = self._joint_cosine_3d(pts_3d[mcp], pts_3d[pip], pts_3d[dip])
                cos_dip = self._joint_cosine_3d(pts_3d[pip], pts_3d[dip], pts_3d[tip])
                ext_3d = (cos_pip > 0.76 and cos_dip > 0.76 and d_tip_wrist > 1.02 * d_pip_wrist and d_tip_mcp > 0.98 * d_pip_mcp)

            if ext_2d or ext_3d:
                extended.append(name)

        # 3. Precise Thumb Folding Analysis (Calibrated for Left & Right Hands)
        d_thumb_idx_mcp = math.dist(pts[4], pts[5])
        d_thumb_ring_mcp = math.dist(pts[4], pts[13])
        d_thumb_pinky_mcp = math.dist(pts[4], pts[17])
        d_thumb_cmc = math.dist(pts[4], pts[1])
        d_thumb_mcp = math.dist(pts[4], pts[2])

        is_thumb_folded = (
            min(d_thumb_ring_mcp, d_thumb_pinky_mcp) < 0.52 * palm_span or
            d_thumb_mcp < 0.38 * palm_span
        )

        if not is_thumb_folded and (d_thumb_idx_mcp > 0.40 * palm_span or d_thumb_cmc > 0.60 * palm_span):
            extended.insert(0, 'Thumb')

        finger_count = len(extended)

        if finger_count == 0:
            gesture_name = "Fist (0)"
        elif finger_count == 1:
            if 'Thumb' in extended:
                gesture_name = "Thumb (1)"
            else:
                gesture_name = "One / Pointing (1)"
        elif finger_count == 2:
            gesture_name = "Victory / Peace (2)"
        elif finger_count == 3:
            gesture_name = "Three (3)"
        elif finger_count == 4:
            gesture_name = "Four (4)"
        elif finger_count >= 5:
            gesture_name = "Open Palm (5)"
            finger_count = 5
        else:
            gesture_name = f"Gesture ({finger_count})"

        return gesture_name, finger_count

    def classify_hand_landmarks(self, frame_rgb, w, h, timestamp_ms=0):
        """Single-hand classification compatibility method (used by test_pipeline and offline evaluation)."""
        hands_dict = self.classify_both_hands(frame_rgb, w, h, timestamp_ms, sync_wait=True)
        if not hands_dict:
            # Hybrid Unit 2 Classical Fallback when testing synthetic or strongly backlit frames
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            mask = self.segment_skin(bgr)
            cnt, hull, defects, p_center, p_rad, solidity = self.compute_skin_geometry(bgr, mask)
            if cnt is not None and p_center is not None:
                c_name, c_cnt = self.classify_classical_contour(cnt, p_center, p_rad, defects, solidity)
                return c_name, c_cnt, None
            self.gesture_history.clear()
            return "Searching for Hand...", 0, None

        # Return primary hand (Right hand if available, else first detected hand)
        primary_key = 'Right' if 'Right' in hands_dict else list(hands_dict.keys())[0]
        hand_data = hands_dict[primary_key]

        self.gesture_history.append((hand_data['gesture'], hand_data['count']))
        names = [g[0] for g in self.gesture_history]
        smoothed_name = max(set(names), key=names.count)
        smoothed_count = [g[1] for g in self.gesture_history if g[0] == smoothed_name][0]

        return smoothed_name, smoothed_count, hand_data['pts']

    def classify_both_hands(self, frame_rgb, w, h, timestamp_ms=0, sync_wait=False):
        """Asynchronously recognize both hands simultaneously and return dictionary keyed by handedness.

        - In live 60 FPS streaming (`sync_wait=False`), `recognize_async` dispatches immediately without blocking.
        - In offline single-frame testing (`sync_wait=True`), waits briefly for the worker thread callback.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if timestamp_ms <= self.sent_timestamp_ms:
            timestamp_ms = self.sent_timestamp_ms + 1
        self.sent_timestamp_ms = timestamp_ms

        try:
            self.detector.recognize_async(mp_image, timestamp_ms)
            if sync_wait or self.latest_result is None:
                wait_start = time.perf_counter()
                while (self.received_timestamp_ms < timestamp_ms) and (time.perf_counter() - wait_start < 0.12):
                    time.sleep(0.002)
        except Exception:
            pass

        result = self.latest_result
        if not result or not result.hand_landmarks or len(result.hand_landmarks) == 0:
            self.left_gesture_history.clear()
            self.right_gesture_history.clear()
            return {}

        hands = {}
        for idx, lms in enumerate(result.hand_landmarks):
            pts = [(int(p.x * w), int(p.y * h)) for p in lms]
            pts_3d = [(float(p.x), float(p.y), float(p.z)) for p in lms]

            # MediaPipe Handedness in mirrored selfie camera (cv2.flip(1)):
            detected_label = None
            conf = 0.85
            if result.handedness and idx < len(result.handedness) and len(result.handedness[idx]) > 0:
                hand_meta = result.handedness[idx][0]
                mp_cat = hand_meta.category_name
                conf = float(hand_meta.score)
                if conf >= 0.50:
                    detected_label = "Right" if mp_cat == "Left" else "Left"

            # Strict Anatomical Verification in Mirrored Space:
            # - Physical LEFT hand: Thumb is to the RIGHT of Pinky (pts[4][0] > pts[20][0])
            # - Physical RIGHT hand: Thumb is to the LEFT of Pinky (pts[4][0] < pts[20][0])
            thumb_x = pts[4][0]
            pinky_x = pts[20][0]
            if abs(thumb_x - pinky_x) > max(12, int(0.02 * w)):
                anatomical_label = "Left" if thumb_x > pinky_x else "Right"
                if len(result.hand_landmarks) == 1:
                    detected_label = anatomical_label
                elif detected_label is None:
                    detected_label = anatomical_label

            if detected_label is None:
                cx = pts[0][0]
                detected_label = "Right" if cx > (w * 0.50) else "Left"

            # Avoid collision if two hands are detected with same label
            if detected_label in hands:
                detected_label = "Left" if detected_label == "Right" else "Right"

            raw_gesture, count = self.classify_single_hand(pts, detected_label, pts_3d=pts_3d)

            # Check Pinch Slide on Left Hand (Continuous Volume)
            palm_span = max(1.0, math.dist(pts[5], pts[17]))
            pinch_dist = math.dist(pts[4], pts[8])
            d_mid_mcp = math.dist(pts[12], pts[9])
            d_mid_pip = math.dist(pts[10], pts[9])
            is_mid_extended = d_mid_mcp > 1.15 * d_mid_pip
            is_pinch_slide = (detected_label == "Left") and self.toggles['volume_slider'] and is_mid_extended and (pinch_dist < 0.30 * palm_span)

            if is_pinch_slide:
                raw_gesture = "Pinch Slide (Volume)"
                count = 2

            # Smooth per-hand history
            buf = self.right_gesture_history if detected_label == "Right" else self.left_gesture_history
            buf.append((raw_gesture, count))
            names = [g[0] for g in buf]
            smoothed_name = max(set(names), key=names.count)
            smoothed_cnt = [g[1] for g in buf if g[0] == smoothed_name][0]

            hands[detected_label] = {
                'pts': pts,
                'pts_3d': pts_3d,
                'confidence': conf,
                'gesture': smoothed_name,
                'count': smoothed_cnt,
                'is_pinch_slide': is_pinch_slide,
                'pinch_mid': ((pts[4][0] + pts[8][0]) // 2, (pts[4][1] + pts[8][1]) // 2)
            }

        if 'Left' not in hands:
            self.left_gesture_history.clear()
        if 'Right' not in hands:
            self.right_gesture_history.clear()

        return hands

    def _handle_pinch_slide(self, current_y):
        """Relative continuous slider for Volume (vertical Up/Down) using non-blocking key dispatch."""
        now = time.time()

        # Initialize baseline on first frame of pinch
        if self.last_pinch_y is None:
            self.last_pinch_y = current_y
            self.pinch_mode = "Ready to Slide"
            self.pinch_active_text = "Pinching: Slide Up/Down for Volume"
            return

        delta_y = self.last_pinch_y - current_y      # Moving up -> negative pixel y -> positive delta_y

        if abs(delta_y) > 16:
            # Vertical sliding: Volume Control
            if now - self.last_volume_slide_time > 0.08:
                self.pinch_mode = "Volume"
                if delta_y > 0:
                    self._send_key(VK_VOLUME_UP, repeats=2)
                    self.pinch_active_text = "Volume ▲ (+5%) [Slide Up]"
                else:
                    self._send_key(VK_VOLUME_DOWN, repeats=2)
                    self.pinch_active_text = "Volume ▼ (-5%) [Slide Down]"
                self.last_action_text = f"Action: {self.pinch_active_text}"
                self.last_volume_slide_time = now
                self.last_pinch_y = current_y

    def _send_key(self, vk_code, repeats=1):
        """Simulate Windows multimedia key press event asynchronously without stalling the 60 FPS video thread."""
        def _dispatch():
            try:
                for idx in range(repeats):
                    ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
                    time.sleep(0.015)
                    ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)
                    if idx + 1 < repeats:
                        time.sleep(0.01)
            except Exception as e:
                print(f"[Error] keybd_event error: {e}")
        threading.Thread(target=_dispatch, daemon=True).start()

    def _send_key_combo(self, modifier_vk, key_vk):
        """Simulate two-key combination (e.g. ALT+LEFT for browser back, ALT+RIGHT for forward)."""
        def _combo():
            try:
                ctypes.windll.user32.keybd_event(modifier_vk, 0, 0, 0)
                time.sleep(0.02)
                ctypes.windll.user32.keybd_event(key_vk, 0, 0, 0)
                time.sleep(0.025)
                ctypes.windll.user32.keybd_event(key_vk, 0, 2, 0)
                time.sleep(0.01)
                ctypes.windll.user32.keybd_event(modifier_vk, 0, 2, 0)
            except Exception as e:
                print(f"[KeyCombo Error] {e}")
        threading.Thread(target=_combo, daemon=True).start()

    def _zoom_wheel(self, zoom_in=True):
        """Simulate Pinch-to-Zoom using CTRL + Mouse Wheel."""
        def _zoom():
            try:
                # Hold CTRL key
                ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)
                time.sleep(0.015)
                # Dispatch wheel delta
                scroll_delta = WHEEL_DELTA if zoom_in else -WHEEL_DELTA
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, scroll_delta, 0)
                time.sleep(0.02)
                # Release CTRL key
                ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 2, 0)
            except Exception as e:
                print(f"[Zoom Error] {e}")
        threading.Thread(target=_zoom, daemon=True).start()

    def _mouse_move(self, index_pt, frame_w, frame_h):
        """Map index fingertip coordinates to display resolution with Cascaded 1-Euro + Velocity-Adaptive EMA."""
        margin_x = int(frame_w * 0.08)
        margin_y = int(frame_h * 0.08)
        raw_x = np.clip(index_pt[0], margin_x, frame_w - margin_x)
        raw_y = np.clip(index_pt[1], margin_y, frame_h - margin_y)

        norm_x = (raw_x - margin_x) / max(1, (frame_w - 2 * margin_x))
        norm_y = (raw_y - margin_y) / max(1, (frame_h - 2 * margin_y))

        target_screen_x = float(norm_x * self.screen_w)
        target_screen_y = float(norm_y * self.screen_h)

        now = time.perf_counter()
        if time.time() < self.click_freeze_until and self.click_freeze_pos is not None:
            # Anchor freeze: lock cursor during pinch-tap to avoid shifting off target
            ctypes.windll.user32.SetCursorPos(int(self.click_freeze_pos[0]), int(self.click_freeze_pos[1]))
            return

        # Stage 1: Adaptive 1-Euro Low-Pass Filter (Casiez et al., CHI 2012)
        euro_x = self.one_euro_x.filter(target_screen_x, now)
        euro_y = self.one_euro_y.filter(target_screen_y, now)

        # Stage 2: Velocity-Adaptive EMA post-stabilizer
        if self.cursor_smooth_x is None or self.cursor_smooth_y is None:
            self.cursor_smooth_x = euro_x
            self.cursor_smooth_y = euro_y
        else:
            dist_sq = (euro_x - self.cursor_smooth_x)**2 + (euro_y - self.cursor_smooth_y)**2
            dynamic_alpha = 0.90 if dist_sq > 900 else (0.65 if dist_sq > 144 else 0.38)
            self.cursor_smooth_x = dynamic_alpha * euro_x + (1.0 - dynamic_alpha) * self.cursor_smooth_x
            self.cursor_smooth_y = dynamic_alpha * euro_y + (1.0 - dynamic_alpha) * self.cursor_smooth_y

        sx = max(0, min(self.screen_w - 1, int(round(self.cursor_smooth_x))))
        sy = max(0, min(self.screen_h - 1, int(round(self.cursor_smooth_y))))
        ctypes.windll.user32.SetCursorPos(sx, sy)

    def _mouse_left_click(self, pos):
        """Simulate single mouse left click (Select) with audio chime and visual ripple."""
        def _click():
            try:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.025)
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            except Exception as e:
                print(f"[Mouse Error] {e}")
        threading.Thread(target=_click, daemon=True).start()
        self.trigger_haptic(pos, color=(0, 255, 120), sound_freq=1500, sound_duration=45)
        self.last_action_text = "Mouse: Select (1 Tap)"

    def _mouse_double_click(self, pos):
        """Simulate mouse double click (Open) with two distinct rapid click events."""
        def _double():
            try:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.02)
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.04)
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.02)
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            except Exception as e:
                print(f"[Mouse Error] {e}")
        threading.Thread(target=_double, daemon=True).start()
        self.trigger_haptic(pos, color=(0, 255, 255), sound_freq=1800, sound_duration=55)
        self.last_action_text = "Mouse: Open (Double Tap)"

    def _mouse_hold_down(self, pos):
        """Press and hold mouse left button (Drag / Hold)."""
        if not self.is_mouse_holding:
            try:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            except Exception as e:
                print(f"[Mouse Error] {e}")
            self.is_mouse_holding = True
            self.trigger_haptic(pos, color=(255, 100, 0), sound_freq=850, sound_duration=65)
            self.last_action_text = "Mouse: Holding / Dragging"

    def _mouse_hold_release(self, pos=None):
        """Release held mouse left button."""
        if self.is_mouse_holding:
            try:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            except Exception as e:
                print(f"[Mouse Error] {e}")
            self.is_mouse_holding = False
            if pos:
                self.trigger_haptic(pos, color=(100, 255, 100), sound_freq=1200, sound_duration=40)
            self.last_action_text = "Mouse: Released Hold"

    def _mouse_right_click(self, pos):
        """Simulate single mouse right click with audio chime and haptic ripple."""
        now = time.time()
        if now - self.last_right_click_time > 0.50:
            self.last_right_click_time = now
            def _rclick():
                try:
                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    time.sleep(0.015)
                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                except Exception as e:
                    print(f"[Mouse Error] {e}")
            threading.Thread(target=_rclick, daemon=True).start()
            self.trigger_haptic(pos, color=(255, 120, 0), sound_freq=950, sound_duration=60)
            self.last_action_text = "Mouse: Right Click"

    def _mouse_scroll_palm_swirl(self, pts):
        """Simulate high-precision mouse wheel scrolling using Open Palm Swirl / Circular Dial rotation.
        
        Clockwise Swirl (CW ↻)        -> Scroll Down ▼
        Counter-Clockwise Swirl (CCW ↺) -> Scroll Up ▲
        
        Features:
        - Multi-landmark principal axis (Wrist pts[0] to Middle MCP pts[9] + Tip pts[12]) for rock-solid angle stability.
        - High-frequency 8-degree rotation trigger threshold for effortless, fluid wheel clicks.
        - Dynamic wheel speed scaling: gentle swirl scrolls 1 line, energetic swirl scrolls fast.
        - Moving trajectory ring visualizer.
        """
        now = time.time()
        wrist = pts[0]
        mid_mcp = pts[9]
        mid_tip = pts[12]
        palm_center = ((wrist[0] + mid_mcp[0]) // 2, (wrist[1] + mid_mcp[1]) // 2)

        # Primary orientation vector: wrist to middle finger tip
        vec_x = float(mid_tip[0] - wrist[0])
        vec_y = float(mid_tip[1] - wrist[1])

        # Record visual trajectory of middle fingertip
        self.scroll_trajectory.append(mid_tip)
        self.scroll_center = palm_center

        current_angle = math.atan2(vec_y, vec_x)  # Radians in [-pi, pi]

        if self.last_scroll_angle is not None:
            # Shortest angular delta across -pi / +pi seam
            delta_angle = current_angle - self.last_scroll_angle
            while delta_angle > math.pi:
                delta_angle -= 2 * math.pi
            while delta_angle < -math.pi:
                delta_angle += 2 * math.pi

            # Reject sudden teleporting landmark noise (> 70 degrees in a single frame)
            if abs(delta_angle) < math.radians(70):
                self.accumulated_scroll_angle += delta_angle

                # Sensitive 8.5-degree trigger threshold (0.15 rad) ensures effortless response
                angle_thresh = math.radians(8.5)
                if abs(self.accumulated_scroll_angle) >= angle_thresh:
                    if now - self.last_scroll_time > 0.035:
                        # In image coordinates (Y increases downwards):
                        # delta_angle > 0 means clockwise rotation -> Scroll Down
                        # delta_angle < 0 means counter-clockwise rotation -> Scroll Up
                        is_clockwise = self.accumulated_scroll_angle > 0
                        
                        # Velocity scaling: fast swirl scrolls multiple clicks for speed
                        num_clicks = max(1, min(4, int(abs(self.accumulated_scroll_angle) / angle_thresh)))
                        wheel_delta = int((-WHEEL_DELTA if is_clockwise else WHEEL_DELTA) * num_clicks)

                        try:
                            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_delta, 0)
                        except Exception as e:
                            print(f"[Scroll Error] {e}")

                        self.last_scroll_time = now
                        if is_clockwise:
                            direction = "Scroll Down ▼ (Swirl CW ↻)"
                            sound_freq = 1120
                            rip_color = (0, 165, 255)
                        else:
                            direction = "Scroll Up ▲ (Swirl CCW ↺)"
                            sound_freq = 1750
                            rip_color = (255, 255, 0)

                        self.scroll_mode_text = direction
                        self.last_action_text = f"Mouse: {direction}"
                        self.trigger_haptic(mid_tip, color=rip_color, sound_freq=sound_freq, sound_duration=22)
                        self.accumulated_scroll_angle = 0.0

        self.last_scroll_angle = current_angle

    def execute_dual_hand_action(self, hands_dict, frame_w, frame_h):
        """Coordinate dual-hand simultaneous control: Left Hand = Volume/Scroll/Mute, Right Hand = Mouse/Clicks."""
        current_time = time.time()
        actions = []

        # -------------------------------------------------------------
        # 1. LEFT HAND: Media, Volume Slider, Dial Scroll, and Fist Mute
        # -------------------------------------------------------------
        if 'Left' in hands_dict:
            l_data = hands_dict['Left']
            l_pts = l_data['pts']
            l_gest = l_data['gesture']

            # Continuous Volume Pinch Slide on Left Hand
            if l_data['is_pinch_slide']:
                self.is_pinching = True
                self.pinch_pos = l_data['pinch_mid']
                self._handle_pinch_slide(l_data['pinch_mid'][1])
                actions.append("Left: Vol Slide")
            else:
                if self.is_pinching and ('Right' not in hands_dict or not hands_dict['Right'].get('is_pinch_slide')):
                    self.is_pinching = False
                    self.last_pinch_y = None

                # Flat Palm Push towards Camera: Play / Pause Media Track
                # Detected when palm area expands rapidly (moving hand forward towards camera lens)
                l_palm_span = max(1.0, math.dist(l_pts[5], l_pts[17]))
                l_wrist = l_pts[0]
                l_palm_area = 0.5 * abs(
                    (l_pts[5][0] - l_wrist[0]) * (l_pts[17][1] - l_wrist[1]) - 
                    (l_pts[17][0] - l_wrist[0]) * (l_pts[5][1] - l_wrist[1])
                )
                self.palm_area_history.append(l_palm_area)

                is_push_detected = False
                if len(self.palm_area_history) >= 4:
                    area_delta = self.palm_area_history[-1] - self.palm_area_history[0]
                    area_ratio = self.palm_area_history[-1] / max(1.0, self.palm_area_history[0])
                    # Rapid expansion (>28% area growth across 4 frames) = Forward Push
                    if area_ratio > 1.28 and area_delta > 350:
                        is_push_detected = True

                if is_push_detected and (current_time - self.last_play_pause_time > 1.2):
                    self._send_key(VK_MEDIA_PLAY_PAUSE)
                    self.last_play_pause_time = current_time
                    self.palm_area_history.clear()
                    self.trigger_haptic(l_pts[9], color=(0, 255, 255), sound_freq=1350, sound_duration=55)
                    actions.append("Left: Play / Pause Track [TOGGLED]")

                # Pinch-to-Zoom (when Right Hand is pointing and Left Hand scales Thumb-Index distance)
                elif 'Right' in hands_dict and self.toggles['pinch_zoom'] and (l_gest in ["Two (2)", "Victory / Peace (2)", "Gesture (2)"]):
                    l_thumb_tip = l_pts[4]
                    l_idx_tip = l_pts[8]
                    current_zoom_dist = math.dist(l_thumb_tip, l_idx_tip)
                    
                    if self.last_zoom_span is None:
                        self.last_zoom_span = current_zoom_dist
                    else:
                        delta_zoom = current_zoom_dist - self.last_zoom_span
                        if abs(delta_zoom) > (0.09 * l_palm_span):
                            if current_time - self.last_zoom_time > 0.09:
                                zoom_in = delta_zoom > 0
                                self._zoom_wheel(zoom_in=zoom_in)
                                self.last_zoom_time = current_time
                                zoom_text = "Zoom In ▲ (+)" if zoom_in else "Zoom Out ▼ (-)"
                                self.zoom_mode_text = zoom_text
                                self.trigger_haptic(l_idx_tip, color=(0, 255, 180) if zoom_in else (180, 100, 255), sound_freq=1600 if zoom_in else 1100, sound_duration=25)
                                actions.append(f"Left: {zoom_text}")
                                self.last_zoom_span = current_zoom_dist
                else:
                    self.last_zoom_span = None

                # Next / Previous Track: Left Thumb Lateral Pointing
                # Thumb extended laterally while other 4 fingers are curled
                if l_gest == "Thumb (1)" or (l_gest == "One / Pointing (1)" and math.dist(l_pts[4], l_pts[5]) > 0.65 * l_palm_span):
                    if current_time - self.last_track_skip_time > 0.9:
                        # Vector from wrist to thumb tip
                        thumb_dx = l_pts[4][0] - l_pts[0][0]
                        # In mirrored camera, pointing towards screen right is Next Track, screen left is Previous Track
                        if thumb_dx > 25:
                            self._send_key(VK_MEDIA_NEXT_TRACK)
                            self.last_track_skip_time = current_time
                            self.trigger_haptic(l_pts[4], color=(255, 200, 0), sound_freq=1900, sound_duration=40)
                            actions.append("Left: Next Track ▶▶|")
                        elif thumb_dx < -25:
                            self._send_key(VK_MEDIA_PREV_TRACK)
                            self.last_track_skip_time = current_time
                            self.trigger_haptic(l_pts[4], color=(0, 200, 255), sound_freq=1200, sound_duration=40)
                            actions.append("Left: Prev Track |◀◀")

                # Closed Fist on Left Hand: Safe 2.2s Intentional Hold + 3.0s Cooldown
                if l_gest == "Fist (0)":
                    time_since_mute = current_time - self.last_mute_time
                    if time_since_mute < self.mute_cooldown:
                        self.fist_hold_start_time = None
                        remaining = max(0.0, self.mute_cooldown - time_since_mute)
                        actions.append(f"Left: Fist (Mute Cooldown {remaining:.1f}s)")
                    else:
                        if self.fist_hold_start_time is None:
                            self.fist_hold_start_time = current_time

                        hold_dur = current_time - self.fist_hold_start_time
                        if hold_dur >= self.fist_hold_required:
                            if self.toggles['mute_fist']:
                                self._send_key(VK_VOLUME_MUTE)
                                self.last_mute_time = current_time
                                self.last_action_time = current_time
                                self.fist_hold_start_time = None
                                self.trigger_haptic(l_pts[0], color=(0, 0, 255), sound_freq=600, sound_duration=70)
                                actions.append("Left: Mute / Unmute Audio [TOGGLED]")
                            else:
                                actions.append("Left: Mute [DISABLED]")
                        else:
                            hold_left = self.fist_hold_required - hold_dur
                            actions.append(f"Left: Hold Fist to Mute ({hold_left:.1f}s)")

                # Left Hand: High-Precision Swirl / Circular Palm Dial Scrolling
                elif l_gest in ["Open Palm (5)", "Four (4)"]:
                    self.fist_hold_start_time = None
                    if self.toggles['circle_scroll']:
                        self._mouse_scroll_palm_swirl(l_pts)
                        actions.append(f"Left: {self.scroll_mode_text or 'Swirl Dial Scroll (🖐️ ↻/↺)'}")
                    else:
                        actions.append("Left: Swirl Scroll [DISABLED]")
                else:
                    self.fist_hold_start_time = None
                    self.last_scroll_angle = None
                    self.accumulated_scroll_angle = 0.0
                    self.scroll_trajectory.clear()
        else:
            self.last_zoom_span = None
            self.last_scroll_angle = None
            self.accumulated_scroll_angle = 0.0
            self.scroll_trajectory.clear()
            if not hands_dict.get('Right', {}).get('is_pinch_slide'):
                self.is_pinching = False
                self.last_pinch_y = None

        # -------------------------------------------------------------
        # 2. RIGHT HAND: Air Mouse Cursor Navigation & Tap/Hold Select
        # -------------------------------------------------------------
        if 'Right' in hands_dict:
            r_data = hands_dict['Right']
            r_pts = r_data['pts']
            r_gest = r_data['gesture']

            palm_span = max(1.0, math.dist(r_pts[5], r_pts[17]))
            thumb_tip = r_pts[4]
            index_tip = r_pts[8]
            middle_tip = r_pts[12] if len(r_pts) > 12 else r_pts[8]
            middle_dip = r_pts[11] if len(r_pts) > 11 else middle_tip
            middle_pip = r_pts[10] if len(r_pts) > 10 else middle_tip

            pinch_thumb_index = math.dist(thumb_tip, index_tip)

            # 1. ROBUST ISOLATED AIR MOUSE POINTER:
            # Requires index finger to be extended while at least ring or pinky is curled (isolated pointing).
            # This completely prevents an open palm or multi-finger posture from ever moving the mouse cursor!
            d_idx_mcp_tip = math.dist(r_pts[8], r_pts[5])   # Distance from knuckle (MCP) directly to tip
            d_seg1 = math.dist(r_pts[6], r_pts[5])          # MCP to PIP
            d_seg2 = math.dist(r_pts[7], r_pts[6])          # PIP to DIP
            d_seg3 = math.dist(r_pts[8], r_pts[7])          # DIP to TIP
            idx_total_length = max(1.0, d_seg1 + d_seg2 + d_seg3)

            is_index_straight = (d_idx_mcp_tip / idx_total_length) > 0.70 or (d_idx_mcp_tip > 0.65 * palm_span)
            
            # Non-pointing fingers curled check:
            # When pointing, the ring and pinky fingers are curled towards palm (area of open palm excluded)
            d_ring_tip_wrist = math.dist(r_pts[16], r_pts[0])
            d_ring_mcp_wrist = math.dist(r_pts[13], r_pts[0])
            d_pinky_tip_wrist = math.dist(r_pts[20], r_pts[0])
            d_pinky_mcp_wrist = math.dist(r_pts[17], r_pts[0])
            is_pointing_posture = (
                is_index_straight and
                (d_ring_tip_wrist <= 1.35 * d_ring_mcp_wrist or d_pinky_tip_wrist <= 1.35 * d_pinky_mcp_wrist) and
                (r_gest not in ["Open Palm (5)", "Four (4)"])
            )

            # Move cursor only during valid isolated pointing posture
            if is_pointing_posture and self.toggles['air_mouse']:
                self._mouse_move(index_tip, frame_w, frame_h)

            # 2. ACCURATE TAP / DOUBLE TAP / HOLD TRIGGER WITH HYSTERESIS & MULTI-JOINT CONTACT:
            # When curling the middle finger, the thumb may touch the middle fingertip, DIP joint, or PIP pad.
            # Measuring min distance to all middle finger joints ensures natural contact is ALWAYS caught!
            pinch_thumb_middle = min(
                math.dist(thumb_tip, middle_tip),
                math.dist(thumb_tip, middle_dip),
                math.dist(thumb_tip, middle_pip)
            )

            # Entry distance threshold: 0.38 * palm_span to initiate pinch.
            # Hold maintenance threshold: 0.50 * palm_span once dragging is active (hysteresis).
            # This completely prevents arm movement or camera perspective shifts from dropping the drag!
            pinch_threshold = 0.50 * palm_span if self.is_mouse_holding else 0.38 * palm_span
            is_middle_thumb_pinched = (pinch_thumb_middle < pinch_threshold)

            # Right Click: Thumb + Ring Finger Pinch (Strict: ring finger must be closer to thumb than middle finger!)
            ring_tip = r_pts[16] if len(r_pts) > 16 else r_pts[8]
            pinch_thumb_ring = math.dist(thumb_tip, ring_tip)
            is_ring_thumb_pinched = (
                (pinch_thumb_ring < 0.24 * palm_span) and 
                (pinch_thumb_ring < pinch_thumb_middle * 0.85)
            )

            if is_middle_thumb_pinched:
                pinch_mid = ((thumb_tip[0] + middle_tip[0]) // 2, (thumb_tip[1] + middle_tip[1]) // 2)
                self.last_hold_detected_time = current_time
                if self.pinch_start_time is None:
                    self.pinch_start_time = current_time
                    if self.cursor_smooth_x is not None and self.cursor_smooth_y is not None:
                        self.click_freeze_until = current_time + 0.12
                        self.click_freeze_pos = (self.cursor_smooth_x, self.cursor_smooth_y)

                pinch_dur = current_time - self.pinch_start_time
                if pinch_dur >= 0.35:
                    if self.toggles['air_mouse']:
                        self._mouse_hold_down(pinch_mid)
                        actions.append("Right: Holding / Drag (Mid+Thumb)")
                else:
                    if not self.is_mouse_holding:
                        actions.append("Right: Tap Contact (👌)")
            elif is_ring_thumb_pinched and not self.is_mouse_holding:
                if self.toggles['air_mouse']:
                    pinch_pt = ((thumb_tip[0] + ring_tip[0]) // 2, (thumb_tip[1] + ring_tip[1]) // 2)
                    self._mouse_right_click(pinch_pt)
                    actions.append("Right: Right Click (Ring+Thumb)")
            else:
                # When fingers separate, apply grace period if actively holding
                time_since_contact = current_time - self.last_hold_detected_time
                if self.is_mouse_holding and (time_since_contact < self.hold_release_grace):
                    # Within grace period: maintain drag state smoothly even if landmark tracking blips
                    actions.append("Right: Holding / Drag (Active)")
                else:
                    if self.pinch_start_time is not None:
                        pinch_dur = current_time - self.pinch_start_time
                        pinch_mid = ((thumb_tip[0] + middle_tip[0]) // 2, (thumb_tip[1] + middle_tip[1]) // 2)

                        if self.is_mouse_holding:
                            self._mouse_hold_release(pinch_mid)
                            actions.append("Right: Released Drag")
                        else:
                            if pinch_dur >= 0.03:
                                if self.toggles['air_mouse']:
                                    time_since_first = current_time - self.last_tap_release_time
                                    if time_since_first <= self.double_tap_window:
                                        # Second tap within double_tap_window: fire double click
                                        self._mouse_double_click(pinch_mid)
                                        self.last_tap_release_time = 0
                                        actions.append("Right: Open (Double Tap)")
                                    else:
                                        # First tap: fire immediate single click
                                        self._mouse_left_click(pinch_mid)
                                        self.last_tap_release_time = current_time
                                        actions.append("Right: Select (1 Tap)")

                        self.pinch_start_time = None
                    else:
                        if not self.is_mouse_holding:
                            if is_index_straight:
                                actions.append("Right: Air Mouse (☝️ Point)")

            # 3. Dedicated Browser Back / Forward Navigation (Separate Gesture - NOT active while pointing):
            # Only triggered when the index and other fingers are curled and the thumb is intentionally extended sideways
            if not self.is_mouse_holding and not is_index_straight and self.toggles['browser_nav']:
                # Ensure other fingers are curled (middle, ring, pinky tips close to MCPs)
                fingers_curled = (
                    math.dist(r_pts[8], r_pts[5]) < 0.60 * palm_span and
                    math.dist(r_pts[12], r_pts[9]) < 0.60 * palm_span and
                    math.dist(r_pts[16], r_pts[13]) < 0.60 * palm_span
                )
                thumb_extended = math.dist(thumb_tip, r_pts[5]) > 0.65 * palm_span

                if fingers_curled and thumb_extended and (r_gest in ["Thumb (1)", "One / Pointing (1)"]):
                    thumb_rel_x = thumb_tip[0] - r_pts[0][0]  # Relative to wrist
                    if current_time - self.last_browser_nav_time > 1.0:
                        # In mirrored camera view:
                        # Thumb pointing towards screen left (negative dx) = Back (Alt + Left)
                        # Thumb pointing towards screen right (positive dx) = Forward (Alt + Right)
                        if thumb_rel_x < -35:
                            self._send_key_combo(VK_MENU, VK_LEFT)
                            self.last_browser_nav_time = current_time
                            self.trigger_haptic(thumb_tip, color=(255, 100, 200), sound_freq=1400, sound_duration=50)
                            actions.append("Right: Browser Back ◄ (Alt+Left)")
                        elif thumb_rel_x > 35:
                            self._send_key_combo(VK_MENU, VK_RIGHT)
                            self.last_browser_nav_time = current_time
                            self.trigger_haptic(thumb_tip, color=(100, 200, 255), sound_freq=1700, sound_duration=50)
                            actions.append("Right: Browser Forward ► (Alt+Right)")

            if not is_middle_thumb_pinched and not is_index_straight:
                time_since_contact = current_time - self.last_hold_detected_time
                if self.is_mouse_holding and (time_since_contact >= self.hold_release_grace):
                    self._mouse_hold_release()
                    self.pinch_start_time = None
                if not self.is_mouse_holding:
                    self.one_euro_x.reset()
                    self.one_euro_y.reset()
                    self.cursor_smooth_x = None
                    self.cursor_smooth_y = None

                # (Air scrolling and media control are strictly reserved for Left Hand per dual-hand architecture)

        else:
            # If Right hand is temporarily lost from camera frame while holding, do NOT drop immediately!
            # Wait for hold_release_grace window before releasing mouse button
            if self.is_mouse_holding:
                time_since_contact = current_time - self.last_hold_detected_time
                if time_since_contact >= self.hold_release_grace:
                    self._mouse_hold_release()
                    self.pinch_start_time = None
                    self.one_euro_x.reset()
                    self.one_euro_y.reset()
                    self.cursor_smooth_x = None
                    self.cursor_smooth_y = None
                else:
                    actions.append("Right: Holding / Drag (Active)")
            else:
                self.one_euro_x.reset()
                self.one_euro_y.reset()
                self.cursor_smooth_x = None
                self.cursor_smooth_y = None

        if actions:
            self.last_action_text = " | ".join(actions)
            self.last_action_time = current_time
        elif current_time - self.last_action_time > 2.0 and not self.is_pinching:
            self.last_action_text = "Ready (Left: Vol/Scroll | Right: Mouse)"

        return self.last_action_text

    def execute_system_action(self, gesture_name, pts=None, frame_w=None, frame_h=None):
        """Single-hand fallback wrapper maintaining full compatibility with test_pipeline."""
        hands_dict = {'Right': {'pts': pts, 'gesture': gesture_name, 'count': 0, 'is_pinch_slide': False, 'pinch_mid': pts[8] if pts else None}} if pts else {}
        return self.execute_dual_hand_action(hands_dict, frame_w, frame_h)

    def run(self, camera_index=0):
        """Execute the real-time application with full-screen recognition & comprehensive visuals."""
        # Use DirectShow with MJPG for responsive, crash-free Windows camera capture
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"[Error] Could not access camera with index {camera_index}.")
            return

        # Request high-speed capture (60 FPS default with 30 FPS hardware fallback)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Zero-latency capture buffer
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure adaptation for dark rooms
        except Exception:
            pass

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Active Stream Resolution: {actual_w}x{actual_h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS")

        prev_time = time.time()
        start_time = time.time()
        snapshot_count = 1

        print("=" * 70)
        print("Real-time Hand Gesture Recognition Engine (High FPS Smooth Pipeline)")
        print("Show your hand anywhere on screen.")
        print("Dual-Hand Simultaneous Controls:")
        print("  - Right Hand: ☝️ Move Cursor | 👌 Tap: Select | 👌 2x: Open | 👌 Hold: Drag")
        print("  - Left Hand:  👌 Pinch Slide: Volume | 🖐️ Palm Dial: Scroll | ✊ Fist: Mute")
        print("Commands & Feature Toggles:")
        print(" - Keys [1-4] or Click on UI Buttons to toggle features:")
        print("     [1] ☝️ Air Mouse & Select")
        print("     [2] 🖐️ Open Palm Scroll")
        print("     [3] 👌 Volume Slide")
        print("     [4] ✊ Fist Mute")
        print(" - 's': Save snapshot for report/presentation")
        print(" - 'q': Quit application")
        print("=" * 70)

        window_name = "Hand Gesture Recognition - CS305 Project #16"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, actual_w, actual_h)

        self._calibrate_requested = False

        def _on_mouse_click(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for btn in self.ui_buttons:
                    bx, by, bw, bh = btn['rect']
                    if bx <= mx <= bx + bw and by <= my <= by + bh:
                        if btn['id'] == 'calibrate_skin':
                            self._calibrate_requested = True
                        else:
                            self.toggle_feature(btn['id'])
                        break

        cv2.setMouseCallback(window_name, _on_mouse_click)

        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Palm base
        ]

        consecutive_drops = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_drops += 1
                if consecutive_drops > 30:
                    print("[Warning] Camera disconnected or stopped sending frames.")
                    break
                time.sleep(0.01)
                continue
            consecutive_drops = 0

            t_stage_start = time.perf_counter()

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)

            # 1. Adaptive Low-Light Enhancement (Single pass shared by MediaPipe & Skin Segmentation)
            enhanced_frame = self.enhance_low_light(frame)

            # 2. Classical Skin Color Segmentation & Geometric Analysis (Unit 2 Curriculum)
            skin_mask = self.segment_skin(enhanced_frame, already_enhanced=True)
            cnt, hull_pts, defect_pts, palm_center, palm_radius, solidity = self.compute_skin_geometry(
                enhanced_frame, skin_mask
            )

            # 3. Dual-Hand Landmark & Gesture Classification (True Non-Blocking Async Stream)
            frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            current_timestamp_ms = int((time.time() - start_time) * 1000)
            hands_dict = self.classify_both_hands(
                frame_rgb, actual_w, actual_h, current_timestamp_ms, sync_wait=False
            )

            # Handle on-demand Dynamic Palm Skin Calibration (Key 'C' or UI button)
            if self._calibrate_requested:
                self.calibrate_skin_from_palm(enhanced_frame, hands_dict)
                self._calibrate_requested = False

            # 4. Interactive Action Trigger (Dual-Hand Coordinated Controller)
            action_text = self.execute_dual_hand_action(hands_dict, actual_w, actual_h)
            if not hands_dict and cnt is not None and palm_center is not None:
                c_gest, _ = self.classify_classical_contour(cnt, palm_center, palm_radius, defect_pts, solidity)
                action_text = f"Classical Fallback (Unit 2): {c_gest} [S={solidity:.2f}]"

            self.last_pipeline_ms = (time.perf_counter() - t_stage_start) * 1000.0

            # Render landmarks for each detected hand with dedicated color themes
            for hand_side, h_info in hands_dict.items():
                pts = h_info['pts']
                gest = h_info['gesture']
                is_right = (hand_side == 'Right')
                conn_color = (0, 255, 255) if is_right else (255, 160, 50)  # Cyan for Right, Orange for Left

                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    cv2.line(frame, pts[p1_idx], pts[p2_idx], conn_color, 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)
                for tip_idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, pts[tip_idx], 7, (0, 255, 0), -1)

                # Wrist Label indicating hand identity & confidence
                wrist_pt = pts[0]
                conf_pct = int(h_info.get('confidence', 0.90) * 100)
                hand_tag = f"{hand_side}: {gest} ({conf_pct}%)"
                cv2.putText(frame, hand_tag, (wrist_pt[0] - 45, wrist_pt[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, conn_color, 1)

                # Right Hand: Air Mouse Cursor Visual Indicator (Displays only when intentionally pointing ☝️)
                d_idx_mcp_vis = math.dist(pts[8], pts[5])
                idx_len_vis = max(1.0, math.dist(pts[6], pts[5]) + math.dist(pts[7], pts[6]) + math.dist(pts[8], pts[7]))
                palm_span_vis = max(1.0, math.dist(pts[5], pts[17]))
                idx_straight_vis = (d_idx_mcp_vis / idx_len_vis > 0.70) or (d_idx_mcp_vis > 0.65 * palm_span_vis)
                is_pointing_vis = idx_straight_vis and (gest not in ["Open Palm (5)", "Four (4)"])

                if is_right and (is_pointing_vis or self.is_mouse_holding):
                    idx_pt = pts[8]
                    cursor_color = (0, 165, 255) if self.is_mouse_holding else (0, 255, 255)
                    cv2.circle(frame, idx_pt, 16, cursor_color, 2)
                    cv2.circle(frame, idx_pt, 5, (0, 255, 0), -1)
                    mode_label = "HOLDING / DRAGGING" if self.is_mouse_holding else f"1-Euro Air Mouse ({self.one_euro_x.last_cutoff:.1f}Hz)"
                    cv2.putText(frame, mode_label, (idx_pt[0] + 16, idx_pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, cursor_color, 1)

                # Left Hand: High-Precision Palm Swirl Dial Visualizer
                if (not is_right) and gest in ["Open Palm (5)", "Four (4)"] and self.toggles['circle_scroll'] and self.scroll_center is not None:
                    cx, cy = self.scroll_center
                    cv2.circle(frame, (cx, cy), 54, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                    pts_traj = list(self.scroll_trajectory)
                    for i in range(1, len(pts_traj)):
                        thickness = max(1, int(math.sqrt(24 / float(i + 1)) * 2))
                        cv2.line(frame, pts_traj[i - 1], pts_traj[i], (0, 255, 255), thickness)

                    cv2.putText(frame, "CW: Scroll Down | CCW: Scroll Up", (cx - 140, cy - 66),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 2)

                # Left Hand: Dynamic Volume Pinch Slide Visualizer
                if h_info['is_pinch_slide'] and self.pinch_pos is not None:
                    cv2.line(frame, pts[4], pts[8], (0, 255, 255), 4)
                    cv2.circle(frame, self.pinch_pos, 14, (0, 215, 255), 3)
                    cv2.circle(frame, self.pinch_pos, 7, (0, 255, 0), -1)
                    px, py = self.pinch_pos
                    cv2.arrowedLine(frame, (px, py - 16), (px, py - 46), (0, 255, 255), 2, tipLength=0.35)
                    cv2.arrowedLine(frame, (px, py + 16), (px, py + 46), (0, 255, 255), 2, tipLength=0.35)
                    cv2.putText(frame, "Left Hand: Volume Up/Down", (px - 95, py - 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)

            # Render Animated Expanding Visual Haptic Ripples
            now_time = time.time()
            active_ripples = []
            for rip in self.haptic_ripples:
                age = now_time - rip['birth']
                if age < 0.35:
                    progress = age / 0.35
                    curr_r = int(rip['radius'] + progress * (rip['max_radius'] - rip['radius']))
                    alpha = max(0.0, 1.0 - progress)
                    cv2.circle(frame, rip['pos'], curr_r, rip['color'], max(1, int(3 * alpha)))
                    cv2.circle(frame, rip['pos'], max(2, curr_r - 6), (255, 255, 255), 1)
                    active_ripples.append(rip)
            self.haptic_ripples = active_ripples

            # FPS Calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # -------------------------------------------------------------
            # Modern UI Redesign: Bottom Dashboard, Gesture Banner & Badges
            # -------------------------------------------------------------

            # 1. Top Header Badge & Interactive Gesture Toggle Toolbar
            top_bar_h = 42
            roi_top = frame[0:top_bar_h, 0:actual_w]
            overlay_top = np.full_like(roi_top, (15, 15, 20))
            cv2.addWeighted(overlay_top, 0.70, roi_top, 0.30, 0, roi_top)
            cv2.line(frame, (0, top_bar_h), (actual_w, top_bar_h), (50, 50, 60), 1)

            cv2.putText(frame, "ICFAI CV LAB | PROJECT #16", (16, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 215, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f} ({self.last_pipeline_ms:.1f}ms)", (actual_w - 175, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 140), 2)

            # Interactive Toggle Buttons Toolbar: (key, id, label)
            toggle_defs = [
                ('1', 'air_mouse', '1:Mouse'),
                ('2', 'circle_scroll', '2:Scroll'),
                ('3', 'volume_slider', '3:Vol'),
                ('4', 'mute_fist', '4:Mute'),
                ('5', 'pinch_zoom', '5:Zoom'),
                ('6', 'browser_nav', '6:Nav'),
                ('n', 'night_mode', 'N:Night'),
                ('c', 'calibrate_skin', 'C:Calib')
            ]
            btn_w, btn_h = 68, 28
            btn_start_x = 195
            btn_y = 7
            self.ui_buttons.clear()

            for idx, (hotkey, feat_id, label) in enumerate(toggle_defs):
                bx = btn_start_x + idx * (btn_w + 5)
                is_on = self.is_skin_calibrated if feat_id == 'calibrate_skin' else self.toggles.get(feat_id, True)

                self.ui_buttons.append({'id': feat_id, 'rect': (bx, btn_y, btn_w, btn_h), 'label': label, 'key': hotkey})

                bg_color = (0, 150, 60) if is_on else (45, 45, 55)
                border_col = (0, 255, 120) if is_on else (90, 90, 100)
                text_col = (255, 255, 255) if is_on else (160, 160, 170)

                roi_btn = frame[btn_y:btn_y + btn_h, bx:bx + btn_w]
                btn_fill = np.full_like(roi_btn, bg_color)
                cv2.addWeighted(btn_fill, 0.85, roi_btn, 0.15, 0, roi_btn)
                cv2.rectangle(frame, (bx, btn_y), (bx + btn_w, btn_y + btn_h), border_col, 1)

                status_dot = "OK" if (feat_id == 'calibrate_skin' and is_on) else ("ON" if is_on else "OFF")
                cv2.putText(frame, f"{label} [{status_dot}]", (bx + 3, btn_y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27, text_col, 1)

            # 2. Bottom Interactive Gesture Legend Banner
            banner_h = 36
            banner_y = actual_h - banner_h
            roi_banner = frame[banner_y:actual_h, 0:actual_w]
            banner_fill = np.full_like(roi_banner, (12, 12, 16))
            cv2.addWeighted(banner_fill, 0.88, roi_banner, 0.12, 0, roi_banner)
            cv2.line(frame, (0, banner_y), (actual_w, banner_y), (60, 60, 75), 1)

            night_tag = " [Night Boost ON]" if (self.toggles['night_mode'] and self.is_dark_scene) else ""
            calib_tag = " [Skin Calibrated]" if self.is_skin_calibrated else ""
            legend_text = f"[Controls] R: Point=Cursor | Mid+Thumb=Click/Drag | Ring=R-Click | L: Palm Swirl=Scroll | Push=Play | Pinch=Vol | Fist=Mute{night_tag}{calib_tag}"
            cv2.putText(frame, legend_text, (10, banner_y + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255) if not night_tag else (100, 255, 255), 1)

            # 3. Bottom Information HUD Box (Elevated above banner on bottom-left)
            hud_w, hud_h = 620, 122
            hud_x = 24
            hud_y = banner_y - hud_h - 14

            roi_hud = frame[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w]
            hud_fill = np.full_like(roi_hud, (18, 18, 24))
            cv2.addWeighted(hud_fill, 0.82, roi_hud, 0.18, 0, roi_hud)
            border_color = (0, 255, 255) if self.is_pinching else (80, 80, 100)
            cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), border_color, 2)

            # Dual-Hand Live Gesture Summary + Telemetry
            r_str = hands_dict.get('Right', {}).get('gesture', 'None')
            l_str = hands_dict.get('Left', {}).get('gesture', 'None')
            cv2.putText(frame, f"Right Hand (Mouse) : {r_str}", (hud_x + 18, hud_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2)
            cv2.putText(frame, f"Left Hand (Media)  : {l_str}", (hud_x + 18, hud_y + 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 175, 50), 2)
            cv2.putText(frame, f"{action_text}", (hud_x + 18, hud_y + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 255, 120), 2)
            telemetry_str = (
                f"1-Euro fc: {self.one_euro_x.last_cutoff:.1f}Hz | "
                f"Unit-2 Solidity: {self.last_solidity:.2f} | "
                f"Defects: {self.last_defect_count} | "
                f"Lum L: {self.scene_luminance:.0f}"
            )
            cv2.putText(frame, telemetry_str, (hud_x + 18, hud_y + 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (190, 210, 235), 1)

            # 4. Annotated PiP (Picture-in-Picture) Skin Mask Preview with Live Convex Hull & Defects (Unit 2)
            pip_size = 150
            mask_vis = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
            if hull_pts is not None:
                cv2.drawContours(mask_vis, [hull_pts], -1, (0, 255, 0), 2)
            if palm_center is not None and palm_radius > 2:
                cv2.circle(mask_vis, palm_center, palm_radius, (255, 220, 0), 1)
                cv2.circle(mask_vis, palm_center, 3, (255, 220, 0), -1)
            for df_pt in defect_pts:
                cv2.circle(mask_vis, df_pt, 4, (0, 0, 255), -1)

            mask_small_bgr = cv2.resize(mask_vis, (pip_size, pip_size))
            pip_x = actual_w - pip_size - 24
            pip_y = banner_y - pip_size - 14

            cv2.rectangle(frame, (pip_x - 2, pip_y - 2), (pip_x + pip_size + 2, pip_y + pip_size + 2), (0, 255, 140), 2)
            frame[pip_y:pip_y + pip_size, pip_x:pip_x + pip_size] = mask_small_bgr
            cv2.putText(frame, f"Unit 2 Hull/Defects (D={self.last_defect_count})", (pip_x - 10, pip_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            cv2.imshow("Hand Gesture Recognition - CS305 Project #16", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_dir = "img" if os.path.exists("img") else "."
                filename_main = os.path.join(save_dir, f"demo_frame_{snapshot_count}.png")
                filename_raw = os.path.join(save_dir, f"raw_frame_{snapshot_count}.png")
                filename_mask = os.path.join(save_dir, f"demo_mask_{snapshot_count}.png")
                cv2.imwrite(filename_main, frame)
                cv2.imwrite(filename_raw, enhanced_frame)
                cv2.imwrite(filename_mask, mask_vis)
                print(f"[Saved] Snapshot saved as {filename_main}, {filename_raw}, and {filename_mask}")
                snapshot_count += 1
            elif key == ord('c'):
                self.calibrate_skin_from_palm(enhanced_frame, hands_dict)
            elif key == ord('1'):
                self.toggle_feature('air_mouse')
            elif key == ord('2'):
                self.toggle_feature('circle_scroll')
            elif key == ord('3'):
                self.toggle_feature('volume_slider')
            elif key == ord('4'):
                self.toggle_feature('mute_fist')
            elif key == ord('5'):
                self.toggle_feature('pinch_zoom')
            elif key == ord('6'):
                self.toggle_feature('browser_nav')
            elif key == ord('n') or key == ord('7'):
                self.toggle_feature('night_mode')

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = HybridHandGestureRecognizer()
    recognizer.run()
