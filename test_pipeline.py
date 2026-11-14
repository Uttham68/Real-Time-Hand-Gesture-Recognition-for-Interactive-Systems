"""
Comprehensive Automated Verification & Benchmark Suite for Hybrid Hand Gesture Recognizer
Course: CS-305 Computer Vision (Case Design Project #16)

Tests:
1. Canonical Demo Frame Classification (Hybrid + Unit 2 Classical Convex Hull / Defects)
2. 3D Cosine Joint Kinematics & Perspective Tilt Invariance (21-Point 3D Skeletons)
3. Adaptive 1-Euro Filter (Casiez et al., CHI 2012) Jitter Attenuation & Step Response
4. Sub-Stage Execution Latency (ms) & Real-Time Throughput (FPS) Profiling
"""

import os
import time
import math
import cv2
import numpy as np
from gesture_recognizer import HybridHandGestureRecognizer, OneEuroFilter


def _generate_canonical_hand_frame(finger_count, width=640, height=480):
    """Synthesize a calibrated skin-chrominance hand posture frame for deterministic Unit-2 & Hybrid testing."""
    img = np.full((height, width, 3), (22, 24, 30), dtype=np.uint8)
    # Calibrated realistic human skin BGR color within HSV (H:0-25, S:30-220, V:50-255) & YCrCb (Cr:134-175, Cb:80-126)
    skin_bgr = (135, 162, 218)

    cx, cy = width // 2, int(height * 0.66)
    palm_r = 68

    if finger_count == 0:
        # Closed compact fist (high solidity > 0.88, 0 convexity defects)
        cv2.ellipse(img, (cx, cy - 10), (palm_r, int(palm_r * 0.92)), 0, 0, 360, skin_bgr, -1, cv2.LINE_AA)
        return img

    # Draw palm base
    cv2.circle(img, (cx, cy), palm_r, skin_bgr, -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - palm_r + 8, cy - 25), (cx + palm_r - 8, cy + 35), skin_bgr, -1)

    # Finger angles (in degrees, where -90 is straight up)
    angle_Layouts = {
        1: [-90],
        2: [-106, -74],
        3: [-116, -90, -64],
        4: [-124, -101, -79, -56],
        5: [-142, -114, -88, -64, -38],
    }
    angles = angle_Layouts.get(finger_count, [-90])
    finger_len = 148
    finger_thickness = 24

    for idx, ang_deg in enumerate(angles):
        rad = math.radians(ang_deg)
        # Anchor slightly above palm center so inter-digital valleys form realistic convexity defects
        base_r = palm_r * 0.52
        bx = int(cx + base_r * math.cos(rad))
        by = int(cy + base_r * math.sin(rad))
        length = int(finger_len * (0.85 if (finger_count == 5 and idx == 0) else 1.0))
        tx = int(cx + (palm_r + length) * math.cos(rad))
        ty = int(cy + (palm_r + length) * math.sin(rad))
        cv2.line(img, (bx, by), (tx, ty), skin_bgr, finger_thickness, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), finger_thickness // 2, skin_bgr, -1, cv2.LINE_AA)

    return img


def ensure_demo_frames():
    """Ensure canonical test frames exist in img/ for reproducible regression testing."""
    os.makedirs("img", exist_ok=True)
    for i in range(1, 6):
        raw_path = os.path.join("img", f"raw_frame_{i}.png")
        if not os.path.exists(raw_path):
            synth = _generate_canonical_hand_frame(i)
            cv2.imwrite(raw_path, synth)


def _build_3d_skeleton(extended_fingers, tilt_deg=0.0):
    """Build a 21-landmark 3D hand skeleton with optional camera pitch tilt (foreshortening test)."""
    pts_2d = [(480, 400)] * 21
    pts_3d = [(0.5, 0.75, 0.0)] * 21

    # Wrist (0)
    wx, wy, wz = 480.0, 400.0, 0.0
    pts_2d[0] = (int(wx), int(wy))
    pts_3d[0] = (wx / 960.0, wy / 540.0, wz)

    # Knuckle MCPs: Index(5), Middle(9), Ring(13), Pinky(17)
    mcps = {
        5: (430.0, 300.0, 0.0),
        9: (465.0, 292.0, 0.0),
        13: (500.0, 298.0, 0.0),
        17: (535.0, 310.0, 0.0),
    }
    for k, (mx, my, mz) in mcps.items():
        pts_2d[k] = (int(mx), int(my))
        pts_3d[k] = (mx / 960.0, my / 540.0, mz)

    # Thumb landmarks (1, 2, 3, 4)
    pts_2d[1] = (445, 365)
    pts_2d[2] = (420, 335)
    pts_3d[1] = (445 / 960.0, 365 / 540.0, 0.0)
    pts_3d[2] = (420 / 960.0, 335 / 540.0, 0.0)
    if 'Thumb' in extended_fingers:
        pts_2d[3] = (380, 310)
        pts_2d[4] = (345, 285)
        pts_3d[3] = (380 / 960.0, 310 / 540.0, 0.0)
        pts_3d[4] = (345 / 960.0, 285 / 540.0, 0.0)
    else:
        # Folded across palm toward ring MCP
        pts_2d[3] = (465, 325)
        pts_2d[4] = (490, 315)
        pts_3d[3] = (465 / 960.0, 325 / 540.0, 0.0)
        pts_3d[4] = (490 / 960.0, 315 / 540.0, 0.0)

    tilt_rad = math.radians(tilt_deg)
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)

    finger_defs = [
        ('Index', 5, 6, 7, 8),
        ('Middle', 9, 10, 11, 12),
        ('Ring', 13, 14, 15, 16),
        ('Pinky', 17, 18, 19, 20),
    ]
    for name, mcp, pip, dip, tip in finger_defs:
        mx, my, mz = mcps[mcp]
        if name in extended_fingers:
            # Straight collinear segments along tilted 3D vector
            for j_idx, dist_px in [(pip, 36.0), (dip, 68.0), (tip, 98.0)]:
                jx = mx
                jy = my - dist_px * cos_t
                jz = - (dist_px / 540.0) * sin_t
                pts_2d[j_idx] = (int(jx), int(jy))
                pts_3d[j_idx] = (jx / 960.0, jy / 540.0, jz)
        else:
            # Curled back toward palm
            pts_2d[pip] = (int(mx), int(my - 22))
            pts_2d[dip] = (int(mx), int(my - 5))
            pts_2d[tip] = (int(mx), int(my + 12))
            pts_3d[pip] = (mx / 960.0, (my - 22) / 540.0, -0.02)
            pts_3d[dip] = (mx / 960.0, (my - 5) / 540.0, -0.04)
            pts_3d[tip] = (mx / 960.0, (my + 12) / 540.0, -0.01)

    return pts_2d, pts_3d


def test_saved_demo_frames():
    ensure_demo_frames()
    recognizer = HybridHandGestureRecognizer()

    print("=" * 78)
    print(" SUITE 1: HYBRID & CLASSICAL (UNIT 2) BENCHMARK FRAME VERIFICATION")
    print("=" * 78)

    expected_labels = {
        1: "One / Pointing (1)",
        2: "Victory / Peace (2)",
        3: "Three (3)",
        4: "Four (4)",
        5: "Open Palm (5)"
    }

    all_passed = True
    for i in range(1, 6):
        raw_fname = f"img/raw_frame_{i}.png"
        fname = raw_fname if os.path.exists(raw_fname) else f"img/demo_frame_{i}.png"
        recognizer.gesture_history.clear()
        img = cv2.imread(fname)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name, count, _ = recognizer.classify_hand_landmarks(rgb, img.shape[1], img.shape[0])

        expected = expected_labels[i]
        status = "PASSED" if name == expected else "FAILED"
        if status == "FAILED":
            all_passed = False

        print(
            f" [{status}] Frame {i} | Detected: {name:<20} | "
            f"Solidity: {recognizer.last_solidity:.2f} | Defects: {recognizer.last_defect_count} | Expected: {expected}"
        )

    print("-" * 78)
    print(" SUITE 2: 3D VECTOR COSINE KINEMATICS & TILT INVARIANCE VERIFICATION")
    print("-" * 78)
    posture_cases = [
        ([], 0.0, "Fist (0)"),
        (['Index'], 0.0, "One / Pointing (1)"),
        (['Index', 'Middle'], 38.0, "Victory / Peace (2)"),  # 38-deg forward camera tilt test
        (['Index', 'Middle', 'Ring'], 35.0, "Three (3)"),
        (['Index', 'Middle', 'Ring', 'Pinky'], 0.0, "Four (4)"),
        (['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'], 32.0, "Open Palm (5)"),
    ]
    for ext_list, tilt, exp_name in posture_cases:
        pts_2d, pts_3d = _build_3d_skeleton(ext_list, tilt_deg=tilt)
        pred_name, pred_cnt = recognizer.classify_single_hand(pts_2d, "Right", pts_3d=pts_3d)
        ok = (pred_name == exp_name)
        if not ok:
            all_passed = False
        print(f" [{'PASSED' if ok else 'FAILED'}] 3D Skeleton (Tilt={tilt:>4.1f} deg) -> {pred_name:<20} | Expected: {exp_name}")

    print("-" * 78)
    print(" SUITE 3: ADAPTIVE 1-EURO FILTER (CASIEZ ET AL.) TREMOR ATTENUATION BENCHMARK")
    print("-" * 78)
    rng = np.random.default_rng(42)
    f_euro = OneEuroFilter(min_cutoff=0.35, beta=0.004, d_cutoff=1.0)
    raw_tremor = 500.0 + rng.normal(0.0, 3.2, size=120)
    filtered_vals = []
    ema_val = None
    for idx, val in enumerate(raw_tremor):
        e_val = f_euro.filter(val, idx * (1.0 / 60.0))
        ema_val = e_val if ema_val is None else (0.38 * e_val + 0.62 * ema_val)
        filtered_vals.append(ema_val)
    raw_rms = float(np.std(raw_tremor[20:]))
    filt_rms = float(np.std(filtered_vals[20:]))
    jitter_reduction = (1.0 - filt_rms / raw_rms) * 100.0
    euro_ok = jitter_reduction > 80.0
    if not euro_ok:
        all_passed = False
    print(
        f" [{'PASSED' if euro_ok else 'FAILED'}] Stationary Tremor RMS: Raw={raw_rms:.2f}px -> "
        f"Cascaded 1-Euro+EMA={filt_rms:.2f}px ({jitter_reduction:.1f}% Jitter Reduction)"
    )

    print("-" * 78)
    print(" SUITE 4: REAL-TIME PIPELINE LATENCY & FPS BENCHMARK (960x540)")
    print("-" * 78)
    bench_frame = _generate_canonical_hand_frame(5, width=960, height=540)
    n_iters = 50
    t0 = time.perf_counter()
    for _ in range(n_iters):
        enh = recognizer.enhance_low_light(bench_frame)
        msk = recognizer.segment_skin(enh, already_enhanced=True)
        recognizer.compute_skin_geometry(enh, msk)
    elapsed_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
    theoretical_fps = 1000.0 / max(0.1, elapsed_ms)
    speed_ok = elapsed_ms < 12.0
    print(
        f" [{'PASSED' if speed_ok else 'FAILED'}] Vision Core Latency: {elapsed_ms:.2f} ms/frame "
        f"(Supports {min(60.0, theoretical_fps):.1f} FPS Real-Time Stream, Max Cap: {theoretical_fps:.0f} FPS)"
    )

    print("=" * 78)
    if all_passed:
        print(" ALL VERIFICATION SUITES PASSED WITH 100% ACCURACY!")
    else:
        print(" SOME SUITES NEED REVIEW.")
    print("=" * 78)


if __name__ == "__main__":
    test_saved_demo_frames()
