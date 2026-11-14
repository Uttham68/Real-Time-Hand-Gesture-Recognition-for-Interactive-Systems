# Quick Start Guide: CS-305 Hand Gesture Recognition Project (#16)

### Project Structure
- [`gesture_recognizer.py`](file:///d:/Practice/CVProject/gesture_recognizer.py) : Main 60 FPS real-time OpenCV + MediaPipe hybrid application with 1-Euro adaptive cursor filtering, 3D cosine joint kinematics, YCrCb Luma CLAHE night boost, dynamic palm skin calibration, annotated Unit-2 Convex Hull/Defects PiP preview, and dual-hand Win32 action triggers.
- [`test_pipeline.py`](file:///d:/Practice/CVProject/test_pipeline.py) : Multi-suite automated verification & performance benchmark script (tests Unit-2 classical geometry, 3D tilt-invariant kinematics, 1-Euro jitter reduction, and sub-6ms pipeline latency).
- [`Faculty_Presentation_CS305_Project16.pptx`](file:///d:/Practice/CVProject/Faculty_Presentation_CS305_Project16.pptx) : Complete 13-slide professional widescreen presentation for faculty evaluation.
- [`Complete_Project_Overview_CS305.pdf`](file:///d:/Practice/CVProject/Complete_Project_Overview_CS305.pdf) : Comprehensive student study guide & technical overview PDF explaining every component, mathematical formula, and viva Q&A.

---

### How to Run the Project

1. **Run the 4-Suite Automated Verification & Benchmark Pipeline:**
   ```bash
   python test_pipeline.py
   ```

2. **Launch Real-Time 60 FPS Dual-Hand Gesture Recognition:**
   ```bash
   python gesture_recognizer.py
   ```

3. **Interactive Toolbar & Hotkey Controls while Running:**
   - **`1–6`** (or click top UI pills) : Toggle individual gesture engines (`1:Mouse`, `2:Scroll`, `3:Vol`, `4:Mute`, `5:Zoom`, `6:Nav`).
   - **`n`** (or **`7`**) : Toggle Low-Light Night Vision Boost (`YCrCb Luma CLAHE + Pre-computed Gamma LUT`).
   - **`c`** (or click **`C:Calib`**) : Dynamically calibrate HSV + YCrCb skin chrominance thresholds ($\mu \pm 2.5\sigma$) from your live palm.
   - **`s`** : Save a live snapshot (`demo_frame_*.png`, `raw_frame_*.png`, and annotated `demo_mask_*.png`).
   - **`q`** : Quit the application cleanly.

4. **Coordinated Dual-Hand Gestures & OS Actions:**
   - **Right Hand (Spatial Pointer & Clicks):**
     - **Isolated Index Point (1)** $\rightarrow$ Move Mouse Cursor (`Cascaded 1-Euro + Velocity-Adaptive EMA`)
     - **Thumb + Middle Single Tap** $\rightarrow$ Left Click / Select (Zero cursor drift)
     - **Thumb + Middle Double Tap** $\rightarrow$ Double Click / Open File
     - **Thumb + Middle Hold (>0.35s)** $\rightarrow$ Click & Drag Window (`0.50` hysteresis + `450ms` grace)
     - **Thumb + Ring Pinch** $\rightarrow$ Right Click Context Menu
     - **Thumb Lateral Flick (Key 6)** $\rightarrow$ Browser Back / Forward (`Alt + Left/Right`)
   - **Left Hand (Multimedia, Scroll & Zoom):**
     - **Open Palm Swirl (CW $\circlearrowright$ / CCW $\circlearrowleft$)** $\rightarrow$ Continuous Dial Scroll Down / Up (`atan2` kinematics)
     - **Vertical Pinch Slide** $\rightarrow$ Continuous Volume Up / Down ($\pm 5\%$)
     - **Flat Palm Forward Push** $\rightarrow$ Play / Pause Media (`Z-depth triangle expansion > 28%`)
     - **Left Thumb Lateral Point** $\rightarrow$ Next / Previous Media Track
     - **Closed Fist (0) [2.2s Hold]** $\rightarrow$ Safe Audio Mute / Unmute (`3.0s` cooldown)
     - **Left Pinch Scale + Right Point** $\rightarrow$ Bimanual Pinch-to-Zoom (`Ctrl + Mouse Wheel`)
