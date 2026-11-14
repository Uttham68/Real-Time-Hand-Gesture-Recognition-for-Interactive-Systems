# 5 to 8-Minute Video Demonstration Script & Submission Checklist
**Course:** CS-305 Computer Vision | Case Design Project #16  
**Student Name:** Porandla Uttham  
**Enrollment ID:** 23STUCHH010053  
**Section:** E (Data Science & Artificial Intelligence)  
**Institution:** ICFAI Foundation for Higher Education (IFHE), Hyderabad  

---

## 📋 Pre-Recording Setup & Checklist (Takes 3 Minutes)

1. **Screen Recording Tool:**
   - Use **OBS Studio** or **Windows Game Bar** (`Win + G` or `Win + Alt + R`).
   - Set microphone audio and system audio to recorded.
   - Recommended resolution: 1080p (Full HD) at 30 or 60 FPS.
2. **Setup Your Desk & Lighting:**
   - Sit in a comfortable position about 1.5 to 2.5 feet away from your webcam.
   - Make sure your background is clear and your hands are in view of the webcam.
3. **Open Applications to Demonstrate:**
   - Have a web browser open (e.g., YouTube video or Wikipedia page to show scrolling and play/pause).
   - Have a folder open with sample files (to show double-click opening and drag-and-drop).
4. **Terminal Ready:**
   - Keep a PowerShell or CMD terminal open in `d:\Practice\CVProject`.
   - Run: `python test_pipeline.py` first to show all tests pass.
   - Run: `python gesture_recognizer.py` for the live webcam demonstration.

---

## 🎬 Word-for-Word Video Script (5 to 8 Minutes)

### Part 1: Introduction & Project Overview (0:00 - 1:15)
> **[Camera / Screen showing title slide or terminal]**
>
> *"Good morning / afternoon respected faculty members.*  
> *My name is **Porandla Uttham**, Enrollment ID **23STUCHH010053**, from B.Tech Computer Science (Data Science & AI), Section E.*  
> 
> *Today, I am presenting my CS-305 Computer Vision Project, Case Design Project Number 16:*  
> ***'Real-Time Hand Gesture Recognition for Interactive Systems using Dual Classical and Landmark Models'***.  
> 
> *The main goal of this project is to turn any ordinary, low-cost laptop webcam into a fast, 60-Frames-Per-Second touchless mouse and multimedia controller.*  
> 
> *Most existing vision projects have four big problems:*  
> *First, when users try to click by pinching their index finger and thumb, the cursor jumps off target right before clicking.*  
> *Second, trying to control the mouse, volume, and scrolling all on one single hand causes constant mistakes.*  
> *Third, standard camera projects lag at 15 to 20 frames per second.*  
> *And fourth, they stop working in dim lighting or when your hand tilts toward the camera.*  
> 
> *In this project, I built a hybrid system that solves all of these problems. It combines classical skin color modeling and convexity analysis from our CS-305 Unit 2 curriculum with modern 3D hand landmark AI and sub-pixel filtering.*  
> *Let me first show you our automated test verification, and then I will demonstrate the live software."*

---

### Part 2: Automated Verification Test Suite (1:15 - 2:00)
> **[Action: Switch to terminal window. Type `python test_pipeline.py` and press Enter]**
>
> *"Before starting the live camera, let us run our automated verification pipeline: `test_pipeline.py`.*  
> 
> *As you can see on the screen, this script runs four complete test suites:*  
> - ***Suite 1*** *tests our Unit 2 classical skin segmentation and convexity defects across five benchmark hand postures: Pointing, Victory, Three, Four, and Open Palm. All five passed with 100% accuracy.*  
> - ***Suite 2*** *tests our 3D vector cosine joint kinematics. Even when the hand is tilted up to 38 degrees toward the camera, all postures pass.*  
> - ***Suite 3*** *tests our adaptive 1-Euro low-pass filter, proving that it removes over 80% of natural hand tremor.*  
> - ***Suite 4*** *benchmarks our vision core latency at just 5.46 milliseconds per frame, which theoretically supports up to 183 FPS and easily locks our camera at a solid 60 FPS.*  
> 
> *All test suites pass with 100% accuracy. Now, let us launch the real-time application."*

---

### Part 3: Launching the App & Explaining the UI (2:00 - 3:00)
> **[Action: In terminal, run `python gesture_recognizer.py`. The OpenCV window appears]**
>
> *"Here is the live application running in real time.*  
> *Notice three important parts of our interface:*  
> 1. *At the top, we have our live FPS counter running smoothly at 60 FPS, with clickable feature toggle buttons from 1 to 6, plus Night Mode and Calibration.*  
> 2. *At the bottom-left, our Information HUD displays live telemetry: the exact gesture recognized on each hand, the active action, and our 1-Euro filter frequency.*  
> 3. *At the bottom-right, you see the Picture-in-Picture window showing our CS-305 Unit 2 classical vision pipeline. It displays the skin mask, the green Convex Hull, the cyan palm circle found via Distance Transform, and the red inter-digital defect valleys.*  
> 
> *Our architecture divides work naturally between both hands:*  
> - *The **Right Hand** is the pointer and mouse.*  
> - *The **Left Hand** is the media and scroll controller.*  
> *Let us demonstrate each one."*

---

### Part 4: Right Hand Demonstration — Air Mouse & Clicks (3:00 - 4:30)
> **[Action: Raise your Right Hand with index finger pointed ☝️]**
>
> *"First, let us look at the Right Hand.*  
> 
> *1. **Air Mouse Movement:**  
> When I point my index finger with other fingers curled, the cyan cursor ring appears. As I move my hand, the mouse cursor moves smoothly across my monitor. Thanks to our 1-Euro filter and velocity-adaptive smoothing, when I stop moving, the cursor is completely still with zero trembling.*  
> 
> **[Action: Move cursor over a desktop folder or browser button. Tap Thumb and Middle finger together 👌]**  
> *2. **Select / Left Click:**  
> Notice our key innovation here: to click, I touch my **Thumb and Middle finger** together. My index finger remains pointing, so the cursor never jumps. A green ripple pulse appears and a chime sounds. It clicked cleanly.*  
> 
> **[Action: Tap Thumb and Middle finger twice quickly 👌👌 on a folder]**  
> *3. **Double Click / Open:**  
> When I tap my Thumb and Middle finger twice rapidly within 0.38 seconds, it triggers a native OS double-click and opens the window, with an 1800 Hertz confirmation chime.*  
> 
> **[Action: Pinch Thumb and Middle finger together and hold them closed for half a second, then move your hand]**  
> *4. **Click & Drag:**  
> When I hold the Thumb and Middle finger closed for more than 0.35 seconds, it enters Drag Mode. An orange indicator appears, and I can drag windows across my desktop. Even if my hand moves quickly, our 450-millisecond grace window prevents accidental drops.*  
> 
> **[Action: Open fingers. Pinch Thumb and Ring finger together 🖐️]**  
> *5. **Right Click:**  
> Touching my Thumb and Ring finger triggers a right-click, opening the context menu instantly."*

---

### Part 5: Left Hand Demonstration — Multimedia & Dial Scroll (4:30 - 6:00)
> **[Action: Raise your Left Hand with an open palm 🖐️]**
>
> *"Now let us look at the Left Hand, which controls multimedia and scrolling without interfering with our mouse cursor.*  
> 
> **[Action: Rotate your open left palm clockwise like turning a volume knob ↻]**  
> *1. **Circular Dial Scroll:**  
> When I show an open palm, a circular dial appears. Rotating my hand clockwise scrolls down web pages and documents smoothly. Rotating counter-clockwise scrolls up. It feels just like a physical wheel.*  
> 
> **[Action: Pinch Left Thumb and Index together, keeping middle finger extended, and slide hand up and down]**  
> *2. **Continuous Volume Slider:**  
> When I pinch my left thumb and index finger together, a vertical slider appears. Sliding up raises the system volume by 5%. Sliding down lowers the volume. This runs on a background thread so the camera feed never stutters.*  
> 
> **[Action: Push flat left palm forward toward camera lens]**  
> *3. **Push to Play / Pause:**  
> When I push my flat palm forward toward the camera, the system detects a 28% increase in palm area and toggles Play / Pause on our media player.*  
> 
> **[Action: Close Left Hand into a tight fist ✊ and hold for 2 seconds]**  
> *4. **Safe Hold-to-Mute:**  
> If I close my left hand into a fist, notice the screen shows a countdown. You must hold the fist continuously for 2.2 seconds before the audio mutes. This intentional safety delay prevents accidental mutes."*

---

### Part 6: Night Mode, Auto-Calibration & Hotkeys (6:00 - 7:00)
> **[Action: Press 'c' on keyboard or click 'C:Calib' on top toolbar]**
>
> *"Let us look at two special engineering features:*  
> 
> *1. **Dynamic Skin Calibration:**  
> If lighting changes or a user has a different skin tone, pressing the **'C' key** samples five regions of the palm and automatically calibrates the HSV and YCrCb color boundaries to that person's hand.*  
> 
> **[Action: Press 'n' on keyboard or turn off room light]**  
> *2. **Adaptive Low-Light Night Mode:**  
> Pressing **'N'** enables our low-light engine. It measures scene brightness and uses fast YCrCb CLAHE and gamma curve adjustment, allowing gestures to be tracked even in a dark room.*  
> 
> *3. **Live Snapshots:**  
> Pressing the **'S' key** saves a full snapshot of the clean frame, the live detection frame, and the annotated skin mask directly into our project folder for evaluation."*

---

### Part 7: Conclusion & Wrap-Up (7:00 - 7:45)
> **[Action: Press 'q' to cleanly exit the app. Return to presentation slides or webcam]**
>
> *"In summary:*  
> *This project successfully bridges classical Computer Vision theory from CS-305 Unit 2 with state-of-the-art 3D AI and real-time Windows OS integration.*  
> *It achieves:*  
> - *A fast 60 FPS real-time pipeline with sub-6-millisecond vision latency,*  
> - *Over 80% reduction in cursor jitter using the 1-Euro filter,*  
> - *Zero cursor drift during clicks through decoupled finger roles, and*  
> - *A 100% automated test pass rate.*  
> 
> *All code, report documents, slides, and test scripts are organized in the project folder.*  
> *Thank you very much for your time and guidance, respected faculty members."*

---

## 📌 Submission Checklist for Faculty
- [ ] Uploaded `Complete_Project_Overview_CS305.pdf` (Project report & overview).
- [ ] Uploaded `Faculty_Presentation_CS305_Project16.pptx` (Presentation slides).
- [ ] Uploaded `gesture_recognizer.py` and `test_pipeline.py` (Source code).
- [ ] Recorded 5–8 minute MP4 video demonstration following this script.
- [ ] Included Student Name: **Porandla Uttham**, ID: **23STUCHH010053**, Section: **E (DS & AI)**.

