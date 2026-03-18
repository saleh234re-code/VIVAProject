import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import psutil
import threading

# ══════════════════════════════════════════════════════════════════
#  VIVASUM PRO ANALYZER
#  Optimized for: Intel Core i7-5600U | Python 3.10 | VS Code
#  Camera Resolution: 854x480 | FPS: 25
# ══════════════════════════════════════════════════════════════════

mp_holistic = mp.solutions.holistic


# ──────────────────────────────────────────────────────────────────
#  Hardware Monitor Class
# ──────────────────────────────────────────────────────────────────

class HardwareMonitor:
    """
    Monitors CPU and RAM usage every second in a separate thread.

    Pressure Levels (tuned for i7-5600U):
        LOW    -> CPU < 45%  and RAM < 65%  : Normal operation
        MEDIUM -> CPU 45-65% or  RAM 65-80% : Reduce model complexity
        HIGH   -> CPU 65-80% or  RAM 80-90% : Skip frames
        DANGER -> CPU > 80%  or  RAM > 90%  : Pause processing
    """

    # Thresholds tuned specifically for i7-5600U
    THRESHOLDS = {
        'cpu': {'low': 45, 'medium': 65, 'high': 80},
        'ram': {'low': 65, 'medium': 80, 'high': 90},
    }

    def __init__(self, interval: float = 1.0):
        self.interval    = interval
        self.cpu_usage   = 0.0
        self.ram_usage   = 0.0
        self.level       = 'LOW'
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def _monitor_loop(self):
        """Continuously monitors hardware usage in background."""
        while not self._stop_event.is_set():
            self.cpu_usage = psutil.cpu_percent(interval=self.interval)
            self.ram_usage = psutil.virtual_memory().percent
            self.level     = self._evaluate()

    def _evaluate(self) -> str:
        """Returns current pressure level based on CPU and RAM usage."""
        cpu = self.cpu_usage
        ram = self.ram_usage
        t   = self.THRESHOLDS
        if cpu > t['cpu']['high'] or ram > t['ram']['high']:
            return 'DANGER'
        if cpu > t['cpu']['medium'] or ram > t['ram']['medium']:
            return 'HIGH'
        if cpu > t['cpu']['low'] or ram > t['ram']['low']:
            return 'MEDIUM'
        return 'LOW'

    @property
    def recommended_complexity(self) -> int:
        """Returns the recommended model complexity based on hardware state."""
        return {'LOW': 1, 'MEDIUM': 0, 'HIGH': 0, 'DANGER': 0}[self.level]

    @property
    def pause_processing(self) -> bool:
        """Returns True if processing should be paused to protect hardware."""
        return self.level == 'DANGER'

    def stop(self):
        """Stops the background monitoring thread."""
        self._stop_event.set()

    def status_line(self) -> str:
        """Returns a formatted status string for display."""
        return (f"CPU: {self.cpu_usage:.0f}%  "
                f"RAM: {self.ram_usage:.0f}%  "
                f"Level: {self.level}")


# ──────────────────────────────────────────────────────────────────
#  Main Analyzer Class
# ──────────────────────────────────────────────────────────────────

class VivasumProAnalyzer:
    """
    Body Language Analysis Engine — VIVASUM PRO

    Measures three core axes:
        Eye Contact      40%
        Hand Gestures    30%
        Posture          30%

    Hardware Protection Layers:
        1. HardwareMonitor     -> Monitors CPU/RAM in a separate thread
        2. Adaptive Complexity -> Reduces model complexity automatically
        3. Frame Skipping      -> Analyzes every Nth frame
        4. Resolution Scaling  -> Downscales frame before processing
        5. Danger Pause        -> Halts processing if CPU exceeds 80%
        6. Camera Cap          -> Limits camera to 854x480 @ 25fps
    """

    # Frame downscale ratios per pressure level
    RESOLUTION_SCALE = {
        'LOW':    1.0,    # Full resolution
        'MEDIUM': 0.75,   # 75% of original
        'HIGH':   0.5,    # 50% of original
        'DANGER': 0.5,
    }

    # Analyze every Nth frame per pressure level
    SKIP_RATIO = {
        'LOW':    1,   # Every frame
        'MEDIUM': 2,   # Every 2nd frame
        'HIGH':   3,   # Every 3rd frame
        'DANGER': 4,   # Every 4th frame
    }

    def __init__(self, window_size: int = 20):
        # Always start with complexity=0 for safety
        self.current_complexity = 0
        self.holistic           = self._build_holistic(0)

        # Hardware monitor running in background thread
        self.hw = HardwareMonitor(interval=1.0)

        # Moving average buffers (size 20 to save RAM)
        self.eye_buffer     = deque(maxlen=window_size)
        self.hand_buffer    = deque(maxlen=window_size)
        self.posture_buffer = deque(maxlen=window_size)

        # Session history for final report
        self.history = {
            'eye': [], 'hand': [], 'posture': [], 'total': [], 'time': []
        }

        # Wrist positions for velocity tracking
        self.prev_wrists = {'left': None, 'right': None}

        # Session statistics
        self.total_frames    = 0
        self.analyzed_frames = 0
        self.skipped_frames  = 0
        self.paused_frames   = 0
        self.no_face_frames  = 0

        self.start_time = time.time()

    # ── Holistic Model Builder ────────────────────────────────────

    def _build_holistic(self, complexity: int):
        """Builds and returns a MediaPipe Holistic model instance."""
        return mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=complexity,
            enable_segmentation=False,
            refine_face_landmarks=True,   # Required for iris tracking (point 468)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _maybe_update_complexity(self):
        """Updates model complexity if hardware conditions have changed."""
        recommended = self.hw.recommended_complexity
        if recommended != self.current_complexity:
            self.holistic.close()
            self.holistic           = self._build_holistic(recommended)
            self.current_complexity = recommended

    # ── Utility ───────────────────────────────────────────────────

    @staticmethod
    def _euclidean(p1, p2) -> float:
        """Calculates Euclidean distance between two MediaPipe landmarks."""
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # ── Core Metric Calculators ───────────────────────────────────

    def _eye_contact_score(self, face_landmarks) -> float:
        """
        Calculates eye contact score (0-100).
        Uses iris landmarks (468, 473) if available, falls back to eye corners.
        """
        if face_landmarks is None:
            self.no_face_frames += 1
            return 0.0

        landmarks = face_landmarks.landmark

        if len(landmarks) > 473:
            # Primary: use iris center (average of both eyes)
            l_iris = landmarks[468]
            r_iris = landmarks[473]
            avg_x  = (l_iris.x + r_iris.x) / 2
            avg_y  = (l_iris.y + r_iris.y) / 2
            dev    = np.sqrt((avg_x - 0.5) ** 2 + (avg_y - 0.5) ** 2)
            return max(0.0, 100.0 - dev * 500)
        else:
            # Fallback: use eye corner midpoint (landmarks 33, 263)
            lc  = landmarks[33]
            rc  = landmarks[263]
            cx  = (lc.x + rc.x) / 2
            cy  = (lc.y + rc.y) / 2
            dev = np.sqrt((cx - 0.5) ** 2 + (cy - 0.42) ** 2)
            return max(0.0, 100.0 - dev * 300)

    def _hand_gesture_score(self, left_lm, right_lm) -> float:
        """
        Calculates hand gesture score (0-100).
        Uses an optimal range model:
            - Hand presence  -> bonus points (max 50)
            - Natural motion -> best score
            - Excessive motion or stillness -> penalized
        """
        hand_energy   = 0.0
        hands_visible = 0

        if left_lm is not None:
            hands_visible += 1
            curr = left_lm.landmark[0]
            if self.prev_wrists['left'] is not None:
                hand_energy += self._euclidean(curr, self.prev_wrists['left'])
            self.prev_wrists['left'] = curr

        if right_lm is not None:
            hands_visible += 1
            curr = right_lm.landmark[0]
            if self.prev_wrists['right'] is not None:
                hand_energy += self._euclidean(curr, self.prev_wrists['right'])
            self.prev_wrists['right'] = curr

        # No hands detected -> heavy penalty
        if hands_visible == 0:
            return 10.0

        # Presence bonus: 25 points per visible hand (max 50)
        presence_bonus = hands_visible * 25

        # Motion score based on optimal movement range
        if hand_energy < 0.001:   motion_score = 15.0   # Completely still
        elif hand_energy < 0.008: motion_score = 40.0   # Slight movement
        elif hand_energy < 0.025: motion_score = 50.0   # Natural expressive motion (best)
        elif hand_energy < 0.06:  motion_score = 30.0   # Slightly excessive
        else:                     motion_score = max(5.0, 50.0 - hand_energy * 600)  # Too much

        return min(100.0, presence_bonus + motion_score)

    def _posture_score(self, pose_landmarks) -> float:
        """
        Calculates posture score (0-100).
        Measures:
            - Shoulder tilt   (horizontal alignment) -> 60% weight
            - Forward lean    (depth axis balance)   -> 40% weight
        """
        if pose_landmarks is None:
            return 0.0

        l_sh = pose_landmarks.landmark[11]   # Left shoulder
        r_sh = pose_landmarks.landmark[12]   # Right shoulder

        tilt_score    = max(0.0, 100.0 - abs(l_sh.y - r_sh.y) * 1500)
        forward_score = max(0.0, 100.0 - abs(l_sh.z - r_sh.z) * 800)
        return (tilt_score * 0.6) + (forward_score * 0.4)

    # ── Grade Evaluation ──────────────────────────────────────────

    @staticmethod
    def _evaluate_grade(total, eye, hand, posture):
        """
        Returns (grade, emoji, tips) based on total score.
        Tips are personalized based on the weakest axis.
        """
        # Determine grade level
        if total >= 85:
            grade, emoji = "Excellent", "🏆"
        elif total >= 70:
            grade, emoji = "Very Good", "🌟"
        elif total >= 55:
            grade, emoji = "Good", "✅"
        elif total >= 40:
            grade, emoji = "Acceptable", "📈"
        else:
            grade, emoji = "Needs Improvement", "💪"

        # Generate personalized tips based on weakest axes
        tips = []
        if eye < 55:
            tips.append("Maintain more direct eye contact with the camera")
        elif eye < 70:
            tips.append("Eye contact is decent — try to keep it more consistent")

        if hand < 40:
            tips.append("Keep your hands visible and use them to express ideas")
        elif hand < 60:
            tips.append("Good hand presence — try to use more natural gestures")

        if posture < 55:
            tips.append("Sit upright and avoid leaning or tilting")
        elif posture < 70:
            tips.append("Posture is acceptable — work on keeping it more stable")

        if not tips:
            tips.append("Outstanding performance across all axes — keep it up!")

        return grade, emoji, tips

    # ── Frame Processing ──────────────────────────────────────────

    def _scale_frame(self, frame):
        """Downscales frame based on current hardware pressure level."""
        scale = self.RESOLUTION_SCALE[self.hw.level]
        if scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _process_frame(self, frame):
        """
        Processes a single frame with all hardware protection layers active.
        Stores smoothed scores into session history.
        """
        self.total_frames += 1

        # Protection 1: Pause if hardware is in danger zone
        if self.hw.pause_processing:
            self.paused_frames += 1
            return

        # Protection 2: Frame skipping based on pressure level
        if self.total_frames % self.SKIP_RATIO[self.hw.level] != 0:
            self.skipped_frames += 1
            return

        # Protection 3: Downscale frame before sending to model
        small_frame = self._scale_frame(frame)
        rgb         = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Protection 4: Adaptive complexity check every 60 frames
        if self.total_frames % 60 == 0:
            self._maybe_update_complexity()

        # Run MediaPipe inference
        results = self.holistic.process(rgb)
        self.analyzed_frames += 1

        # Calculate raw scores
        raw_eye     = self._eye_contact_score(results.face_landmarks)
        raw_hand    = self._hand_gesture_score(
                          results.left_hand_landmarks,
                          results.right_hand_landmarks)
        raw_posture = self._posture_score(results.pose_landmarks)

        # Apply moving average smoothing
        self.eye_buffer.append(raw_eye)
        self.hand_buffer.append(raw_hand)
        self.posture_buffer.append(raw_posture)

        smooth_e = float(np.mean(self.eye_buffer))
        smooth_h = float(np.mean(self.hand_buffer))
        smooth_p = float(np.mean(self.posture_buffer))

        # Weighted total score
        total = (smooth_e * 0.40) + (smooth_h * 0.30) + (smooth_p * 0.30)

        # Store in history
        self.history['eye'].append(smooth_e)
        self.history['hand'].append(smooth_h)
        self.history['posture'].append(smooth_p)
        self.history['total'].append(total)
        self.history['time'].append(time.time() - self.start_time)

    # ── Session Runner ─────────────────────────────────────────────

    def run_session(self):
        """Starts the live analysis session."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            self.hw.stop()
            return

        # Protection 5: Cap camera resolution to 854x480 @ 25fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  854)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)

        print("╔══════════════════════════════════════╗")
        print("║     VIVASUM PRO — Ready ✅            ║")
        print("║  Press  q  or  ESC  to end session   ║")
        print("╚══════════════════════════════════════╝\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)

            # Run silent background analysis
            self._process_frame(frame)

            # ── Student-facing UI (no scores shown) ───────────────
            cv2.rectangle(frame, (0, 0), (480, 60), (20, 20, 20), -1)
            cv2.putText(frame, "VIVASUM LIVE ANALYSIS ACTIVE",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (56, 189, 248), 2)

            # ── Hardware status indicator (bottom bar) ─────────────
            hw_color = {
                'LOW':    (0, 255, 100),
                'MEDIUM': (0, 200, 255),
                'HIGH':   (0, 100, 255),
                'DANGER': (0, 0, 255),
            }[self.hw.level]
            cv2.rectangle(frame,
                          (0, frame.shape[0] - 30),
                          (frame.shape[1], frame.shape[0]),
                          (20, 20, 20), -1)
            cv2.putText(frame, self.hw.status_line(),
                        (10, frame.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, hw_color, 1)

            # Warning overlay if hardware is in danger zone
            if self.hw.level == 'DANGER':
                cv2.putText(frame, "WARNING: HIGH LOAD — ANALYSIS PAUSED",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 255), 2)

            cv2.imshow('VIVASUM - Presentation Mode', frame)

            # Exit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hw.stop()
        self.generate_pro_report()

    # ── Report Generator ───────────────────────────────────────────

    def generate_pro_report(self):
        """Generates the final report: text summary + charts + saved image."""
        if not self.history['total']:
            print("WARNING: Not enough data to generate report.")
            return

        avg_eye     = float(np.mean(self.history['eye']))
        avg_hand    = float(np.mean(self.history['hand']))
        avg_posture = float(np.mean(self.history['posture']))
        final_score = float(np.mean(self.history['total']))
        face_vis    = ((self.total_frames - self.no_face_frames)
                       / max(1, self.total_frames)) * 100
        analysis_rate = (self.analyzed_frames
                         / max(1, self.total_frames)) * 100

        # ── Text Report ───────────────────────────────────────────
        print("\n" + "═" * 50)
        print("         VIVASUM PRO — FINAL REPORT          ")
        print("═" * 50)
        print(f"  Overall Confidence Score  :  {final_score:.1f}%")
        print("─" * 50)
        print(f"  1. Eye Contact    (40%)   :  {avg_eye:.1f}%")
        print(f"  2. Hand Gestures  (30%)   :  {avg_hand:.1f}%")
        print(f"  3. Posture        (30%)   :  {avg_posture:.1f}%")
        print("─" * 50)
        print(f"  Total Frames              :  {self.total_frames}")
        print(f"  Analyzed Frames           :  {self.analyzed_frames}"
              f"  ({analysis_rate:.0f}%)")
        print(f"  Skipped  (Frame Skip)     :  {self.skipped_frames}")
        print(f"  Paused   (DANGER mode)    :  {self.paused_frames}")
        print(f"  Face Visibility           :  {face_vis:.1f}%")
        if face_vis < 70:
            print("  WARNING: Face not detected enough — check lighting")
        if analysis_rate < 50:
            print("  WARNING: Less than 50% frames analyzed — high system load")
        print("═" * 50)

        # ── Grade & Recommendations ───────────────────────────────
        grade, emoji, tips = self._evaluate_grade(
            final_score, avg_eye, avg_hand, avg_posture
        )
        print()
        print("╔" + "═" * 48 + "╗")
        print(f"║  {emoji}  Final Grade : {grade:<37}║")
        print("╠" + "═" * 48 + "╣")
        print("║  Recommendations:                               ║")
        for tip in tips:
            # Truncate tip if too long for box width
            print(f"║   • {tip[:43]:<43}║")
        print("╚" + "═" * 48 + "╝")
        print()

        # ── Charts ────────────────────────────────────────────────
        t = self.history['time']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plt.style.use('dark_background')

        # Chart 1: Performance trend over time
        ax1 = axes[0]
        ax1.plot(t, self.history['total'],
                 color='#38bdf8', lw=2.5,
                 label=f'Total  ({final_score:.1f}%)')
        ax1.plot(t, self.history['eye'],
                 color='#00e5ff', lw=1.2, alpha=0.7,
                 label=f'Eye    ({avg_eye:.1f}%)')
        ax1.plot(t, self.history['hand'],
                 color='#f97316', lw=1.2, alpha=0.7,
                 label=f'Hands  ({avg_hand:.1f}%)')
        ax1.plot(t, self.history['posture'],
                 color='#a855f7', lw=1.2, alpha=0.7,
                 label=f'Posture({avg_posture:.1f}%)')
        ax1.fill_between(t, self.history['total'],
                         color='#38bdf8', alpha=0.08)
        ax1.set_title("Performance Trend Over Time", fontsize=13, pad=12)
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Score (0 – 100)")
        ax1.set_ylim(0, 110)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(alpha=0.2)

        # Chart 2: Category averages bar chart
        ax2 = axes[1]
        categories = ['Eye Contact\n(40%)', 'Hand Gestures\n(30%)', 'Posture\n(30%)']
        values     = [avg_eye, avg_hand, avg_posture]
        colors_bar = ['#00e5ff', '#f97316', '#a855f7']
        bars = ax2.bar(categories, values, color=colors_bar,
                       alpha=0.85, edgecolor='white', lw=0.8)
        ax2.axhline(y=final_score, color='#38bdf8',
                    linestyle='--', lw=2,
                    label=f'Final Score: {final_score:.1f}%  |  {grade}  {emoji}')
        ax2.set_title("Category Averages", fontsize=13, pad=12)
        ax2.set_ylabel("Average Score")
        ax2.set_ylim(0, 115)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.2)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 2,
                     f'{val:.1f}%', ha='center', va='bottom',
                     fontsize=11, fontweight='bold', color='white')

        plt.suptitle("VIVASUM PRO — Body Language Analysis Report",
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig("vivasum_report.png", dpi=150,
                    bbox_inches='tight', facecolor='#0f172a')
        plt.show()
        print("  Report saved -> vivasum_report.png")


# ══════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    viva = VivasumProAnalyzer(window_size=20)
    viva.run_session()
