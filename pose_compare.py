import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import threading
import queue
import time
import cv2
import mediapipe as mp
import numpy as np
import yt_dlp

from PyQt5 import QtWidgets, QtCore, QtGui

# ── config ────────────────────────────────────────────────────────────────────

TRAINER_URL = "https://www.youtube.com/watch?v=LnCQ-MECZSw"  # Video-1
TRAINEE_URL = "https://www.youtube.com/watch?v=was4RtzpfJs"  # Video-2 — replace with trainee URL
SKIP_SEC    = 10
ANGLE_THRESHOLD = 20.0  # degrees — differences below this are ignored

# joint triplets (A, VERTEX, B) — angle is measured at VERTEX
JOINT_ANGLES = {
    "Right Elbow":    (12, 14, 16),
    "Left Elbow":     (11, 13, 15),
    "Right Shoulder": (14, 12, 24),
    "Left Shoulder":  (13, 11, 23),
    "Right Hip":      (12, 24, 26),
    "Left Hip":       (11, 23, 25),
    "Right Knee":     (24, 26, 28),
    "Left Knee":      (23, 25, 27),
}

# human-readable correction hints per joint
CORRECTIONS = {
    "Right Elbow":    ("extend your right arm more",        "bend your right elbow more"),
    "Left Elbow":     ("extend your left arm more",         "bend your left elbow more"),
    "Right Shoulder": ("raise your right arm higher",       "lower your right arm"),
    "Left Shoulder":  ("raise your left arm higher",        "lower your left arm"),
    "Right Hip":      ("open your right hip more",          "close your right hip"),
    "Left Hip":       ("open your left hip more",           "close your left hip"),
    "Right Knee":     ("straighten your right leg more",    "bend your right knee more"),
    "Left Knee":      ("straighten your left leg more",     "bend your left knee more"),
}

mp_pose = mp.solutions.pose

# ── helpers ───────────────────────────────────────────────────────────────────

def get_stream_url(url):
    with yt_dlp.YoutubeDL({"format": "best[height=720][vcodec^=avc1][ext=mp4]/best[height<=720][vcodec^=avc1]/best[height<=720]", "quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        fmt = next(
            (f for f in reversed(info.get("formats", []))
             if f.get("url") and f.get("vcodec", "").startswith("avc1") and (f.get("height") or 0) <= 720),
            None
        ) or next(
            (f for f in reversed(info.get("formats", []))
             if f.get("url") and f.get("vcodec", "none") != "none" and (f.get("height") or 0) <= 720),
            None
        )
        return fmt["url"] if fmt else info["formats"][-1]["url"]


def angle_at_vertex(lm, a, v, b):
    """Angle in degrees at landmark v between landmarks a and b."""
    A = np.array([lm[a].x, lm[a].y, lm[a].z])
    V = np.array([lm[v].x, lm[v].y, lm[v].z])
    B = np.array([lm[b].x, lm[b].y, lm[b].z])
    va = A - V
    vb = B - V
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def compare_poses(trainer_lm, trainee_lm):
    """Return list of (joint_name, delta, suggestion) for joints that differ."""
    feedback = []
    for joint, (a, v, b) in JOINT_ANGLES.items():
        t_angle  = angle_at_vertex(trainer_lm, a, v, b)
        s_angle  = angle_at_vertex(trainee_lm, a, v, b)
        delta    = t_angle - s_angle          # positive = trainee too closed
        if abs(delta) >= ANGLE_THRESHOLD:
            more, less   = CORRECTIONS[joint]
            suggestion   = more if delta > 0 else less
            feedback.append((joint, delta, suggestion))
    return feedback


# ── producer ──────────────────────────────────────────────────────────────────

def make_producer(url, out_queue, stop_event, paused_ref, seek_ref, cap_ref):
    def producer():
        cap = cv2.VideoCapture(get_stream_url(url))
        if not cap.isOpened():
            out_queue.put(None)
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 24)
        cap_ref[0] = cap

        pose       = mp_pose.Pose(model_complexity=0)
        frame_time = 1.0 / 24
        last_time  = time.time()

        while not stop_event.is_set() and cap.isOpened():
            if seek_ref[0] is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_ref[0])
                seek_ref[0] = None
            if paused_ref[0]:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now = time.time()
            elapsed = now - last_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_time = time.time()

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            lm      = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None

            if out_queue.full():
                try:
                    out_queue.get_nowait()
                except queue.Empty:
                    pass
            out_queue.put((frame, lm))

        cap.release()
        pose.close()
        out_queue.put(None)
    return producer


# ── video panel widget ────────────────────────────────────────────────────────

class VideoPanel(QtWidgets.QWidget):
    def __init__(self, title, cap_ref, seek_ref, paused_ref):
        super().__init__()
        self.cap_ref    = cap_ref
        self.seek_ref   = seek_ref
        self.paused_ref = paused_ref

        self.setStyleSheet("background:#111;")

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet("color:white; font-size:14px; font-weight:bold; padding:4px;")
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)

        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)

        btn_style = """QPushButton {
            background:#333; color:white; border:none;
            padding:6px 16px; font-size:18px; border-radius:4px;
        } QPushButton:hover { background:#555; }"""

        self.btn_rew = QtWidgets.QPushButton("⏪")
        self.btn_pp  = QtWidgets.QPushButton("⏸")
        self.btn_fwd = QtWidgets.QPushButton("⏩")
        for b in (self.btn_rew, self.btn_pp, self.btn_fwd):
            b.setStyleSheet(btn_style)

        controls = QtWidgets.QHBoxLayout()
        controls.addStretch()
        controls.addWidget(self.btn_rew)
        controls.addWidget(self.btn_pp)
        controls.addWidget(self.btn_fwd)
        controls.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(title_lbl)
        layout.addWidget(self.label)
        layout.addLayout(controls)

        self.btn_rew.clicked.connect(self.rewind)
        self.btn_pp.clicked.connect(self.toggle_pause)
        self.btn_fwd.clicked.connect(self.forward)

    def show_frame(self, frame):
        h, w, ch = frame.shape
        img = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_BGR888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.label.width(), self.label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def toggle_pause(self):
        self.paused_ref[0] = not self.paused_ref[0]
        self.btn_pp.setText("▶" if self.paused_ref[0] else "⏸")

    def rewind(self):
        if self.cap_ref[0]:
            cur = int(self.cap_ref[0].get(cv2.CAP_PROP_POS_FRAMES))
            self.seek_ref[0] = max(0, cur - SKIP_SEC * 24)

    def forward(self):
        if self.cap_ref[0]:
            cur = int(self.cap_ref[0].get(cv2.CAP_PROP_POS_FRAMES))
            self.seek_ref[0] = cur + SKIP_SEC * 24


# ── feedback panel ────────────────────────────────────────────────────────────

class FeedbackPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:#0a0a0a;")
        self.setMinimumWidth(340)

        title = QtWidgets.QLabel("📋 Trainee Feedback")
        title.setStyleSheet("color:white; font-size:15px; font-weight:bold; padding:8px;")
        title.setAlignment(QtCore.Qt.AlignCenter)

        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("""
            QTextEdit {
                background:#111; color:#eee;
                font-size:13px; border:none; padding:8px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(title)
        layout.addWidget(self.text)

    def update_feedback(self, feedback):
        if not feedback:
            self.text.setHtml("<span style='color:#4caf50;'>✅ Pose looks good!</span>")
            return
        lines = []
        for joint, delta, suggestion in feedback:
            diff = abs(delta)
            severity = "🔴" if diff > 40 else "🟡"
            lines.append(
                f"{severity} <b>{joint}</b><br>"
                f"&nbsp;&nbsp;Trainer: {'+' if delta > 0 else ''}{delta:.0f}° difference<br>"
                f"&nbsp;&nbsp;→ <i>{suggestion}</i><br>"
            )
        self.text.setHtml("<br>".join(lines))


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Comparison — Trainer vs Trainee")
        self.setStyleSheet("background:#0d0d0d;")

        self.stop_event = threading.Event()

        self.trainer_queue   = queue.Queue(maxsize=2)
        self.trainee_queue   = queue.Queue(maxsize=2)
        self.trainer_cap_ref = [None]
        self.trainee_cap_ref = [None]
        self.trainer_seek    = [None]
        self.trainee_seek    = [None]
        self.trainer_paused  = [False]
        self.trainee_paused  = [False]

        self.trainer_panel  = VideoPanel("🎓 Trainer (Video 1)", self.trainer_cap_ref, self.trainer_seek, self.trainer_paused)
        self.trainee_panel  = VideoPanel("🏋️ Trainee (Video 2)", self.trainee_cap_ref, self.trainee_seek, self.trainee_paused)
        self.feedback_panel = FeedbackPanel()

        videos = QtWidgets.QHBoxLayout()
        videos.addWidget(self.trainer_panel)
        videos.addWidget(self.trainee_panel)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(videos, stretch=3)
        main_layout.addWidget(self.feedback_panel, stretch=1)

        # latest landmarks for comparison
        self.trainer_lm = None
        self.trainee_lm = None

        # start producers
        threading.Thread(target=make_producer(
            TRAINER_URL, self.trainer_queue, self.stop_event,
            self.trainer_paused, self.trainer_seek, self.trainer_cap_ref
        ), daemon=True).start()

        threading.Thread(target=make_producer(
            TRAINEE_URL, self.trainee_queue, self.stop_event,
            self.trainee_paused, self.trainee_seek, self.trainee_cap_ref
        ), daemon=True).start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.consume)
        self.timer.start(42)

    def consume(self):
        if self.stop_event.is_set():
            QtWidgets.QApplication.quit()
            return

        # drain trainer queue
        try:
            item = self.trainer_queue.get_nowait()
            if item is None:
                self.stop_event.set()
                return
            frame, lm = item
            self.trainer_panel.show_frame(frame)
            if lm:
                self.trainer_lm = lm
        except queue.Empty:
            pass

        # drain trainee queue
        try:
            item = self.trainee_queue.get_nowait()
            if item is None:
                self.stop_event.set()
                return
            frame, lm = item
            self.trainee_panel.show_frame(frame)
            if lm:
                self.trainee_lm = lm
        except queue.Empty:
            pass

        # compare and update feedback
        if self.trainer_lm and self.trainee_lm:
            feedback = compare_poses(self.trainer_lm, self.trainee_lm)
            self.feedback_panel.update_feedback(feedback)

    def closeEvent(self, event):
        self.stop_event.set()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(1600, 700)
    win.show()
    app.exec_()
