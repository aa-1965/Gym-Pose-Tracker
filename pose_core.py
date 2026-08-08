"""Shared pose analysis engine used by the PyQt apps and the web UI."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yt_dlp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINER_VIDEO = str(BASE_DIR / "vids" / "perfect_pushups.mp4")
DEFAULT_TRAINEE_VIDEO = "/Users/aaron/Desktop/CS /GYM/vids/imperfect_pusshups.mp4"

ANGLE_THRESHOLD = 25.0
DTW_WINDOW = 60
HEIGHT_THRESHOLD = 6
CONSECUTIVE_FRAMES = 15
DEBOUNCE_SEC = 2.0
COOLDOWN_SEC = 3.0

CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)
LEFT = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

JOINT_ANGLES = {
    "Left Knee": (23, 25, 27),
    "Right Knee": (24, 26, 28),
    "Left Hip": (11, 23, 25),
    "Right Hip": (12, 24, 26),
    "Left Elbow": (11, 13, 15),
    "Right Elbow": (12, 14, 16),
}

CORRECTIONS = {
    "Right Elbow": ("Straighten your right arm", "Bend your right elbow more"),
    "Left Elbow": ("Straighten your left arm", "Bend your left elbow more"),
    "Right Hip": ("Push your right hip back/open", "Bring your right hip forward"),
    "Left Hip": ("Push your left hip back/open", "Bring your left hip forward"),
    "Right Knee": ("Straighten your right knee", "Bend your right knee deeper"),
    "Left Knee": ("Straighten your left knee", "Bend your left knee deeper"),
}

HEIGHT_CHECKS = {
    "Hips": (23, 24),
    "Shoulders": (11, 12),
    "Knees": (25, 26),
}

mp_pose = mp.solutions.pose


def get_stream_url(url: str) -> str:
    if not url.startswith("http"):
        return url
    with yt_dlp.YoutubeDL(
        {
            "format": "best[height=720][vcodec^=avc1][ext=mp4]/best[height<=720][vcodec^=avc1]/best[height<=720]",
            "quiet": True,
        }
    ) as ydl:
        info = ydl.extract_info(url, download=False)
        fmt = next(
            (
                f
                for f in reversed(info.get("formats", []))
                if f.get("url")
                and f.get("vcodec", "").startswith("avc1")
                and (f.get("height") or 0) <= 720
            ),
            None,
        ) or next(
            (
                f
                for f in reversed(info.get("formats", []))
                if f.get("url")
                and f.get("vcodec", "none") != "none"
                and (f.get("height") or 0) <= 720
            ),
            None,
        )
        return fmt["url"] if fmt else info["formats"][-1]["url"]


def joint_angles_from_lm(lm) -> list[float]:
    angles = []
    for a, v, b in JOINT_ANGLES.values():
        a_pt = np.array([lm[a].x, lm[a].y, lm[a].z])
        v_pt = np.array([lm[v].x, lm[v].y, lm[v].z])
        b_pt = np.array([lm[b].x, lm[b].y, lm[b].z])
        va, vb = a_pt - v_pt, b_pt - v_pt
        cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)
        angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))
    return angles


def dtw_feedback(trainer_hist: list, trainee_hist: list) -> list[tuple]:
    if len(trainer_hist) < 2 or len(trainee_hist) < 2:
        return []
    t_arr = np.array(trainer_hist, dtype=np.float32)
    s_arr = np.array(trainee_hist, dtype=np.float32)
    _, path = fastdtw(t_arr, s_arr, dist=euclidean)

    joint_keys = list(JOINT_ANGLES.keys())
    n = len(joint_keys)
    sum_t = np.zeros(n)
    sum_s = np.zeros(n)
    for ti, si in path:
        sum_t += t_arr[ti]
        sum_s += s_arr[si]
    mean_t = sum_t / len(path)
    mean_s = sum_s / len(path)
    deltas = mean_t - mean_s

    feedback = []
    for j, joint in enumerate(joint_keys):
        delta = deltas[j]
        if abs(delta) >= ANGLE_THRESHOLD:
            more, less = CORRECTIONS[joint]
            feedback.append((joint, delta, more if delta > 0 else less, float(mean_t[j]), float(mean_s[j])))
    return feedback


def height_feedback(trainer_lm, trainee_lm) -> list[dict]:
    messages = []
    for label, (l, r) in HEIGHT_CHECKS.items():
        t_h = (trainer_lm[l].y + trainer_lm[r].y) / 2
        s_h = (trainee_lm[l].y + trainee_lm[r].y) / 2
        diff = (s_h - t_h) * 100
        if abs(diff) >= HEIGHT_THRESHOLD:
            direction = "lower" if diff < 0 else "raise"
            messages.append(
                {
                    "type": "height",
                    "joint": label,
                    "message": f"{direction.capitalize()} your {label.lower()}",
                    "trainee": round(s_h * 100),
                    "target": round(t_h * 100),
                }
            )
    return messages


def landmarks_to_dict(lm) -> list[dict]:
    return [{"x": p.x, "y": p.y, "z": p.z, "v": p.visibility} for p in lm]


def draw_pose(frame, lm, error_joints: set | None = None):
    if lm is None:
        return frame
    error_joints = error_joints or set()
    h, w = frame.shape[:2]

    def pt(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    for a, b in CONNECTIONS:
        color = (0, 255, 120)
        if a in error_joints or b in error_joints:
            color = (0, 80, 255)
        cv2.line(frame, pt(a), pt(b), color, 2, cv2.LINE_AA)

    for i in range(len(lm)):
        color = (0, 255, 120) if i not in error_joints else (0, 80, 255)
        cv2.circle(frame, pt(i), 3, color, -1, cv2.LINE_AA)
    return frame


def joint_marker_positions(lm, bad_joints: list[str]) -> list[dict]:
    markers = []
    for joint in bad_joints:
        _, v, _ = JOINT_ANGLES[joint]
        markers.append({"joint": joint, "x": lm[v].x, "y": lm[v].y})
    return markers


def compute_score(feedback: list) -> int:
    if not feedback:
        return 100
    penalty = sum(min(abs(item[1]), 60) for item in feedback)
    return max(0, min(100, int(100 - penalty * 0.8)))


class PoseSession:
    """Tracks trainer/trainee histories and debounced live feedback."""

    def __init__(
        self,
        trainer_video: str = DEFAULT_TRAINER_VIDEO,
        trainee_video: str = DEFAULT_TRAINEE_VIDEO,
    ):
        self.trainer_video = trainer_video
        self.trainee_video = trainee_video
        self.trainer_hist: list[list[float]] = []
        self.trainee_hist: list[list[float]] = []
        self.trainer_lm = None
        self.trainee_lm = None
        self.tracking_active = False
        self._streaks: dict[str, int] = {}
        self._last_notify = 0.0
        self._last_shown: dict[str, float] = {}
        self._active_warnings: list = []
        self._summary_issues: list[dict] = []

    def reset_tracking(self):
        self.trainee_hist.clear()
        self._streaks.clear()
        self._active_warnings.clear()
        self._last_notify = 0.0

    def ingest_trainer(self, lm):
        if lm is None:
            return
        self.trainer_lm = lm
        self.trainer_hist.append(joint_angles_from_lm(lm))
        if len(self.trainer_hist) > DTW_WINDOW:
            self.trainer_hist.pop(0)

    def analyze_trainee(self, lm) -> dict:
        if lm is None:
            return {"ok": False, "message": "No pose detected"}

        self.trainee_lm = lm
        self.trainee_hist.append(joint_angles_from_lm(lm))
        if len(self.trainee_hist) > DTW_WINDOW:
            self.trainee_hist.pop(0)

        if not self.tracking_active:
            return {
                "ok": True,
                "tracking": False,
                "landmarks": landmarks_to_dict(lm),
                "feedback": [],
                "markers": [],
                "score": None,
            }

        raw = dtw_feedback(self.trainer_hist, self.trainee_hist)
        if self.trainer_lm is not None:
            height_msgs = height_feedback(self.trainer_lm, lm)
        else:
            height_msgs = []

        now = time.time()
        confirmed = []
        frame_results = []
        raw_joints = {item[0] for item in raw}

        for joint in JOINT_ANGLES.keys():
            if joint in raw_joints:
                continue
            self._streaks[joint] = 0
            frame_results.append({"joint": joint, "isCorrect": True})

        for joint, delta, action, t_angle, s_angle in raw:
            streak = self._streaks.get(joint, 0) + 1
            self._streaks[joint] = streak
            is_bad = streak >= CONSECUTIVE_FRAMES
            frame_results.append({"joint": joint, "isCorrect": not is_bad, "message": action})
            if is_bad and now - self._last_shown.get(joint, 0) >= COOLDOWN_SEC:
                confirmed.append(
                    {
                        "joint": joint,
                        "action": action,
                        "delta": float(delta),
                        "trainee_angle": float(s_angle),
                        "target_angle": float(t_angle),
                        "severity": "major" if abs(delta) > 50 else "minor",
                    }
                )
                self._last_shown[joint] = now

        notify = None
        if confirmed and now - self._last_notify >= DEBOUNCE_SEC:
            self._last_notify = now
            self._active_warnings = confirmed
            notify = confirmed[0]
            self._summary_issues.append({**confirmed[0], "timestamp": now})
        elif not confirmed:
            self._active_warnings = []

        bad_joints = [item["joint"] for item in self._active_warnings]
        markers = joint_marker_positions(lm, bad_joints)
        for msg in height_msgs:
            markers.append({"joint": msg["joint"], "x": 0.5, "y": 0.2, "message": msg["message"]})

        score = compute_score(raw)
        return {
            "ok": True,
            "tracking": True,
            "landmarks": landmarks_to_dict(lm),
            "feedback": self._active_warnings,
            "height_feedback": height_msgs,
            "markers": markers,
            "score": score,
            "notify": notify,
            "frame_results": frame_results,
            "good_form": len(raw) == 0,
        }

    def get_summary(self) -> dict:
        return {"issues": self._summary_issues, "count": len(self._summary_issues)}


class PoseDetector:
    """Lightweight MediaPipe wrapper for single-frame analysis."""

    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=0, static_image_mode=False)

    def process_bgr(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        return result.pose_landmarks.landmark if result.pose_landmarks else None

    def close(self):
        self.pose.close()
