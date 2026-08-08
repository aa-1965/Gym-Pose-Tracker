#!/usr/bin/env python3
"""Flask API connecting the GymForm web UI to MediaPipe pose analysis."""

from __future__ import annotations

import base64
import copy
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from pose_core import (
    DEFAULT_TRAINEE_VIDEO,
    DEFAULT_TRAINER_VIDEO,
    PoseDetector,
    PoseSession,
    draw_pose,
    get_stream_url,
)

UI_DIR = Path(__file__).resolve().parent / "ui"
BASE_FPS = 24

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="")

session = PoseSession(DEFAULT_TRAINER_VIDEO, DEFAULT_TRAINEE_VIDEO)
_detector = None
_detector_lock = threading.Lock()

_trainee_source = "camera"  # "camera" | "video"

_analysis_lock = threading.Lock()
_latest_analysis: dict | None = None


def get_detector() -> PoseDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = PoseDetector()
        return _detector


# ── Shared playback (trainer + optional trainee video) ─────────────────────────
_video_lock = threading.Lock()
_cap = None
_trainee_cap = None
_total_frames = 0
_trainee_total_frames = 0
_paused = True
_speed = 1.0
_seek_frame = None
_trainee_seek_frame = None
_current_frame = None
_current_lm = None
_trainee_current_frame = None
_replay_requested = False
_trainee_replay_requested = False


def _open_cap():
    global _cap, _total_frames
    path = get_stream_url(session.trainer_video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open trainer video: {session.trainer_video}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _cap = cap
    _total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    return cap


def _open_trainee_cap():
    global _trainee_cap, _trainee_total_frames
    path = get_stream_url(session.trainee_video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open trainee video: {session.trainee_video}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _trainee_cap = cap
    _trainee_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    return cap


def _read_trainer_frame():
    global _current_frame, _current_lm, _replay_requested, _seek_frame

    with _video_lock:
        if _cap is None:
            _open_cap()

        if _replay_requested:
            _cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _replay_requested = False

        if _seek_frame is not None:
            _cap.set(cv2.CAP_PROP_POS_FRAMES, _seek_frame)
            _seek_frame = None

        if _paused:
            frame = _current_frame.copy() if _current_frame is not None else None
            lm = _current_lm
        else:
            ret, frame = _cap.read()
            if not ret or frame is None:
                _cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = _cap.read()
            if frame is None:
                return None, None, 0, BASE_FPS
            lm = get_detector().process_bgr(frame)
            _current_frame = frame.copy()
            _current_lm = lm
            session.ingest_trainer(lm)

        if frame is None:
            return None, None, 0, BASE_FPS

        pos = int(_cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = _cap.get(cv2.CAP_PROP_FPS) or BASE_FPS
        return frame, lm, pos, fps


def _read_trainee_frame():
    global _trainee_current_frame, _replay_requested, _trainee_replay_requested
    global _seek_frame, _trainee_seek_frame, _latest_analysis

    with _video_lock:
        if _trainee_cap is None:
            _open_trainee_cap()

        if _replay_requested or _trainee_replay_requested:
            _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _trainee_replay_requested = False

        if _trainee_seek_frame is not None:
            _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, _trainee_seek_frame)
            _trainee_seek_frame = None
        elif _seek_frame is not None:
            _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, _seek_frame)

        if _paused:
            frame = _trainee_current_frame.copy() if _trainee_current_frame is not None else None
            lm = None
            if frame is not None and session.tracking_active:
                lm = get_detector().process_bgr(frame)
        else:
            ret, frame = _trainee_cap.read()
            if not ret or frame is None:
                _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = _trainee_cap.read()
            if frame is None:
                return None, None, 0, BASE_FPS
            lm = get_detector().process_bgr(frame)
            _trainee_current_frame = frame.copy()

        if frame is None:
            return None, None, 0, BASE_FPS

        if lm is not None and session.tracking_active:
            result = session.analyze_trainee(lm)
            with _analysis_lock:
                _latest_analysis = result

        pos = int(_trainee_cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = _trainee_cap.get(cv2.CAP_PROP_FPS) or BASE_FPS
        return frame, lm, pos, fps


def _trainer_mjpeg(draw_overlay: bool = True):
    frame_time = 1.0 / BASE_FPS
    while True:
        start = time.time()
        try:
            frame, lm, pos, fps = _read_trainer_frame()
        except Exception:
            break
        if frame is None:
            time.sleep(0.05)
            continue

        display = draw_pose(frame.copy(), lm) if draw_overlay else frame
        ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

        if not _paused:
            delay = frame_time / max(abs(_speed), 0.25)
            elapsed = time.time() - start
            if elapsed < delay:
                time.sleep(delay - elapsed)


def _trainee_mjpeg():
    frame_time = 1.0 / BASE_FPS
    while True:
        start = time.time()
        try:
            frame, lm, pos, fps = _read_trainee_frame()
        except Exception:
            break
        if frame is None:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

        if not _paused:
            delay = frame_time / max(abs(_speed), 0.25)
            elapsed = time.time() - start
            if elapsed < delay:
                time.sleep(delay - elapsed)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/stream/trainer")
def stream_trainer():
    draw_overlay = request.args.get("overlay", "1") != "0"
    return Response(
        _trainer_mjpeg(draw_overlay),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/trainee")
def stream_trainee():
    if _trainee_source != "video":
        return jsonify({"error": "Trainee source is not video"}), 400
    return Response(
        _trainee_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    with _video_lock:
        pos = int(_cap.get(cv2.CAP_PROP_POS_FRAMES)) if _cap else 0
        fps = (_cap.get(cv2.CAP_PROP_FPS) if _cap else None) or BASE_FPS
        t_pos = int(_trainee_cap.get(cv2.CAP_PROP_POS_FRAMES)) if _trainee_cap else 0
        t_fps = (_trainee_cap.get(cv2.CAP_PROP_FPS) if _trainee_cap else None) or BASE_FPS
    return jsonify(
        {
            "paused": _paused,
            "speed": _speed,
            "tracking": session.tracking_active,
            "frame": pos,
            "total_frames": _total_frames,
            "time_sec": round(pos / fps, 2) if fps else 0,
            "duration_sec": round(_total_frames / fps, 2) if _total_frames and fps else 0,
            "trainer_video": session.trainer_video,
            "trainee_source": _trainee_source,
            "trainee_video": session.trainee_video,
            "trainee_frame": t_pos,
            "trainee_total_frames": _trainee_total_frames,
            "trainee_time_sec": round(t_pos / t_fps, 2) if t_fps else 0,
        }
    )


@app.route("/api/analysis")
def api_analysis():
    with _analysis_lock:
        if _latest_analysis is None:
            return jsonify({"ok": False, "tracking": session.tracking_active})
        return jsonify(copy.deepcopy(_latest_analysis))


@app.route("/api/control", methods=["POST"])
def api_control():
    global _paused, _speed, _seek_frame, _trainee_seek_frame, _replay_requested
    global _trainee_replay_requested, _trainee_source, _latest_analysis

    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")

    with _video_lock:
        if action == "play":
            _paused = False
        elif action == "pause":
            _paused = True
        elif action == "stop":
            _paused = True
            _speed = 1.0
            _seek_frame = 0
            _trainee_seek_frame = 0
        elif action == "replay":
            _paused = False
            _replay_requested = True
            _trainee_replay_requested = True
        elif action == "speed":
            _speed = float(data.get("speed", _speed))
            _speed = max(-2.0, min(2.0, _speed))
        elif action == "seek":
            _seek_frame = int(data.get("frame", 0))
            _trainee_seek_frame = int(data.get("frame", 0))
        elif action == "tracking":
            session.tracking_active = bool(data.get("active", False))
            if not session.tracking_active:
                session.reset_tracking()
                with _analysis_lock:
                    _latest_analysis = None
        elif action == "trainee_source":
            src = data.get("source", "camera")
            if src not in ("camera", "video"):
                return jsonify({"ok": False, "error": "source must be camera or video"}), 400
            _trainee_source = src
            if src == "video" and _trainee_cap is None:
                try:
                    _open_trainee_cap()
                except RuntimeError as e:
                    return jsonify({"ok": False, "error": str(e)}), 400
        else:
            return jsonify({"ok": False, "error": f"Unknown action: {action}"}), 400

    return jsonify(
        {
            "ok": True,
            "paused": _paused,
            "speed": _speed,
            "tracking": session.tracking_active,
            "trainee_source": _trainee_source,
        }
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze a trainee webcam frame (base64 JPEG or raw bytes)."""
    if _trainee_source == "video":
        return jsonify(
            {
                "ok": False,
                "error": "Trainee source is test video; use GET /api/analysis",
            }
        ), 400

    frame = None

    if request.content_type and "application/json" in request.content_type:
        payload = request.get_json(force=True) or {}
        img_b64 = payload.get("image", "")
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[1]
        raw = base64.b64decode(img_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif request.files.get("frame"):
        raw = request.files["frame"].read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        raw = request.get_data()
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    lm = get_detector().process_bgr(frame)
    result = session.analyze_trainee(lm)
    with _analysis_lock:
        global _latest_analysis
        _latest_analysis = result
    return jsonify(result)


@app.route("/api/summary")
def api_summary():
    return jsonify(session.get_summary())


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    global _cap, _trainee_cap
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        video = data.get("trainer_video")
        if video:
            session.trainer_video = video
            with _video_lock:
                if _cap:
                    _cap.release()
                    _cap = None
        t_video = data.get("trainee_video")
        if t_video:
            session.trainee_video = t_video
            with _video_lock:
                if _trainee_cap:
                    _trainee_cap.release()
                    _trainee_cap = None
        return jsonify(
            {
                "ok": True,
                "trainer_video": session.trainer_video,
                "trainee_video": session.trainee_video,
            }
        )
    return jsonify(
        {
            "trainer_video": session.trainer_video,
            "trainee_video": session.trainee_video,
        }
    )


def main():
    import webbrowser

    port = 8080
    url = f"http://localhost:{port}"
    print(f"GymForm API + UI at {url}")
    print(f"Trainer video: {session.trainer_video}")
    print(f"Trainee test video: {session.trainee_video}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
