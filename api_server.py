#!/usr/bin/env python3
"""Flask API connecting the GymForm web UI to MediaPipe pose analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
import base64
import copy
import json
import shutil
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
    resolve_video_source,
)

UI_DIR = Path(__file__).resolve().parent / "ui"
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
PROGRESS_PATH = RECORDINGS_DIR / "progress.json"
BASE_FPS = 24
SERVER_BUILD = "trainer-frame-v3"

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
_trainer_stream_error: str | None = None

_recording_active = False
_recording_writer: cv2.VideoWriter | None = None
_recording_path: Path | None = None
_recording_frames = 0
_last_recording_meta: dict = {}


def _load_progress_sessions() -> list[dict]:
    if not PROGRESS_PATH.exists():
        return []
    try:
        data = json.loads(PROGRESS_PATH.read_text())
        sessions = data.get("sessions", [])
        return sessions if isinstance(sessions, list) else []
    except json.JSONDecodeError:
        return []


def _save_progress_sessions(sessions: list[dict]) -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps({"sessions": sessions}, indent=2))


def _append_progress_session(entry: dict) -> dict:
    sessions = _load_progress_sessions()
    sessions.append(entry)
    # Keep a rolling window so the file stays manageable.
    if len(sessions) > 500:
        sessions = sessions[-500:]
    _save_progress_sessions(sessions)
    return entry


def _session_datetime(session: dict) -> datetime:
    raw = session.get("completed_at") or session.get("recorded_at")
    if not raw:
        return datetime.now()
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return datetime.now()


def _compute_streak(sessions: list[dict]) -> int:
    if not sessions:
        return 0
    days = sorted({_session_datetime(s).date() for s in sessions}, reverse=True)
    today = datetime.now().date()
    if days[0] < today - timedelta(days=1):
        return 0
    streak = 0
    expected = days[0]
    for day in days:
        if day == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif day < expected:
            break
    return streak


def _aggregate_progress(period: str) -> dict:
    sessions = _load_progress_sessions()
    now = datetime.now()
    bars = [0.0] * 7
    counts = [0] * 7
    labels = ["M", "T", "W", "T", "F", "S", "S"]
    period_sessions: list[dict] = []

    if period == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        labels = ["12a", "3a", "6a", "9a", "12p", "3p", "6p"]
        for entry in sessions:
            t = _session_datetime(entry)
            if t < start:
                continue
            period_sessions.append(entry)
            bucket = min(t.hour // 3, 6)
            bars[bucket] += float(entry.get("score", 0))
            counts[bucket] += 1
    elif period == "month":
        start = (now - timedelta(days=6)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        labels = [(start + timedelta(days=i)).strftime("%d") for i in range(7)]
        for entry in sessions:
            t = _session_datetime(entry)
            if t < start:
                continue
            period_sessions.append(entry)
            day_index = (t.date() - start.date()).days
            if 0 <= day_index < 7:
                bars[day_index] += float(entry.get("score", 0))
                counts[day_index] += 1
    else:
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        labels = ["M", "T", "W", "T", "F", "S", "S"]
        for entry in sessions:
            t = _session_datetime(entry)
            if t < start:
                continue
            period_sessions.append(entry)
            wd = t.weekday()
            bars[wd] += float(entry.get("score", 0))
            counts[wd] += 1

    chart_bars = [round(bars[i] / counts[i]) if counts[i] else 0 for i in range(7)]
    avg_score = (
        round(sum(float(s.get("score", 0)) for s in period_sessions) / len(period_sessions))
        if period_sessions
        else 0
    )

    return {
        "ok": True,
        "period": period,
        "bars": chart_bars,
        "labels": labels,
        "sessions": len(period_sessions),
        "streak": _compute_streak(sessions),
        "avg": f"{avg_score}%",
    }


def _placeholder_frame(message: str) -> np.ndarray:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (40, 40, 48)
    lines = message.replace("\r", "").split("\n")[:6]
    y = 280
    for line in lines:
        cv2.putText(
            img,
            line[:70],
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (220, 220, 230),
            2,
            cv2.LINE_AA,
        )
        y += 44
    return img


def _open_cap():
    global _cap, _total_frames, _current_frame, _current_lm, _trainer_stream_error
    path = resolve_video_source(session.trainer_video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open trainer video: {session.trainer_video}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _cap = cap
    _total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    _trainer_stream_error = None
    ret, frame = cap.read()
    if ret and frame is not None:
        _current_frame = frame.copy()
        _current_lm = get_detector().process_bgr(frame)
        session.ingest_trainer(_current_lm)
    return cap


def _open_trainee_cap():
    global _trainee_cap, _trainee_total_frames, _trainee_current_frame
    path = resolve_video_source(session.trainee_video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open trainee video: {session.trainee_video}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _trainee_cap = cap
    _trainee_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    _trainee_current_frame = None
    return cap


def _reload_trainee_cap():
    global _trainee_cap, _trainee_total_frames, _trainee_current_frame
    if _trainee_cap:
        _trainee_cap.release()
        _trainee_cap = None
    _trainee_total_frames = 0
    _trainee_current_frame = None
    if _trainee_source == "video":
        _open_trainee_cap()


def _prepare_recording_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if (w, h) != (1280, 720):
        return cv2.resize(frame, (1280, 720))
    return frame


def _start_recording():
    global _recording_active, _recording_writer, _recording_path, _recording_frames
    if _recording_active:
        _stop_recording()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    _recording_path = RECORDINGS_DIR / f"session_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(_recording_path), fourcc, BASE_FPS, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("Could not start MP4 recording")
    _recording_writer = writer
    _recording_active = True
    _recording_frames = 0


def _write_recording_frame(frame: np.ndarray):
    global _recording_frames
    if not _recording_active or _recording_writer is None or frame is None:
        return
    _recording_writer.write(_prepare_recording_frame(frame))
    _recording_frames += 1


def _stop_recording(extra_meta: dict | None = None) -> dict:
    global _recording_active, _recording_writer, _recording_path, _last_recording_meta
    global _recording_frames

    _recording_active = False
    if _recording_writer is not None:
        _recording_writer.release()
        _recording_writer = None

    if _recording_frames < 1 or _recording_path is None or not _recording_path.exists():
        if _recording_path and _recording_path.exists():
            _recording_path.unlink(missing_ok=True)
        _recording_path = None
        _recording_frames = 0
        return {"ok": False, "error": "No frames recorded"}

    last_mp4 = RECORDINGS_DIR / "last_session.mp4"
    shutil.copy2(_recording_path, last_mp4)
    meta_path = RECORDINGS_DIR / "last_session.json"
    _last_recording_meta = {
        "ok": True,
        "filename": last_mp4.name,
        "path": str(last_mp4.resolve()),
        "session_file": _recording_path.name,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames": _frame_count(last_mp4),
        "trainer_video": session.trainer_video,
        "speed": _speed,
    }
    if extra_meta:
        for key, value in extra_meta.items():
            if value is not None:
                _last_recording_meta[key] = value
    meta_path.write_text(json.dumps(_last_recording_meta, indent=2))

    session.trainee_video = str(last_mp4.resolve())
    _reload_trainee_cap()

    info = dict(_last_recording_meta)
    _recording_path = None
    _recording_frames = 0
    return info


def _frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return count


def _load_last_recording_meta() -> dict:
    global _last_recording_meta
    meta_path = RECORDINGS_DIR / "last_session.json"
    last_mp4 = RECORDINGS_DIR / "last_session.mp4"
    if meta_path.exists():
        try:
            _last_recording_meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            _last_recording_meta = {}
    elif last_mp4.exists():
        _last_recording_meta = {
            "ok": True,
            "filename": last_mp4.name,
            "path": str(last_mp4.resolve()),
            "recorded_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(last_mp4.stat().st_mtime)
            ),
        }
    if _last_recording_meta.get("ok") and last_mp4.exists():
        _last_recording_meta["download_url"] = "/api/recordings/last/download"
    return _last_recording_meta


def _analyze_trainee_frame(frame: np.ndarray | None, lm):
    global _latest_analysis
    if lm is None or not session.tracking_active:
        return
    result = session.analyze_trainee(lm)
    with _analysis_lock:
        _latest_analysis = result


def _advance_trainee_frame():
    """Keep trainee file in sync with trainer during recorded replay analysis."""
    global _trainee_current_frame, _latest_analysis
    global _trainee_replay_requested, _trainee_seek_frame, _seek_frame

    if _trainee_source != "video" or _trainee_cap is None:
        return

    if _trainee_replay_requested:
        _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _trainee_replay_requested = False

    if _trainee_seek_frame is not None:
        _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, _trainee_seek_frame)
        _trainee_seek_frame = None
    elif _seek_frame is not None:
        _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, _seek_frame)

    ret, frame = _trainee_cap.read()
    if not ret or frame is None:
        _trainee_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = _trainee_cap.read()
    if frame is None:
        return

    lm = get_detector().process_bgr(frame)
    _trainee_current_frame = frame.copy()
    _analyze_trainee_frame(frame, lm)


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
            if _current_frame is None and _cap is not None:
                ret, frame = _cap.read()
                if ret and frame is not None:
                    _current_frame = frame.copy()
                    _current_lm = get_detector().process_bgr(frame)
                    session.ingest_trainer(_current_lm)
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
            if _trainee_source == "video" and session.tracking_active:
                _advance_trainee_frame()

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

        # During replay compare, trainer playback advances the trainee cap.
        if (
            session.tracking_active
            and _trainee_source == "video"
            and _trainee_current_frame is not None
        ):
            pos = int(_trainee_cap.get(cv2.CAP_PROP_POS_FRAMES)) if _trainee_cap else 0
            fps = (_trainee_cap.get(cv2.CAP_PROP_FPS) if _trainee_cap else None) or BASE_FPS
            return _trainee_current_frame.copy(), None, pos, fps

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
            _analyze_trainee_frame(frame, lm)

        pos = int(_trainee_cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = _trainee_cap.get(cv2.CAP_PROP_FPS) or BASE_FPS
        return frame, lm, pos, fps


def _trainer_mjpeg(draw_overlay: bool = True):
    global _trainer_stream_error
    frame_time = 1.0 / BASE_FPS
    while True:
        start = time.time()
        frame = None
        lm = None
        try:
            frame, lm, pos, fps = _read_trainer_frame()
        except Exception as exc:
            _trainer_stream_error = str(exc)
            frame = _placeholder_frame(f"Trainer video error:\n{_trainer_stream_error}")
            lm = None
        if frame is None:
            msg = _trainer_stream_error or "Waiting for trainer video…"
            frame = _placeholder_frame(msg)
        display = draw_pose(frame.copy(), lm) if draw_overlay and lm else frame
        ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(0.05)
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
    draw_overlay = request.args.get("overlay", "0") == "1"
    return Response(
        _trainer_mjpeg(draw_overlay),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/health")
def api_health():
    return jsonify(
        {
            "ok": True,
            "build": SERVER_BUILD,
            "trainer_frame": True,
        }
    )


@app.route("/api/trainer/frame")
@app.route("/api/trainer/frame.jpg")
def trainer_frame_jpg():
    """Single JPEG frame — works in Safari (multipart MJPEG in <img> does not)."""
    global _trainer_stream_error
    draw_overlay = request.args.get("overlay", "0") == "1"
    frame = None
    lm = None
    try:
        frame, lm, _pos, _fps = _read_trainer_frame()
    except Exception as exc:
        _trainer_stream_error = str(exc)
        frame = _placeholder_frame(f"Trainer video error:\n{_trainer_stream_error}")
    if frame is None:
        frame = _placeholder_frame(_trainer_stream_error or "Waiting for trainer video…")
    display = draw_pose(frame.copy(), lm) if draw_overlay and lm else frame
    ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return Response(status=500)
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


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
    recording_info = None

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
        elif action == "recording":
            active = bool(data.get("active", False))
            if active:
                try:
                    _start_recording()
                except RuntimeError as exc:
                    return jsonify({"ok": False, "error": str(exc)}), 500
            else:
                extra_meta = data.get("meta")
                if extra_meta is not None and not isinstance(extra_meta, dict):
                    return jsonify({"ok": False, "error": "meta must be an object"}), 400
                recording_info = _stop_recording(extra_meta)
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

    payload = {
        "ok": True,
        "paused": _paused,
        "speed": _speed,
        "tracking": session.tracking_active,
        "trainee_source": _trainee_source,
        "recording": _recording_active,
    }
    if recording_info is not None:
        payload["recording_info"] = recording_info
    return jsonify(payload)


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

    lm = get_detector().process_bgr_snapshot(frame)
    if _recording_active:
        _write_recording_frame(frame)
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
    global _cap, _trainee_cap, _total_frames
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        video = data.get("trainer_video")
        if video:
            video = video.strip()
            try:
                path = resolve_video_source(video)
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            test = cv2.VideoCapture(path)
            if not test.isOpened():
                test.release()
                return jsonify(
                    {"ok": False, "error": f"Cannot open trainer video: {video}"}
                ), 400
            test.release()

            session.trainer_video = video
            session.trainer_hist.clear()
            session.trainer_lm = None
            global _current_frame, _current_lm
            _current_frame = None
            _current_lm = None
            with _video_lock:
                if _cap:
                    _cap.release()
                    _cap = None
                _total_frames = 0
                try:
                    _open_cap()
                except Exception as exc:
                    return jsonify({"ok": False, "error": str(exc)}), 400
        t_video = data.get("trainee_video")
        if t_video:
            session.trainee_video = t_video.strip()
            with _video_lock:
                _reload_trainee_cap()
        return jsonify(
            {
                "ok": True,
                "trainer_video": session.trainer_video,
                "trainee_video": session.trainee_video,
            }
        )
    return jsonify(
        {
            "ok": True,
            "trainer_video": session.trainer_video,
            "trainee_video": session.trainee_video,
        }
    )


@app.route("/api/progress", methods=["GET", "POST"])
def api_progress():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        try:
            score = int(data.get("score", 0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "score must be a number"}), 400
        score = max(0, min(100, score))
        duration = data.get("duration_sec")
        try:
            duration_sec = int(duration) if duration is not None else 0
        except (TypeError, ValueError):
            duration_sec = 0

        entry = {
            "completed_at": data.get("completed_at")
            or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exercise": (data.get("exercise") or "Exercise").strip()[:80],
            "score": score,
            "duration_sec": max(0, duration_sec),
            "recording_path": data.get("recording_path"),
        }
        _append_progress_session(entry)
        return jsonify({"ok": True, "session": entry})

    period = request.args.get("period", "week")
    if period not in ("day", "week", "month"):
        return jsonify({"ok": False, "error": "period must be day, week, or month"}), 400
    return jsonify(_aggregate_progress(period))


@app.route("/api/recordings/last")
def api_recording_last():
    meta = _load_last_recording_meta()
    if not meta.get("ok"):
        return jsonify({"ok": False, "error": "No recording saved yet"}), 404
    return jsonify(meta)


@app.route("/api/recordings/last/download")
def api_recording_download():
    last_mp4 = RECORDINGS_DIR / "last_session.mp4"
    if not last_mp4.exists():
        return jsonify({"ok": False, "error": "No recording saved yet"}), 404
    return send_from_directory(
        RECORDINGS_DIR,
        last_mp4.name,
        as_attachment=True,
        download_name=f"gymform_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
    )


def main():
    import webbrowser

    def _warm_trainer():
        try:
            with _video_lock:
                _open_cap()
            print("Trainer video ready.")
        except Exception as exc:
            print(f"Trainer video warm-up failed: {exc}")

    threading.Thread(target=_warm_trainer, daemon=True).start()

    port = 8080
    url = f"http://localhost:{port}"
    print(f"GymForm API + UI at {url}")
    print(f"Server build: {SERVER_BUILD}  (health: {url}/api/health)")
    print(f"Trainer frame: {url}/api/trainer/frame")
    print(f"Trainer video: {session.trainer_video}")
    print(f"Trainee test video: {session.trainee_video}")
    print(f"Recordings folder: {RECORDINGS_DIR.resolve()}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
