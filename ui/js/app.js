/**
 * GymForm UI — connected to Flask + MediaPipe backend
 */
(function () {
  "use strict";

  const state = {
    cameraActive: false,
    isPlaying: false,
    speed: 1.0,
    isMuted: false,
    speakerOn: true,
    mirrorOn: false,
    showCorrections: false,
    mirrorExerciseActive: false,
    exerciseIndex: 0,
    progressPeriod: "week",
    cameraStream: null,
    backendConnected: false,
    traineeSource: "camera", // "camera" | "video"
    inputActive: false,
    smoothLandmarks: null,
    viewMode: "trainer", // "trainer" | "trainee" (trainee = camera adjust only)
    isRecording: false,
    lastRecordingPath: null,
    lastRecordingMeta: null,
    replayCompareActive: false,
    activeSession: null,
  };

  const EMPTY_PROGRESS = {
    bars: [0, 0, 0, 0, 0, 0, 0],
    labels: ["M", "T", "W", "T", "F", "S", "S"],
    sessions: 0,
    streak: 0,
    avg: "0%",
  };

  const EXERCISES = [
    { name: "Push-ups", sets: "Set 1/3", instruction: "Keep your core tight. Lower until chest nearly touches the floor." },
    { name: "Squats", sets: "Set 1/3", instruction: "Feet shoulder-width apart. Push hips back and keep knees aligned." },
    { name: "Lunges", sets: "Set 1/3", instruction: "Step forward, lower back knee toward floor. Keep torso upright." },
    { name: "Plank", sets: "Set 1/3", instruction: "Hold a straight line from head to heels. Don't let hips sag." },
  ];

  const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
    [24, 26], [26, 28], [15, 17], [15, 19], [15, 21],
    [16, 18], [16, 20], [16, 22],
  ];

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const JOINT_VERTICES = {
    "Left Knee": [23, 25, 27],
    "Right Knee": [24, 26, 28],
    "Left Hip": [11, 23, 25],
    "Right Hip": [12, 24, 26],
    "Left Elbow": [11, 13, 15],
    "Right Elbow": [12, 14, 16],
  };

  const els = {
    appShell: document.querySelector(".app-shell"),
    cameraPlaceholder: $("#cameraPlaceholder"),
    btnFlipLabel: $("#btnFlipLabel"),
    trainerCanvas: $("#trainerCanvas"),
    trainerPane: $("#trainerPane"),
    stageTrainer: $("#stageTrainer"),
    stageCamera: $("#stageCamera"),
    videoStage: $("#videoStage"),
    formOverlay: $("#formOverlay"),
    formOverlayLabel: $("#formOverlayLabel"),
    replayCornerStream: $("#replayCornerStream"),
    skeletonHint: $("#skeletonHint"),
    instructionPanel: $("#instructionPanel"),
    instructionBar: $("#instructionBar"),
    instructionText: $("#instructionText"),
    controlBar: $("#controlBar"),
    trainerControls: $("#trainerControls"),
    traineeControls: $("#traineeControls"),
    btnFlip: $("#btnFlip"),
    btnPlayPause: $("#btnPlayPause"),
    btnRewind: $("#btnRewind"),
    btnFastForward: $("#btnFastForward"),
    btnStop: $("#btnStop"),
    btnReplay: $("#btnReplay"),
    btnPrev: $("#btnPrev"),
    btnNext: $("#btnNext"),
    btnMirror: $("#btnMirror"),
    btnReplayLast: $("#btnReplayLast"),
    recBadge: $("#recBadge"),
    replayOverlay: $("#replayOverlay"),
    replayTitle: $("#replayTitle"),
    replayMeta: $("#replayMeta"),
    btnCloseReplay: $("#btnCloseReplay"),
    btnCompareRecording: $("#btnCompareRecording"),
    btnDownloadRecording: $("#btnDownloadRecording"),
    btnShowCorrections: $("#btnShowCorrections"),
    btnMirrorExercise: $("#btnMirrorExercise"),
    btnStopExercise: $("#btnStopExercise"),
    btnStopExerciseFloat: $("#btnStopExerciseFloat"),
    btnSpeaker: $("#btnSpeaker"),
    btnMute: $("#btnMute"),
    volumeSlider: $("#volumeSlider"),
    speedBadge: $("#speedBadge"),
    speedSelect: $("#speedSelect"),
    exerciseTag: $("#exerciseTag"),
    correctionToast: $("#correctionToast"),
    toastText: $("#toastText"),
    skeletonCanvas: $("#skeletonCanvas"),
    suggestionsList: $("#suggestionsList"),
    suggestionsStatus: $("#suggestionsStatus"),
    suggestionsEmpty: $("#suggestionsEmpty"),
    suggestionsPanel: null,
    traineeCamera: $("#traineeCamera"),
    traineeStream: $("#traineeStream"),
    btnSourceCamera: $("#btnSourceCamera"),
    btnSourceVideo: $("#btnSourceVideo"),
    trainerVideoUrl: $("#trainerVideoUrl"),
    btnLoadTrainer: $("#btnLoadTrainer"),
    trainerVideoHint: $("#trainerVideoHint"),
    scoreRing: $("#scoreRing"),
    scoreNum: $("#scoreNum"),
    progressChart: $("#progressChart"),
    sessionsCount: $("#sessionsCount"),
    streakCount: $("#streakCount"),
    avgScore: $("#avgScore"),
    formScore: $("#formScore"),
  };

  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");

  const notifier = new PoseNotifier({
    consecutiveThreshold: 15,
    debounceMs: 2000,
    onNotify(issue) {
      showCorrection(issue.message);
      pushSuggestionItem(issue.joint, issue.message);
    },
    onClear() {
      hideCorrection();
    },
  });

  let analyzeInterval = null;
  let statusInterval = null;
  let trainerFrameTimer = null;
  let trainerFrameBusy = false;
  let analyzing = false;
  let suggestionSnapshot = [];
  const SKIP_SECONDS = 5;

  function resetActiveSession() {
    state.activeSession = null;
  }

  function beginActiveSession() {
    state.activeSession = { startedAt: Date.now(), scores: [] };
  }

  function recordSessionScore(score) {
    if (!state.activeSession || score == null) return;
    state.activeSession.scores.push(score);
  }

  function buildSessionPayload() {
    const session = state.activeSession;
    if (!session?.startedAt) return null;

    const durationSec = Math.round((Date.now() - session.startedAt) / 1000);
    const scores = session.scores;
    if (durationSec < 5 && scores.length < 3) return null;

    const avgScore = scores.length
      ? Math.round(scores.reduce((sum, value) => sum + value, 0) / scores.length)
      : null;
    if (avgScore == null) return null;

    return {
      exercise: EXERCISES[state.exerciseIndex]?.name || "Exercise",
      score: avgScore,
      duration_sec: durationSec,
      completed_at: new Date().toISOString(),
      recording_path: state.lastRecordingPath || null,
    };
  }

  async function saveProgressFromSession(payload) {
    if (!payload) return;
    try {
      const res = await GymAPI.saveProgressSession(payload);
      if (!res.ok) {
        console.warn("progress save", res.error);
        return;
      }
      await refreshProgressChart(state.progressPeriod);
    } catch (err) {
      console.warn("progress save", err);
    }
  }

  async function refreshProgressChart(period = state.progressPeriod) {
    try {
      const data = await GymAPI.getProgress(period);
      if (data.ok) {
        updateProgressChart(period, data);
        return;
      }
    } catch {
      /* fall back to empty chart */
    }
    updateProgressChart(period, EMPTY_PROGRESS);
  }

  function formatRecordingTime(iso) {
    if (!iso) return "";
    try {
      return new Date(iso).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  }

  function updateReplayLastButton() {
    const hasRecording = Boolean(state.lastRecordingPath);
    if (els.btnReplayLast) {
      els.btnReplayLast.disabled = !hasRecording;
    }
    if (els.btnDownloadRecording) {
      els.btnDownloadRecording.href = hasRecording
        ? `${GymAPI.recordingDownloadUrl()}?t=${Date.now()}`
        : "#";
    }
  }

  function showRecordingIndicator(on) {
    els.recBadge?.classList.toggle("hidden", !on);
  }

  function applyLastRecording(meta) {
    if (!meta?.ok) return;
    state.lastRecordingPath = meta.path || null;
    state.lastRecordingMeta = meta;
    updateReplayLastButton();
  }

  async function loadLastRecording() {
    try {
      const meta = await GymAPI.getLastRecording();
      if (meta.ok) applyLastRecording(meta);
    } catch {
      /* no recording yet */
    }
  }

  function triggerRecordingDownload() {
    if (!state.lastRecordingPath) return;
    const a = document.createElement("a");
    a.href = `${GymAPI.recordingDownloadUrl()}?t=${Date.now()}`;
    a.download = `gymform_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.mp4`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  async function startExerciseRecording() {
    if (state.traineeSource !== "camera" || !state.cameraActive) return false;
    try {
      await GymAPI.control("recording", { active: true });
      state.isRecording = true;
      showRecordingIndicator(true);
      return true;
    } catch {
      return false;
    }
  }

  async function stopExerciseRecording() {
    if (!state.isRecording) return null;
    showRecordingIndicator(false);
    state.isRecording = false;
    try {
      const res = await GymAPI.control("recording", {
        active: false,
        meta: { exercise: EXERCISES[state.exerciseIndex]?.name },
      });
      const info = res.recording_info;
      if (info?.ok) {
        applyLastRecording(info);
        triggerRecordingDownload();
        setInstruction(
          "Saved as MP4 (recordings/ folder + downloaded). Tap Replay last to compare with trainer.",
          "success"
        );
      } else {
        setInstruction(info?.error || "Recording was empty — try a longer session.", "correction");
      }
      return info;
    } catch {
      setInstruction("Could not save recording — check server is running.", "correction");
      return null;
    }
  }

  function updateReplayCompareUI() {
    const active = state.replayCompareActive;
    els.trainerPane?.classList.toggle("replay-compare", active);
    if (els.formOverlayLabel) {
      els.formOverlayLabel.textContent = active ? "Your recording" : "Your form";
    }
    if (els.replayCornerStream) {
      if (active) {
        els.replayCornerStream.src = `/stream/trainee?t=${Date.now()}`;
        els.replayCornerStream.classList.remove("hidden");
      } else {
        els.replayCornerStream.src = "";
        els.replayCornerStream.classList.add("hidden");
      }
    }
    if (els.traineeStream) {
      if (active) {
        els.traineeStream.src = `/stream/trainee?t=${Date.now()}`;
      } else if (state.traineeSource !== "video") {
        els.traineeStream.src = "";
      }
    }
  }

  async function startReplayCompare() {
    const meta = state.lastRecordingMeta || (await GymAPI.getLastRecording());
    if (!meta?.ok || !meta.path) {
      setInstruction("No recording yet. Complete a Mirror & Do session first.", "correction");
      return;
    }
    applyLastRecording(meta);
    closeReplayOverlay();

    const configPayload = { trainee_video: meta.path };
    if (meta.trainer_video) configPayload.trainer_video = meta.trainer_video;
    try {
      const cfg = await GymAPI.setConfig(configPayload);
      if (!cfg.ok) {
        setInstruction(cfg.error || "Could not load videos for replay.", "correction");
        return;
      }
      if (meta.trainer_video && els.trainerVideoUrl) {
        els.trainerVideoUrl.value = meta.trainer_video;
        els.trainerVideoHint.textContent = `Loaded: ${shortVideoLabel(meta.trainer_video)}`;
      }
    } catch {
      setInstruction("Could not load videos for replay — check the server.", "correction");
      return;
    }

    state.traineeSource = "video";
    state.replayCompareActive = true;
    state.mirrorExerciseActive = true;
    state.showCorrections = true;
    els.btnShowCorrections?.classList.add("active");
    updateMirrorExerciseButton();
    updateSourceButtons();
    updateReplayCompareUI();

    await sendControl("trainee_source", { source: "video" });
    await sendControl("tracking", { active: true });
    if (meta.speed != null) {
      state.speed = Number(meta.speed) || 1;
      await sendControl("speed", { speed: state.speed });
      updateSpeedBadge();
      updateSpeedSelect();
    }
    await sendControl("replay");
    await sendControl("play");
    reloadTrainerStream();

    if (state.viewMode !== "trainer") {
      state.viewMode = "trainer";
      updateViewMode();
    }
    state.inputActive = true;
    startAnalyzeLoop();
    setInstruction(
      "Replaying your recording with the trainer — both videos play in sync. Red joints show mistakes.",
      "success"
    );
  }

  async function restoreLiveCameraInput() {
    if (els.traineeStream) {
      els.traineeStream.src = "";
      els.traineeStream.classList.add("hidden");
    }
    state.traineeSource = "camera";
    updateSourceButtons();
    await sendControl("trainee_source", { source: "camera" });

    if (state.cameraStream && els.traineeCamera.srcObject) {
      hidePlaceholder();
      try {
        await els.traineeCamera.play();
      } catch {
        /* ignore */
      }
      state.inputActive = true;
      state.cameraActive = true;
      return true;
    }

    const ok = await startCamera();
    state.inputActive = ok;
    state.cameraActive = ok;
    return ok;
  }

  async function stopReplayCompare() {
    if (!state.replayCompareActive) return;
    state.replayCompareActive = false;
    updateReplayCompareUI();
    await sendControl("pause");
    await restoreLiveCameraInput();
  }

  function openReplayOverlay() {
    if (!state.lastRecordingPath) {
      setInstruction("No recording yet. Complete a Mirror & Do session first.", "correction");
      return;
    }
    const meta = state.lastRecordingMeta;
    if (els.replayTitle) {
      els.replayTitle.textContent = meta?.exercise
        ? `Your ${meta.exercise}`
        : "Your last exercise";
    }
    if (els.replayMeta) {
      const when = formatRecordingTime(meta?.recorded_at || meta?.recordedAt);
      const loc = meta?.path ? `Saved: ${meta.path}` : "";
      els.replayMeta.textContent = [when && `Recorded ${when}`, loc].filter(Boolean).join(" · ");
    }
    els.replayOverlay?.classList.remove("hidden");
  }

  function closeReplayOverlay() {
    els.replayOverlay?.classList.add("hidden");
  }


  async function skipRelative(seconds) {
    try {
      const s = await GymAPI.status();
      const fps =
        s.total_frames && s.duration_sec ? s.total_frames / s.duration_sec : 24;
      const current = s.frame || 0;
      const total = s.total_frames || 0;
      const delta = Math.round(seconds * fps);
      const frame = Math.max(0, Math.min(Math.max(total - 1, 0), current + delta));
      await sendControl("seek", { frame });
      reloadTrainerStream();
    } catch {
      setInstruction("Could not skip — check that the server is running.", "correction");
    }
  }

  async function syncStatus() {
    try {
      const s = await GymAPI.status();
      state.backendConnected = true;
      state.isPlaying = !s.paused;
      state.speed = s.speed;
      updatePlayButton();
      updateSpeedBadge();
      updateSpeedSelect();
      scheduleTrainerFrame();
      if (s.trainee_source && s.trainee_source !== state.traineeSource) {
        state.traineeSource = s.trainee_source;
        updateSourceButtons();
      }
      // Only mirror server tracking when the server stops — never re-enable
      // exercise mode from tracking alone (corrections/replay also use tracking).
      if (!s.tracking && state.mirrorExerciseActive) {
        state.mirrorExerciseActive = false;
        updateMirrorExerciseButton();
      }
    } catch {
      state.backendConnected = false;
    }
  }

  async function sendControl(action, extra = {}) {
    try {
      await GymAPI.control(action, extra);
      await syncStatus();
    } catch {
      setInstruction("Cannot reach pose server. Start api_server.py", "correction");
    }
  }

  // ── Trainee frame analysis ──────────────────────────────────────
  function startAnalyzeLoop() {
    stopAnalyzeLoop();
    analyzeInterval = setInterval(analyzeCurrentFrame, 1000 / 12);
  }

  function stopAnalyzeLoop() {
    if (analyzeInterval) {
      clearInterval(analyzeInterval);
      analyzeInterval = null;
    }
    analyzing = false;
    notifier.reset();
    resetSmoothLandmarks();
  }

  function applyAnalysisResult(result) {
    if (!result || !result.ok) {
      if (state.viewMode === "trainer") {
        clearSkeletonOverlay();
        updateSkeletonHint("Step into frame so your skeleton appears here.");
      }
      if (state.viewMode === "trainee" && state.cameraActive) {
        const msg =
          result?.message ||
          result?.error ||
          "No pose detected — step back so your full body is visible.";
        setInstruction(msg, "correction");
      }
      return;
    }

    if (result.waiting_for_trainer && result.message) {
      setInstruction(result.message, "correction");
      if (state.viewMode === "trainer") {
        drawLandmarks(result.landmarks, []);
        updateSkeletonHint(null);
      }
      updateSuggestionsPanel(result);
      return;
    }

    if (result.score != null) {
      updateScore(result.score);
      if (state.mirrorExerciseActive && !state.replayCompareActive) {
        recordSessionScore(result.score);
      }
    }
    if (els.formScore && result.score != null) els.formScore.textContent = `${result.score}%`;

    if (result.frame_results && result.frame_results.length) {
      notifier.processFrame(result.frame_results);
    } else if (result.good_form) {
      notifier.processFrame(
        Object.keys(JOINT_VERTICES).map((joint) => ({ joint, isCorrect: true }))
      );
    }

    if (result.notify && state.viewMode === "trainer") {
      showCorrection(result.notify.action);
      pushSuggestionItem(result.notify.joint, result.notify.action);
    }

    const feedbackForSkeleton =
      state.mirrorExerciseActive || state.showCorrections
        ? result.feedback?.length
          ? result.feedback
          : (result.live_suggestions || []).map((s) => ({
              joint: s.joint,
              action: s.action,
            }))
        : (result.live_suggestions || []).map((s) => ({
            joint: s.joint,
            action: s.action,
          }));

    if (state.viewMode === "trainer" && result.landmarks?.length) {
      drawLandmarks(result.landmarks, feedbackForSkeleton);
      updateSkeletonHint(null);
    }

    updateSuggestionsPanel(result);
  }

  function updateSkeletonHint(message) {
    const hint = els.skeletonHint;
    if (!hint) return;
    if (!message) {
      hint.classList.add("hidden");
      return;
    }
    hint.textContent = message;
    hint.classList.remove("hidden");
  }

  function collectSuggestionItems(result) {
    const merged = [];
    const seen = new Set();
    const add = (joint, action) => {
      if (!joint || !action) return;
      const key = `${joint}:${action}`;
      if (seen.has(key)) return;
      seen.add(key);
      merged.push({ joint, action });
    };

    for (const item of result?.feedback || []) add(item.joint, item.action);
    for (const item of result?.live_suggestions || []) add(item.joint, item.action);
    for (const h of result?.height_feedback || []) add(h.joint, h.message);
    for (const fr of result?.frame_results || []) {
      if (!fr.isCorrect) add(fr.joint, fr.message);
    }
    if (result?.notify) add(result.notify.joint, result.notify.action);

    return merged;
  }

  function pushSuggestionItem(joint, action) {
    if (!joint || !action) return;
    const merged = collectSuggestionItems({
      frame_results: [{ joint, isCorrect: false, message: action }],
    });
    for (const item of suggestionSnapshot) {
      const key = `${item.joint}:${item.action}`;
      if (!merged.some((m) => `${m.joint}:${m.action}` === key)) merged.push(item);
    }
    renderSuggestionList(merged);
  }

  function renderSuggestionList(items) {
    const list = els.suggestionsList;
    const empty = els.suggestionsEmpty;
    const status = els.suggestionsStatus;
    if (!list) return;

    suggestionSnapshot = items;

    if (!items.length) {
      list.innerHTML = "";
      empty?.classList.remove("hidden");
      if (empty) empty.textContent = "Looking good — no fixes needed right now.";
      if (status) status.textContent = "Aligned with the trainer on this frame.";
      return;
    }

    empty?.classList.add("hidden");
    if (status) status.textContent = "Live form tips (updates as you move).";
    list.innerHTML = items
      .map((item) => `<li><strong>${item.joint}</strong>${item.action}</li>`)
      .join("");
  }

  function updateSuggestionsPanel(result) {
    const status = els.suggestionsStatus;
    const trackingLive = Boolean(result?.tracking || state.mirrorExerciseActive);

    if (!trackingLive && !state.showCorrections) {
      if (status) {
        status.textContent = state.inputActive
          ? "Press Mirror & Do to compare your form to the trainer."
          : "Start your camera, then turn on Mirror & Do.";
      }
      suggestionSnapshot = [];
      renderSuggestionList([]);
      return;
    }

    if (result?.waiting_for_trainer) {
      if (status) status.textContent = result.message || "Waiting for trainer video…";
      suggestionSnapshot = [];
      renderSuggestionList([]);
      els.suggestionsEmpty?.classList.remove("hidden");
      if (els.suggestionsEmpty) {
        els.suggestionsEmpty.textContent = "Play the trainer video on the left.";
      }
      return;
    }

    const merged = collectSuggestionItems(result);
    renderSuggestionList(merged);
  }

  function clearSuggestionsPanel() {
    suggestionSnapshot = [];
    renderSuggestionList([]);
    if (els.suggestionsEmpty) {
      els.suggestionsEmpty.classList.remove("hidden");
      els.suggestionsEmpty.textContent = "Suggestions appear here during form comparison.";
    }
    if (els.suggestionsStatus) {
      els.suggestionsStatus.textContent = "Turn on Mirror & Do for live tips.";
    }
  }

  async function analyzeCurrentFrame() {
    if (analyzing) return;
    if (!state.inputActive && !state.replayCompareActive) return;

    analyzing = true;
    try {
      let result;
      if (state.traineeSource === "video" || state.replayCompareActive) {
        result = await GymAPI.getAnalysis();
      } else {
        const video = els.traineeCamera;
        if (!video?.srcObject && !video?.src) return;
        if (!video.videoWidth || video.readyState < 2) return;
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        captureCtx.save();
        if (state.mirrorOn) {
          captureCtx.translate(captureCanvas.width, 0);
          captureCtx.scale(-1, 1);
        }
        captureCtx.drawImage(video, 0, 0);
        captureCtx.restore();
        result = await GymAPI.analyzeFrame(captureCanvas);
      }
      applyAnalysisResult(result);
    } catch {
      if (state.inputActive) {
        setInstruction("Pose tracking needs the server — run ./run_gymform.sh", "correction");
      }
    } finally {
      analyzing = false;
    }
  }

  function clearSkeletonOverlay() {
    const canvas = els.skeletonCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function clearOverlay() {
    clearSkeletonOverlay();
    clearSuggestionsPanel();
  }

  const SMOOTH_ALPHA = 0.35;

  function containRect(srcW, srcH, dstW, dstH) {
    const srcAspect = srcW / srcH;
    const dstAspect = dstW / dstH;
    let drawW, drawH, ox, oy;
    if (srcAspect > dstAspect) {
      drawW = dstW;
      drawH = dstW / srcAspect;
      ox = 0;
      oy = (dstH - drawH) / 2;
    } else {
      drawH = dstH;
      drawW = dstH * srcAspect;
      ox = (dstW - drawW) / 2;
      oy = 0;
    }
    return { ox, oy, drawW, drawH };
  }

  function smoothLandmarks(incoming) {
    if (!incoming?.length) return null;
    if (!state.smoothLandmarks || state.smoothLandmarks.length !== incoming.length) {
      state.smoothLandmarks = incoming.map((p) => ({
        x: p.x,
        y: p.y,
        z: p.z,
        v: p.v ?? p.visibility,
      }));
      return state.smoothLandmarks;
    }
    const a = SMOOTH_ALPHA;
    for (let i = 0; i < incoming.length; i++) {
      const p = incoming[i];
      const s = state.smoothLandmarks[i];
      s.x = s.x * (1 - a) + p.x * a;
      s.y = s.y * (1 - a) + p.y * a;
      if (p.v !== undefined) s.v = p.v;
      else if (p.visibility !== undefined) s.v = p.visibility;
    }
    return state.smoothLandmarks;
  }

  function resetSmoothLandmarks() {
    state.smoothLandmarks = null;
  }

  function errorVertexSet(feedback) {
    const verts = new Set();
    for (const f of feedback || []) {
      const joint = f.joint || f;
      const indices = JOINT_VERTICES[joint];
      if (indices) indices.forEach((i) => verts.add(i));
    }
    return verts;
  }

  function syncCanvasPixels(canvas, pane) {
    const host = pane || canvas?.parentElement;
    if (!canvas || !host) return { w: 0, h: 0 };
    const w = Math.max(Math.floor(host.clientWidth), 1);
    const h = Math.max(Math.floor(host.clientHeight), 1);
    if (canvas._paintW !== w || canvas._paintH !== h) {
      canvas._paintW = w;
      canvas._paintH = h;
      canvas.width = w;
      canvas.height = h;
    }
    return { w, h };
  }

  function drawLandmarks(landmarks, feedback) {
    if (!landmarks || !landmarks.length || !els.skeletonCanvas) return;

    const smoothed = smoothLandmarks(landmarks);
    const canvas = els.skeletonCanvas;
    const { w, h } = syncCanvasPixels(canvas, els.formOverlay);
    if (w < 2 || h < 2) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!state.replayCompareActive) {
      ctx.fillStyle = "#0a0e14";
      ctx.fillRect(0, 0, w, h);
    }

    const pad = 8;
    const drawW = w - pad * 2;
    const drawH = h - pad * 2;
    const mirrorDraw = state.traineeSource === "camera" && state.mirrorOn;
    const badVerts = errorVertexSet(feedback);

    const pt = (i) => {
      let nx = smoothed[i].x;
      const ny = smoothed[i].y;
      if (mirrorDraw) nx = 1 - nx;
      return {
        x: pad + nx * drawW,
        y: pad + ny * drawH,
      };
    };

    const connBad = (a, b) => badVerts.has(a) || badVerts.has(b);

    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const [a, b] of POSE_CONNECTIONS) {
      if (!smoothed[a] || !smoothed[b]) continue;
      const p1 = pt(a);
      const p2 = pt(b);
      const bad = connBad(a, b);
      ctx.strokeStyle = bad ? "#ff4757" : "#00ffcc";
      ctx.shadowColor = bad ? "rgba(255,71,87,0.45)" : "rgba(0,255,204,0.3)";
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    for (let i = 0; i < smoothed.length; i++) {
      const vis = smoothed[i].v ?? landmarks[i]?.visibility;
      if (vis !== undefined && vis < 0.25) continue;
      const p = pt(i);
      ctx.fillStyle = badVerts.has(i) ? "#ff4757" : "#00ffcc";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── UI helpers ──────────────────────────────────────────────────
  function setInstruction(text, type = "normal") {
    els.instructionText.textContent = text;
    els.instructionBar.className = "instruction-bar";
    els.instructionPanel?.classList.remove("correction", "success");
    if (type === "correction") {
      els.instructionBar.classList.add("correction");
      els.instructionPanel?.classList.add("correction");
    }
    if (type === "success") {
      els.instructionBar.classList.add("success");
      els.instructionPanel?.classList.add("success");
    }
  }

  function showCorrection(message) {
    els.toastText.textContent = message;
    els.correctionToast.classList.remove("hidden");
  }

  function hideCorrection() {
    els.correctionToast.classList.add("hidden");
    if (state.mirrorExerciseActive) setInstruction("Good form! Keep going.", "success");
  }

  function updateExercise() {
    const ex = EXERCISES[state.exerciseIndex];
    els.exerciseTag.textContent = `${ex.name} · ${ex.sets}`;
    if (!state.isPlaying && !state.mirrorExerciseActive) setInstruction(ex.instruction);
  }

  function updateSpeedBadge() {
    const sign = state.speed > 0 && state.speed !== 1 ? "+" : "";
    const label = state.speed === 1 ? "1.0" : state.speed.toFixed(1);
    els.speedBadge.textContent = `${sign}${label}×`;
  }

  function updateSpeedSelect() {
    if (!els.speedSelect) return;
    const val = String(state.speed);
    const hasOption = Array.from(els.speedSelect.options).some((o) => o.value === val);
    els.speedSelect.value = hasOption ? val : "1";
  }

  function updatePlayButton() {
    const btn = els.btnPlayPause;
    if (!btn) return;
    if (state.isPlaying) {
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
      btn.setAttribute("aria-label", "Pause");
      btn.classList.add("playing");
    } else {
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>`;
      btn.setAttribute("aria-label", "Play");
      btn.classList.remove("playing");
    }
  }

  function updateScore(score) {
    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - score / 100);
    els.scoreRing.style.strokeDashoffset = offset;
    els.scoreNum.textContent = score;
    if (els.formScore) els.formScore.textContent = `${score}%`;
  }

  function updateProgressChart(period, data = EMPTY_PROGRESS) {
    const bars = data.bars || EMPTY_PROGRESS.bars;
    const labels = data.labels || EMPTY_PROGRESS.labels;
    const maxBar = Math.max(...bars, 1);
    els.progressChart.innerHTML = bars
      .map(
        (h, i) =>
          `<div class="bar${h === maxBar && h > 0 ? " active" : ""}" style="--h:${h}%"><span>${labels[i] || ""}</span></div>`
      )
      .join("");
    els.sessionsCount.textContent = data.sessions ?? 0;
    els.streakCount.textContent = data.streak ?? 0;
    els.avgScore.textContent = data.avg ?? "0%";
  }

  function updateFlipLabel() {
    els.btnFlipLabel.textContent =
      state.viewMode === "trainer" ? "Flip to you" : "Flip to trainer";
  }

  function updateViewMode() {
    const isCameraAdjust = state.viewMode === "trainee";
    els.stageTrainer?.classList.toggle("hidden", isCameraAdjust);
    els.stageCamera?.classList.toggle("camera-adjust-active", isCameraAdjust);
    els.stageCamera?.classList.toggle("capture-only", !isCameraAdjust);
    els.videoStage?.classList.toggle("camera-adjust-view", isCameraAdjust);
    els.videoStage?.classList.toggle("trainer-view", !isCameraAdjust);
    els.trainerControls?.classList.toggle("hidden", isCameraAdjust);
    els.traineeControls?.classList.toggle("hidden", !isCameraAdjust);
    els.controlBar?.classList.toggle("trainee-mode", isCameraAdjust);
    els.controlBar?.classList.toggle("trainer-mode", !isCameraAdjust);
    updateFlipLabel();

    if (isCameraAdjust) {
      clearSkeletonOverlay();
      resumeTraineePreview();
      setInstruction("Position yourself in frame, then flip back to the trainer.");
    } else {
      syncCanvasPixels(els.trainerCanvas, els.trainerPane);
      syncCanvasPixels(els.skeletonCanvas, els.formOverlay);
      if (!state.inputActive) {
        updateSkeletonHint("Flip to you to set up camera");
      }
    }

    if (state.inputActive) startAnalyzeLoop();
  }

  async function syncTrackingState() {
    const active = state.mirrorExerciseActive || state.showCorrections;
    await sendControl("tracking", { active });
  }

  function updateMirrorExerciseButton() {
    const active = state.mirrorExerciseActive;
    els.trainerPane?.classList.toggle("exercise-active", active);
    if (els.btnMirrorExercise) {
      els.btnMirrorExercise.classList.toggle("active", active);
      els.btnMirrorExercise.disabled = active;
    }
    if (els.btnStopExercise) {
      els.btnStopExercise.disabled = !active;
      els.btnStopExercise.classList.toggle("hidden", !active);
    }
    if (els.btnStopExerciseFloat) {
      els.btnStopExerciseFloat.disabled = !active;
      els.btnStopExerciseFloat.classList.toggle("hidden", !active);
    }
  }

  async function stopMirrorExercise() {
    if (!state.mirrorExerciseActive && !state.isRecording && !state.replayCompareActive) return;

    const wasReplay = state.replayCompareActive;
    state.mirrorExerciseActive = false;
    if (state.isRecording) await stopExerciseRecording();
    if (state.replayCompareActive) await stopReplayCompare();
    const sessionPayload = wasReplay ? null : buildSessionPayload();

    if (wasReplay) {
      state.showCorrections = false;
      els.btnShowCorrections?.classList.remove("active");
    }

    if (state.traineeSource !== "camera") {
      await restoreLiveCameraInput();
    }

    hideCorrection();
    clearSuggestionsPanel();
    updateMirrorExerciseButton();
    await syncTrackingState();
    resetActiveSession();
    if (sessionPayload) {
      await saveProgressFromSession(sessionPayload);
      setInstruction(
        `Exercise saved — score ${sessionPayload.score}%. Progress chart updated.`,
        "success"
      );
    } else {
      setInstruction("Exercise stopped. Recording saved if REC was on.");
    }
  }

  async function startMirrorExercise() {
    if (state.mirrorExerciseActive) return;

    if (state.replayCompareActive) {
      await stopReplayCompare();
    }

    if (state.traineeSource !== "camera") {
      state.inputActive = false;
      state.cameraActive = false;
      await restoreLiveCameraInput();
    }

    if (!state.inputActive || !state.cameraActive) {
      const ok = await ensureTraineePreview();
      if (!ok) return;
      startAnalyzeLoop();
    }
    beginActiveSession();
    if (state.viewMode === "trainee") {
      state.viewMode = "trainer";
      updateViewMode();
    }

    state.mirrorExerciseActive = true;
    updateMirrorExerciseButton();
    await syncTrackingState();
    await sendControl("play");
    const recording = await startExerciseRecording();
    setInstruction(
      recording
        ? "Recording MP4 — follow the trainer. Tap Stop exercise when done."
        : "Mirror & Do started. (MP4 recording unavailable.)"
    );
  }

  function updateSourceButtons() {
    updateFlipLabel();
    els.btnSourceCamera.classList.toggle("active", state.traineeSource === "camera");
    els.btnSourceVideo.classList.toggle("active", state.traineeSource === "video");
    els.btnMirror.disabled = state.traineeSource === "video";
  }

  async function setTraineeSource(source) {
    if (state.inputActive) await stopTraineeInput();
    state.traineeSource = source;
    updateSourceButtons();
    await sendControl("trainee_source", { source });
    setInstruction(
      source === "video"
        ? "Test video mode — flip to you to preview the test clip."
        : "Live camera — flip to you to adjust your position."
    );
  }

  function showPlaceholder(message) {
    els.cameraPlaceholder.classList.remove("hidden");
    if (message && els.cameraPlaceholder.querySelector("p")) {
      els.cameraPlaceholder.querySelector("p").textContent = message;
    }
    els.traineeStream.classList.add("hidden");
  }

  function hidePlaceholder() {
    els.cameraPlaceholder.classList.add("hidden");
  }

  async function ensureTraineePreview() {
    if (state.traineeSource === "video") {
      if (!els.traineeStream.src) {
        els.traineeStream.src = `/stream/trainee?t=${Date.now()}`;
      }
      els.traineeStream.classList.remove("hidden");
      hidePlaceholder();
      await sendControl("trainee_source", { source: "video" });
      state.inputActive = true;
      state.cameraActive = true;
      return true;
    }

    els.traineeStream.classList.add("hidden");
    await sendControl("trainee_source", { source: "camera" });

    if (state.cameraStream && els.traineeCamera.srcObject) {
      hidePlaceholder();
      try {
        await els.traineeCamera.play();
      } catch {
        /* ignore */
      }
      state.inputActive = true;
      state.cameraActive = true;
      return true;
    }

    const ok = await startCamera();
    state.inputActive = ok;
    state.cameraActive = ok;
    return ok;
  }

  function resumeTraineePreview() {
    if (state.traineeSource === "video") {
      if (!els.traineeStream.src) {
        els.traineeStream.src = `/stream/trainee?t=${Date.now()}`;
      }
      els.traineeStream.classList.remove("hidden");
      hidePlaceholder();
      return;
    }
    els.traineeStream.classList.add("hidden");
    if (state.cameraStream) {
      hidePlaceholder();
      els.traineeCamera.play().catch(() => {});
    }
  }

  async function startTraineeInput() {
    const ok = await ensureTraineePreview();
    if (ok) startAnalyzeLoop();
    return ok;
  }

  async function stopTraineeInput() {
    if (state.traineeSource === "camera") {
      stopCamera();
    } else {
      els.traineeStream.src = "";
      els.traineeStream.classList.add("hidden");
    }
    state.inputActive = false;
    state.cameraActive = false;
    stopAnalyzeLoop();
    state.mirrorExerciseActive = false;
    state.showCorrections = false;
    if (state.isRecording) await stopExerciseRecording();
    if (state.replayCompareActive) await stopReplayCompare();
    els.btnShowCorrections?.classList.remove("active");
    updateMirrorExerciseButton();
    await sendControl("tracking", { active: false });
    hideCorrection();
    clearOverlay();
    showPlaceholder();
  }

  async function flipView() {
    if (state.viewMode === "trainer") {
      const ready = await ensureTraineePreview();
      state.viewMode = "trainee";
      updateViewMode();
      if (!ready) {
        setInstruction(
          "Allow camera access in your browser to adjust your position.",
          "correction"
        );
      }
    } else {
      state.viewMode = "trainer";
      updateViewMode();
      updateExercise();
      setInstruction(
        state.mirrorExerciseActive
          ? "Follow the trainer — your form appears in the corner. Red joints = mistakes."
          : "Watch the trainer. Flip to you to set up your camera, then your form shows in the corner."
      );
      analyzeCurrentFrame();
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setInstruction(
        "Camera needs a secure context. Open http://localhost:8080 (not a file:// URL).",
        "correction"
      );
      showPlaceholder("Camera unavailable in this browser.");
      return false;
    }
    try {
      resetSmoothLandmarks();
      if (state.cameraStream) {
        state.cameraStream.getTracks().forEach((t) => t.stop());
        state.cameraStream = null;
      }
      state.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const video = els.traineeCamera;
      video.srcObject = state.cameraStream;
      hidePlaceholder();
      await new Promise((resolve, reject) => {
        if (video.readyState >= 2) {
          resolve();
          return;
        }
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error("video error"));
      });
      await video.play();
      if (!state.mirrorOn) {
        state.mirrorOn = true;
        els.btnMirror.classList.add("active");
        video.classList.add("mirrored");
      }
      state.cameraActive = true;
      return true;
    } catch (err) {
      console.error(err);
      showPlaceholder("Camera blocked — allow access for localhost in browser settings.");
      setInstruction(
        "Camera unavailable. Allow camera access for localhost in browser settings.",
        "correction"
      );
      state.cameraActive = false;
      return false;
    }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((t) => t.stop());
      state.cameraStream = null;
      els.traineeCamera.srcObject = null;
    }
    state.cameraActive = false;
  }

  function trainerPollIntervalMs() {
    if (!state.isPlaying) return 400;
    const spd = Math.max(0.25, Math.abs(state.speed || 1));
    return Math.round(1000 / (24 * spd));
  }

  function resizeTrainerCanvas() {
    syncCanvasPixels(els.trainerCanvas, els.trainerPane);
  }

  function resizeVideoPanes() {
    resizeTrainerCanvas();
    syncCanvasPixels(els.skeletonCanvas, els.formOverlay);
  }

  async function pullTrainerFrame() {
    if (!els.trainerCanvas || trainerFrameBusy) return;
    trainerFrameBusy = true;
    try {
      const res = await fetch(`/api/trainer/frame?overlay=0&t=${Date.now()}`, {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const bmp = await createImageBitmap(blob);
      const canvas = els.trainerCanvas;
      const { w, h } = syncCanvasPixels(canvas, els.trainerPane);
      if (w < 2 || h < 2) {
        bmp.close();
        return;
      }
      const ctx = canvas.getContext("2d");
      const rect = containRect(bmp.width, bmp.height, w, h);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(bmp, rect.ox, rect.oy, rect.drawW, rect.drawH);
      bmp.close();
    } catch (err) {
      console.warn("trainer frame", err);
    } finally {
      trainerFrameBusy = false;
    }
  }

  function scheduleTrainerFrame() {
    if (trainerFrameTimer) clearTimeout(trainerFrameTimer);
    trainerFrameTimer = setTimeout(async () => {
      await pullTrainerFrame();
      scheduleTrainerFrame();
    }, trainerPollIntervalMs());
  }

  function startTrainerDisplay() {
    scheduleTrainerFrame();
    pullTrainerFrame();
  }

  function reloadTrainerStream() {
    pullTrainerFrame();
  }

  function shortVideoLabel(url) {
    if (!url) return "";
    if (url.startsWith("http")) {
      try {
        const u = new URL(url);
        return u.hostname + u.pathname.slice(0, 40) + (u.pathname.length > 40 ? "…" : "");
      } catch {
        return url.slice(0, 48) + (url.length > 48 ? "…" : "");
      }
    }
    const parts = url.split("/");
    return parts[parts.length - 1] || url;
  }

  async function loadTrainerVideoFromInput() {
    const url = els.trainerVideoUrl.value.trim();
    if (!url) {
      setInstruction("Enter a YouTube URL or path to a video file.", "correction");
      return;
    }
    els.btnLoadTrainer.disabled = true;
    els.btnLoadTrainer.textContent = "Loading…";
    setInstruction(
      "Loading trainer video… First YouTube load can take up to a minute (downloading in the server)."
    );
    try {
      const res = await GymAPI.setConfig({ trainer_video: url });
      if (!res.ok) {
        setInstruction(res.error || "Could not load trainer video.", "correction");
        return;
      }
      reloadTrainerStream();
      await sendControl("replay");
      els.trainerVideoHint.textContent = `Loaded: ${shortVideoLabel(res.trainer_video || url)}`;
      setInstruction("Trainer video updated. Press Play if it does not start automatically.", "success");
    } catch {
      setInstruction("Could not load trainer video. Check the URL and server logs.", "correction");
    } finally {
      els.btnLoadTrainer.disabled = false;
      els.btnLoadTrainer.textContent = "Load";
    }
  }

  async function loadInitialConfig() {
    try {
      const cfg = await GymAPI.getConfig();
      if (cfg.trainer_video) {
        els.trainerVideoUrl.value = cfg.trainer_video;
        els.trainerVideoHint.textContent = `Current: ${shortVideoLabel(cfg.trainer_video)}`;
      }
    } catch {
      /* ignore */
    }
  }


  els.btnFlip.addEventListener("click", flipView);

  els.btnLoadTrainer.addEventListener("click", loadTrainerVideoFromInput);
  els.trainerVideoUrl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadTrainerVideoFromInput();
  });

  els.btnSourceCamera.addEventListener("click", () => setTraineeSource("camera"));
  els.btnSourceVideo.addEventListener("click", () => setTraineeSource("video"));

  els.btnPlayPause.addEventListener("click", async () => {
    await sendControl(state.isPlaying ? "pause" : "play");
    setInstruction(state.isPlaying ? "Follow the trainer's movement." : "Paused.");
  });

  els.btnRewind.addEventListener("click", () => skipRelative(-SKIP_SECONDS));

  els.btnFastForward.addEventListener("click", () => skipRelative(SKIP_SECONDS));

  els.btnStop.addEventListener("click", async () => {
    await sendControl("stop");
    state.speed = 1.0;
    updateSpeedBadge();
    updateSpeedSelect();
    setInstruction("Stopped.");
  });

  els.btnReplay.addEventListener("click", async () => {
    await sendControl("replay");
    setInstruction("Replaying from the start.");
  });

  els.speedSelect?.addEventListener("change", async () => {
    state.speed = parseFloat(els.speedSelect.value);
    await sendControl("speed", { speed: state.speed });
    updateSpeedBadge();
  });

  els.btnPrev.addEventListener("click", () => {
    state.exerciseIndex = (state.exerciseIndex - 1 + EXERCISES.length) % EXERCISES.length;
    updateExercise();
  });

  els.btnNext.addEventListener("click", () => {
    state.exerciseIndex = (state.exerciseIndex + 1) % EXERCISES.length;
    updateExercise();
  });

  els.btnMirror.addEventListener("click", () => {
    state.mirrorOn = !state.mirrorOn;
    els.btnMirror.classList.toggle("active", state.mirrorOn);
    els.traineeCamera.classList.toggle("mirrored", state.mirrorOn);
  });

  els.btnShowCorrections.addEventListener("click", async () => {
    state.showCorrections = !state.showCorrections;
    els.btnShowCorrections.classList.toggle("active", state.showCorrections);
    if (state.showCorrections && !state.inputActive) {
      const ok = await ensureTraineePreview();
      if (!ok) {
        state.showCorrections = false;
        els.btnShowCorrections.classList.remove("active");
        return;
      }
      startAnalyzeLoop();
    }
    await syncTrackingState();
    setInstruction(
      state.showCorrections
        ? "Corrections on — mistakes show in red on your corner form window."
        : "Corrections off."
    );
  });

  els.btnMirrorExercise.addEventListener("click", () => {
    startMirrorExercise();
  });

  els.btnStopExercise?.addEventListener("click", () => {
    stopMirrorExercise();
  });

  els.btnStopExerciseFloat?.addEventListener("click", () => {
    stopMirrorExercise();
  });

  els.btnReplayLast.addEventListener("click", () => {
    startReplayCompare();
  });

  els.btnCompareRecording?.addEventListener("click", () => {
    startReplayCompare();
  });

  els.btnDownloadRecording?.addEventListener("click", (e) => {
    if (!state.lastRecordingPath) {
      e.preventDefault();
      setInstruction("No recording to download yet.", "correction");
    }
  });

  els.btnCloseReplay?.addEventListener("click", closeReplayOverlay);
  els.replayOverlay?.addEventListener("click", (e) => {
    if (e.target === els.replayOverlay) closeReplayOverlay();
  });

  els.btnSpeaker.addEventListener("click", () => {
    state.speakerOn = !state.speakerOn;
    els.btnSpeaker.classList.toggle("active", state.speakerOn);
  });

  els.btnMute.addEventListener("click", () => {
    state.isMuted = !state.isMuted;
    els.btnMute.classList.toggle("muted", state.isMuted);
    els.volumeSlider.value = state.isMuted ? 0 : 75;
  });

  els.volumeSlider.addEventListener("input", () => {
    state.isMuted = els.volumeSlider.value === "0";
    els.btnMute.classList.toggle("muted", state.isMuted);
  });

  $$(".time-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".time-tabs .tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.progressPeriod = tab.dataset.period;
      refreshProgressChart(state.progressPeriod);
    });
  });

  async function checkServerBuild() {
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      if (data.build !== "trainer-frame-v3") {
        setInstruction(
          "Server is outdated. Stop the terminal (Ctrl+C) and run ./run_gymform.sh again.",
          "correction"
        );
      }
    } catch {
      setInstruction(
        "Cannot reach GymForm server on this port. Run ./run_gymform.sh from the Gym Pose Tracker folder.",
        "correction"
      );
    }
  }

  // ── Init ────────────────────────────────────────────────────────
  updateExercise();
  updateSpeedBadge();
  updateSpeedSelect();
  updatePlayButton();
  updateScore(75);
  refreshProgressChart("week");
  syncStatus();
  loadInitialConfig();
  checkServerBuild();
  startTrainerDisplay();
  updateViewMode();
  updateMirrorExerciseButton();
  updateReplayLastButton();
  updateSkeletonHint("Flip to you to set up camera");
  loadLastRecording();
  window.addEventListener("resize", resizeVideoPanes);
  sendControl("trainee_source", { source: "camera" }).catch(() => {});
  sendControl("play").catch(() => {});
  statusInterval = setInterval(syncStatus, 2000);
  updateSourceButtons();
  showPlaceholder("Tap Flip to you to start your camera.");
})();
