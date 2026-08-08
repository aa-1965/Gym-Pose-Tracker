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
  };

  const EXERCISES = [
    { name: "Push-ups", sets: "Set 1/3", instruction: "Keep your core tight. Lower until chest nearly touches the floor." },
    { name: "Squats", sets: "Set 1/3", instruction: "Feet shoulder-width apart. Push hips back and keep knees aligned." },
    { name: "Lunges", sets: "Set 1/3", instruction: "Step forward, lower back knee toward floor. Keep torso upright." },
    { name: "Plank", sets: "Set 1/3", instruction: "Hold a straight line from head to heels. Don't let hips sag." },
  ];

  const PROGRESS_DATA = {
    day: { bars: [30, 45, 60, 75, 55, 40, 20], sessions: 1, streak: 5, avg: "75%" },
    week: { bars: [45, 62, 58, 75, 40, 30, 20], sessions: 12, streak: 5, avg: "68%" },
    month: { bars: [55, 48, 62, 70, 65, 58, 72], sessions: 48, streak: 5, avg: "71%" },
  };

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
    trainerStream: $("#trainerStream"),
    instructionBar: $("#instructionBar"),
    instructionText: $("#instructionText"),
    trainerControls: $("#trainerControls"),
    traineeControls: $("#traineeControls"),
    btnFlip: $("#btnFlip"),
    btnPlayPause: $("#btnPlayPause"),
    btnStop: $("#btnStop"),
    btnReplay: $("#btnReplay"),
    btnPrev: $("#btnPrev"),
    btnNext: $("#btnNext"),
    btnMirror: $("#btnMirror"),
    btnReplayLast: $("#btnReplayLast"),
    btnShowCorrections: $("#btnShowCorrections"),
    btnMirrorExercise: $("#btnMirrorExercise"),
    btnSpeaker: $("#btnSpeaker"),
    btnMute: $("#btnMute"),
    volumeSlider: $("#volumeSlider"),
    speedBadge: $("#speedBadge"),
    exerciseTag: $("#exerciseTag"),
    correctionToast: $("#correctionToast"),
    toastText: $("#toastText"),
    skeletonCanvas: $("#skeletonCanvas"),
    skeletonFeedback: $("#skeletonFeedback"),
    traineeCamera: $("#traineeCamera"),
    traineeStream: $("#traineeStream"),
    traineePaneLabel: $("#traineePaneLabel"),
    btnSourceCamera: $("#btnSourceCamera"),
    btnSourceVideo: $("#btnSourceVideo"),
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
    },
    onClear() {
      hideCorrection();
    },
  });

  let analyzeInterval = null;
  let statusInterval = null;
  let analyzing = false;

  // ── Backend sync ────────────────────────────────────────────────
  async function syncStatus() {
    try {
      const s = await GymAPI.status();
      state.backendConnected = true;
      state.isPlaying = !s.paused;
      state.speed = s.speed;
      updatePlayButton();
      updateSpeedBadge();
      if (s.trainee_source && s.trainee_source !== state.traineeSource) {
        state.traineeSource = s.trainee_source;
        updateSourceButtons();
      }
      if (s.tracking !== state.mirrorExerciseActive) {
        state.mirrorExerciseActive = s.tracking;
        els.btnMirrorExercise.classList.toggle("active", s.tracking);
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
    analyzeInterval = setInterval(analyzeCurrentFrame, 1000 / 10);
  }

  function stopAnalyzeLoop() {
    if (analyzeInterval) {
      clearInterval(analyzeInterval);
      analyzeInterval = null;
    }
    analyzing = false;
    notifier.reset();
  }

  function applyAnalysisResult(result) {
    if (!result || !result.ok) {
      if (state.traineeSource === "camera") {
        setInstruction("No pose detected — step back so your full body is visible.", "correction");
      }
      return;
    }

    if (result.score != null) updateScore(result.score);
    if (els.formScore && result.score != null) els.formScore.textContent = `${result.score}%`;

    if (result.frame_results && result.frame_results.length) {
      notifier.processFrame(result.frame_results);
    } else if (result.good_form) {
      notifier.processFrame(
        Object.keys(JOINT_VERTICES).map((joint) => ({ joint, isCorrect: true }))
      );
    }

    if (result.notify) showCorrection(result.notify.action);

    if (state.showCorrections || state.mirrorExerciseActive) {
      drawLandmarks(result.landmarks, result.feedback || []);
      renderSkeletonFeedback(result.feedback || [], result.height_feedback || []);
    } else {
      drawLandmarks(result.landmarks, []);
      renderSkeletonFeedback([], []);
    }
  }

  async function analyzeCurrentFrame() {
    if (analyzing || !state.mirrorExerciseActive) return;

    analyzing = true;
    try {
      let result;
      if (state.traineeSource === "video") {
        result = await GymAPI.getAnalysis();
      } else {
        const video = els.traineeCamera;
        if (!video.videoWidth) return;
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
      setInstruction("Analysis error — check that api_server.py is running.", "correction");
    } finally {
      analyzing = false;
    }
  }

  function clearOverlay() {
    const canvas = els.skeletonCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (els.skeletonFeedback) {
      els.skeletonFeedback.innerHTML = "";
      els.skeletonFeedback.classList.add("hidden");
    }
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

  function drawLandmarks(landmarks, feedback) {
    if (!landmarks || !landmarks.length || !els.skeletonCanvas) return;
    const canvas = els.skeletonCanvas;
    const pane = canvas.parentElement;
    const w = pane.clientWidth || 200;
    const h = Math.max(pane.clientHeight - (els.skeletonFeedback?.offsetHeight || 0), 160);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);

    const badVerts = errorVertexSet(feedback);

    const pt = (i) => ({
      x: landmarks[i].x * w,
      y: landmarks[i].y * h,
    });

    const connBad = (a, b) => badVerts.has(a) || badVerts.has(b);

    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    for (const [a, b] of POSE_CONNECTIONS) {
      if (!landmarks[a] || !landmarks[b]) continue;
      const p1 = pt(a);
      const p2 = pt(b);
      ctx.strokeStyle = connBad(a, b) ? "#ff4757" : "#00d4aa";
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    for (let i = 0; i < landmarks.length; i++) {
      if (landmarks[i].visibility !== undefined && landmarks[i].visibility < 0.4) continue;
      const p = pt(i);
      ctx.fillStyle = badVerts.has(i) ? "#ff4757" : "#00d4aa";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderSkeletonFeedback(feedback, heightFeedback) {
    const list = els.skeletonFeedback;
    if (!list) return;
    const items = [
      ...(feedback || []).map((f) => f.action),
      ...(heightFeedback || []).map((h) => h.message),
    ];
    if (!state.showCorrections || items.length === 0) {
      list.innerHTML = "";
      list.classList.add("hidden");
      return;
    }
    list.innerHTML = items.map((t) => `<li>⚠ ${t}</li>`).join("");
    list.classList.remove("hidden");
  }

  // ── UI helpers ──────────────────────────────────────────────────
  function setInstruction(text, type = "normal") {
    els.instructionText.textContent = text;
    els.instructionBar.className = "instruction-bar";
    if (type === "correction") els.instructionBar.classList.add("correction");
    if (type === "success") els.instructionBar.classList.add("success");
  }

  function showCorrection(message) {
    els.toastText.textContent = message;
    els.correctionToast.classList.remove("hidden");
    setInstruction(message, "correction");
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
    const sign = state.speed >= 0 ? "+" : "";
    els.speedBadge.textContent = `${sign}${state.speed.toFixed(1)}×`;
  }

  function updatePlayButton() {
    const btn = els.btnPlayPause;
    if (state.isPlaying) {
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" class="play-icon"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg><span>Pause</span>`;
      btn.classList.add("playing");
    } else {
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" class="play-icon"><path d="M8 5v14l11-7z"/></svg><span>Play</span>`;
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

  function updateProgressChart(period) {
    const data = PROGRESS_DATA[period];
    const days = ["M", "T", "W", "T", "F", "S", "S"];
    els.progressChart.innerHTML = data.bars
      .map((h, i) => `<div class="bar${h === Math.max(...data.bars) ? " active" : ""}" style="--h:${h}%"><span>${days[i]}</span></div>`)
      .join("");
    els.sessionsCount.textContent = data.sessions;
    els.streakCount.textContent = data.streak;
    els.avgScore.textContent = data.avg;
  }

  function updateFlipLabel() {
    if (!state.inputActive) {
      els.btnFlipLabel.textContent =
        state.traineeSource === "video" ? "Start test video" : "Start camera";
    }
  }

  function updateSourceButtons() {
    updateFlipLabel();
    els.btnSourceCamera.classList.toggle("active", state.traineeSource === "camera");
    els.btnSourceVideo.classList.toggle("active", state.traineeSource === "video");
    els.btnMirror.disabled = state.traineeSource === "video";
    els.traineePaneLabel.textContent = state.traineeSource === "video" ? "Test video" : "You";
  }

  async function setTraineeSource(source) {
    if (state.inputActive) await stopTraineeInput();
    state.traineeSource = source;
    updateSourceButtons();
    await sendControl("trainee_source", { source });
    setInstruction(
      source === "video"
        ? "Test video mode: use Start to play imperfect_pushups vs the trainer."
        : "Live camera mode: use Start camera when ready."
    );
  }

  function showPlaceholder() {
    els.cameraPlaceholder.classList.remove("hidden");
    els.traineeCamera.classList.add("hidden");
    els.traineeStream.classList.add("hidden");
  }

  async function startTraineeInput() {
    if (state.traineeSource === "camera") {
      await startCamera();
      els.btnFlipLabel.textContent = "Stop camera";
    } else {
      els.traineeStream.src = `/stream/trainee?t=${Date.now()}`;
      els.traineeStream.classList.remove("hidden");
      els.traineeCamera.classList.add("hidden");
      els.cameraPlaceholder.classList.add("hidden");
      els.btnFlipLabel.textContent = "Stop test video";
      await sendControl("play");
    }
    state.inputActive = true;
    setInstruction(
      state.traineeSource === "video"
        ? "Comparing test video to trainer. Press Mirror & Do to analyze form."
        : "Trainer on the left. Your camera on the right."
    );
  }

  async function stopTraineeInput() {
    if (state.traineeSource === "camera") {
      stopCamera();
      els.btnFlipLabel.textContent = "Start camera";
    } else {
      els.traineeStream.src = "";
      els.traineeStream.classList.add("hidden");
      els.btnFlipLabel.textContent = "Start test video";
    }
    state.inputActive = false;
    stopAnalyzeLoop();
    await sendControl("tracking", { active: false });
    state.mirrorExerciseActive = false;
    els.btnMirrorExercise.classList.remove("active");
    hideCorrection();
    clearOverlay();
    showPlaceholder();
  }

  // ── Flip / camera ───────────────────────────────────────────────
  async function toggleTraineeInput() {
    if (state.inputActive) {
      await stopTraineeInput();
      updateExercise();
    } else {
      await startTraineeInput();
    }
  }

  async function startCamera() {
    try {
      state.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 720 }, height: { ideal: 1280 } },
        audio: false,
      });
      els.traineeCamera.srcObject = state.cameraStream;
      els.traineeCamera.classList.remove("hidden");
      els.cameraPlaceholder.classList.add("hidden");
    } catch {
      setInstruction("Camera unavailable. Grant permission and reload.", "correction");
    }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((t) => t.stop());
      state.cameraStream = null;
      els.traineeCamera.srcObject = null;
    }
  }

  // ── Events ──────────────────────────────────────────────────────
  els.btnFlip.addEventListener("click", toggleTraineeInput);

  els.btnSourceCamera.addEventListener("click", () => setTraineeSource("camera"));
  els.btnSourceVideo.addEventListener("click", () => setTraineeSource("video"));

  els.btnPlayPause.addEventListener("click", async () => {
    await sendControl(state.isPlaying ? "pause" : "play");
    setInstruction(state.isPlaying ? "Follow the trainer's movement." : "Paused.");
  });

  els.btnStop.addEventListener("click", async () => {
    await sendControl("stop");
    state.speed = 1.0;
    updateSpeedBadge();
    setInstruction("Stopped.");
  });

  els.btnReplay.addEventListener("click", async () => {
    await sendControl("replay");
    setInstruction("Replaying from the start.");
  });

  els.btnPrev.addEventListener("click", () => {
    state.exerciseIndex = (state.exerciseIndex - 1 + EXERCISES.length) % EXERCISES.length;
    updateExercise();
  });

  els.btnNext.addEventListener("click", () => {
    state.exerciseIndex = (state.exerciseIndex + 1) % EXERCISES.length;
    updateExercise();
  });

  $$(".speed-row .ctrl-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const delta = parseFloat(btn.dataset.speed);
      state.speed = Math.max(-2, Math.min(2, state.speed + delta));
      await sendControl("speed", { speed: state.speed });
      updateSpeedBadge();
    });
  });

  els.btnMirror.addEventListener("click", () => {
    state.mirrorOn = !state.mirrorOn;
    els.btnMirror.classList.toggle("active", state.mirrorOn);
    els.traineeCamera.classList.toggle("mirrored", state.mirrorOn);
  });

  els.btnShowCorrections.addEventListener("click", () => {
    state.showCorrections = !state.showCorrections;
    els.btnShowCorrections.classList.toggle("active", state.showCorrections);
    if (!state.showCorrections && els.skeletonFeedback) {
      els.skeletonFeedback.classList.add("hidden");
    }
  });

  els.btnMirrorExercise.addEventListener("click", async () => {
    state.mirrorExerciseActive = !state.mirrorExerciseActive;
    els.btnMirrorExercise.classList.toggle("active", state.mirrorExerciseActive);
    await sendControl("tracking", { active: state.mirrorExerciseActive });

    if (state.mirrorExerciseActive) {
      setInstruction("Comparing your pose to the trainer (MediaPipe + DTW)…");
      startAnalyzeLoop();
    } else {
      stopAnalyzeLoop();
      hideCorrection();
      clearOverlay();
      setInstruction("Exercise mode stopped.");
    }
  });

  els.btnReplayLast.addEventListener("click", async () => {
    await sendControl("replay");
    setInstruction("Replaying trainer demo…");
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
      updateProgressChart(state.progressPeriod);
    });
  });

  // ── Init ────────────────────────────────────────────────────────
  updateExercise();
  updateSpeedBadge();
  updatePlayButton();
  updateScore(75);
  updateProgressChart("week");
  syncStatus();
  sendControl("play").catch(() => {});
  statusInterval = setInterval(syncStatus, 2000);
  updateSourceButtons();
  els.btnFlipLabel.textContent = "Start camera";
  showPlaceholder();
})();
