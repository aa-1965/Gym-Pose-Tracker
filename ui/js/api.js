/**
 * GymForm API client — talks to the Flask/MediaPipe backend.
 */
const GymAPI = (() => {
  const BASE = ""; // same origin

  async function control(action, extra = {}) {
    const res = await fetch(`${BASE}/api/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }),
    });
    return res.json();
  }

  async function status() {
    const res = await fetch(`${BASE}/api/status`);
    return res.json();
  }

  async function analyzeFrame(canvas) {
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.7)
    );
    if (!blob) return null;

    const b64 = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });

    const res = await fetch(`${BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64 }),
    });
    return res.json();
  }

  async function summary() {
    const res = await fetch(`${BASE}/api/summary`);
    return res.json();
  }

  async function getAnalysis() {
    const res = await fetch(`${BASE}/api/analysis`);
    return res.json();
  }

  async function getConfig() {
    const res = await fetch(`${BASE}/api/config`);
    return res.json();
  }

  async function setConfig(config) {
    const res = await fetch(`${BASE}/api/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { ok: false, error: data.error || `Server error (${res.status})` };
    }
    return data;
  }

  async function getLastRecording() {
    const res = await fetch(`${BASE}/api/recordings/last`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { ok: false, error: data.error || "No recording" };
    }
    return data;
  }

  function recordingDownloadUrl() {
    return `${BASE}/api/recordings/last/download`;
  }

  async function getProgress(period = "week") {
    const res = await fetch(`${BASE}/api/progress?period=${encodeURIComponent(period)}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { ok: false, error: data.error || "Could not load progress" };
    }
    return data;
  }

  async function saveProgressSession(session) {
    const res = await fetch(`${BASE}/api/progress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(session),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { ok: false, error: data.error || "Could not save progress" };
    }
    return data;
  }

  return {
    control,
    status,
    analyzeFrame,
    getAnalysis,
    summary,
    getConfig,
    setConfig,
    getLastRecording,
    recordingDownloadUrl,
    getProgress,
    saveProgressSession,
  };
})();

window.GymAPI = GymAPI;
