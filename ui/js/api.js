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

  return { control, status, analyzeFrame, getAnalysis, summary };
})();

window.GymAPI = GymAPI;
