/**
 * PoseNotifier — debounced correction alerts per sketch spec:
 *   • Only notify after 15 consecutive incorrect frames
 *   • 2-second debounce between notifications
 */
class PoseNotifier {
  constructor(options = {}) {
    this.consecutiveThreshold = options.consecutiveThreshold ?? 15;
    this.debounceMs = options.debounceMs ?? 2000;
    this.onNotify = options.onNotify ?? (() => {});
    this.onClear = options.onClear ?? (() => {});

    this._streaks = new Map();       // joint → consecutive bad frames
    this._lastNotifyTime = 0;
    this._pendingTimer = null;
    this._activeIssue = null;
  }

  /**
   * Feed one frame of joint analysis.
   * @param {Array<{joint: string, isCorrect: boolean, message?: string}>} frameResults
   */
  processFrame(frameResults) {
    const badJoints = [];

    for (const { joint, isCorrect, message } of frameResults) {
      if (!isCorrect) {
        const streak = (this._streaks.get(joint) ?? 0) + 1;
        this._streaks.set(joint, streak);
        if (streak >= this.consecutiveThreshold) {
          badJoints.push({ joint, message: message ?? `Fix your ${joint.toLowerCase()}` });
        }
      } else {
        this._streaks.set(joint, 0);
      }
    }

    if (badJoints.length === 0) {
      this._clearPending();
      if (this._activeIssue) {
        this._activeIssue = null;
        this.onClear();
      }
      return;
    }

    const primary = badJoints[0];
    const now = Date.now();
    const elapsed = now - this._lastNotifyTime;

    if (elapsed >= this.debounceMs) {
      this._fire(primary);
    } else if (!this._pendingTimer) {
      this._pendingTimer = setTimeout(() => {
        this._pendingTimer = null;
        this._fire(primary);
      }, this.debounceMs - elapsed);
    }
  }

  _fire(issue) {
    this._lastNotifyTime = Date.now();
    this._activeIssue = issue;
    this.onNotify(issue);
  }

  _clearPending() {
    if (this._pendingTimer) {
      clearTimeout(this._pendingTimer);
      this._pendingTimer = null;
    }
  }

  reset() {
    this._streaks.clear();
    this._clearPending();
    this._activeIssue = null;
    this._lastNotifyTime = 0;
  }
}

window.PoseNotifier = PoseNotifier;
