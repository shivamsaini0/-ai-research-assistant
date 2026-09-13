/**
 * App — Main application controller. Initializes everything and
 * provides the global `app` object used by HTML event handlers.
 */

// ── Toast Notification Utility ─────────────────────────────────
const Toast = {
  show(message, type = "info", duration = 4000) {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;

    const icons = { success: "✓", error: "✕", info: "ℹ" };
    toast.innerHTML = `<span>${icons[type] || "ℹ"}</span><span>${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = "toastOut 0.3s ease forwards";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  },
};

// ── App Controller ──────────────────────────────────────────────
const app = {
  sessionId: null,

  async init() {
    // Generate a unique session ID for this browser session
    this.sessionId = "session_" + Math.random().toString(36).substr(2, 9);

    // Initialize modules
    UploadManager.init();
    ChatManager.init(this.sessionId);

    // Check backend health
    await this.checkHealth();

    // Poll health every 30s
    setInterval(() => this.checkHealth(), 30000);

    console.log("✅ ResearchAI initialized | session:", this.sessionId);
  },

  async checkHealth() {
    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");
    try {
      const data = await ApiClient.health();
      dot.className = "status-dot online";
      text.textContent = `Online · ${data.total_chunks} chunks`;
    } catch {
      dot.className = "status-dot error";
      text.textContent = "Backend offline";
    }
  },

  async sendMessage() {
    const input = document.getElementById("chatInput");
    const text = input.value.trim();
    if (!text) return;

    // Clear input
    input.value = "";
    input.style.height = "auto";
    document.getElementById("charCount").textContent = "0 / 2000";

    await ChatManager.sendMessage(text);
  },

  clearChat() {
    if (!confirm("Clear conversation history?")) return;
    ChatManager.clear();
    ChatManager.showWelcome(true);
    ApiClient.clearSession(this.sessionId).catch(() => {});
    Toast.show("Chat cleared", "info");
  },

  insertSampleQuestion(btn) {
    const input = document.getElementById("chatInput");
    input.value = btn.textContent;
    input.focus();
    document.getElementById("charCount").textContent =
      `${input.value.length} / 2000`;
  },
};

// ── Boot ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => app.init());
