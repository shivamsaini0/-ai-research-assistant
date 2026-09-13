/**
 * Chat Module — Handles message rendering and sending.
 */

const ChatManager = {
  sessionId: null,

  init(sessionId) {
    this.sessionId = sessionId;

    // Auto-resize textarea
    const input = document.getElementById("chatInput");
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 120) + "px";
      document.getElementById("charCount").textContent =
        `${input.value.length} / 2000`;
    });

    // Send on Enter (Shift+Enter = newline)
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        app.sendMessage();
      }
    });
  },

  async sendMessage(text) {
    if (!text.trim()) return;

    const docIds = UploadManager.getSelectedDocIds();

    // Show user message
    this.addMessage("user", text);
    this.showWelcome(false);

    // Show thinking indicator
    const thinking = document.getElementById("thinking");
    thinking.style.display = "flex";

    // Disable input
    this.setInputEnabled(false);

    try {
      const data = await ApiClient.chat(this.sessionId, text, docIds);

      thinking.style.display = "none";
      this.addMessage("assistant", data.answer);
      this.renderSources(data.sources);
    } catch (err) {
      thinking.style.display = "none";
      this.addMessage("assistant", `⚠️ Error: ${err.message}`);
      Toast.show(err.message, "error");
    } finally {
      this.setInputEnabled(true);
      document.getElementById("chatInput").focus();
    }
  },

  addMessage(role, content) {
    const messages = document.getElementById("messages");
    messages.classList.add("has-messages");

    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const initial = role === "user" ? "U" : "AI";

    const msg = document.createElement("div");
    msg.className = `message ${role}`;
    msg.innerHTML = `
      <div class="msg-avatar">${initial}</div>
      <div class="msg-content">
        <div class="msg-bubble">${this.formatMarkdown(content)}</div>
        <div class="msg-time">${now}</div>
      </div>
    `;

    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
  },

  formatMarkdown(text) {
    // Simple markdown → HTML conversion for assistant messages
    return text
      // Code blocks (```...```)
      .replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
      // Inline code
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      // Bold **text**
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      // Italic *text*
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      // Headers ### ## #
      .replace(/^### (.*$)/gm, "<h4>$1</h4>")
      .replace(/^## (.*$)/gm, "<h3>$1</h3>")
      .replace(/^# (.*$)/gm, "<h2>$1</h2>")
      // Bullet points
      .replace(/^[-•] (.*$)/gm, "<li>$1</li>")
      .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
      // Newlines → paragraphs
      .split("\n\n")
      .map((para) => (para.startsWith("<") ? para : `<p>${para}</p>`))
      .join("");
  },

  renderSources(sources) {
    const list = document.getElementById("sourcesList");
    const empty = document.getElementById("emptySources");
    const count = document.getElementById("sourcesCount");

    // Clear previous
    list.querySelectorAll(".source-card").forEach((el) => el.remove());

    count.textContent = `${sources.length} citation${sources.length !== 1 ? "s" : ""}`;

    if (!sources.length) {
      empty.style.display = "flex";
      return;
    }

    empty.style.display = "none";

    sources.forEach((src, i) => {
      const score = (src.relevance_score * 100).toFixed(1);
      const card = document.createElement("div");
      card.className = "source-card";
      card.innerHTML = `
        <div class="source-card-header">
          <span class="source-rank">#${i + 1}</span>
          <span class="source-filename" title="${src.filename}">${src.filename}</span>
          <span class="source-score">${score}%</span>
        </div>
        <div class="source-meta">Chunk ${src.chunk_index}</div>
        <div class="source-preview">${src.content_preview}</div>
      `;
      list.appendChild(card);
    });
  },

  clear() {
    const messages = document.getElementById("messages");
    messages.innerHTML = "";
    messages.classList.remove("has-messages");

    // Reset sources
    document.querySelectorAll(".source-card").forEach((el) => el.remove());
    document.getElementById("emptySources").style.display = "flex";
    document.getElementById("sourcesCount").textContent = "0 citations";
  },

  showWelcome(show) {
    const welcome = document.getElementById("welcomeScreen");
    welcome.style.display = show ? "flex" : "none";
  },

  setInputEnabled(enabled) {
    const input = document.getElementById("chatInput");
    const btn = document.getElementById("sendBtn");
    input.disabled = !enabled;
    btn.disabled = !enabled;
  },
};
