/**
 * Upload Module — Handles file drag-and-drop, validation, and upload UI.
 */

const UploadManager = {
  documents: [],   // Cached document list

  init() {
    const zone = document.getElementById("uploadZone");
    const input = document.getElementById("fileInput");

    // Drag & Drop
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
      this.handleFiles(Array.from(e.dataTransfer.files));
    });

    // File input change
    input.addEventListener("change", (e) => {
      this.handleFiles(Array.from(e.target.files));
      input.value = "";  // Reset so same file can be re-uploaded
    });

    // Load existing documents
    this.loadDocuments();
  },

  async handleFiles(files) {
    const allowed = ["pdf", "docx", "txt"];
    const valid = files.filter((f) => {
      const ext = f.name.split(".").pop().toLowerCase();
      if (!allowed.includes(ext)) {
        Toast.show(`Skipped '${f.name}' — unsupported type`, "error");
        return false;
      }
      if (f.size > 50 * 1024 * 1024) {
        Toast.show(`Skipped '${f.name}' — exceeds 50MB`, "error");
        return false;
      }
      return true;
    });

    if (!valid.length) return;

    for (const file of valid) {
      await this.uploadFile(file);
    }
  },

  async uploadFile(file) {
    const progress = document.getElementById("uploadProgress");
    const fill = document.getElementById("uploadProgressFill");
    const text = document.getElementById("uploadProgressText");

    progress.style.display = "block";
    fill.style.width = "0%";
    text.textContent = `Uploading ${file.name}...`;

    try {
      const result = await ApiClient.uploadFile(file, (pct) => {
        fill.style.width = `${pct}%`;
        if (pct === 100) text.textContent = "Processing & indexing...";
      });

      fill.style.width = "100%";
      text.textContent = `✓ Indexed ${result.total_chunks} chunks`;

      Toast.show(`'${file.name}' ingested — ${result.total_chunks} chunks`, "success");

      setTimeout(() => {
        progress.style.display = "none";
      }, 2000);

      await this.loadDocuments();
    } catch (err) {
      text.textContent = `Error: ${err.message}`;
      Toast.show(err.message, "error");
      setTimeout(() => (progress.style.display = "none"), 3000);
    }
  },

  async loadDocuments() {
    try {
      const data = await ApiClient.listDocuments();
      this.documents = data.documents || [];
      this.renderDocList();
    } catch {
      // Backend might not be ready yet
    }
  },

  renderDocList() {
    const list = document.getElementById("docList");
    const empty = document.getElementById("emptyDocs");
    const count = document.getElementById("docCount");

    count.textContent = `${this.documents.length} doc${this.documents.length !== 1 ? "s" : ""}`;

    if (!this.documents.length) {
      empty.style.display = "block";
      // Remove all doc items
      list.querySelectorAll(".doc-item").forEach((el) => el.remove());
      return;
    }

    empty.style.display = "none";
    list.querySelectorAll(".doc-item").forEach((el) => el.remove());

    this.documents.forEach((doc) => {
      const ext = doc.filename.split(".").pop().toLowerCase();
      const iconMap = { pdf: "PDF", docx: "DOC", txt: "TXT" };
      const icon = iconMap[ext] || "DOC";

      const kb = Math.round(doc.file_size_bytes / 1024);
      const size = kb > 1024 ? `${(kb / 1024).toFixed(1)}MB` : `${kb}KB`;

      const item = document.createElement("div");
      item.className = "doc-item";
      item.dataset.docId = doc.doc_id;
      item.innerHTML = `
        <input type="checkbox" class="doc-checkbox" id="doc-${doc.doc_id}" value="${doc.doc_id}" />
        <div class="doc-icon ${ext}">${icon}</div>
        <div class="doc-info">
          <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
          <div class="doc-meta">${doc.total_chunks} chunks · ${size}</div>
        </div>
        <button class="doc-delete" title="Delete document" onclick="UploadManager.deleteDoc('${doc.doc_id}', event)">✕</button>
      `;

      item.querySelector(".doc-checkbox").addEventListener("change", (e) => {
        item.classList.toggle("selected", e.target.checked);
      });

      list.appendChild(item);
    });
  },

  async deleteDoc(docId, event) {
    event.stopPropagation();
    if (!confirm("Delete this document and all its chunks?")) return;
    try {
      await ApiClient.deleteDocument(docId);
      Toast.show("Document deleted", "info");
      await this.loadDocuments();
    } catch (err) {
      Toast.show(err.message, "error");
    }
  },

  getSelectedDocIds() {
    const filterEnabled = document.getElementById("filterByDocs").checked;
    if (!filterEnabled) return null;
    const checked = document.querySelectorAll(".doc-checkbox:checked");
    return checked.length ? Array.from(checked).map((c) => c.value) : null;
  },
};
