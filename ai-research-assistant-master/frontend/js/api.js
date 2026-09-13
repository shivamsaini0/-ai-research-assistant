/**
 * API Client — Handles all HTTP calls to the FastAPI backend.
 * Central place to set the base URL and handle errors.
 */

const API_BASE = "http://localhost:8000/api";

const ApiClient = {
  /**
   * Upload a file to the backend for ingestion.
   * @param {File} file
   * @param {function} onProgress - optional progress callback (0-100)
   */
  async uploadFile(file, onProgress) {
    const formData = new FormData();
    formData.append("file", file);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/upload`);

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      });

      xhr.addEventListener("load", () => {
        try {
          const data = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(data);
          } else {
            reject(new Error(data.detail || `Upload failed (${xhr.status})`));
          }
        } catch {
          reject(new Error("Invalid server response"));
        }
      });

      xhr.addEventListener("error", () => reject(new Error("Network error")));
      xhr.send(formData);
    });
  },

  /**
   * Send a chat message and get a RAG response.
   */
  async chat(sessionId, message, documentIds = null) {
    const body = {
      session_id: sessionId,
      message: message,
      document_ids: documentIds,
    };

    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Chat failed (${res.status})`);
    }

    return res.json();
  },

  /**
   * List all ingested documents.
   */
  async listDocuments() {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error("Failed to fetch documents");
    return res.json();
  },

  /**
   * Delete a document by ID.
   */
  async deleteDocument(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete document");
    return res.json();
  },

  /**
   * Clear a chat session.
   */
  async clearSession(sessionId) {
    const res = await fetch(`${API_BASE}/chat/${sessionId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to clear session");
    return res.json();
  },

  /**
   * Health check.
   */
  async health() {
    const res = await fetch("http://localhost:8000/health");
    if (!res.ok) throw new Error("Backend unhealthy");
    return res.json();
  },
};
