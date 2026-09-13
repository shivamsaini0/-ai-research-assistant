/**
 * Evaluation Dashboard — JavaScript Controller
 */

let sampleCount = 0;

// ── Initialize ────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  addSample();  // Start with one sample row
});

// ── Sample Management ─────────────────────────────────────────────────────────

function addSample() {
  sampleCount++;
  const container = document.getElementById("samplesContainer");

  const row = document.createElement("div");
  row.className = "sample-row";
  row.id = `sample-${sampleCount}`;
  row.innerHTML = `
    <div class="sample-row-header">
      <span class="sample-num">Q${sampleCount}</span>
      <button class="sample-remove" onclick="removeSample(${sampleCount})" title="Remove">✕</button>
    </div>
    <span class="sample-label">Question *</span>
    <textarea class="sample-input" id="q-${sampleCount}" placeholder="e.g. What are the main findings?" rows="2"></textarea>
    <span class="sample-label">Ground Truth (optional — enables Context Recall)</span>
    <textarea class="sample-input" id="gt-${sampleCount}" placeholder="Expected answer..." rows="2"></textarea>
  `;
  container.appendChild(row);
}

function removeSample(id) {
  const row = document.getElementById(`sample-${id}`);
  if (row) row.remove();
}

function getSamples() {
  const rows = document.querySelectorAll(".sample-row");
  const samples = [];
  rows.forEach((row) => {
    const id = row.id.replace("sample-", "");
    const q = document.getElementById(`q-${id}`)?.value?.trim();
    const gt = document.getElementById(`gt-${id}`)?.value?.trim();
    if (q) samples.push({ question: q, ground_truth: gt || null });
  });
  return samples;
}

// ── Run Evaluations ───────────────────────────────────────────────────────────

async function runQuickEval() {
  setLoading(true, "Running 5 generic questions against your documents...");
  try {
    const data = await fetchEval(`${API_BASE}/evaluate/quick`, {});
    displayResults(data);
  } catch (err) {
    Toast.show(err.message, "error");
  } finally {
    setLoading(false);
  }
}

async function runCustomEval() {
  const samples = getSamples();
  if (!samples.length) {
    Toast.show("Add at least one question first.", "error");
    return;
  }

  setLoading(true, `Evaluating ${samples.length} question(s)...`);
  try {
    const data = await fetchEval(`${API_BASE}/evaluate`, { samples });
    displayResults(data);
  } catch (err) {
    Toast.show(err.message, "error");
  } finally {
    setLoading(false);
  }
}

async function fetchEval(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// ── Display Results ───────────────────────────────────────────────────────────

const METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"];

function displayResults(data) {
  const agg = data.aggregate || {};
  const perQ = data.per_question || [];

  // Hide empty state
  document.getElementById("evalEmpty").style.display = "none";

  // Update metric cards
  let totalScore = 0;
  let validMetrics = 0;

  METRIC_KEYS.forEach((key) => {
    const val = agg[key];
    const valEl = document.getElementById(`val-${key}`);
    const fillEl = document.getElementById(`fill-${key}`);
    const cardEl = document.getElementById(`card-${key}`);

    if (val !== undefined && val !== null) {
      valEl.className = "metric-value";
      valEl.textContent = (val * 100).toFixed(1) + "%";
      fillEl.style.width = (val * 100) + "%";
      cardEl.classList.add("scored");
      totalScore += val;
      validMetrics++;
    } else {
      valEl.className = "metric-value na";
      valEl.textContent = "N/A";
      fillEl.style.width = "0%";
    }
  });

  // Show summary bar
  const summaryEl = document.getElementById("evalSummary");
  summaryEl.style.display = "flex";

  const overallScore = validMetrics > 0 ? totalScore / validMetrics : 0;
  document.getElementById("overallScore").textContent = (overallScore * 100).toFixed(1) + "%";

  const noteText = data.note ? `<br/><em>${data.note}</em>` : "";
  document.getElementById("summaryMeta").innerHTML =
    `${data.num_samples} sample(s) evaluated · ${data.status}${noteText}`;

  // Breakdown table
  if (perQ.length) {
    document.getElementById("breakdownSection").style.display = "flex";
    const tbody = document.getElementById("breakdownBody");
    tbody.innerHTML = "";

    perQ.forEach((row, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td style="max-width:280px">${row.question}</td>
        ${scoreCell(row.faithfulness)}
        ${scoreCell(row.answer_relevancy)}
        ${scoreCell(row.context_precision)}
        ${scoreCell(row.context_recall)}
      `;
      tbody.appendChild(tr);
    });
  }
}

function scoreCell(val) {
  if (val === null || val === undefined) {
    return `<td class="score-cell score-na">N/A</td>`;
  }
  const pct = (val * 100).toFixed(1);
  const cls = val >= 0.7 ? "score-high" : val >= 0.4 ? "score-medium" : "score-low";
  return `<td class="score-cell ${cls}">${pct}%</td>`;
}

// ── UI Helpers ────────────────────────────────────────────────────────────────

function setLoading(loading, text = "Running evaluation...") {
  const statusEl = document.getElementById("evalStatus");
  const statusText = document.getElementById("evalStatusText");
  const quickBtn = document.getElementById("quickEvalBtn");
  const customBtn = document.getElementById("customEvalBtn");

  statusEl.style.display = loading ? "flex" : "none";
  statusText.textContent = text;
  quickBtn.disabled = loading;
  customBtn.disabled = loading;
}

// Reuse Toast from app.js (already loaded)
