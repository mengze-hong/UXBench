// UXBench Leaderboard — Data & Logic
// All numbers aligned with tab_main_results.tex
// 26 frontier LLMs + Pointwise GRM (our trained model)
// Test set: Task1 N=2000 (1000 Good + 1000 Bad), Task2 N=4900, Task3 N=500

// ── Task 1: UX Judge (Avg-Acc = (Good-Acc + Bad-Acc) / 2) ──────────────────
// 26 frontier LLMs + Pointwise GRM (our trained model, shown at top)
const TASK1_DATA = [
  { rank:1,  model:"Pointwise GRM (Hunyuan 3)", org:"Ours",      good:82.1, bad:72.4, avg:77.2, trained:true,  date:"2026-05" },
  { rank:2,  model:"Claude Opus 4.7",      org:"Anthropic", good:89.1, bad:61.5, avg:75.3, date:"2026-05" },
  { rank:3,  model:"GPT-5.2",              org:"OpenAI",    good:85.0, bad:65.1, avg:75.0, date:"2026-05" },
  { rank:4,  model:"GPT-5.5",              org:"OpenAI",    good:92.7, bad:55.7, avg:74.2, date:"2026-05" },
  { rank:5,  model:"GPT-5",               org:"OpenAI",    good:89.5, bad:56.2, avg:72.9, date:"2026-05" },
  { rank:6,  model:"GPT-5.1",             org:"OpenAI",    good:94.8, bad:50.1, avg:72.5, date:"2026-05" },
  { rank:7,  model:"Claude Opus 4.6",     org:"Anthropic", good:92.6, bad:51.5, avg:72.0, date:"2026-05" },
  { rank:8,  model:"Gemini 3.1 Pro",      org:"Google",    good:91.6, bad:49.3, avg:70.4, date:"2026-05" },
  { rank:9,  model:"Claude Sonnet 4.5",   org:"Anthropic", good:89.6, bad:49.0, avg:69.3, date:"2026-05" },
  { rank:10, model:"Kimi K2.6",           org:"Moonshot",  good:96.1, bad:41.2, avg:68.7, date:"2026-05" },
  { rank:11, model:"GLM-5.1",             org:"Zhipu AI",  good:96.1, bad:40.9, avg:68.5, date:"2026-05" },
  { rank:12, model:"Gemini 3.0 Flash",    org:"Google",    good:97.7, bad:37.1, avg:67.4, date:"2026-05" },
  { rank:13, model:"Claude Opus 4.5",     org:"Anthropic", good:96.7, bad:36.7, avg:66.7, date:"2026-05" },
  { rank:14, model:"GLM-5",              org:"Zhipu AI",  good:96.9, bad:36.4, avg:66.7, date:"2026-05" },
  { rank:15, model:"Qwen3.6-Plus",        org:"Alibaba",   good:96.8, bad:34.9, avg:65.8, date:"2026-05" },
  { rank:16, model:"GPT-5 mini",          org:"OpenAI",    good:93.4, bad:36.9, avg:65.2, date:"2026-05" },
  { rank:17, model:"Kimi K2.5",           org:"Moonshot",  good:96.8, bad:32.7, avg:64.8, date:"2026-05" },
  { rank:18, model:"DeepSeek V4 Pro",     org:"DeepSeek",  good:97.4, bad:31.7, avg:64.5, date:"2026-05" },
  { rank:19, model:"DeepSeek V3.2",       org:"DeepSeek",  good:95.7, bad:33.3, avg:64.5, date:"2026-05" },
  { rank:20, model:"Hunyuan 3",           org:"Tencent",   good:95.6, bad:33.1, avg:64.3, date:"2026-05" },
  { rank:21, model:"Gemini 2.5 Pro",      org:"Google",    good:96.8, bad:28.7, avg:62.8, date:"2026-05" },
  { rank:22, model:"Doubao Seed 2.0 Pro", org:"ByteDance", good:98.8, bad:22.9, avg:60.8, date:"2026-05" },
  { rank:23, model:"Gemini 2.5 Flash",    org:"Google",    good:97.5, bad:21.2, avg:59.4, date:"2026-05" },
  { rank:24, model:"DeepSeek R1",         org:"DeepSeek",  good:98.7, bad:18.3, avg:58.5, date:"2026-05" },
  { rank:25, model:"Doubao Seed 1.6",     org:"ByteDance", good:99.1, bad:16.2, avg:57.6, date:"2026-05" },
  { rank:26, model:"Doubao Seed 2.0 Lite",org:"ByteDance", good:98.7, bad:16.0, avg:57.4, date:"2026-05" },
  { rank:27, model:"DeepSeek V3",         org:"DeepSeek",  good:99.7, bad:11.6, avg:55.6, date:"2026-05" },
];

// ── Task 2: UX Eval  (Good% = fraction rated ≥ threshold, ranked desc) ──────
const TASK2_DATA = [
  { rank:1,  model:"Gemini 3.1 Pro",      org:"Google",    good:57.1, date:"2026-05" },
  { rank:2,  model:"GLM-5.1",             org:"Zhipu AI",  good:56.6, date:"2026-05" },
  { rank:3,  model:"GLM-5",              org:"Zhipu AI",  good:53.0, date:"2026-05" },
  { rank:4,  model:"Gemini 3.0 Flash",    org:"Google",    good:52.7, date:"2026-05" },
  { rank:5,  model:"Kimi K2.6",           org:"Moonshot",  good:52.3, date:"2026-05" },
  { rank:6,  model:"Qwen3.6-Plus",        org:"Alibaba",   good:52.3, date:"2026-05" },
  { rank:7,  model:"Gemini 2.5 Pro",      org:"Google",    good:50.8, date:"2026-05" },
  { rank:8,  model:"Kimi K2.5",           org:"Moonshot",  good:50.3, date:"2026-05" },
  { rank:9,  model:"DeepSeek V4 Pro",     org:"DeepSeek",  good:49.7, date:"2026-05" },
  { rank:10, model:"Hunyuan 3",           org:"Tencent",   good:48.8, date:"2026-05" },
  { rank:11, model:"Doubao Seed 2.0 Pro", org:"ByteDance", good:48.7, date:"2026-05" },
  { rank:12, model:"Doubao Seed 2.0 Lite",org:"ByteDance", good:46.3, date:"2026-05" },
  { rank:13, model:"Claude Opus 4.7",     org:"Anthropic", good:44.5, date:"2026-05" },
  { rank:14, model:"Claude Opus 4.6",     org:"Anthropic", good:44.3, date:"2026-05" },
  { rank:15, model:"GPT-5.5",             org:"OpenAI",    good:41.2, date:"2026-05" },
  { rank:16, model:"DeepSeek V3.2",       org:"DeepSeek",  good:41.2, date:"2026-05" },
  { rank:17, model:"DeepSeek R1",         org:"DeepSeek",  good:39.5, date:"2026-05" },
  { rank:18, model:"Claude Opus 4.5",     org:"Anthropic", good:37.9, date:"2026-05" },
  { rank:19, model:"GPT-5.1",             org:"OpenAI",    good:37.1, date:"2026-05" },
  { rank:20, model:"Doubao Seed 1.6",     org:"ByteDance", good:36.8, date:"2026-05" },
  { rank:21, model:"Gemini 2.5 Flash",    org:"Google",    good:36.1, date:"2026-05" },
  { rank:22, model:"Claude Sonnet 4.5",   org:"Anthropic", good:36.0, date:"2026-05" },
  { rank:23, model:"DeepSeek V3",         org:"DeepSeek",  good:35.9, date:"2026-05" },
  { rank:24, model:"GPT-5",              org:"OpenAI",    good:34.7, date:"2026-05" },
  { rank:25, model:"GPT-5.2",             org:"OpenAI",    good:30.8, date:"2026-05" },
  { rank:26, model:"GPT-5 mini",          org:"OpenAI",    good:24.0, date:"2026-05" },
];

// ── Task 3: UX Recovery (Good% = recovery rate, ranked desc) ────────────────
const TASK3_DATA = [
  { rank:1,  model:"Claude Opus 4.6",     org:"Anthropic", good:12.8, date:"2026-05" },
  { rank:2,  model:"Gemini 3.0 Flash",    org:"Google",    good:12.7, date:"2026-05" },
  { rank:3,  model:"Claude Opus 4.7",     org:"Anthropic", good:12.4, date:"2026-05" },
  { rank:4,  model:"Qwen3.6-Plus",        org:"Alibaba",   good:12.0, date:"2026-05" },
  { rank:5,  model:"Kimi K2.6",           org:"Moonshot",  good:11.4, date:"2026-05" },
  { rank:6,  model:"GLM-5.1",             org:"Zhipu AI",  good:11.2, date:"2026-05" },
  { rank:7,  model:"Kimi K2.5",           org:"Moonshot",  good:11.2, date:"2026-05" },
  { rank:8,  model:"DeepSeek V4 Pro",     org:"DeepSeek",  good:11.0, date:"2026-05" },
  { rank:9,  model:"Doubao Seed 2.0 Pro", org:"ByteDance", good:10.8, date:"2026-05" },
  { rank:10, model:"GLM-5",              org:"Zhipu AI",  good:10.4, date:"2026-05" },
  { rank:11, model:"Doubao Seed 2.0 Lite",org:"ByteDance", good:10.4, date:"2026-05" },
  { rank:12, model:"GPT-5.5",             org:"OpenAI",    good:9.9,  date:"2026-05" },
  { rank:13, model:"Claude Opus 4.5",     org:"Anthropic", good:9.4,  date:"2026-05" },
  { rank:14, model:"Gemini 3.1 Pro",      org:"Google",    good:9.2,  date:"2026-05" },
  { rank:15, model:"Claude Sonnet 4.5",   org:"Anthropic", good:9.0,  date:"2026-05" },
  { rank:16, model:"Hunyuan 3",           org:"Tencent",   good:7.6,  date:"2026-05" },
  { rank:17, model:"GPT-5.1",             org:"OpenAI",    good:7.4,  date:"2026-05" },
  { rank:18, model:"DeepSeek V3.2",       org:"DeepSeek",  good:7.0,  date:"2026-05" },
  { rank:19, model:"GPT-5",              org:"OpenAI",    good:6.9,  date:"2026-05" },
  { rank:20, model:"Doubao Seed 1.6",     org:"ByteDance", good:6.8,  date:"2026-05" },
  { rank:21, model:"DeepSeek R1",         org:"DeepSeek",  good:5.8,  date:"2026-05" },
  { rank:22, model:"GPT-5.2",             org:"OpenAI",    good:5.4,  date:"2026-05" },
  { rank:23, model:"Gemini 2.5 Flash",    org:"Google",    good:4.6,  date:"2026-05" },
  { rank:24, model:"Gemini 2.5 Pro",      org:"Google",    good:4.0,  date:"2026-05" },
  { rank:25, model:"GPT-5 mini",          org:"OpenAI",    good:3.6,  date:"2026-05" },
  { rank:26, model:"DeepSeek V3",         org:"DeepSeek",  good:3.6,  date:"2026-05" },
];

// ── Org color map ─────────────────────────────────────────────────────────────
const ORG_COLORS = {
  "Anthropic": "#d97706",
  "OpenAI":    "#10b981",
  "Google":    "#3b82f6",
  "Zhipu AI":  "#8b5cf6",
  "DeepSeek":  "#06b6d4",
  "ByteDance": "#ec4899",
  "Moonshot":  "#f59e0b",
  "Alibaba":   "#ef4444",
  "Tencent":   "#6366f1",
  "Ours":      "#4f46e5",
};

function getOrgColor(org) {
  return ORG_COLORS[org] || "#64748b";
}

// ── Medal helper ──────────────────────────────────────────────────────────────
function medal(rank) {
  if (rank === 1) return '<span class="medal">🥇</span>';
  if (rank === 2) return '<span class="medal">🥈</span>';
  if (rank === 3) return '<span class="medal">🥉</span>';
  return `<span class="rank-num">${rank}</span>`;
}

// ── Model cell helper ─────────────────────────────────────────────────────────
function modelCell(d) {
  const orgColor = getOrgColor(d.org);
  const trainedBadge = d.trained
    ? `<span class="trained-badge">Ours</span>`
    : `<span class="org-tag" style="background:${orgColor}20;color:${orgColor}">${d.org}</span>`;
  return `<td class="model-cell">
    <span class="model-name">${d.model}</span>
    ${trainedBadge}
  </td>`;
}

// ── Bar cell ──────────────────────────────────────────────────────────────────
function barCell(value, max, color) {
  const pct = Math.min(Math.round((value / max) * 100), 100);
  return `<td class="bar-cell">
    <div class="bar-wrap">
      <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
      <span class="bar-label">${value.toFixed(1)}%</span>
    </div>
  </td>`;
}

// ── Badge helper ──────────────────────────────────────────────────────────────
function avgBadge(val, hi, mid) {
  const cls = val >= hi ? 'high' : val >= mid ? 'mid' : 'low';
  return `<span class="avg-badge ${cls}">${val.toFixed(1)}%</span>`;
}

// ── Render Task 1 ─────────────────────────────────────────────────────────────
function renderTask1(data) {
  const tbody = document.getElementById("t1-tbody");
  if (!tbody) return;
  tbody.innerHTML = data.map(d => `
    <tr class="lb-row ${d.rank <= 3 ? 'top3' : ''} ${d.trained ? 'trained-row' : ''}">
      <td class="rank-cell">${medal(d.rank)}</td>
      ${modelCell(d)}
      ${barCell(d.good, 100, "#10b981")}
      ${barCell(d.bad, 100, "#ef4444")}
      <td class="avg-cell">${avgBadge(d.avg, 70, 65)}</td>
    </tr>`).join("");
}

// ── Render Task 2 ─────────────────────────────────────────────────────────────
function renderTask2(data) {
  const tbody = document.getElementById("t2-tbody");
  if (!tbody) return;
  tbody.innerHTML = data.map(d => `
    <tr class="lb-row ${d.rank <= 3 ? 'top3' : ''}">
      <td class="rank-cell">${medal(d.rank)}</td>
      ${modelCell(d)}
      <td class="avg-cell">${avgBadge(d.good, 50, 40)}</td>
    </tr>`).join("");
}

// ── Render Task 3 ─────────────────────────────────────────────────────────────
function renderTask3(data) {
  const tbody = document.getElementById("t3-tbody");
  if (!tbody) return;
  tbody.innerHTML = data.map(d => `
    <tr class="lb-row ${d.rank <= 3 ? 'top3' : ''}">
      <td class="rank-cell">${medal(d.rank)}</td>
      ${modelCell(d)}
      <td class="avg-cell">${avgBadge(d.good, 10, 7)}</td>
    </tr>`).join("");
}

// ── Sort ──────────────────────────────────────────────────────────────────────
let t1SortKey = "avg",  t1SortAsc = false;
let t2SortKey = "good", t2SortAsc = false;
let t3SortKey = "good", t3SortAsc = false;

function sortTable(task, key) {
  if (task === 1) {
    if (t1SortKey === key) t1SortAsc = !t1SortAsc;
    else { t1SortKey = key; t1SortAsc = false; }
    renderTask1([...TASK1_DATA].sort((a,b) => t1SortAsc ? a[key]-b[key] : b[key]-a[key]));
  } else if (task === 2) {
    if (t2SortKey === key) t2SortAsc = !t2SortAsc;
    else { t2SortKey = key; t2SortAsc = false; }
    renderTask2([...TASK2_DATA].sort((a,b) => t2SortAsc ? a[key]-b[key] : b[key]-a[key]));
  } else {
    if (t3SortKey === key) t3SortAsc = !t3SortAsc;
    else { t3SortKey = key; t3SortAsc = false; }
    renderTask3([...TASK3_DATA].sort((a,b) => t3SortAsc ? a[key]-b[key] : b[key]-a[key]));
  }
  document.querySelectorAll(`#task${task}-table th.sortable`).forEach(th => {
    th.classList.toggle("active", th.dataset.key === key);
  });
}

// ── Filter ────────────────────────────────────────────────────────────────────
function filterTable(task, query) {
  const q = query.toLowerCase();
  const match = d => d.model.toLowerCase().includes(q) || d.org.toLowerCase().includes(q);
  if (task === 1) renderTask1(TASK1_DATA.filter(match));
  else if (task === 2) renderTask2(TASK2_DATA.filter(match));
  else renderTask3(TASK3_DATA.filter(match));
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(tab) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.getElementById(`tab-${tab}`).classList.add("active");
  document.getElementById(`panel-${tab}`).classList.add("active");
}
// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  renderTask1(TASK1_DATA);
  renderTask2(TASK2_DATA);
  renderTask3(TASK3_DATA);

  // sort headers (works for all 3 tables via data-task attribute)
  document.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const task = parseInt(th.closest("table").dataset.task);
      sortTable(task, th.dataset.key);
    });
  });

  // search boxes
  const searches = [
    { id: "search-t1", task: 1 },
    { id: "search-t2", task: 2 },
    { id: "search-t3", task: 3 },
  ];
  searches.forEach(({ id, task }) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", e => filterTable(task, e.target.value));
  });

  // ── Mobile nav hamburger ──────────────────────────────────────────
  const toggle = document.getElementById("nav-toggle");
  const drawer = document.getElementById("nav-drawer");
  if (toggle && drawer) {
    toggle.addEventListener("click", () => {
      const open = drawer.classList.toggle("open");
      toggle.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", open);
    });
    drawer.querySelectorAll("a").forEach(a => {
      a.addEventListener("click", () => {
        drawer.classList.remove("open");
        toggle.classList.remove("open");
        toggle.setAttribute("aria-expanded", false);
      });
    });
    document.addEventListener("click", e => {
      if (!toggle.contains(e.target) && !drawer.contains(e.target)) {
        drawer.classList.remove("open");
        toggle.classList.remove("open");
        toggle.setAttribute("aria-expanded", false);
      }
    });
  }
});
