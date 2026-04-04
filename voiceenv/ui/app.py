"""
VoiceEnv Rating UI — a simple web app for community judge validation.

Run with:
  voiceenv judge serve
  # or directly:
  python -m voiceenv.ui.app

Then open http://localhost:8910 in your browser.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voiceenv.core.human_ratings import HumanRating, RatingStore
from voiceenv.core.judge_correlation import compute_correlation, format_correlation_report

app = FastAPI(title="VoiceEnv Community Rating")

RATINGS_DIR = "ratings"
_store: RatingStore | None = None


def get_store() -> RatingStore:
    global _store
    if _store is None:
        _store = RatingStore(RATINGS_DIR)
    return _store


def configure(ratings_dir: str):
    global RATINGS_DIR, _store
    RATINGS_DIR = ratings_dir
    _store = RatingStore(ratings_dir)


# ── API Routes ──


@app.get("/api/runs")
def list_runs():
    store = get_store()
    runs = store.list_runs()
    run_list = []
    for run_id in runs:
        try:
            run = store.load_run_for_rating(run_id)
            existing_ratings = store.get_ratings_for_run(run_id)
            raters = set(r.rater_id for r in existing_ratings)
            run_list.append({
                "run_id": run_id,
                "environment": run.environment_name,
                "turns": len(run.transcript),
                "criteria_count": len(run.criteria_to_rate),
                "ratings_count": len(existing_ratings),
                "raters": len(raters),
            })
        except Exception:
            pass
    return run_list


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    store = get_store()
    try:
        run = store.load_run_for_rating(run_id)
        return run.to_dict()
    except FileNotFoundError:
        return JSONResponse({"error": "Run not found"}, status_code=404)


@app.post("/api/runs/{run_id}/rate")
async def submit_rating(run_id: str, request: Request):
    store = get_store()
    body = await request.json()

    rater_id = body.get("rater_id", "anonymous")
    scores = body.get("scores", {})

    ratings = []
    for criterion_name, data in scores.items():
        ratings.append(HumanRating(
            run_id=run_id,
            criterion_name=criterion_name,
            rater_id=rater_id,
            score=float(data.get("score", 0.5)),
            reasoning=data.get("reasoning", ""),
            audio_listened=data.get("audio_listened", False),
        ))

    store.submit_ratings(ratings)

    # Return comparison with LLM scores
    try:
        run = store.load_run_for_rating(run_id)
        comparison = []
        for r in ratings:
            llm = run.llm_scores.get(r.criterion_name)
            comparison.append({
                "criterion": r.criterion_name,
                "your_score": r.score,
                "llm_score": llm,
                "delta": round(r.score - llm, 3) if llm is not None else None,
            })
        return {"submitted": len(ratings), "comparison": comparison}
    except Exception:
        return {"submitted": len(ratings), "comparison": []}


@app.get("/api/stats")
def get_stats():
    store = get_store()
    stats = store.get_rating_stats()
    return stats


@app.get("/api/correlation")
def get_correlation():
    store = get_store()
    all_ratings = store.load_all_ratings()

    if not all_ratings:
        return {"error": "No ratings yet", "total_ratings": 0}

    llm_scores: dict[str, dict[str, float]] = {}
    for run_id in store.list_runs():
        try:
            run = store.load_run_for_rating(run_id)
            if run.llm_scores:
                llm_scores[run_id] = run.llm_scores
        except Exception:
            pass

    if not llm_scores:
        return {"error": "No LLM scores available", "total_ratings": len(all_ratings)}

    report = compute_correlation(all_ratings, llm_scores)
    return report.to_dict()


# ── HTML Page ──


@app.get("/", response_class=HTMLResponse)
def home():
    return _render_page()


def _render_page() -> str:
    return PAGE_HTML


PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VoiceEnv — Help Improve Voice AI</title>
<style>
  :root {
    --bg: #fafafa; --surface: #fff; --text: #1a1a2e; --muted: #6b7280;
    --accent: #6366f1; --accent-light: #e0e7ff; --green: #10b981;
    --red: #ef4444; --yellow: #f59e0b; --border: #e5e7eb;
    --radius: 12px; --shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }
  .container { max-width: 720px; margin: 0 auto; padding: 20px; }

  /* Header */
  .header {
    text-align: center; padding: 40px 20px 30px;
  }
  .header h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
  .header p { color: var(--muted); font-size: 16px; max-width: 500px; margin: 0 auto; }

  /* Stats bar */
  .stats-bar {
    display: flex; gap: 16px; justify-content: center; margin: 24px 0;
    flex-wrap: wrap;
  }
  .stat {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 24px; text-align: center;
    box-shadow: var(--shadow); min-width: 120px;
  }
  .stat-value { font-size: 28px; font-weight: 700; color: var(--accent); }
  .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }

  /* Cards */
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; margin-bottom: 16px;
    box-shadow: var(--shadow); transition: transform 0.1s;
  }
  .card:hover { transform: translateY(-1px); }
  .card h3 { font-size: 16px; margin-bottom: 4px; }
  .card .meta { color: var(--muted); font-size: 13px; }

  /* Run list */
  .run-card { cursor: pointer; }
  .run-card .badge {
    display: inline-block; background: var(--accent-light); color: var(--accent);
    font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 99px;
  }

  /* Conversation */
  .conversation { margin: 20px 0; }
  .bubble {
    max-width: 85%; padding: 12px 16px; border-radius: 16px;
    margin-bottom: 8px; font-size: 14px; line-height: 1.5;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; } }
  .bubble.agent {
    background: var(--accent); color: white; margin-right: auto;
    border-bottom-left-radius: 4px;
  }
  .bubble.user {
    background: #f3f4f6; margin-left: auto;
    border-bottom-right-radius: 4px;
  }
  .bubble-role {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    margin-bottom: 4px; opacity: 0.7;
  }

  /* Rating form */
  .rating-section { margin: 24px 0; }
  .criterion-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; margin-bottom: 12px;
  }
  .criterion-name { font-weight: 600; font-size: 15px; margin-bottom: 4px; }
  .criterion-desc { color: var(--muted); font-size: 13px; margin-bottom: 16px; }

  /* Emoji rating */
  .emoji-row {
    display: flex; gap: 8px; justify-content: center; margin: 12px 0;
  }
  .emoji-btn {
    font-size: 32px; cursor: pointer; padding: 8px 12px;
    border: 2px solid transparent; border-radius: 12px;
    background: none; transition: all 0.15s;
    filter: grayscale(0.6); opacity: 0.6;
  }
  .emoji-btn:hover { filter: grayscale(0); opacity: 1; transform: scale(1.15); }
  .emoji-btn.selected {
    filter: grayscale(0); opacity: 1; transform: scale(1.2);
    border-color: var(--accent); background: var(--accent-light);
  }
  .emoji-label {
    display: flex; justify-content: space-between;
    font-size: 11px; color: var(--muted); padding: 0 8px;
  }

  /* Reasoning textarea */
  .reasoning {
    width: 100%; border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 12px; font-size: 13px; font-family: inherit;
    resize: vertical; min-height: 40px; margin-top: 8px;
  }
  .reasoning:focus { outline: none; border-color: var(--accent); }

  /* Buttons */
  .btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 12px 28px; border-radius: 99px; border: none;
    font-size: 15px; font-weight: 600; cursor: pointer;
    transition: all 0.15s;
  }
  .btn-primary {
    background: var(--accent); color: white;
  }
  .btn-primary:hover { background: #4f46e5; transform: translateY(-1px); }
  .btn-primary:disabled { background: #c7c7c7; cursor: not-allowed; transform: none; }
  .btn-secondary {
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { background: #f3f4f6; }

  /* Comparison results */
  .comparison { margin: 24px 0; }
  .comparison-row {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
  }
  .comparison-row:last-child { border-bottom: none; }
  .comparison-name { flex: 1; font-weight: 500; }
  .score-pill {
    padding: 4px 12px; border-radius: 99px; font-size: 13px; font-weight: 600;
  }
  .score-you { background: var(--accent-light); color: var(--accent); }
  .score-ai { background: #f3f4f6; color: var(--muted); }
  .delta-good { color: var(--green); }
  .delta-ok { color: var(--yellow); }
  .delta-bad { color: var(--red); }

  /* Correlation dashboard */
  .corr-bar {
    height: 8px; border-radius: 4px; background: #e5e7eb; overflow: hidden;
    margin: 8px 0;
  }
  .corr-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .corr-high { background: var(--green); }
  .corr-mod { background: var(--yellow); }
  .corr-low { background: var(--red); }

  /* Name input */
  .name-input {
    border: 2px solid var(--border); border-radius: 99px;
    padding: 10px 20px; font-size: 15px; width: 100%;
    max-width: 300px; text-align: center; font-family: inherit;
  }
  .name-input:focus { outline: none; border-color: var(--accent); }

  /* Sections */
  .section-title {
    font-size: 20px; font-weight: 700; margin: 32px 0 16px;
  }

  .text-center { text-align: center; }
  .mt-16 { margin-top: 16px; }
  .mt-24 { margin-top: 24px; }
  .hidden { display: none; }

  /* Tabs */
  .tabs {
    display: flex; gap: 0; border-bottom: 2px solid var(--border);
    margin-bottom: 24px;
  }
  .tab {
    padding: 10px 20px; font-size: 14px; font-weight: 600;
    cursor: pointer; border-bottom: 2px solid transparent;
    margin-bottom: -2px; color: var(--muted); transition: all 0.15s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .empty-state {
    text-align: center; padding: 60px 20px; color: var(--muted);
  }
  .empty-state .icon { font-size: 48px; margin-bottom: 16px; }
</style>
</head>
<body>

<div class="header">
  <h1>Help Improve Voice AI</h1>
  <p>Listen to AI agent conversations, rate how well they did, and help us build better voice AI for everyone.</p>
</div>

<div class="container">

  <!-- Stats -->
  <div class="stats-bar" id="stats-bar"></div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" onclick="showTab('rate')">Rate Conversations</div>
    <div class="tab" onclick="showTab('dashboard')">Leaderboard</div>
  </div>

  <!-- Tab: Rate -->
  <div id="tab-rate">

    <!-- Step 1: Name -->
    <div id="step-name" class="text-center">
      <p style="margin-bottom: 16px; color: var(--muted);">First, tell us who you are:</p>
      <input class="name-input" id="rater-name" placeholder="Your name or nickname" />
      <div class="mt-16">
        <button class="btn btn-primary" onclick="setName()">Get Started</button>
      </div>
    </div>

    <!-- Step 2: Pick a run -->
    <div id="step-pick" class="hidden">
      <p class="section-title">Pick a conversation to rate</p>
      <div id="run-list"></div>
    </div>

    <!-- Step 3: Rate -->
    <div id="step-rate" class="hidden">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <p class="section-title" id="rate-title"></p>
        <button class="btn btn-secondary" onclick="backToList()">Back</button>
      </div>
      <div class="conversation" id="conversation"></div>
      <p class="section-title">Your Ratings</p>
      <div id="criteria-list"></div>
      <div class="text-center mt-24">
        <button class="btn btn-primary" id="submit-btn" onclick="submitRatings()" disabled>
          Submit Ratings
        </button>
      </div>
    </div>

    <!-- Step 4: Results -->
    <div id="step-results" class="hidden">
      <div class="text-center">
        <div style="font-size: 48px; margin-bottom: 8px;">&#10024;</div>
        <p class="section-title">Thank you!</p>
        <p style="color: var(--muted);">Here's how your ratings compare to the AI judge:</p>
      </div>
      <div class="comparison card" id="comparison-results"></div>
      <div class="text-center mt-24">
        <button class="btn btn-primary" onclick="backToList()">Rate Another</button>
      </div>
    </div>

  </div>

  <!-- Tab: Dashboard -->
  <div id="tab-dashboard" class="hidden">
    <div id="correlation-dashboard"></div>
  </div>

</div>

<script>
let raterId = '';
let currentRun = null;
let ratings = {};

// ── Tab switching ──
function showTab(tab) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', t.textContent.includes(tab === 'rate' ? 'Rate' : 'Leaderboard'));
  });
  document.getElementById('tab-rate').classList.toggle('hidden', tab !== 'rate');
  document.getElementById('tab-dashboard').classList.toggle('hidden', tab !== 'dashboard');
  if (tab === 'dashboard') loadCorrelation();
}

// ── Load stats ──
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const s = await res.json();
    document.getElementById('stats-bar').innerHTML = `
      <div class="stat"><div class="stat-value">${s.total_ratings || 0}</div><div class="stat-label">Ratings</div></div>
      <div class="stat"><div class="stat-value">${s.unique_raters || 0}</div><div class="stat-label">Raters</div></div>
      <div class="stat"><div class="stat-value">${s.unique_runs_rated || 0}</div><div class="stat-label">Conversations</div></div>
    `;
  } catch(e) {
    document.getElementById('stats-bar').innerHTML = `
      <div class="stat"><div class="stat-value">0</div><div class="stat-label">Ratings</div></div>
      <div class="stat"><div class="stat-value">0</div><div class="stat-label">Raters</div></div>
      <div class="stat"><div class="stat-value">0</div><div class="stat-label">Conversations</div></div>
    `;
  }
}

// ── Step 1: Set name ──
function setName() {
  const name = document.getElementById('rater-name').value.trim();
  if (!name) { document.getElementById('rater-name').style.borderColor = 'var(--red)'; return; }
  raterId = name;
  document.getElementById('step-name').classList.add('hidden');
  document.getElementById('step-pick').classList.remove('hidden');
  loadRuns();
}

// ── Step 2: Load runs ──
async function loadRuns() {
  const res = await fetch('/api/runs');
  const runs = await res.json();
  const el = document.getElementById('run-list');

  if (!runs.length) {
    el.innerHTML = `<div class="empty-state"><div class="icon">&#128172;</div>
      <p>No conversations to rate yet.</p>
      <p style="font-size: 13px; margin-top: 8px;">Run <code>voiceenv judge save-run results.json</code> to add some.</p></div>`;
    return;
  }

  el.innerHTML = runs.map(r => `
    <div class="card run-card" onclick="loadRun('${r.run_id}')">
      <div style="display: flex; justify-content: space-between; align-items: start;">
        <div>
          <h3>${r.environment}</h3>
          <div class="meta">${r.turns} turns &middot; ${r.criteria_count} criteria to rate</div>
        </div>
        <div>
          <span class="badge">${r.ratings_count} ratings from ${r.raters} people</span>
        </div>
      </div>
    </div>
  `).join('');
}

// ── Step 3: Load and rate a run ──
async function loadRun(runId) {
  const res = await fetch('/api/runs/' + runId);
  currentRun = await res.json();
  ratings = {};

  document.getElementById('step-pick').classList.add('hidden');
  document.getElementById('step-rate').classList.remove('hidden');
  document.getElementById('rate-title').textContent = currentRun.environment_name;

  // Render conversation
  const conv = document.getElementById('conversation');
  conv.innerHTML = currentRun.transcript.map((t, i) => `
    <div class="bubble ${t.role}" style="animation-delay: ${i * 0.05}s">
      <div class="bubble-role">${t.role === 'agent' ? 'AI Agent' : 'Caller'}</div>
      ${t.content}
    </div>
  `).join('');

  // Render criteria
  const emojis = [
    { emoji: '&#128078;', label: 'Bad', value: 0.0 },
    { emoji: '&#128533;', label: 'Poor', value: 0.25 },
    { emoji: '&#128528;', label: 'OK', value: 0.5 },
    { emoji: '&#128522;', label: 'Good', value: 0.75 },
    { emoji: '&#11088;', label: 'Great', value: 1.0 },
  ];

  const cl = document.getElementById('criteria-list');
  cl.innerHTML = currentRun.criteria_to_rate.map((c, idx) => `
    <div class="criterion-card">
      <div class="criterion-name">${c.name.replace(/_/g, ' ')}</div>
      <div class="criterion-desc">${c.description || ''}</div>
      <div class="emoji-row">
        ${emojis.map(e => `
          <button class="emoji-btn" data-criterion="${c.name}" data-value="${e.value}"
                  onclick="rate('${c.name}', ${e.value}, this)"
                  title="${e.label}">${e.emoji}</button>
        `).join('')}
      </div>
      <div class="emoji-label"><span>Bad</span><span>Great</span></div>
      <textarea class="reasoning" data-criterion="${c.name}"
                placeholder="Why? (optional, a few words is fine)"></textarea>
    </div>
  `).join('');

  checkSubmittable();
}

function rate(criterion, value, btn) {
  // Deselect siblings
  btn.parentElement.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  ratings[criterion] = value;
  checkSubmittable();
}

function checkSubmittable() {
  const needed = currentRun ? currentRun.criteria_to_rate.length : 0;
  const have = Object.keys(ratings).length;
  document.getElementById('submit-btn').disabled = have < needed;
}

// ── Step 4: Submit ──
async function submitRatings() {
  const scores = {};
  for (const [name, score] of Object.entries(ratings)) {
    const textarea = document.querySelector(`textarea[data-criterion="${name}"]`);
    scores[name] = {
      score: score,
      reasoning: textarea ? textarea.value : '',
    };
  }

  const res = await fetch('/api/runs/' + currentRun.run_id + '/rate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rater_id: raterId, scores }),
  });
  const result = await res.json();

  document.getElementById('step-rate').classList.add('hidden');
  document.getElementById('step-results').classList.remove('hidden');

  const el = document.getElementById('comparison-results');
  if (result.comparison && result.comparison.length) {
    el.innerHTML = result.comparison.map(c => {
      const deltaAbs = c.delta !== null ? Math.abs(c.delta) : null;
      const deltaClass = deltaAbs === null ? '' : deltaAbs < 0.2 ? 'delta-good' : deltaAbs < 0.4 ? 'delta-ok' : 'delta-bad';
      const deltaText = c.delta !== null ? (c.delta >= 0 ? '+' : '') + c.delta.toFixed(2) : 'n/a';
      const match = deltaAbs !== null && deltaAbs < 0.2 ? '&#9989;' : deltaAbs !== null && deltaAbs < 0.4 ? '&#128993;' : '&#128308;';
      return `
        <div class="comparison-row">
          <span class="comparison-name">${c.criterion.replace(/_/g, ' ')}</span>
          <span class="score-pill score-you">${c.your_score.toFixed(2)} you</span>
          <span class="score-pill score-ai">${c.llm_score !== null ? c.llm_score.toFixed(2) : '?'} AI</span>
          <span class="${deltaClass}">${match} ${deltaText}</span>
        </div>`;
    }).join('');
  } else {
    el.innerHTML = '<p style="text-align: center; padding: 20px; color: var(--muted);">No AI scores to compare against.</p>';
  }

  loadStats();
}

function backToList() {
  document.getElementById('step-rate').classList.add('hidden');
  document.getElementById('step-results').classList.add('hidden');
  document.getElementById('step-pick').classList.remove('hidden');
  currentRun = null;
  ratings = {};
  loadRuns();
}

// ── Dashboard ──
async function loadCorrelation() {
  const el = document.getElementById('correlation-dashboard');
  try {
    const res = await fetch('/api/correlation');
    const data = await res.json();

    if (data.error) {
      el.innerHTML = `<div class="empty-state"><div class="icon">&#128202;</div>
        <p>${data.error}</p>
        <p style="font-size: 13px; margin-top: 8px;">Rate some conversations first to see how the AI judge compares to humans.</p></div>`;
      return;
    }

    const overall = data.overall_pearson;
    const overallPct = overall !== null ? Math.round(Math.abs(overall) * 100) : 0;
    const overallClass = overall !== null ? (Math.abs(overall) >= 0.7 ? 'corr-high' : Math.abs(overall) >= 0.4 ? 'corr-mod' : 'corr-low') : 'corr-low';

    let html = `
      <div class="card text-center">
        <div style="font-size: 14px; color: var(--muted); margin-bottom: 4px;">Overall Human-AI Agreement</div>
        <div style="font-size: 42px; font-weight: 700; color: var(--accent);">${overall !== null ? (overall * 100).toFixed(0) + '%' : 'N/A'}</div>
        <div class="corr-bar"><div class="corr-fill ${overallClass}" style="width: ${overallPct}%"></div></div>
        <div style="font-size: 12px; color: var(--muted);">${data.total_comparisons} comparisons from ${data.total_raters} raters</div>
      </div>`;

    if (data.criteria && data.criteria.length) {
      html += '<p class="section-title">Per-Criterion Breakdown</p>';
      for (const c of data.criteria) {
        const pct = c.pearson_r !== null ? Math.round(Math.abs(c.pearson_r) * 100) : 0;
        const cls = c.pearson_r !== null ? (Math.abs(c.pearson_r) >= 0.7 ? 'corr-high' : Math.abs(c.pearson_r) >= 0.4 ? 'corr-mod' : 'corr-low') : 'corr-low';
        const statusEmoji = c.status === 'high' ? '&#9989;' : c.status === 'moderate' ? '&#128993;' : c.status === 'low' ? '&#128308;' : '&#9898;';
        html += `
          <div class="card" style="padding: 16px 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <div>
                <span style="font-weight: 600;">${c.criterion.replace(/_/g, ' ')}</span>
                <span style="font-size: 12px; color: var(--muted); margin-left: 8px;">${c.n_comparisons} ratings</span>
              </div>
              <span>${statusEmoji} ${c.pearson_r !== null ? (c.pearson_r * 100).toFixed(0) + '%' : 'need data'}</span>
            </div>
            <div class="corr-bar"><div class="corr-fill ${cls}" style="width: ${pct}%"></div></div>
          </div>`;
      }
    }

    if (data.flagged_criteria && data.flagged_criteria.length) {
      html += `<div class="card" style="border-color: var(--red); background: #fef2f2;">
        <div style="font-weight: 600; color: var(--red); margin-bottom: 8px;">Needs Improvement</div>
        <p style="font-size: 13px; color: var(--muted);">These criteria have low agreement between humans and the AI judge. They need better expert reference recordings.</p>
        <ul style="margin-top: 8px; padding-left: 20px;">${data.flagged_criteria.map(c => `<li>${c.replace(/_/g, ' ')}</li>`).join('')}</ul>
      </div>`;
    }

    el.innerHTML = html;
  } catch(e) {
    el.innerHTML = `<div class="empty-state"><div class="icon">&#128202;</div><p>Could not load correlation data.</p></div>`;
  }
}

// ── Init ──
loadStats();

// Enter key on name input
document.getElementById('rater-name').addEventListener('keydown', e => {
  if (e.key === 'Enter') setName();
});
</script>
</body>
</html>"""


def run_server(host: str = "0.0.0.0", port: int = 8910, ratings_dir: str = "ratings"):
    """Start the rating UI server."""
    import uvicorn
    configure(ratings_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
