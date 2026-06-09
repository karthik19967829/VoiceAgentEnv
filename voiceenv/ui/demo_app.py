"""
Single-page demo UI for VoiceEnv: end-to-end showcase of the full pipeline.

Stages streamed live via Server-Sent Events:
  1. pick a real WAV
  2. autonomous ingest        (whisper + LLM extract → VoiceEnvironment)
  3. slice caller audio       (per-turn clips for stateless eval)
  4. speech-LLM eval          (gpt-audio in parallel, side-by-side)
  5. verifiable scoring       (deterministic Python rubric)
  6. grounded judge           (Gemini multimodal vs expert WAV)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse


app = FastAPI(title="VoiceEnv Demo")

# Serve any audio file under the project root (we use absolute paths in messages).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from voiceenv.ui import showcase as _showcase  # noqa: E402


def _load_dotenv():
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


# ── Sample discovery ──


@app.get("/api/mode")
def api_mode():
    return {
        "showcase": _showcase.showcase_enabled(),
        "live": not _showcase.showcase_enabled(),
    }


@app.get("/api/samples")
def list_samples():
    """Return HVB sample WAVs available in the workspace."""
    samples = []
    sample_dir = PROJECT_ROOT / "hvb_samples" / "audio" / "agent"
    transcript_dir = PROJECT_ROOT / "hvb_samples" / "transcript"
    # Demo: only surface the curated sample we use on stage.
    ALLOWED_IDS = {"00d676d7058c49bb"}
    if sample_dir.exists():
        for wav in sorted(sample_dir.glob("*.wav")):
            sid = wav.stem
            if sid not in ALLOWED_IDS:
                continue
            tj = transcript_dir / f"{sid}.json"
            label = sid[:8]
            if tj.exists():
                try:
                    raw = json.loads(tj.read_text())
                    text = " ".join(t.get("human_transcript", "") for t in raw[:5] if t.get("speaker_role") == "caller")
                    if text:
                        label = f"{sid[:8]}  ·  {text[:60]}…"
                except Exception:
                    pass
            samples.append({"id": sid, "path": str(wav.relative_to(PROJECT_ROOT)), "label": label})
    return {"samples": samples}


# ── Audio passthrough (paths are relative to project root) ──


@app.get("/audio")
def serve_audio(path: str):
    full = (PROJECT_ROOT / path).resolve()
    if not str(full).startswith(str(PROJECT_ROOT)):
        return {"error": "forbidden"}
    if not full.exists():
        return {"error": "not found"}
    return FileResponse(str(full), media_type="audio/wav")


# ── Streaming pipeline (SSE) ──


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


# In-memory cache of the most-recent run per WAV so /api/improve can re-run
# eval+judge with an updated agent_system_prompt without re-doing ingest/slice.
_RUN_CACHE: dict[str, dict] = {}


@app.get("/api/run")
async def run_pipeline(
    wav: str = Query(..., description="Path to source WAV (relative to project root)"),
    model: str = Query("gpt-audio-mini"),
    max_turns: int = Query(6),
    grounded: bool = Query(True),
):
    """SSE stream that runs the full demo pipeline and emits per-stage updates."""

    if _showcase.showcase_enabled():
        async def gen_showcase() -> AsyncGenerator[bytes, None]:
            async for chunk in _showcase.replay_sse("run_events.json"):
                yield chunk

        return StreamingResponse(gen_showcase(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            from voiceenv.ingest.from_call import (
                _load_hvb_transcript,
                _whisper_transcribe,
                _merge_consecutive,
                _llm_extract,
                _build_environment,
            )
            from voiceenv.ingest import ingest_call
            from voiceenv.demo import (
                slice_caller_turns,
                slice_human_responses,
                run_stateless_eval,
                run_grounded_eval,
            )

            wav_abs = (PROJECT_ROOT / wav).resolve()
            yield _sse("stage", {"stage": "start", "wav": str(wav_abs.relative_to(PROJECT_ROOT))})
            await asyncio.sleep(0)

            # ── 1. Ingest ──
            t0 = time.time()
            yield _sse("log", {"stage": "ingest", "msg": "Autonomous ingest: transcribing + extracting…"})
            await asyncio.sleep(0)

            ingest_out = PROJECT_ROOT / "environments" / f"auto_{wav_abs.stem[:8]}"
            result = await asyncio.to_thread(
                ingest_call,
                wav_path=wav_abs,
                output_dir=ingest_out,
                extraction_model="gpt-4o-mini",
                on_log=lambda m: None,
            )

            env = result.env
            ingest_seconds = time.time() - t0
            yield _sse("env", {
                "name": env.name,
                "description": env.description,
                "vertical": env.vertical.value,
                "difficulty": env.difficulty.value,
                "n_turns": result.n_turns,
                "duration_seconds": result.duration_seconds,
                "n_tools": len(env.tools),
                "n_criteria": len(env.rubric.all_criteria()),
                "ingest_cost_usd": result.cost_usd,
                "ingest_seconds": round(ingest_seconds, 1),
                "task_goal": env.task.goal,
                "persona": env.simulator.persona_description,
                "tools": [{"name": t.name, "description": t.description} for t in env.tools],
                "criteria": [
                    {"name": c.name, "category": cat, "check": c.deterministic_check}
                    for cat, lst in [
                        ("task_success", env.rubric.task_success),
                        ("compliance", env.rubric.compliance),
                        ("efficiency", env.rubric.efficiency),
                    ]
                    for c in lst
                ],
                "expert_audio_url": f"/audio?path={ingest_out.relative_to(PROJECT_ROOT)}/expert_reference/source_call.wav",
            })
            await asyncio.sleep(0)

            # ── 2. Slice caller audio ──
            yield _sse("log", {"stage": "slice", "msg": "Slicing real caller audio per turn…"})
            await asyncio.sleep(0)

            transcript_json = None
            for up in (1, 2, 3):
                try:
                    cand = wav_abs.parents[up] / "transcript" / (wav_abs.stem + ".json")
                    if cand.exists():
                        transcript_json = cand
                        break
                except IndexError:
                    break
            if transcript_json:
                all_turns = _load_hvb_transcript(transcript_json)
            else:
                all_turns = await asyncio.to_thread(_whisper_transcribe, wav_abs)
            all_turns = _merge_consecutive(all_turns)

            clips_dir = ingest_out / "caller_clips"
            clips = await asyncio.to_thread(
                slice_caller_turns, wav_abs, all_turns, clips_dir, max_turns
            )

            human_resp_dir = ingest_out / "human_response_clips"
            human_responses = await asyncio.to_thread(
                slice_human_responses, wav_abs, all_turns, clips, human_resp_dir,
            )

            yield _sse("clips", {
                "n_clips": len(clips),
                "clips": [
                    {
                        "turn_idx": c["turn_idx"],
                        "text": c["text"],
                        "audio_url": f"/audio?path={Path(c['audio_path']).relative_to(PROJECT_ROOT)}",
                        "duration_ms": c["duration_ms"],
                    }
                    for c in clips
                ],
            })
            await asyncio.sleep(0)

            # ── 3. Stateless speech-LLM eval ──
            yield _sse("log", {"stage": "eval", "msg": f"Running stateless eval with {model} (parallel)…"})
            await asyncio.sleep(0)

            t0 = time.time()
            ai_audio_dir = ingest_out / "ai_clips"
            results, cost = await asyncio.to_thread(
                run_stateless_eval,
                env, all_turns, clips, model, 4, lambda m: None,
                True,                # capture_audio
                ai_audio_dir,        # audio_out_dir
            )
            eval_seconds = time.time() - t0

            yield _sse("eval", {
                "model": model,
                "n_turns": len(results),
                "wall_seconds": round(eval_seconds, 2),
                "cost_usd": round(cost, 4),
                "turns": [
                    {
                        "turn_idx": r.turn_idx,
                        "caller_text": r.caller_text,
                        "caller_audio_url": f"/audio?path={Path(r.caller_audio_path).relative_to(PROJECT_ROOT)}",
                        "prior_context": [
                            {"role": "agent" if t.speaker == "agent" else "user", "text": t.text}
                            for t in all_turns[: r.turn_idx]
                        ],
                        "human_response": r.human_response,
                        "human_response_audio_url": (
                            f"/audio?path={Path(human_responses[r.turn_idx]['audio_path']).relative_to(PROJECT_ROOT)}"
                            if r.turn_idx in human_responses else None
                        ),
                        "ai_response": r.ai_response,
                        "ai_audio_url": (
                            f"/audio?path={Path(r.ai_audio_path).relative_to(PROJECT_ROOT)}"
                            if r.ai_audio_path else None
                        ),
                        "ai_tool_calls": r.ai_tool_calls,
                        "latency_ms": r.latency_ms,
                        "error": r.error,
                    }
                    for r in results
                ],
            })
            await asyncio.sleep(0)

            # ── Grounded judge ──
            gres = None
            expert_wav = ingest_out / env.expert_references[0].audio_path if env.expert_references else None
            if grounded and expert_wav and expert_wav.exists():
                yield _sse("log", {"stage": "grounded", "msg": "Grounded judge (Gemini, multimodal)…"})
                await asyncio.sleep(0)
                try:
                    gres = await asyncio.to_thread(
                        run_grounded_eval, env, results, str(expert_wav),
                        "gemini-2.5-flash", all_turns,
                    )
                    yield _sse("grounded", gres)
                except Exception as e:
                    yield _sse("log", {"stage": "grounded", "msg": f"grounded judge failed: {e}", "level": "error"})
            elif grounded:
                yield _sse("log", {"stage": "grounded", "msg": "no expert audio found", "level": "warn"})

            # Stash run state so /api/improve can re-use it.
            _RUN_CACHE[wav] = {
                "env": env,
                "all_turns": all_turns,
                "clips": clips,
                "human_responses": human_responses,
                "results_v1": results,
                "grounded_v1": gres,
                "expert_wav": str(expert_wav) if expert_wav else None,
                "ingest_out": ingest_out,
                "model": model,
            }

            yield _sse("done", {})
        except Exception as e:
            yield _sse("log", {"stage": "fatal", "msg": f"pipeline error: {e}", "level": "error"})
            yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.get("/api/improve")
async def improve_pipeline(wav: str = Query(...)):
    """SSE stream: use the grounded judge feedback (esp. human_likeness) to
    rewrite the agent's system prompt, then re-run eval + judge with it,
    and stream the deltas."""

    if _showcase.showcase_enabled():
        async def gen_showcase() -> AsyncGenerator[bytes, None]:
            async for chunk in _showcase.replay_sse("improve_events.json"):
                yield chunk

        return StreamingResponse(gen_showcase(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            if wav not in _RUN_CACHE:
                yield _sse("log", {"stage": "improve", "msg": "no cached run for this WAV — click 'Run demo' first", "level": "error"})
                yield _sse("done", {})
                return

            state = _RUN_CACHE[wav]
            env = state["env"]
            all_turns = state["all_turns"]
            clips = state["clips"]
            human_responses = state["human_responses"]
            results_v1 = state["results_v1"]
            grounded_v1 = state["grounded_v1"]
            expert_wav = state["expert_wav"]
            ingest_out = state["ingest_out"]
            model = state["model"]

            if grounded_v1 is None:
                yield _sse("log", {"stage": "improve", "msg": "need grounded judge result before improving", "level": "error"})
                yield _sse("done", {})
                return

            from voiceenv.demo import (
                run_stateless_eval, run_grounded_eval, improve_agent_prompt,
            )

            # ── 1. Improve prompt from judge feedback ──
            yield _sse("log", {"stage": "improve", "msg": "Rewriting agent system prompt from judge feedback…"})
            await asyncio.sleep(0)

            improvement = await asyncio.to_thread(
                improve_agent_prompt, env, results_v1, grounded_v1, all_turns, "gpt-4o-mini",
            )
            yield _sse("prompt_diff", improvement)
            await asyncio.sleep(0)

            # ── 2. Re-run stateless eval with the improved prompt ──
            yield _sse("log", {"stage": "improve", "msg": "Re-running stateless eval with improved prompt…"})
            await asyncio.sleep(0)

            ai_audio_dir_v2 = ingest_out / "ai_clips_v2"
            t0 = time.time()
            results_v2, cost_v2 = await asyncio.to_thread(
                run_stateless_eval,
                env, all_turns, clips, model, 4, lambda m: None,
                True,                      # capture_audio
                ai_audio_dir_v2,           # audio_out_dir
                "alloy",                   # voice
                improvement["improved_prompt"],  # override_system_prompt
            )
            eval_v2_seconds = time.time() - t0

            yield _sse("eval_v2", {
                "model": model,
                "n_turns": len(results_v2),
                "wall_seconds": round(eval_v2_seconds, 2),
                "cost_usd": round(cost_v2, 4),
                "turns": [
                    {
                        "turn_idx": r.turn_idx,
                        "caller_text": r.caller_text,
                        "ai_response_v1": next((x.ai_response for x in results_v1 if x.turn_idx == r.turn_idx), ""),
                        "ai_response_v2": r.ai_response,
                        "ai_audio_url_v2": (
                            f"/audio?path={Path(r.ai_audio_path).relative_to(PROJECT_ROOT)}"
                            if r.ai_audio_path else None
                        ),
                        "ai_tool_calls_v2": r.ai_tool_calls,
                        "error": r.error,
                    }
                    for r in results_v2
                ],
            })
            await asyncio.sleep(0)

            # ── 3. Re-run grounded judge ──
            if expert_wav:
                yield _sse("log", {"stage": "improve", "msg": "Re-judging with Gemini…"})
                await asyncio.sleep(0)
                try:
                    gres_v2 = await asyncio.to_thread(
                        run_grounded_eval, env, results_v2, expert_wav,
                        "gemini-2.5-flash", all_turns,
                    )
                    yield _sse("grounded_v2", {
                        "v1": grounded_v1,
                        "v2": gres_v2,
                    })
                except Exception as e:
                    yield _sse("log", {"stage": "improve", "msg": f"re-judge failed: {e}", "level": "error"})

            yield _sse("done", {})
        except Exception as e:
            import traceback; traceback.print_exc()
            yield _sse("log", {"stage": "improve", "msg": f"improve error: {e}", "level": "error"})
            yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# ── HTML ──


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VoiceEnv — Live Demo</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#21262d; --fg:#e6edf3; --dim:#8b949e;
          --accent:#58a6ff; --good:#3fb950; --warn:#d29922; --bad:#f85149; --magenta:#d2a8ff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system,Segoe UI,sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding: 22px 28px; border-bottom: 1px solid var(--line); background:#010409; }
  header h1 { margin:0; font-size:20px; letter-spacing:-0.01em; }
  header p { margin: 4px 0 0; color:var(--dim); font-size:13px; }
  main { padding: 22px 28px; max-width: 1200px; margin: 0 auto; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:18px 20px; margin-bottom:18px; }
  .card h2 { margin:0 0 12px; font-size:14px; color:var(--dim); text-transform:uppercase;
             letter-spacing: 0.06em; font-weight: 600; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap: wrap; }
  select, button { background:#0d1117; border:1px solid var(--line); color:var(--fg);
                   padding:9px 14px; border-radius:6px; font-size:13px; font-family:inherit; }
  select { min-width: 360px; }
  button { background:var(--accent); border-color:var(--accent); color:#000;
           font-weight:600; cursor:pointer; }
  button:hover { filter: brightness(1.1); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .stages { display:flex; gap:8px; flex-wrap:wrap; margin-top: 10px; }
  .stage { padding: 4px 10px; border-radius: 999px; font-size: 12px;
           background:#21262d; color:var(--dim); border: 1px solid var(--line); }
  .stage.active { background: rgba(88,166,255,0.15); color: var(--accent); border-color: var(--accent); }
  .stage.done { background: rgba(63,185,80,0.15); color: var(--good); border-color: var(--good); }
  .stage.error { background: rgba(248,81,73,0.15); color: var(--bad); border-color: var(--bad); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--line);
           vertical-align: top; }
  th { color:var(--dim); font-weight: 600; font-size: 11px; text-transform: uppercase;
       letter-spacing: 0.06em; }
  tr:last-child td { border-bottom: none; }
  audio { height: 28px; vertical-align: middle; }
  .turn-table td { font-size: 12.5px; line-height: 1.45; }
  .pass { color: var(--good); font-weight: 600; }
  .fail { color: var(--bad); font-weight: 600; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
          background: #21262d; color: var(--dim); margin-right: 4px; }
  .score { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
  .score-small { font-size: 13px; color: var(--dim); }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap: 12px; }
  .stat { background:#0d1117; border:1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
  .stat .k { font-size: 11px; color:var(--dim); text-transform: uppercase; letter-spacing: 0.06em; }
  .stat .v { font-size: 18px; font-weight: 600; margin-top: 2px; }
  .yaml { font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: var(--dim);
          white-space: pre-wrap; max-height: 240px; overflow-y: auto;
          background:#0d1117; padding: 12px; border-radius: 6px; border: 1px solid var(--line); }
  .reason { color: var(--dim); font-size: 12px; }
  .tool-call { font-family: ui-monospace, Menlo, monospace; font-size: 11.5px;
               color: var(--magenta); }
  .dim-score-5 { color: var(--good); font-weight:700; }
  .dim-score-4 { color: var(--good); }
  .dim-score-3 { color: var(--warn); }
  .dim-score-2 { color: var(--bad); }
  .dim-score-1 { color: var(--bad); font-weight:700; }
  .col-caller { width: 26%; } .col-human { width: 26%; } .col-ai { width: 38%; }
  .hidden { display: none; }
</style>
</head>
<body>
<header>
  <h1>VoiceEnv — Autonomous RL environment from a single WAV</h1>
  <p>Real call → ingest → speech-LLM eval → verifiable + grounded judging</p>
  <p id="showcase-banner" class="hidden" style="margin-top:10px;padding:8px 12px;border-radius:6px;
     background:rgba(88,166,255,0.12);border:1px solid var(--accent);font-size:12.5px;color:var(--fg);">
    <strong>Interactive replay</strong> — pre-recorded from a real banking call (no API keys).
    <a href="https://github.com/karthik19967829/VoiceAgentEnv" style="color:var(--accent);">GitHub</a>
  </p>
</header>

<main>
  <div class="card">
    <h2>1 · Pick a real call</h2>
    <div class="row">
      <select id="sample"></select>
      <label><input type="checkbox" id="grounded" checked> grounded judge (Gemini)</label>
      <button id="run">Run demo</button>
    </div>
    <div class="stages" id="stages"></div>
  </div>

  <div class="card hidden" id="env-card">
    <h2>2 · Auto-extracted environment</h2>
    <div class="grid" id="env-stats"></div>
    <div style="margin-top:14px;">
      <div class="score-small" style="margin-bottom:6px;">Original call (expert reference)</div>
      <audio id="expert-audio" controls></audio>
    </div>
    <details style="margin-top:14px;">
      <summary style="color:var(--dim);font-size:12px;cursor:pointer;">view extracted task / persona / tools / rubric</summary>
      <div class="yaml" id="env-detail"></div>
    </details>
  </div>

  <div class="card hidden" id="turns-card">
    <h2>3 · Per-turn comparison · real human caller / human agent / AI agent</h2>
    <div style="background:#0d1117;border:1px solid var(--line);border-radius:8px;
                padding:12px 14px;margin-bottom:14px;font-size:12.5px;line-height:1.55;
                color:var(--dim);">
      <span style="color:var(--accent);font-weight:600;">Stateless eval design.</span>
      Each row is an <em>independent</em> test case. The AI agent receives the
      <strong style="color:var(--fg);">same conversation prefix the human agent had at that moment</strong>
      (the original transcript up to that point) plus the caller's real audio
      for that turn. The AI generates a fresh response — never sees what the
      human said next, never conditions on its own prior turns. This is
      offline policy evaluation: same context in, two different actions out,
      side-by-side. Click <em>“show context”</em> on any row to see exactly
      what was sent.
    </div>
    <table class="turn-table"><thead><tr>
      <th class="col-caller">Caller (real audio in)</th>
      <th class="col-human">Human agent (Elizabeth)</th>
      <th class="col-ai">AI agent (<span id="model-name">gpt-audio-mini</span>)</th>
    </tr></thead><tbody id="turns-body"></tbody></table>
  </div>

  <div class="card hidden" id="grounded-card">
    <h2>4 · Grounded judge · Gemini 2.5 Flash, multimodal, anchored on real human call</h2>
    <div class="row" style="margin-bottom: 12px;">
      <div class="stat"><div class="k">Wall time (parallel)</div><div class="v" id="wall">—</div></div>
      <div class="stat"><div class="k">LLM cost</div><div class="v" id="cost">—</div></div>
    </div>
    <div class="row" style="margin-bottom: 12px;">
      <div class="stat"><div class="k">Average</div><div class="v score" id="gscore">—</div></div>
      <div class="score-small">Each dimension reasoned by listening to the original WAV.</div>
    </div>
    <table><thead><tr><th>Dimension</th><th>Score</th><th>Reasoning (vs human expert)</th></tr></thead>
      <tbody id="grounded-body"></tbody></table>
    <div style="margin-top:14px;padding:12px 14px;background:#0d1117;border:1px solid var(--line);
                border-radius:8px;font-size:12.5px;color:var(--dim);line-height:1.55;">
      <span style="color:var(--accent);font-weight:600;">Env-driven improvement.</span>
      The judge just told us where this policy is weak (especially
      <em>human_likeness</em>). Use that signal to rewrite the agent's system
      prompt and re-run — this is the same env → reward → policy-update loop
      RL training uses, demonstrated cheaply on the prompt.
      <div style="margin-top:10px;"><button id="improve-btn">Run env-driven prompt improvement</button></div>
    </div>
  </div>

  <div class="card hidden" id="improve-card">
    <h2>5 · Env-driven improvement · judge feedback → new system prompt → re-eval</h2>
    <div id="prompt-diff-block"></div>
    <div id="improve-stats" class="grid" style="margin: 14px 0;"></div>
    <div id="improve-dim-table"></div>
    <h3 style="font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:0.06em;
               margin: 18px 0 8px;">Per-turn diff · v1 vs v2 (same context, new prompt)</h3>
    <table class="turn-table"><thead><tr>
      <th style="width:30%;">Caller</th>
      <th style="width:35%;">AI v1 (original prompt)</th>
      <th style="width:35%;">AI v2 (improved prompt)</th>
    </tr></thead><tbody id="improve-turns-body"></tbody></table>
  </div>
</main>

<script>
const el = (id) => document.getElementById(id);

const STAGES = ["ingest","slice","eval","grounded","improve"];

let CURRENT_WAV = null;

async function loadSamples() {
  const r = await fetch("/api/samples"); const d = await r.json();
  const sel = el("sample");
  for (const s of d.samples) {
    const opt = document.createElement("option");
    opt.value = s.path; opt.textContent = s.label; sel.appendChild(opt);
  }
}
loadSamples();
fetch("/api/mode").then(r => r.json()).then(m => {
  if (m.showcase) el("showcase-banner").classList.remove("hidden");
});

function setStage(name, status) {
  let s = document.querySelector(`.stage[data-name="${name}"]`);
  if (!s) {
    s = document.createElement("div"); s.className="stage"; s.dataset.name=name; s.textContent=name;
    el("stages").appendChild(s);
  }
  s.className = "stage " + status;
}
function resetStages() {
  el("stages").innerHTML = "";
  for (const s of STAGES) setStage(s, "");
  ["env-card","turns-card","grounded-card","improve-card"].forEach(i => el(i).classList.add("hidden"));
}

el("run").onclick = () => {
  const wav = el("sample").value;
  const grounded = el("grounded").checked;
  if (!wav) return;
  CURRENT_WAV = wav;
  resetStages();
  el("run").disabled = true;
  setStage("ingest","active");
  const url = `/api/run?wav=${encodeURIComponent(wav)}&grounded=${grounded?1:0}`;
  const es = new EventSource(url);

  es.addEventListener("stage", (e) => {});
  es.addEventListener("log", (e) => {
    const d = JSON.parse(e.data);
    if (d.stage && STAGES.includes(d.stage)) setStage(d.stage, d.level==="error"?"error":"active");
  });

  es.addEventListener("env", (e) => {
    const d = JSON.parse(e.data);
    setStage("ingest", "done");
    el("env-card").classList.remove("hidden");
    el("env-stats").innerHTML = `
      <div class="stat"><div class="k">Name</div><div class="v">${d.name}</div></div>
      <div class="stat"><div class="k">Vertical</div><div class="v">${d.vertical}</div></div>
      <div class="stat"><div class="k">Source turns</div><div class="v">${d.n_turns}</div></div>
      <div class="stat"><div class="k">Duration</div><div class="v">${d.duration_seconds}s</div></div>
      <div class="stat"><div class="k">Tools extracted</div><div class="v">${d.n_tools}</div></div>
      <div class="stat"><div class="k">Rubric criteria</div><div class="v">${d.n_criteria}</div></div>
      <div class="stat"><div class="k">Ingest time</div><div class="v">${d.ingest_seconds}s</div></div>
      <div class="stat"><div class="k">Ingest cost</div><div class="v">$${d.ingest_cost_usd.toFixed(4)}</div></div>`;
    el("expert-audio").src = d.expert_audio_url;
    let detail = `goal: ${d.task_goal}\n\npersona: ${d.persona}\n\n`;
    detail += `tools:\n` + d.tools.map(t => `  - ${t.name}: ${t.description}`).join("\n");
    detail += `\n\nverifiable rubric:\n` + d.criteria.map(c => `  - [${c.category}] ${c.name}\n      check: ${c.check}`).join("\n");
    el("env-detail").textContent = detail;
  });

  el("turns-card").classList.add("hidden");

  es.addEventListener("clips", (e) => {
    const d = JSON.parse(e.data);
    setStage("slice", "done");
    el("turns-card").classList.remove("hidden");
    const body = el("turns-body");
    body.innerHTML = "";
    for (const c of d.clips) {
      const tr = document.createElement("tr");
      tr.id = `turn-${c.turn_idx}`;
      tr.innerHTML = `
        <td>
          <audio controls src="${c.audio_url}"></audio>
          <div class="reason" style="margin-top:4px;">"${c.text}"</div>
        </td>
        <td class="human-cell">—</td>
        <td class="ai-cell"><span class="reason">…running</span></td>`;
      body.appendChild(tr);
    }
  });

  es.addEventListener("eval", (e) => {
    const d = JSON.parse(e.data);
    setStage("eval", "done");
    el("model-name").textContent = d.model;
    el("wall").textContent = `${d.wall_seconds}s · ${d.n_turns} parallel`;
    el("cost").textContent = `$${d.cost_usd.toFixed(4)}`;
    for (const t of d.turns) {
      const tr = el(`turn-${t.turn_idx}`); if (!tr) continue;

      // Show what conversation prefix was sent (same for human & AI)
      const callerCell = tr.querySelector("td:first-child");
      const ctx = t.prior_context || [];
      let ctxHtml = "";
      if (ctx.length > 0) {
        const lines = ctx.map(c =>
          `<div style="font-size:11px;color:var(--dim);margin:1px 0;">` +
          `<span style="color:${c.role==='agent'?'var(--good)':'var(--accent)'};">${c.role==='agent'?'human agent':'caller'}:</span> ` +
          `${(c.text||'').replace(/</g,'&lt;')}</div>`
        ).join("");
        ctxHtml = `<details style="margin-top:8px;">
          <summary style="font-size:11px;color:var(--dim);cursor:pointer;">
            shared context shown to both human &amp; AI (${ctx.length} prior turn${ctx.length>1?'s':''})
          </summary>
          <div style="background:#010409;padding:8px 10px;border-radius:4px;margin-top:4px;border:1px solid var(--line);">${lines}</div>
        </details>`;
      } else {
        ctxHtml = `<div style="font-size:11px;color:var(--dim);margin-top:8px;font-style:italic;">cold start · no prior context (this is the first turn)</div>`;
      }
      callerCell.insertAdjacentHTML("beforeend", ctxHtml);

      const human = tr.querySelector(".human-cell");
      let humanHtml = "";
      if (t.human_response_audio_url) {
        humanHtml += `<audio controls src="${t.human_response_audio_url}" style="width:100%;margin-bottom:6px;"></audio>`;
      } else {
        humanHtml += `<div class="reason" style="color:var(--dim);margin-bottom:6px;">— end of call —</div>`;
      }
      humanHtml += `<div class="reason">"${(t.human_response||"—").replace(/</g,"&lt;")}"</div>`;
      human.innerHTML = humanHtml;
      const ai = tr.querySelector(".ai-cell");
      let html = "";
      if (t.ai_audio_url) {
        html += `<audio controls src="${t.ai_audio_url}" style="width:100%;margin-bottom:6px;"></audio>`;
      } else {
        html += `<div class="reason" style="color:var(--warn);margin-bottom:6px;">⚠ no audio captured</div>`;
      }
      const text = (t.ai_response || "").trim();
      if (text) {
        html += `<div>${text.replace(/</g,"&lt;")}</div>`;
      } else if (!t.ai_tool_calls || !t.ai_tool_calls.length) {
        html += `<div class="reason">(empty response)</div>`;
      }
      if (t.ai_tool_calls && t.ai_tool_calls.length) {
        for (const tc of t.ai_tool_calls) {
          html += `<div class="tool-call">→ ${tc.tool}(${JSON.stringify(tc.args)})</div>`;
        }
      }
      if (t.error) {
        html += `<div class="fail" style="margin-top:6px;font-size:11px;">✗ ${t.error}</div>`;
      }
      ai.innerHTML = html;
    }
  });

  es.addEventListener("grounded", (e) => {
    const d = JSON.parse(e.data);
    setStage("grounded", "done");
    el("grounded-card").classList.remove("hidden");
    el("gscore").textContent = `${d.average_score_1_5}/5`;
    const body = el("grounded-body"); body.innerHTML = "";
    for (const [name, x] of Object.entries(d.dimensions || {})) {
      body.insertAdjacentHTML("beforeend",
        `<tr><td>${name}</td><td class="dim-score-${x.score}">${x.score}/5</td>
         <td class="reason">${x.reasoning||""}</td></tr>`);
    }
  });

  es.addEventListener("done", () => { es.close(); el("run").disabled = false; });
  es.onerror = () => { es.close(); el("run").disabled = false; };
};

el("improve-btn").onclick = () => {
  if (!CURRENT_WAV) return;
  el("improve-btn").disabled = true;
  el("improve-btn").textContent = "Running improvement loop…";
  setStage("improve","active");
  el("improve-card").classList.remove("hidden");
  el("prompt-diff-block").innerHTML = `<div class="reason">Asking GPT-4o-mini to rewrite the system prompt based on the judge's feedback…</div>`;
  el("improve-stats").innerHTML = "";
  el("improve-dim-table").innerHTML = "";
  el("improve-turns-body").innerHTML = "";

  const es = new EventSource(`/api/improve?wav=${encodeURIComponent(CURRENT_WAV)}`);

  es.addEventListener("log", (e) => {
    const d = JSON.parse(e.data);
    if (d.level === "error") setStage("improve","error");
  });

  es.addEventListener("prompt_diff", (e) => {
    const d = JSON.parse(e.data);
    el("prompt-diff-block").innerHTML = `
      <div style="background:#0d1117;border:1px solid var(--line);border-radius:8px;
                  padding:10px 12px;font-size:12px;color:var(--accent);margin-bottom:10px;">
        <strong>What changed:</strong> <span style="color:var(--fg);font-weight:400;">${(d.changes_summary||"").replace(/</g,"&lt;")}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div>
          <div style="font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">v1 · original prompt</div>
          <div class="yaml" style="max-height:200px;">${(d.original_prompt||"").replace(/</g,"&lt;")}</div>
        </div>
        <div>
          <div style="font-size:11px;color:var(--good);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">v2 · improved prompt (from judge feedback)</div>
          <div class="yaml" style="max-height:200px;border-color:var(--good);">${(d.improved_prompt||"").replace(/</g,"&lt;")}</div>
        </div>
      </div>`;
  });

  es.addEventListener("eval_v2", (e) => {
    const d = JSON.parse(e.data);
    const body = el("improve-turns-body");
    for (const t of d.turns) {
      const v1 = (t.ai_response_v1 || "(empty)").replace(/</g,"&lt;");
      const v2 = (t.ai_response_v2 || "(empty)").replace(/</g,"&lt;");
      let v2Tools = "";
      if (t.ai_tool_calls_v2 && t.ai_tool_calls_v2.length) {
        v2Tools = t.ai_tool_calls_v2.map(tc =>
          `<div class="tool-call">→ ${tc.tool}(${JSON.stringify(tc.args)})</div>`).join("");
      }
      const v2Audio = t.ai_audio_url_v2
        ? `<audio controls src="${t.ai_audio_url_v2}" style="width:100%;margin-bottom:6px;"></audio>` : "";
      body.insertAdjacentHTML("beforeend",
        `<tr><td>${(t.caller_text||"").replace(/</g,"&lt;")}</td>
             <td style="color:var(--dim);">${v1}</td>
             <td>${v2Audio}<div>${v2}</div>${v2Tools}</td></tr>`);
    }
  });

  es.addEventListener("grounded_v2", (e) => {
    const d = JSON.parse(e.data);
    const v1 = d.v1, v2 = d.v2;
    const delta = (v2.average_score_1_5 - v1.average_score_1_5).toFixed(2);
    const deltaColor = delta >= 0 ? "var(--good)" : "var(--bad)";
    el("improve-stats").innerHTML = `
      <div class="stat"><div class="k">v1 average</div><div class="v">${v1.average_score_1_5}/5</div></div>
      <div class="stat"><div class="k">v2 average</div><div class="v" style="color:${deltaColor};">${v2.average_score_1_5}/5</div></div>
      <div class="stat"><div class="k">Δ</div><div class="v" style="color:${deltaColor};">${delta >= 0 ? "+" : ""}${delta}</div></div>`;
    const allDims = new Set([...Object.keys(v1.dimensions||{}), ...Object.keys(v2.dimensions||{})]);
    let html = `<table><thead><tr><th>Dimension</th><th>v1</th><th>v2</th><th>Δ</th><th>Reasoning (v2)</th></tr></thead><tbody>`;
    for (const name of allDims) {
      const s1 = v1.dimensions[name]?.score ?? 0;
      const s2 = v2.dimensions[name]?.score ?? 0;
      const dd = (s2 - s1);
      const ddCol = dd > 0 ? "var(--good)" : (dd < 0 ? "var(--bad)" : "var(--dim)");
      html += `<tr>
        <td>${name}</td>
        <td class="dim-score-${s1}">${s1}/5</td>
        <td class="dim-score-${s2}">${s2}/5</td>
        <td style="color:${ddCol};font-weight:600;">${dd>0?"+":""}${dd}</td>
        <td class="reason">${(v2.dimensions[name]?.reasoning||"").replace(/</g,"&lt;")}</td>
      </tr>`;
    }
    html += `</tbody></table>`;
    el("improve-dim-table").innerHTML = html;
    setStage("improve","done");
  });

  es.addEventListener("done", () => {
    es.close();
    el("improve-btn").disabled = false;
    el("improve-btn").textContent = "Run env-driven prompt improvement";
  });
  es.onerror = () => {
    es.close();
    el("improve-btn").disabled = false;
    el("improve-btn").textContent = "Run env-driven prompt improvement";
  };
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)


def run_demo_ui(host: str = "0.0.0.0", port: int = 8911):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
