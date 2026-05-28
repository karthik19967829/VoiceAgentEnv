"""
Stateless turn-level evaluation of a speech LLM against a real call.

Design (offline policy evaluation):
  For each caller turn k in the original human-human call, treat it as an
  independent test case. The "context" given to the model under test is the
  ORIGINAL human transcript up to (but not including) turn k, plus the real
  caller audio for turn k. The model produces a fresh response.

  - Embarrassingly parallel (no inter-turn dependency)
  - Apples-to-apples: the AI sees exactly the same prefix the human agent saw
  - Per-turn judging is clean: same caller utterance, two responses to compare

This sidesteps the problems of stateful rollout eval (cascade errors, slow
sequential runs, divergent contexts) and is the right shape for SFT/DPO
data mining and CI-style regression testing.

Compatible model:  gpt-audio, gpt-audio-mini (Chat Completions audio).
The OpenAI Realtime API is *not* used here — Realtime is a streaming,
stateful protocol that fights this design. Audio Chat Completions is the
right primitive for stateless eval.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess as sp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voiceenv.core.sandbox import Sandbox
from voiceenv.core.schema import VoiceEnvironment
from voiceenv.core.scorer import Scorer


# ── Data types ──


@dataclass
class TurnResult:
    turn_idx: int
    caller_text: str             # original caller transcript (reference only)
    caller_audio_path: str       # path to the audio clip we sent to the model
    human_response: str | None   # what the original human agent said next
    ai_response: str             # what the speech LLM said
    ai_audio_path: str | None = None   # path to the AI's spoken response WAV
    ai_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    error: str | None = None


@dataclass
class DemoResult:
    env: VoiceEnvironment
    model: str
    turns: list[TurnResult]
    scorecard: Any               # voiceenv.core.scorer.Scorecard
    total_cost_usd: float
    wall_time_seconds: float


# ── Audio slicing ──


def _find_caller_channel(wav_path: Path) -> Path:
    """If wav_path lives in HVB-style audio/agent/, look for the matching caller
    channel WAV at audio/caller/<same-name>. Otherwise return the WAV itself."""
    if wav_path.parent.name in ("agent", "caller") and wav_path.parents[1].name == "audio":
        candidate = wav_path.parents[1] / "caller" / wav_path.name
        if candidate.exists():
            return candidate
    return wav_path


def _find_agent_channel(wav_path: Path) -> Path:
    """Mirror of _find_caller_channel for the human agent's audio channel."""
    if wav_path.parent.name in ("agent", "caller") and wav_path.parents[1].name == "audio":
        candidate = wav_path.parents[1] / "agent" / wav_path.name
        if candidate.exists():
            return candidate
    return wav_path


def slice_human_responses(
    wav_path: Path,
    all_turns: list,
    caller_clips: list[dict],
    output_dir: Path,
) -> dict[int, dict]:
    """For each caller clip, slice the *next agent turn* from the agent channel.

    Returns: {caller_turn_idx -> {audio_path, text, start_ms, duration_ms}}.
    """
    if not shutil.which("ffmpeg"):
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_wav = _find_agent_channel(Path(wav_path))

    out: dict[int, dict] = {}
    for clip in caller_clips:
        caller_idx = clip["turn_idx"]
        # Find first agent turn after this caller turn
        next_agent = None
        for t in all_turns[caller_idx + 1:]:
            if t.speaker == "agent" and t.duration_ms > 200:
                next_agent = t
                break
        if next_agent is None:
            continue

        clip_path = output_dir / f"human_response_{caller_idx:02d}.wav"
        try:
            sp.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(agent_wav),
                    "-ss", f"{next_agent.start_ms / 1000:.3f}",
                    "-t", f"{max(next_agent.duration_ms / 1000, 0.5):.3f}",
                    "-ar", "16000", "-ac", "1",
                    str(clip_path),
                ],
                check=True, capture_output=True,
            )
        except Exception:
            continue

        out[caller_idx] = {
            "audio_path": str(clip_path),
            "text": next_agent.text,
            "start_ms": next_agent.start_ms,
            "duration_ms": next_agent.duration_ms,
        }
    return out


def slice_caller_turns(
    wav_path: Path,
    transcript_turns: list,
    output_dir: Path,
    max_turns: int | None = None,
) -> list[dict]:
    """Slice the caller channel into per-turn WAV clips using transcript timing.

    Args:
        wav_path: path to the source audio (WAV/agent or caller channel both fine).
        transcript_turns: list of voiceenv.ingest.Turn objects.
        output_dir: where to write the clips.
        max_turns: cap how many caller turns to slice (None = all).

    Returns:
        list of {turn_idx, audio_path, start_ms, duration_ms, text}.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH — required to slice audio.")

    output_dir.mkdir(parents=True, exist_ok=True)
    caller_wav = _find_caller_channel(Path(wav_path))

    clips: list[dict] = []
    for i, t in enumerate(transcript_turns):
        if t.speaker != "caller":
            continue
        if t.duration_ms <= 200:
            continue   # skip near-silent fragments like single "yes"/[noise]
        clip_path = output_dir / f"caller_turn_{i:02d}.wav"
        sp.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(caller_wav),
                "-ss", f"{t.start_ms / 1000:.3f}",
                "-t", f"{max(t.duration_ms / 1000, 0.5):.3f}",
                "-ar", "16000", "-ac", "1",
                str(clip_path),
            ],
            check=True,
            capture_output=True,
        )
        clips.append({
            "turn_idx": i,
            "audio_path": str(clip_path),
            "start_ms": t.start_ms,
            "duration_ms": t.duration_ms,
            "text": t.text,
        })
        if max_turns and len(clips) >= max_turns:
            break
    return clips


# ── Stateless per-turn LLM call ──


def _build_messages(
    env: VoiceEnvironment,
    prefix_turns: list,
    current_audio_b64: str,
    override_system_prompt: str | None = None,
) -> list[dict]:
    """Build the OpenAI chat messages array for one turn-level test case."""
    sys_prompt = override_system_prompt or env.agent_system_prompt or (
        f"You are an AI agent handling a {env.vertical.value} call. "
        f"Goal: {env.task.goal}. Be concise, polite, and use the tools when appropriate."
    )
    # Demo-critical: force a spoken sentence on every turn, even when calling
    # a tool. Without this the audio modality may return null on tool turns.
    sys_prompt += (
        "\n\nIMPORTANT: ALWAYS respond with a short spoken sentence to the user, "
        "EVEN IF you also call a tool. Never reply with only a tool call and "
        "no speech."
    )

    # Interpolate world_state into the system prompt if templated
    try:
        sys_prompt = sys_prompt.format(world_state=env.world_state.fields, **env.world_state.fields)
    except Exception:
        pass

    messages: list[dict] = [{"role": "system", "content": sys_prompt}]

    # Prefix: the original human conversation up to (but not including) this turn.
    # We use TEXT for the prefix to keep token cost manageable; only the current
    # caller turn is sent as audio. (Audio for *all* prior turns blows up cost
    # quickly with no benefit for stateless eval.)
    for t in prefix_turns:
        role = "assistant" if t.speaker == "agent" else "user"
        messages.append({"role": role, "content": t.text})

    # Current caller turn — REAL human audio
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": current_audio_b64, "format": "wav"}},
        ],
    })
    return messages


def _call_model(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    capture_audio: bool = False,
    voice: str = "alloy",
) -> tuple[Any, dict]:
    """One API call. Returns (message, usage)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if capture_audio:
        kwargs["modalities"] = ["text", "audio"]
        kwargs["audio"] = {"voice": voice, "format": "wav"}
    else:
        kwargs["modalities"] = ["text"]
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }
    return resp.choices[0].message, usage


def run_stateless_eval(
    env: VoiceEnvironment,
    all_turns: list,
    caller_clips: list[dict],
    model: str = "gpt-audio-mini",
    parallelism: int = 4,
    on_log=print,
    capture_audio: bool = False,
    audio_out_dir: Path | None = None,
    voice: str = "alloy",
    override_system_prompt: str | None = None,
) -> tuple[list[TurnResult], float]:
    """Run one independent API call per caller clip, in parallel.

    If capture_audio is True, the speech LLM is asked to ALSO emit a spoken
    response (text + audio); the audio is saved as
    `<audio_out_dir>/ai_turn_NN.wav` and the path is attached to TurnResult.

    Returns (turn_results, total_cost_usd).
    """
    from openai import OpenAI

    client = OpenAI()
    sandbox = Sandbox(tools=env.tools, initial_state=env.world_state)
    tool_schemas = sandbox.get_tool_schemas() if env.tools else None
    if capture_audio and audio_out_dir is not None:
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    def _one(clip: dict) -> tuple[TurnResult, dict]:
        try:
            with open(clip["audio_path"], "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            prefix = all_turns[: clip["turn_idx"]]
            messages = _build_messages(env, prefix, audio_b64, override_system_prompt=override_system_prompt)
            t0 = time.time()

            # Retry on transient errors (rate limits, 5xx)
            msg = None
            usage = {}
            last_err = None
            for attempt in range(3):
                try:
                    msg, usage = _call_model(
                        client, model, messages, tool_schemas,
                        capture_audio=capture_audio, voice=voice,
                    )
                    break
                except Exception as e:
                    last_err = e
                    code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                    on_log(f"  attempt {attempt+1} failed (status={code}): {str(e)[:120]}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # 1s, 2s backoff
                    else:
                        raise last_err
            latency = int((time.time() - t0) * 1000)

            ai_audio_path: str | None = None
            ai_text = (msg.content or "").strip()
            audio_obj = getattr(msg, "audio", None)

            tool_calls_raw = msg.tool_calls or []

            # gpt-audio quirk: when it emits tool_calls it returns *no* text and
            # *no* audio. Do a forced follow-up call asking it to verbalize.
            if capture_audio and tool_calls_raw and (
                audio_obj is None or not getattr(audio_obj, "data", None)
            ):
                try:
                    follow_msgs = list(messages) + [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls_raw
                            ],
                            "content": "",
                        },
                    ]
                    for tc in tool_calls_raw:
                        follow_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"success": True}),
                        })
                    follow_msgs.append({
                        "role": "system",
                        "content": "Now speak a single short spoken confirmation to the customer in one sentence.",
                    })
                    msg2, usage2 = _call_model(
                        client, model, follow_msgs, None,
                        capture_audio=True, voice=voice,
                    )
                    audio_obj = getattr(msg2, "audio", None)
                    if not ai_text and getattr(msg2, "content", None):
                        ai_text = msg2.content.strip()
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        usage[k] = usage.get(k, 0) + usage2.get(k, 0)
                except Exception as e:
                    on_log(f"  turn {clip['turn_idx']}: followup speech call failed: {e}")

            if capture_audio and audio_out_dir is not None:
                out = audio_out_dir / f"ai_turn_{clip['turn_idx']:02d}.wav"

                # Prefer native audio from the speech LLM
                if audio_obj is not None and getattr(audio_obj, "data", None):
                    try:
                        out.write_bytes(base64.b64decode(audio_obj.data))
                        ai_audio_path = str(out)
                        if not ai_text and getattr(audio_obj, "transcript", None):
                            ai_text = audio_obj.transcript.strip()
                    except Exception as e:
                        on_log(f"  turn {clip['turn_idx']}: failed to save native audio: {e}")

                # Bulletproof fallback: if there's still no audio but we have
                # *some* text (or a tool call we can describe), TTS it so every
                # turn in the UI has a playable audio cell.
                if ai_audio_path is None:
                    speak_text = ai_text
                    if not speak_text and tool_calls_raw:
                        tc = tool_calls_raw[0]
                        try:
                            args = json.loads(tc.function.arguments)
                        except Exception:
                            args = {}
                        speak_text = f"I'll {tc.function.name.replace('_', ' ')} for you now."
                        if not ai_text:
                            ai_text = speak_text
                    if speak_text:
                        try:
                            tts_resp = client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice=voice,
                                input=speak_text,
                                response_format="wav",
                            )
                            out.write_bytes(tts_resp.content)
                            ai_audio_path = str(out)
                            on_log(f"  turn {clip['turn_idx']}: TTS fallback used")
                        except Exception as e:
                            on_log(f"  turn {clip['turn_idx']}: TTS fallback failed: {e}")
            tool_calls = []
            for tc in (msg.tool_calls or []):
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_calls.append({"tool": tc.function.name, "args": args, "success": True})

            # The original human agent's response = the first agent turn AFTER this caller turn
            human_resp = None
            for t in all_turns[clip["turn_idx"] + 1:]:
                if t.speaker == "agent":
                    human_resp = t.text
                    break

            return (
                TurnResult(
                    turn_idx=clip["turn_idx"],
                    caller_text=clip["text"],
                    caller_audio_path=clip["audio_path"],
                    human_response=human_resp,
                    ai_response=ai_text,
                    ai_audio_path=ai_audio_path,
                    ai_tool_calls=tool_calls,
                    latency_ms=latency,
                ),
                usage,
            )
        except Exception as e:
            return (
                TurnResult(
                    turn_idx=clip["turn_idx"],
                    caller_text=clip["text"],
                    caller_audio_path=clip["audio_path"],
                    human_response=None,
                    ai_response="",
                    error=str(e),
                ),
                {},
            )

    results: list[TurnResult] = []
    total_cost = 0.0
    completed = 0

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(_one, c): c for c in caller_clips}
        for fut in as_completed(futures):
            res, usage = fut.result()
            results.append(res)
            completed += 1
            if usage:
                # Rough gpt-4o-audio-preview pricing: assume $40/M audio-input,
                # $10/M text-input, $20/M text-output. Approximate.
                cost = (
                    usage.get("prompt_tokens", 0) / 1e6 * 15.0
                    + usage.get("completion_tokens", 0) / 1e6 * 20.0
                )
                total_cost += cost
            on_log(f"  [{completed}/{len(caller_clips)}] turn {res.turn_idx}: "
                   f"{res.latency_ms}ms"
                   + (f"  ✗ {res.error}" if res.error else
                      (f"  → tool: {res.ai_tool_calls[0]['tool']}" if res.ai_tool_calls else
                       f"  → {(res.ai_response[:60] + '…') if len(res.ai_response) > 60 else res.ai_response}")))

    results.sort(key=lambda r: r.turn_idx)
    return results, total_cost


# ── Scoring ──


# ── Grounded multimodal judging (Gemini) ──


GROUNDED_PROMPT = """You are evaluating an AI agent's responses to a REAL customer support call.

You have access to TWO things:
  1. The original human-human call recording (audio). This is the EXPERT REFERENCE — \
how a real human agent handled the same task, with the same caller. Listen to it carefully.
  2. The AI agent's text responses to each caller turn (below). The AI was given the \
same context the human agent had at each point in the call.

CRITICAL EVAL DESIGN — READ BEFORE SCORING:
  • This is a STATELESS, per-turn offline evaluation. Each AI response was generated \
INDEPENDENTLY, conditioned ONLY on the original human-human transcript up to that \
point + the caller's current audio. The AI never saw its own previous responses.
  • Therefore: DO NOT penalize the AI for "repetition" or "looping" across turns. \
If the AI re-confirms a request on turn 3 and again on turn 5, that is because each \
turn was a fresh, independent call — the AI had no memory of having said it before.
  • If the caller appears to repeat themselves (e.g. "my debit card", "yes my debit \
card"), it is almost always because the HUMAN agent in the original recording had \
poor phone audio and asked the caller to repeat. The AI is responding correctly to \
the context it was given at each turn. Judge the AI on the QUALITY of each \
individual response given its context, not on cross-turn coherence.
  • The AI does NOT need to match the human's exact behavior. The human may have \
been slow, may have misheard the caller, may have been overly terse. Judge whether \
the AI's response is a GOOD response for that turn's context — using the human \
recording as a sanity anchor for tone/style of a real banking call, not as a \
gold-label ceiling.

For each dimension below, rate the AI agent on a 1-5 scale and give a one-sentence \
reason GROUNDED in the expert recording — but respect the stateless eval design above.

Dimensions:
  - human_likeness:        How closely the AI's responses RESEMBLE a real human agent \
in cadence, word choice, fillers, naturalness, and conversational flow (vs. the actual \
human in the recording). 5 = could pass as the human agent. 1 = obviously a chatbot.
  - tone_appropriateness:  Tone fit for a banking-support call (vs. expert)
  - empathy:               Acknowledges caller's situation (vs. expert)
  - task_focus:            Quality of each individual response toward resolution \
(DO NOT penalize cross-turn repetition — see eval design above)
  - conciseness:           Brevity / on-the-phone style (vs. expert who used 5-10 words/turn)
  - tool_appropriateness:  Used tools at the right moment (replace_card etc.)

Per-turn breakdown. For each turn we show the PRIOR CONTEXT the AI saw \
(the original human-human transcript up to that point), the caller's current \
utterance, and the AI's fresh response generated from that context:
{ai_responses}

Return ONLY a JSON object of shape:
{{
  "human_likeness":       {{"score": <1-5>, "reasoning": "..."}},
  "tone_appropriateness": {{"score": <1-5>, "reasoning": "..."}},
  "empathy":              {{"score": <1-5>, "reasoning": "..."}},
  "task_focus":           {{"score": <1-5>, "reasoning": "..."}},
  "conciseness":          {{"score": <1-5>, "reasoning": "..."}},
  "tool_appropriateness": {{"score": <1-5>, "reasoning": "..."}}
}}
"""


def run_grounded_eval(
    env: VoiceEnvironment,
    results: list[TurnResult],
    expert_audio_path: str,
    model: str = "gemini-2.5-flash",
    all_turns: list | None = None,
) -> dict:
    """Grounded multimodal judging: feed Gemini the expert call audio + AI's
    per-turn responses, get back 1-5 scores on tone, empathy, task_focus,
    conciseness, tool_appropriateness — anchored on a real human recording."""
    from google import genai
    from google.genai import types
    import os

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) must be set in env.")

    client = genai.Client(api_key=api_key)

    # Inline the audio (works for files <20MB; our HVB calls are <1MB and avoids
    # the Files API processing-delay race condition).
    expert_bytes = Path(expert_audio_path).read_bytes()

    lines = []
    for r in results:
        ai = r.ai_response or "(no response)"
        if r.ai_tool_calls:
            ai = ai + f"  [tool_call: {r.ai_tool_calls[0]['tool']}({json.dumps(r.ai_tool_calls[0]['args'])})]"
        ctx_block = ""
        if all_turns is not None:
            prior = all_turns[: r.turn_idx]
            if prior:
                ctx_lines = []
                for t in prior:
                    role = "human-agent" if t.speaker == "agent" else "caller"
                    ctx_lines.append(f"      {role}: {t.text}")
                ctx_block = "    PRIOR CONTEXT (what the AI saw at this turn):\n" + "\n".join(ctx_lines) + "\n"
            else:
                ctx_block = "    PRIOR CONTEXT: (cold start — first turn)\n"
        lines.append(
            f"  --- Turn {r.turn_idx} (independent, stateless) ---\n"
            f"{ctx_block}"
            f"    Caller (current turn): {r.caller_text}\n"
            f"    AI response:           {ai}"
        )
    ai_responses = "\n\n".join(lines)

    prompt = GROUNDED_PROMPT.format(ai_responses=ai_responses)

    # Send only the expert audio (not AI audio) to keep the request fast.
    # The AI's responses go in as text in the prompt above. Audio comparison
    # against AI clips can be added later as a separate, optional stage.
    parts = [
        types.Part(inline_data=types.Blob(data=expert_bytes, mime_type="audio/wav")),
        types.Part(text=prompt),
    ]

    last_err = None
    resp = None
    candidate_models = [model, "gemini-2.0-flash", "gemini-flash-latest"]
    seen: set[str] = set()
    for m_name in candidate_models:
        if m_name in seen:
            continue
        seen.add(m_name)
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=m_name,
                    contents=[types.Content(parts=parts)],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                model = m_name  # record which one actually answered
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(2 ** attempt)
                    continue
                raise
        if resp is not None:
            break
    if resp is None:
        raise last_err  # type: ignore[misc]

    parsed = json.loads(resp.text or "{}")

    avg = sum(d.get("score", 0) for d in parsed.values()) / max(len(parsed), 1)
    return {
        "model": model,
        "expert_reference": expert_audio_path,
        "average_score_1_5": round(avg, 2),
        "dimensions": parsed,
    }


PROMPT_IMPROVER_SYSTEM = """You are a prompt engineer optimizing an AI voice-agent's \
SYSTEM PROMPT based on feedback from a grounded multimodal judge that compared the \
agent's responses to a REAL human agent recording.

You will receive:
  1. The CURRENT system prompt
  2. Per-dimension scores (1-5) with judge reasoning (incl. "human_likeness")
  3. A few example turns: AI response vs. the actual human response

Your job: REWRITE the system prompt so the agent scores HIGHER on weak dimensions \
— especially human_likeness, conciseness, empathy, and tone — while preserving \
tool use and task completion.

BE AGGRESSIVE. The current prompt is too generic. Force a clear stylistic shift.

STYLE THE AGENT MUST ADOPT (this is non-negotiable):
  - SHORT: most responses 5–12 words. Single sentence, no lists.
  - HUMAN FILLERS: occasional "uh", "okay", "alright", "sure", "got it".
  - DROP CORPORATE-SPEAK: NO "I'd be happy to assist", "Certainly!", "Of course!", \
"Please allow me", "I understand your concern". Real phone reps don't talk this way.
  - DIRECT: ask for ONE thing at a time, like a real human collecting info \
slot-by-slot. Don't bundle multiple questions.
  - NO RESTATEMENT: don't echo the customer's request back ("So you'd like to..."). \
Just act.
  - CONFIRM SUCCINCTLY: "alright, that's done" / "okay, transferred". Not paragraphs.

REQUIREMENTS (preserve):
  - Tool use: still call the appropriate tool when all slots are gathered.
  - The "ALWAYS respond with a short spoken sentence even if calling a tool" rule.
  - World-state placeholders if present in the original prompt (keep {placeholders}).
  - Under 250 words total.

INCLUDE in the new prompt:
  - 2-3 concrete EXAMPLE UTTERANCES the agent should sound like (drawn from the \
human responses you saw in the feedback). Format as: GOOD: "..."
  - 2-3 ANTI-EXAMPLES the agent must NOT produce. Format as: BAD: "..."

Return ONLY a JSON object:
{"improved_prompt": "<the new system prompt string>",
 "changes_summary": "<one sentence explaining what you changed and why>"}
"""


def improve_agent_prompt(
    env: VoiceEnvironment,
    results: list[TurnResult],
    grounded: dict,
    all_turns: list | None = None,
    model: str = "gpt-4o",
) -> dict:
    """Use the grounded judge's feedback to rewrite the agent's system prompt.

    Returns dict with keys: improved_prompt, changes_summary, original_prompt.
    """
    from openai import OpenAI

    original = env.agent_system_prompt or "(no explicit system prompt)"
    dims = grounded.get("dimensions", {})
    feedback_lines = [
        f"  - {name}: {d.get('score','?')}/5 — {d.get('reasoning','')}"
        for name, d in dims.items()
    ]

    turn_examples = []
    for r in results[:4]:
        if not r.ai_response:
            continue
        human = r.human_response or "(end of call)"
        turn_examples.append(
            f"  CALLER: {r.caller_text}\n"
            f"  HUMAN:  {human}\n"
            f"  AI:     {r.ai_response}"
        )

    user_msg = (
        f"CURRENT SYSTEM PROMPT:\n{original}\n\n"
        f"JUDGE SCORES (avg {grounded.get('average_score_1_5','?')}/5):\n"
        + "\n".join(feedback_lines)
        + "\n\nSAMPLE TURNS (AI vs HUMAN):\n"
        + "\n\n".join(turn_examples)
        + "\n\nRewrite the system prompt aggressively per the rules in your "
        "system instructions. The improved_prompt MUST be at least 120 words, "
        "MUST include explicit GOOD: \"...\" example utterances drawn from the "
        "human responses above, and MUST include BAD: \"...\" anti-examples "
        "showing corporate-speak to avoid (e.g. \"I'd be happy to assist\"). "
        "If your output does not contain at least 3 GOOD: and 2 BAD: examples, "
        "you have failed the task."
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT_IMPROVER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    return {
        "original_prompt": original,
        "improved_prompt": parsed.get("improved_prompt", original),
        "changes_summary": parsed.get("changes_summary", ""),
    }


def score_eval(env: VoiceEnvironment, results: list[TurnResult]):
    """Build a synthetic transcript+tool_calls from the per-turn results and
    score with the env's verifiable rubric."""
    transcript = []
    tool_calls = []
    for r in results:
        transcript.append({"role": "user", "content": r.caller_text})
        transcript.append({"role": "agent", "content": r.ai_response})
        for tc in r.ai_tool_calls:
            tool_calls.append({"tool": tc["tool"], "args": tc["args"], "success": True})

    scorer = Scorer(env.rubric, skip_soft_scoring=True)
    return scorer.score(
        transcript=transcript,
        tool_calls=tool_calls,
        final_state=dict(env.world_state.fields),
    )
