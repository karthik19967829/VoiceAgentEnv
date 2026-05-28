"""
Autonomous ingest from a raw WAV call recording.

Pipeline stages (each timed and reported):
  1. Load + transcribe         (HVB JSON sibling OR Whisper API per channel)
  2. Segment into turns        (speaker-labelled, ms-aligned)
  3. LLM extraction            (task, persona, tools, rubric, agent prompt)
  4. Build VoiceEnvironment    (validated against schema)
  5. Emit artefacts            (env.yaml, expert_reference/, rollouts/)

The output directory is self-contained and can be passed to:
    voiceenv export <dir>/env.yaml --target both
    voiceenv publish <dir>/env.yaml --target both
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from voiceenv.core.schema import (
    Difficulty,
    ExpertReference,
    ScoringCriterion,
    ScoringRubric,
    SimulatorProfile,
    TaskDefinition,
    ToolDefinition,
    ToolParameter,
    Vertical,
    VoiceConfig,
    VoiceEnvironment,
    WorldState,
)


# ── Data types ──


@dataclass
class Turn:
    speaker: str  # "agent" or "caller"
    text: str
    start_ms: int
    duration_ms: int
    dialog_acts: list[str] = field(default_factory=list)
    emotion: dict[str, float] = field(default_factory=dict)


@dataclass
class IngestResult:
    env: VoiceEnvironment
    output_dir: Path
    n_turns: int
    duration_seconds: float
    cost_usd: float
    timings_ms: dict[str, int]


# ── Stage 1: transcript loading ──


def _load_hvb_transcript(transcript_path: Path) -> list[Turn]:
    """Load a HarperValleyBank-style sibling transcript JSON."""
    raw = json.loads(transcript_path.read_text())
    turns: list[Turn] = []
    for t in raw:
        text = (t.get("human_transcript") or t.get("transcript") or "").strip()
        if not text or text.startswith("[") and text.endswith("]"):
            # Skip [noise] / [silence] etc.
            continue
        turns.append(
            Turn(
                speaker=t["speaker_role"],
                text=text,
                start_ms=int(t.get("start_ms", t.get("offset_ms", 0))),
                duration_ms=int(t.get("duration_ms", 0)),
                dialog_acts=[
                    da.replace("gridspace_", "") for da in t.get("dialog_acts", [])
                ],
                emotion=t.get("emotion", {}),
            )
        )
    turns.sort(key=lambda x: x.start_ms)
    return turns


def _whisper_transcribe(wav_path: Path) -> list[Turn]:
    """
    Fallback: transcribe a stereo WAV via OpenAI Whisper, treating
    left channel as caller and right as agent. For mono WAV, all turns
    are attributed to a single speaker (caller-only fallback).

    This is a pragmatic, demo-grade diarization. For production we'd run
    a proper diarizer (pyannote, NVIDIA NeMo, etc.).
    """
    from openai import OpenAI

    client = OpenAI()

    # Try stereo channel split via ffmpeg if available; else mono.
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()

    turns: list[Turn] = []

    if n_channels >= 2 and shutil.which("ffmpeg"):
        import subprocess as sp
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            for ch_idx, role in [(0, "caller"), (1, "agent")]:
                out = Path(td) / f"ch{ch_idx}.wav"
                sp.run(
                    [
                        "ffmpeg", "-y", "-i", str(wav_path),
                        "-map_channel", f"0.0.{ch_idx}",
                        str(out),
                    ],
                    check=True, capture_output=True,
                )
                with open(out, "rb") as f:
                    resp = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )
                for seg in resp.segments or []:
                    text = seg.text.strip()
                    if not text:
                        continue
                    turns.append(
                        Turn(
                            speaker=role,
                            text=text,
                            start_ms=int(seg.start * 1000),
                            duration_ms=int((seg.end - seg.start) * 1000),
                        )
                    )
    else:
        with open(wav_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        # Mono: alternate speakers heuristically (better than nothing for demo)
        speakers = ["caller", "agent"]
        for i, seg in enumerate(resp.segments or []):
            text = seg.text.strip()
            if not text:
                continue
            turns.append(
                Turn(
                    speaker=speakers[i % 2],
                    text=text,
                    start_ms=int(seg.start * 1000),
                    duration_ms=int((seg.end - seg.start) * 1000),
                )
            )

    turns.sort(key=lambda x: x.start_ms)
    return turns


# ── Stage 2: turn merging (consecutive same-speaker) ──


def _merge_consecutive(turns: list[Turn]) -> list[Turn]:
    """Merge ALL consecutive same-speaker turns into a single dialog turn.

    A "turn" is everything one speaker said between two speaker switches.
    This is the standard dialog-segmentation rule and is what aligns
    caller utterances with the agent's responses one-to-one.
    """
    if not turns:
        return turns
    merged: list[Turn] = [turns[0]]
    for t in turns[1:]:
        last = merged[-1]
        if t.speaker == last.speaker:
            last.text = (last.text + " " + t.text).strip()
            last.duration_ms = (t.start_ms + t.duration_ms) - last.start_ms
            last.dialog_acts = list(set(last.dialog_acts) | set(t.dialog_acts))
        else:
            merged.append(t)
    return merged


# ── Stage 3: LLM-based extraction ──


_EXTRACTION_SYSTEM = """You are an environment designer for VoiceEnv, a platform that turns real call \
recordings into RL training environments for speech LLMs.

You will be given a transcript of a real human-human voice conversation (e.g. a customer \
calling a support agent). Your job is to produce a JSON spec describing the environment \
that could be used to train an AI agent to handle this kind of call.

You MUST output STRICT JSON conforming to this schema:

{
  "name": "snake_case_short_name",
  "description": "one-sentence description",
  "vertical": "support|sales|healthcare|collections|scheduling|recruiting|onboarding|emergency|custom",
  "difficulty": "easy|medium|hard",
  "task": {
    "goal": "what the AI agent must accomplish (1-2 sentences, agent-facing)",
    "context": "background context the agent has at start of call",
    "success_criteria": ["bullet 1", "bullet 2", "..."]
  },
  "world_state": {
    "description": "what the sandboxed world contains",
    "fields": { "key": "value", ... }
  },
  "persona": {
    "persona_description": "natural language description of the human caller",
    "patience": 0.0-1.0,
    "cooperativeness": 0.0-1.0,
    "skepticism": 0.0-1.0,
    "verbosity": 0.0-1.0,
    "emotional_volatility": 0.0-1.0,
    "hidden_goals": ["..."]
  },
  "tools": [
    {
      "name": "snake_case",
      "description": "what it does",
      "parameters": [{"name": "x", "type": "string", "description": "...", "required": true}],
      "side_effects": {"world_field_to_change": "value_or_template"}
    }
  ],
  "agent_system_prompt": "system prompt for the AI agent under test. Keep it tight, role-focused, mention available tools by name.",
  "rubric": {
    "task_success": [
      {
        "name": "snake_case",
        "description": "human-readable",
        "weight": 1.0,
        "scoring_type": "binary",
        "deterministic_check": "python expression evaluated against `state` (a dict with keys: transcript, tool_calls, world_state). e.g. any('refund' in tc['name'] for tc in state['tool_calls'])"
      }
    ],
    "compliance": [ ... ],
    "efficiency": [ ... ]
  }
}

Rules:
- 3-6 verifiable rubric criteria total across categories. Each MUST have a deterministic_check that is a valid one-line Python expression.
- At runtime, these variables are available DIRECTLY (NOT inside any `state` dict):
    transcript: list of {"role": "agent"|"user", "content": str}
    tool_calls: list of {"tool": str, "args": dict, "success": bool, "result": Any}
    agent_turns: list of agent transcript entries
    user_turns: list of user transcript entries
    turns: int (total turn count)
  And these helper functions:
    tool_was_called(name): bool
    transcript_contains(pattern, speaker=None): bool
    all_tools_succeeded(): bool
- IMPORTANT: role is ONLY 'agent' or 'user' (never 'caller', 'customer'). Tool entries use the key 'tool' (not 'name') and 'args' (not 'arguments').
- Prefer SIMPLE, ROBUST checks. Examples of GOOD checks:
    tool_was_called('replace_card')
    any('replace' in t['content'].lower() for t in agent_turns)
    transcript_contains('thank you', speaker='agent')
    len(agent_turns) >= 2
    turns <= 12
- AVOID checking for very specific multi-word phrases inside tool args; that almost never matches.
- Tool-presence checks should match the EXACT tool names you define above.
- 1-3 tools that the AI agent could realistically call to accomplish the task.
- Persona dimensions should reflect the actual caller in the transcript.
- The success_criteria should be *outcomes*, not tool calls."""


_EXTRACTION_USER_TEMPLATE = """Transcript of a real call (speaker | text):

{transcript}

Dialog-act tags observed in this call: {dialog_acts}
Caller-side average emotion: {emotion}

Produce the JSON spec now."""


def _llm_extract(turns: list[Turn], model: str = "gpt-4o-mini") -> tuple[dict, float]:
    """Call the LLM and parse a strict JSON spec. Returns (spec, cost_usd)."""
    from openai import OpenAI

    client = OpenAI()

    transcript_text = "\n".join(f"{t.speaker} | {t.text}" for t in turns)
    all_acts = sorted({a for t in turns for a in t.dialog_acts}) or ["(none)"]

    # Average emotion across caller turns
    caller_turns = [t for t in turns if t.speaker == "caller" and t.emotion]
    avg_emotion: dict[str, float] = {}
    if caller_turns:
        for k in ("positive", "neutral", "negative"):
            vals = [t.emotion.get(k, 0.0) for t in caller_turns]
            avg_emotion[k] = round(sum(vals) / len(vals), 2)

    user_msg = _EXTRACTION_USER_TEMPLATE.format(
        transcript=transcript_text,
        dialog_acts=", ".join(all_acts),
        emotion=avg_emotion or "(unavailable)",
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = resp.choices[0].message.content
    spec = json.loads(content)

    # Rough gpt-4o-mini cost: $0.15/M input + $0.60/M output
    usage = resp.usage
    cost = (usage.prompt_tokens / 1e6 * 0.15) + (usage.completion_tokens / 1e6 * 0.60)
    return spec, cost


# ── Stage 4: build VoiceEnvironment from spec ──


def _safe_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return name or "auto_env"


def _build_environment(spec: dict, source_call_id: str, expert_audio_relpath: str,
                       transcript_text: str) -> VoiceEnvironment:
    """Convert the LLM spec dict into a validated VoiceEnvironment."""

    # Tools
    tools = []
    for t in spec.get("tools", []):
        params = [
            ToolParameter(
                name=p["name"],
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
            )
            for p in t.get("parameters", [])
        ]
        tools.append(
            ToolDefinition(
                name=t["name"],
                description=t.get("description", ""),
                parameters=params,
                side_effects=t.get("side_effects", {}),
            )
        )

    # Rubric
    def _crit(c: dict) -> ScoringCriterion:
        return ScoringCriterion(
            name=c["name"],
            description=c.get("description", ""),
            weight=float(c.get("weight", 1.0)),
            scoring_type=c.get("scoring_type", "binary"),
            deterministic_check=c.get("deterministic_check"),
            llm_judge_prompt=c.get("llm_judge_prompt"),
        )

    rubric_dict = spec.get("rubric", {}) or {}
    rubric = ScoringRubric(
        task_success=[_crit(c) for c in rubric_dict.get("task_success", [])],
        compliance=[_crit(c) for c in rubric_dict.get("compliance", [])],
        efficiency=[_crit(c) for c in rubric_dict.get("efficiency", [])],
        voice_quality=[_crit(c) for c in rubric_dict.get("voice_quality", [])],
        persona_fidelity=[_crit(c) for c in rubric_dict.get("persona_fidelity", [])],
    )

    # Persona
    p = spec.get("persona", {}) or {}
    sim = SimulatorProfile(
        persona_description=p.get("persona_description", "A real human caller."),
        patience=float(p.get("patience", 0.5)),
        cooperativeness=float(p.get("cooperativeness", 0.6)),
        skepticism=float(p.get("skepticism", 0.4)),
        verbosity=float(p.get("verbosity", 0.5)),
        emotional_volatility=float(p.get("emotional_volatility", 0.3)),
        hidden_goals=p.get("hidden_goals", []),
    )

    # Task
    task_d = spec.get("task", {}) or {}
    task = TaskDefinition(
        goal=task_d.get("goal", "Help the caller with their request."),
        context=task_d.get("context", ""),
        success_criteria=task_d.get("success_criteria", []),
    )

    # World state
    ws = spec.get("world_state", {}) or {}
    world = WorldState(
        description=ws.get("description", ""),
        fields=ws.get("fields", {}),
    )

    # Vertical / difficulty (defensive)
    try:
        vertical = Vertical(spec.get("vertical", "support"))
    except ValueError:
        vertical = Vertical.SUPPORT
    try:
        difficulty = Difficulty(spec.get("difficulty", "medium"))
    except ValueError:
        difficulty = Difficulty.MEDIUM

    # Expert reference (the original call IS the expert)
    expert = ExpertReference(
        name=f"source_call_{source_call_id}",
        description=("The original human-human call from which this environment was "
                     "auto-extracted. Use this to ground LLM judges on real human behavior."),
        audio_path=expert_audio_relpath,
        transcript=transcript_text,
        annotations=[
            "Ground truth: this is how a real human agent handled this exact task.",
            "Use for grounded comparison of tone, pacing, and de-escalation.",
        ],
    )

    return VoiceEnvironment(
        name=_safe_name(spec.get("name", "auto_env")),
        description=spec.get("description", "Auto-extracted environment from a real call."),
        version="1.0.0",
        author="voiceenv-ingest",
        tags=["auto-ingested", "real-call", source_call_id[:8]],
        vertical=vertical,
        difficulty=difficulty,
        task=task,
        world_state=world,
        simulator=sim,
        tools=tools,
        rubric=rubric,
        voice=VoiceConfig(),
        expert_references=[expert],
        agent_system_prompt=spec.get("agent_system_prompt", ""),
    )


# ── Stage 5: emit artefacts ──


def _emit_rollout(turns: list[Turn], env: VoiceEnvironment, out_path: Path) -> None:
    """Write a JSONL rollout in the format trainers expect."""
    transcript = [
        {"role": "agent" if t.speaker == "agent" else "user", "content": t.text}
        for t in turns
    ]

    # Evaluate the verifiable rubric on this human-human rollout, using the
    # SAME eval-context shape the runtime scorer uses, so the reward is honest.
    eval_ctx = {
        "transcript": transcript,
        "tool_calls": [],
        "tools": [],
        "turns": len(transcript),
        "agent_turns": [t for t in transcript if t["role"] == "agent"],
        "user_turns": [t for t in transcript if t["role"] == "user"],
        "tool_was_called": lambda name, **kw: False,
        "transcript_contains": lambda pattern, speaker=None, **kw: any(
            pattern.lower() in t["content"].lower()
            for t in transcript if (speaker is None or t["role"] == speaker)
        ),
        "all_tools_succeeded": lambda: True,
        "state": {"transcript": transcript, "tool_calls": []},
    }
    passed = 0
    total = 0
    for crit in env.rubric.all_criteria():
        if not crit.deterministic_check:
            continue
        total += 1
        try:
            if eval(crit.deterministic_check, {"__builtins__": {}}, eval_ctx):
                passed += 1
        except Exception:
            pass
    verifiable_reward = (passed / total) if total else 1.0

    record = {
        "env_name": env.name,
        "transcript": transcript,
        "tool_calls": [],
        "verifiable_reward": round(verifiable_reward, 3),
        "source": "human_human_real_call",
        "n_turns": len(transcript),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(record) + "\n")


# ── Public API ──


def ingest_call(
    wav_path: str | Path,
    output_dir: str | Path,
    transcript_path: str | Path | None = None,
    extraction_model: str = "gpt-4o-mini",
    on_log=print,
) -> IngestResult:
    """
    Ingest a single call recording into a complete VoiceEnv environment package.

    Args:
        wav_path: path to the call recording (wav/mp3/flac).
        output_dir: directory to write the environment package into.
        transcript_path: optional sibling transcript JSON (HarperValleyBank format).
                         If not given, falls back to OpenAI Whisper.
        extraction_model: LLM for env extraction (default: gpt-4o-mini).

    Returns:
        IngestResult with the built env, output path, and timing/cost info.
    """
    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect HVB sibling transcript if not provided.
    # HVB layout: <root>/audio/{agent,caller}/<id>.wav  +  <root>/transcript/<id>.json
    if transcript_path is None:
        for up in (1, 2, 3):
            try:
                candidate = wav_path.parents[up] / "transcript" / (wav_path.stem + ".json")
                if candidate.exists():
                    transcript_path = candidate
                    on_log(f"  ↳ found HVB sibling transcript: {candidate}")
                    break
            except IndexError:
                break

    timings: dict[str, int] = {}

    # 1. transcript
    t0 = time.time()
    if transcript_path is not None:
        on_log("[1/5] loading transcript JSON...")
        turns = _load_hvb_transcript(Path(transcript_path))
    else:
        on_log("[1/5] transcribing audio via Whisper...")
        turns = _whisper_transcribe(wav_path)
    timings["transcript_ms"] = int((time.time() - t0) * 1000)
    on_log(f"      → {len(turns)} raw turns ({timings['transcript_ms']} ms)")

    # 2. merge consecutive
    t0 = time.time()
    on_log("[2/5] segmenting + merging consecutive turns...")
    turns = _merge_consecutive(turns)
    timings["segment_ms"] = int((time.time() - t0) * 1000)
    on_log(f"      → {len(turns)} merged turns ({timings['segment_ms']} ms)")

    # 3. LLM extraction
    t0 = time.time()
    on_log(f"[3/5] extracting persona + task + tools + rubric ({extraction_model})...")
    spec, cost = _llm_extract(turns, model=extraction_model)
    timings["extract_ms"] = int((time.time() - t0) * 1000)
    on_log(f"      → spec: name='{spec.get('name')}', "
           f"{len(spec.get('tools', []))} tools, "
           f"{sum(len(spec.get('rubric', {}).get(k, [])) for k in ('task_success','compliance','efficiency'))} rubric criteria "
           f"({timings['extract_ms']} ms, ${cost:.4f})")

    # 4. build env
    t0 = time.time()
    on_log("[4/5] assembling VoiceEnvironment + expert reference + rollout...")
    expert_audio_relpath = "expert_reference/source_call.wav"
    transcript_text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
    env = _build_environment(spec, wav_path.stem, expert_audio_relpath, transcript_text)
    timings["build_ms"] = int((time.time() - t0) * 1000)

    # 5. emit
    t0 = time.time()
    on_log("[5/5] writing package to disk...")
    env_yaml_path = output_dir / "env.yaml"
    env.to_yaml(env_yaml_path)

    # Copy expert audio
    expert_dir = output_dir / "expert_reference"
    expert_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav_path, expert_dir / "source_call.wav")

    # Write annotations alongside
    (expert_dir / "annotations.json").write_text(json.dumps({
        "source_file": wav_path.name,
        "n_turns": len(turns),
        "duration_seconds": round(
            max((t.start_ms + t.duration_ms) for t in turns) / 1000, 1
        ) if turns else 0,
        "annotations": env.expert_references[0].annotations,
    }, indent=2))

    # Write rollout
    _emit_rollout(turns, env, output_dir / "rollouts" / "source_call.jsonl")

    # README so the package is hub-friendly
    (output_dir / "README.md").write_text(_readme(env, wav_path.name, len(turns)))

    timings["emit_ms"] = int((time.time() - t0) * 1000)

    duration = round(
        max((t.start_ms + t.duration_ms) for t in turns) / 1000, 1
    ) if turns else 0.0

    on_log(f"      → {env_yaml_path}")
    on_log(f"      → {expert_dir}/source_call.wav")
    on_log(f"      → {output_dir / 'rollouts' / 'source_call.jsonl'}")

    return IngestResult(
        env=env,
        output_dir=output_dir,
        n_turns=len(turns),
        duration_seconds=duration,
        cost_usd=cost,
        timings_ms=timings,
    )


def _readme(env: VoiceEnvironment, source_filename: str, n_turns: int) -> str:
    crits = env.rubric.all_criteria()
    crit_lines = "\n".join(
        f"- **{c.name}** ({c.scoring_type}): {c.description}"
        for c in crits
    )
    return f"""# {env.name}

> {env.description}

**Auto-ingested** from a real human-human call recording (`{source_filename}`,
{n_turns} turns) by `voiceenv ingest`. No manual annotation. No template.

## Use

```bash
# Run any speech LLM against this environment:
voiceenv run env.yaml --model gpt-4o-mini -n 10

# Generate RL training rollouts:
voiceenv train rollouts . --model gpt-4o-mini -n 50

# Publish to HuggingFace Spaces (OpenEnv) or Prime Intellect:
voiceenv export env.yaml --target both
voiceenv publish env.yaml --target both
```

## Task

{env.task.goal}

**Success criteria:**
{chr(10).join(f"- {s}" for s in env.task.success_criteria)}

## Verifiable rubric ({len(crits)} criteria)

{crit_lines}

## Tools available to the agent

{chr(10).join(f"- `{t.name}` — {t.description}" for t in env.tools) or "_(none)_"}

## Expert reference

The original call recording is provided in `expert_reference/source_call.wav`
and is used to ground LLM judges on real human behavior (tone, pacing,
de-escalation). This is what makes VoiceEnv judges *grounded* rather than
ungrounded LLM-as-judge.

## License

{env.license}
"""
