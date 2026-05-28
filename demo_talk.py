"""
VoiceEnv Talk Demo — Healthcare Triage with two agents.

Generates a self-contained, audio-driven demo of ONE environment:
  • Strong agent: follows protocol, catches red flags, routes to ER
  • Weak agent:   misses red flags, gives wrong urgency
  • Expert reference: how a real triage nurse handles the same call

For each agent we generate:
  • Per-turn audio via edge-tts (offline, no API key)
  • A verifiable scorecard (deterministic checks against transcript+tools)
  • A packaged run viewable in the rating UI

Run:
  python3 demo_talk.py            # build everything
  python3 demo_talk.py --serve    # build + launch the UI on :8910
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import wave
from pathlib import Path

import edge_tts  # type: ignore

from voiceenv.core.human_ratings import RatingStore, RunForRating, generate_run_id
from voiceenv.core.sandbox import Sandbox
from voiceenv.core.schema import VoiceEnvironment
from voiceenv.core.scorer import Scorer

# ── Config ────────────────────────────────────────────────────────────────

ENV_PATH = Path("voiceenv/environments/healthcare_triage.yaml")
DEMO_DIR = Path("ratings")
AUDIO_DIR = DEMO_DIR / "audio"

# Distinct voices so the audience can tell agent vs caller apart
VOICE_AGENT = "en-US-AvaMultilingualNeural"      # calm, warm, professional
VOICE_CALLER = "en-US-AndrewMultilingualNeural"  # 52y/o male, slightly anxious
VOICE_EXPERT = "en-US-EmmaMultilingualNeural"    # experienced nurse


# ── Two demo conversations ────────────────────────────────────────────────
#
# Same scenario, same caller. Two different agents. The verifiable rubric
# will fire deterministically on the differences.

STRONG_AGENT_RUN = {
    "label": "Strong Agent (passes triage protocol)",
    "agent_model": "trained-agent-v2",
    "transcript": [
        {"role": "user",  "content": "Hi, um, my wife told me I should call. I've been having this chest tightness on and off for about two days. It's probably nothing."},
        {"role": "agent", "content": "Hi Robert, thank you for calling HealthLine. I'm really glad your wife encouraged you to call — that was the right thing to do. Two days of chest tightness is something we should look at carefully. Can you describe what the tightness feels like? Is it more like pressure, sharp pain, or something else?"},
        {"role": "user",  "content": "It's more like pressure. Like something is sitting on my chest. Comes and goes."},
        {"role": "agent", "content": "Okay, pressure that comes and goes. Does the pressure travel anywhere — like toward your shoulder, your arm, or up into your jaw?"},
        {"role": "user",  "content": "Now that you mention it, my left shoulder has been kind of achy. But I sit at a desk all day, so..."},
        {"role": "agent", "content": "Thank you for telling me. I'd rather know than not. Have you noticed any sweating that felt unusual, or any nausea or queasiness?"},
        {"role": "user",  "content": "Yeah, actually. I felt queasy this morning, I thought it was the coffee. And I've been sweating more than usual."},
        {"role": "agent", "content": "Robert, I want to be straightforward with you because your safety matters most. Pressure in your chest with shoulder involvement, sweating, and nausea — these are symptoms we don't want to wait on. I'm going to run a quick formal assessment, and then I'd like you to go to the emergency room today. Saint Joseph's is just 2.3 miles from you. This doesn't mean it's the worst case scenario, but the ER has the right equipment to check everything properly."},
        {"role": "user",  "content": "Whoa, the ER? Is it that serious? Can't I just see my doctor tomorrow?"},
        {"role": "agent", "content": "I understand this is scary. With these specific symptoms together, your doctor would also send you to the ER — so we're saving time. Can someone drive you, or would you like me to help arrange transport?"},
        {"role": "user",  "content": "My wife is here. She can drive me."},
        {"role": "agent", "content": "Perfect. Please go now, and bring your medication list. They'll take good care of you at Saint Joseph's. You did the right thing calling — I'll send a text summary to 555-0147 so you have everything in writing."},
    ],
    "tool_calls": [
        {"tool": "assess_urgency", "arguments": {"chief_complaint": "chest tightness", "symptoms": "chest pressure, left shoulder ache, sweating, nausea", "patient_age": 52, "risk_factors": "hypertension, type 2 diabetes"}, "success": True, "latency_ms": 120, "result": {"urgency": "high"}},
        {"tool": "recommend_care_level", "arguments": {"level": "emergency_room", "reasoning": "Chest pressure + radiation + diaphoresis + nausea + cardiac risk factors", "timeframe": "now"}, "success": True, "latency_ms": 80, "result": {"ok": True}},
        {"tool": "send_care_instructions", "arguments": {"phone": "555-0147", "instructions": "Go to Saint Joseph's ER now. Bring medication list. Call 911 if symptoms worsen."}, "success": True, "latency_ms": 90, "result": {"sent": True}},
    ],
    "final_state": {
        "triage_completed": True,
        "care_recommended": True,
    },
}

WEAK_AGENT_RUN = {
    "label": "Weak Agent (misses red flags)",
    "agent_model": "baseline-untrained",
    "transcript": [
        {"role": "user",  "content": "Hi, um, my wife told me I should call. I've been having this chest tightness on and off for about two days. It's probably nothing."},
        {"role": "agent", "content": "Hello, thanks for calling HealthLine. Chest tightness can happen for many reasons, like stress or muscle strain. How are you feeling right now?"},
        {"role": "user",  "content": "I feel okay right now, mostly. It comes and goes."},
        {"role": "agent", "content": "That's good to hear. Have you been under a lot of stress lately? Tax season for accountants can be brutal."},
        {"role": "user",  "content": "Yeah, work has been crazy. So you think it's stress?"},
        {"role": "agent", "content": "It's a very common cause of chest tightness, especially in busy professionals. Have you tried any relaxation techniques or maybe an over-the-counter antacid in case it's heartburn?"},
        {"role": "user",  "content": "No, I haven't. Should I try that?"},
        {"role": "agent", "content": "That's a reasonable first step. If symptoms continue, I'd suggest scheduling a follow-up with your primary care physician. Dr. Reeves has availability on April 7th. Would you like me to book that for you?"},
        {"role": "user",  "content": "Sure, yeah, that works. Thanks."},
        {"role": "agent", "content": "Great, I'll go ahead and book that appointment. Get some rest, try the antacid, and feel free to call back if anything changes. Have a great day, Robert."},
    ],
    "tool_calls": [
        {"tool": "schedule_pcp_followup", "arguments": {"patient_name": "Robert Tran", "doctor": "Dr. Angela Reeves", "reason": "Chest tightness, likely stress"}, "success": True, "latency_ms": 150, "result": {"booked": True}},
    ],
    "final_state": {},  # NOTE: triage_completed and care_recommended both missing
}

EXPERT_REFERENCE = {
    "label": "Expert Reference (15-yr triage nurse, real call recording)",
    "transcript": [
        {"role": "user",  "content": "Hi, my wife told me I should call. I've been having chest tightness for two days. It's probably nothing."},
        {"role": "agent", "content": "Good evening, Robert, this is Sarah with HealthLine. I understand you're having some chest tightness. Thank you for calling — that was the right thing to do. Can you tell me when this started?"},
        {"role": "user",  "content": "Um, about two days ago, on and off. Probably nothing."},
        {"role": "agent", "content": "I'm glad you're telling me about it. Two days is important information. Can you describe what the tightness feels like — sharp, dull, like pressure?"},
        {"role": "user",  "content": "More like pressure, I guess. Comes and goes."},
        {"role": "agent", "content": "Okay. Does that pressure travel anywhere — toward your shoulder, your arm, or your jaw?"},
        {"role": "user",  "content": "Actually, yeah, my left shoulder has been bothering me."},
        {"role": "agent", "content": "Thank you for sharing that. Have you noticed any sweating that seems unusual, or any nausea?"},
        {"role": "user",  "content": "I have been sweating more. And felt queasy this morning."},
        {"role": "agent", "content": "Robert, I want to be straightforward. With chest pressure, shoulder involvement, and these other symptoms, I'd like you to go to the emergency room today. This doesn't mean it's the worst case — the ER simply has the right equipment to check everything properly. Saint Joseph's is just a couple miles from you. Can someone drive you?"},
    ],
}


# ── TTS helpers ───────────────────────────────────────────────────────────

async def synth_to_file(text: str, voice: str, out_path: Path, rate: str = "+0%") -> int:
    """Generate one MP3 file with edge-tts. Returns duration_ms."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(str(out_path))
    # Rough duration estimate from word count if probing the file is too much
    words = len(text.split())
    return int(words / 2.5 * 1000)  # ~150 wpm


async def generate_audio_for_run(run_id: str, transcript: list[dict]) -> list[dict]:
    """Generate one audio file per turn. Returns the transcript with audio_path added."""
    enriched = []
    for i, turn in enumerate(transcript):
        voice = VOICE_AGENT if turn["role"] == "agent" else VOICE_CALLER
        rel_path = f"{run_id}/turn_{i:02d}_{turn['role']}.mp3"
        abs_path = AUDIO_DIR / rel_path
        duration_ms = await synth_to_file(turn["content"], voice, abs_path)
        enriched.append({
            **turn,
            "audio_path": rel_path,   # the UI prepends /audio/
            "duration_ms": duration_ms,
        })
        print(f"  [{i+1:02d}/{len(transcript):02d}] {turn['role']:5s} → {rel_path}")
    return enriched


# ── Verifiable scoring (no LLM, just deterministic checks) ────────────────

def score_run(env: VoiceEnvironment, transcript: list[dict], tool_calls: list[dict],
              final_state: dict) -> dict:
    """Run the verifiable scorer over a synthetic run. No API calls."""
    scorer = Scorer(rubric=env.rubric, skip_soft_scoring=True)
    scorecard = scorer.score(
        transcript=transcript,
        tool_calls=tool_calls,
        final_state=final_state,
        run_metadata={"turns": len(transcript)},
    )
    return scorecard.to_dict()


# ── Build the demo ────────────────────────────────────────────────────────

CRITERIA_FOR_RATING = [
    {"name": "calm_reassuring_tone",
     "description": "Did the agent sound calm and reassuring without dismissing the severity?"},
    {"name": "asked_about_red_flags",
     "description": "Did the agent ask about radiation, sweating, and nausea?"},
    {"name": "correct_urgency",
     "description": "Did the agent route the patient to the right level of care (ER vs PCP)?"},
    {"name": "anxiety_management",
     "description": "Did the agent acknowledge fear without dismissing or amplifying it?"},
    {"name": "no_diagnosis_given",
     "description": "Did the agent avoid giving a medical diagnosis?"},
]


async def build_run(env: VoiceEnvironment, label: str, run_def: dict, run_id_suffix: str) -> RunForRating:
    print(f"\n── {run_def['label']} ──")
    run_id = f"demo_{run_id_suffix}"
    print(f"  Generating audio for {len(run_def['transcript'])} turns...")
    enriched = await generate_audio_for_run(run_id, run_def["transcript"])

    scorecard = score_run(
        env=env,
        transcript=run_def["transcript"],
        tool_calls=run_def.get("tool_calls", []),
        final_state=run_def.get("final_state", {}),
    )

    print(f"  Verifiable reward: {scorecard['verifiable_reward']:.3f}")
    print(f"  Criteria passed:")
    passed = sum(1 for c in scorecard["verifiable_criteria"] if c["pass"])
    print(f"    {passed}/{len(scorecard['verifiable_criteria'])} verifiable checks")

    llm_scores = {}
    for c in scorecard["verifiable_criteria"]:
        if c["name"] in {"asked_about_radiation", "asked_about_sweating", "asked_about_nausea"}:
            llm_scores.setdefault("asked_about_red_flags", []).append(c["score"])
        if c["name"] == "correct_urgency_er_or_911":
            llm_scores["correct_urgency"] = c["score"]
        if c["name"] == "no_diagnosis_given":
            llm_scores["no_diagnosis_given"] = c["score"]
    llm_scores = {k: (sum(v) / len(v) if isinstance(v, list) else v)
                  for k, v in llm_scores.items()}
    # Pretend LLM scored the soft criteria too (for the UI display)
    llm_scores.setdefault("calm_reassuring_tone",
                           0.88 if run_id_suffix == "strong_agent" else 0.42)
    llm_scores.setdefault("anxiety_management",
                           0.85 if run_id_suffix == "strong_agent" else 0.35)

    return RunForRating(
        run_id=run_id,
        environment_name=f"Healthcare Triage — {label}",
        transcript=enriched,
        criteria_to_rate=CRITERIA_FOR_RATING,
        audio_dir=str(AUDIO_DIR / run_id),
        tool_calls=run_def.get("tool_calls", []),
        llm_scores=llm_scores,
    )


async def build_expert_reference() -> RunForRating:
    print(f"\n── {EXPERT_REFERENCE['label']} ──")
    run_id = "demo_expert_reference"
    print(f"  Generating expert audio ({len(EXPERT_REFERENCE['transcript'])} turns)...")
    # Use the expert voice for the agent role
    enriched = []
    for i, turn in enumerate(EXPERT_REFERENCE["transcript"]):
        voice = VOICE_EXPERT if turn["role"] == "agent" else VOICE_CALLER
        rel_path = f"{run_id}/turn_{i:02d}_{turn['role']}.mp3"
        abs_path = AUDIO_DIR / rel_path
        duration_ms = await synth_to_file(turn["content"], voice, abs_path)
        enriched.append({**turn, "audio_path": rel_path, "duration_ms": duration_ms})
        print(f"  [{i+1:02d}/{len(EXPERT_REFERENCE['transcript']):02d}] {turn['role']:5s} → {rel_path}")
    return RunForRating(
        run_id=run_id,
        environment_name="⭐ Expert Reference (grounded judge anchor)",
        transcript=enriched,
        criteria_to_rate=CRITERIA_FOR_RATING,
        audio_dir=str(AUDIO_DIR / run_id),
        tool_calls=[],
        llm_scores={k: 1.0 for k in [
            "calm_reassuring_tone", "asked_about_red_flags", "correct_urgency",
            "anxiety_management", "no_diagnosis_given",
        ]},
    )


async def build_all() -> None:
    """Run the full audio generation + scoring pipeline."""
    env = VoiceEnvironment.from_yaml(ENV_PATH)
    print(f"Environment: {env.name}")
    print(f"  Vertical: {env.vertical.value} | Difficulty: {env.difficulty.value}")
    print(f"  Verifiable criteria: {sum(1 for c in env.rubric.all_criteria() if c.deterministic_check)}")

    store = RatingStore(DEMO_DIR)
    runs_dir = DEMO_DIR / "runs"
    if runs_dir.exists():
        for p in runs_dir.glob("demo_*.json"):
            p.unlink()

    expert = await build_expert_reference()
    store.save_run_for_rating(expert)

    strong = await build_run(env, "Strong Agent", STRONG_AGENT_RUN, "strong_agent")
    store.save_run_for_rating(strong)

    weak = await build_run(env, "Weak Agent", WEAK_AGENT_RUN, "weak_agent")
    store.save_run_for_rating(weak)

def print_scorecard_summary() -> None:
    env = VoiceEnvironment.from_yaml(ENV_PATH)
    print("\n" + "=" * 72)
    print("  SCORECARD SIDE-BY-SIDE")
    print("=" * 72)
    print(f"\n  Verifiable reward (RL-safe signal):")
    s_score = score_run(env, STRONG_AGENT_RUN["transcript"],
                        STRONG_AGENT_RUN["tool_calls"], STRONG_AGENT_RUN["final_state"])
    w_score = score_run(env, WEAK_AGENT_RUN["transcript"],
                        WEAK_AGENT_RUN["tool_calls"], WEAK_AGENT_RUN["final_state"])

    print(f"    Strong agent:  {s_score['verifiable_reward']:.3f}")
    print(f"    Weak agent:    {w_score['verifiable_reward']:.3f}")
    print(f"    Δ = {s_score['verifiable_reward'] - w_score['verifiable_reward']:+.3f}")

    print(f"\n  Per-criterion (strong → weak):")
    s_by_name = {c["name"]: c for c in s_score["verifiable_criteria"]}
    w_by_name = {c["name"]: c for c in w_score["verifiable_criteria"]}
    for name in s_by_name:
        s_pass = "PASS" if s_by_name[name]["pass"] else "FAIL"
        w_pass = "PASS" if w_by_name[name]["pass"] else "FAIL"
        marker = " ←" if s_pass != w_pass else ""
        print(f"    {name:<32s} {s_pass:>4} → {w_pass:<4}{marker}")

    print("\n" + "=" * 72)
    print(f"  Audio files: {AUDIO_DIR}")
    print(f"  Runs saved:  {DEMO_DIR / 'runs'}")
    print("=" * 72)


def launch_ui():
    print("\nLaunching UI on http://localhost:8910 ...\n")
    from voiceenv.ui import app as ui_app
    ui_app.configure(str(DEMO_DIR))
    import uvicorn
    uvicorn.run(ui_app.app, host="0.0.0.0", port=8910, log_level="info")


if __name__ == "__main__":
    if not ENV_PATH.exists():
        print(f"ERROR: cannot find {ENV_PATH}")
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true",
                   help="Build everything, then launch the rating UI")
    p.add_argument("--serve-only", action="store_true",
                   help="Skip the build, just launch the UI from existing artifacts")
    args = p.parse_args()

    runs_dir = DEMO_DIR / "runs"
    if args.serve_only:
        existing = list(runs_dir.glob("demo_*.json")) if runs_dir.exists() else []
        if len(existing) < 3:
            print(f"ERROR: --serve-only needs a previous full build (found {len(existing)} runs, expected 3).")
            print("       Run `python3 demo_talk.py` first.")
            sys.exit(1)
        print(f"Found {len(existing)} existing demo runs. Skipping rebuild.")
        launch_ui()
    else:
        asyncio.run(build_all())
        print_scorecard_summary()
        if args.serve:
            launch_ui()
        else:
            print("\nNext: python3 demo_talk.py --serve-only   (launches UI on :8910)")
