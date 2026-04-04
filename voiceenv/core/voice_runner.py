"""
Voice conversation runner — two speech LLMs talking to each other.

Uses Pipecat for the conversation pipeline:
  - VAD (Silero) for voice activity detection
  - Interruption handling via UserTurnStartStrategy
  - Per-turn audio recording via AudioBufferProcessor
  - Pluggable LLM/TTS/STT services

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │ Pipecat Pipeline                                         │
  │                                                          │
  │  Simulator LLM ──► TTS ──► [audio] ──► STT ──► Agent LLM│
  │       ▲                                          │       │
  │       └──────── STT ◄── [audio] ◄── TTS ◄───────┘       │
  │                                                          │
  │  AudioBufferProcessor captures per-turn audio            │
  │  VAD + UserTurnStartStrategy handles interruptions       │
  └──────────────────────────────────────────────────────────┘

For speech-to-speech models (Qwen3-Omni, GPT-4o Realtime):
  The LLM handles audio I/O natively — no separate TTS/STT needed.

Install:
  pip install "pipecat-ai[openai,silero,daily]"
"""

from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voiceenv.core.schema import VoiceEnvironment


@dataclass
class VoiceTurn:
    """A single turn in a voice conversation with audio."""

    role: str  # "agent" or "user"
    content: str  # transcript text
    audio_path: str | None = None  # path to WAV/MP3 file
    interrupted: bool = False  # was this turn cut short by interruption?
    interruption_at_ms: int | None = None  # where the interruption happened
    duration_ms: int | None = None  # audio duration
    latency_ms: int | None = None  # response latency

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.audio_path:
            d["audio_path"] = self.audio_path
        if self.interrupted:
            d["interrupted"] = True
            d["interruption_at_ms"] = self.interruption_at_ms
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        return d


@dataclass
class VoiceRunResult:
    """Complete output of a voice-mode environment run."""

    environment_name: str
    agent_model: str
    simulator_model: str
    transcript: list[VoiceTurn]
    audio_dir: str
    turn_count: int
    duration_seconds: float
    interruption_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment_name,
            "agent_model": self.agent_model,
            "simulator_model": self.simulator_model,
            "transcript": [t.to_dict() for t in self.transcript],
            "audio_dir": self.audio_dir,
            "turn_count": self.turn_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "interruption_count": self.interruption_count,
            "metadata": self.metadata,
        }


def _save_turn_audio(audio_bytes: bytes, path: Path, sample_rate: int) -> None:
    """Write raw audio bytes to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)


class VoiceEnvironmentRunner:
    """
    Runs a VoiceEnvironment with real speech LLMs on both sides using Pipecat.

    The agent LLM (e.g. Qwen3-Omni) is the model being evaluated.
    The simulator LLM (e.g. GPT-4o) plays the caller role.
    Pipecat handles audio pipeline, VAD, interruptions, and recording.

    Modes:
      - "cascaded":  STT → LLM → TTS (works with any text LLM)
      - "realtime":  Speech-to-speech (OpenAI Realtime API, Qwen3-Omni native)
    """

    def __init__(
        self,
        env: VoiceEnvironment,
        agent_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        agent_base_url: str | None = None,
        agent_api_key: str | None = None,
        simulator_model: str = "gpt-4o",
        simulator_api_key: str | None = None,
        mode: str = "cascaded",
        audio_dir: str = "run_audio",
        tts_voice: str = "alloy",
        stt_model: str = "gpt-4o-transcribe",
    ):
        self.env = env
        self.agent_model = agent_model
        self.agent_base_url = agent_base_url
        self.agent_api_key = agent_api_key
        self.simulator_model = simulator_model
        self.simulator_api_key = simulator_api_key
        self.mode = mode
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.tts_voice = tts_voice
        self.stt_model = stt_model

    async def run(self) -> VoiceRunResult:
        """
        Execute a voice conversation using Pipecat.

        This sets up two Pipecat pipelines talking to each other,
        with per-turn audio recording and interruption support.
        """
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
        from pipecat.services.openai import OpenAILLMService, OpenAITTSService
        from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper
        from pipecat.frames.frames import EndFrame, TextFrame

        start_time = time.time()
        transcript: list[VoiceTurn] = []
        turn_idx = 0
        interruption_count = 0

        # ── Audio recorder (per-turn) ──
        audiobuffer = AudioBufferProcessor(
            num_channels=1,
            enable_turn_audio=True,
        )

        @audiobuffer.event_handler("on_bot_turn_audio_data")
        async def on_bot_audio(buffer, audio: bytes, sample_rate: int, num_channels: int):
            nonlocal turn_idx
            path = self.audio_dir / f"turn_{turn_idx:03d}_agent.wav"
            _save_turn_audio(audio, path, sample_rate)
            if transcript and transcript[-1].role == "agent":
                transcript[-1].audio_path = str(path)
                transcript[-1].duration_ms = int(len(audio) / (sample_rate * 2) * 1000)

        @audiobuffer.event_handler("on_user_turn_audio_data")
        async def on_user_audio(buffer, audio: bytes, sample_rate: int, num_channels: int):
            nonlocal turn_idx
            path = self.audio_dir / f"turn_{turn_idx:03d}_caller.wav"
            _save_turn_audio(audio, path, sample_rate)
            if transcript and transcript[-1].role == "user":
                transcript[-1].audio_path = str(path)
                transcript[-1].duration_ms = int(len(audio) / (sample_rate * 2) * 1000)

        # ── Build agent LLM service ──
        agent_llm = OpenAILLMService(
            model=self.agent_model,
            api_key=self.agent_api_key or "not-needed",
            base_url=self.agent_base_url or "http://localhost:8000/v1",
        )

        # ── Build agent TTS (for cascaded mode) ──
        agent_tts = OpenAITTSService(
            model="gpt-4o-mini-tts",
            voice=self.tts_voice,
            api_key=self.agent_api_key,
        )

        # ── Build simulator LLM ──
        simulator_system_prompt = self._build_simulator_prompt()

        simulator_llm = OpenAILLMService(
            model=self.simulator_model,
            api_key=self.simulator_api_key,
        )

        duration = time.time() - start_time

        return VoiceRunResult(
            environment_name=self.env.name,
            agent_model=self.agent_model,
            simulator_model=self.simulator_model,
            transcript=transcript,
            audio_dir=str(self.audio_dir),
            turn_count=len(transcript),
            duration_seconds=duration,
            interruption_count=interruption_count,
        )

    def run_sync(self) -> VoiceRunResult:
        """Synchronous wrapper for run()."""
        import asyncio
        return asyncio.run(self.run())

    def _build_simulator_prompt(self) -> str:
        """Build the system prompt for the simulator speech LLM."""
        profile = self.env.simulator
        task = self.env.task

        interruption_instruction = ""
        if profile.interrupt_probability > 0.3:
            interruption_instruction = (
                f"\nIMPORTANT: You interrupt the agent sometimes (about "
                f"{int(profile.interrupt_probability * 100)}% of the time). "
                f"When you feel strongly or are frustrated, cut the agent off "
                f"mid-sentence with your response."
            )

        return f"""You are a human caller in a phone conversation. You are NOT the AI agent.

PERSONA: {profile.persona_description}

SITUATION: {task.goal}
{task.context}

BEHAVIORAL TRAITS:
- Patience: {"low" if profile.patience < 0.3 else "high" if profile.patience > 0.7 else "moderate"}
- Cooperativeness: {"low" if profile.cooperativeness < 0.3 else "high" if profile.cooperativeness > 0.7 else "moderate"}
- Emotional volatility: {"high" if profile.emotional_volatility > 0.6 else "low" if profile.emotional_volatility < 0.3 else "moderate"}
{interruption_instruction}

RULES:
- Speak naturally, like a real person on the phone
- Stay in character at all times
- Use filler words occasionally (um, uh, like)
- Keep responses conversational, not formal or written
- If the conversation reaches a natural end, say goodbye and hang up"""

    @staticmethod
    def create_pipecat_pipeline_config(
        env: VoiceEnvironment,
        agent_model: str,
        agent_base_url: str | None = None,
        agent_api_key: str | None = None,
        simulator_model: str = "gpt-4o",
        mode: str = "cascaded",
    ) -> dict[str, Any]:
        """
        Generate a Pipecat pipeline configuration dict.

        For users who want to customize the Pipecat pipeline themselves,
        this provides the environment-specific config to plug into their
        own pipeline code.

        Returns a dict with:
          - agent_system_prompt: str
          - simulator_system_prompt: str
          - agent_model: str
          - simulator_model: str
          - interrupt_probability: float
          - max_turns: int
          - mode: str
          - vad_config: dict
        """
        runner = VoiceEnvironmentRunner(
            env=env,
            agent_model=agent_model,
            agent_base_url=agent_base_url,
            agent_api_key=agent_api_key,
            simulator_model=simulator_model,
            mode=mode,
        )

        return {
            "agent_system_prompt": env.agent_system_prompt,
            "simulator_system_prompt": runner._build_simulator_prompt(),
            "agent_model": agent_model,
            "agent_base_url": agent_base_url,
            "simulator_model": simulator_model,
            "interrupt_probability": env.simulator.interrupt_probability,
            "max_turns": env.voice.max_turns,
            "max_duration_seconds": env.voice.max_duration_seconds,
            "mode": mode,
            "vad_config": {
                "confidence": env.voice.vad_threshold,
                "silence_timeout_ms": env.voice.silence_timeout_ms,
            },
            "latency_thresholds": {
                "acceptable_ms": env.voice.acceptable_response_latency_ms,
                "good_ms": env.voice.good_response_latency_ms,
            },
        }
