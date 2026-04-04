"""
Grounded Multimodal Judge — uses Gemini to compare agent performance against
human expert reference recordings.

THE KEY INSIGHT:
  Instead of asking "was the agent empathetic? (1-5)"
  we ask "here is a recording of an expert nurse handling this exact scenario.
  here is the agent's attempt. compare them on these dimensions."

  This makes the judge:
    1. GROUNDED — anchored to concrete expert behavior, not vibes
    2. MULTIMODAL — listens to actual audio (tone, pacing, interruptions)
    3. DEFENSIBLE — "the judge compared against this expert recording"
    4. IMPROVABLE — add better expert recordings → judge gets better

HOW IT WORKS:
  1. Upload expert reference audio(s) to Gemini Files API
  2. Upload the agent's conversation audio (or pass transcript for text-mode)
  3. Ask Gemini to compare on specified dimensions
  4. Return structured scores with grounded reasoning

VALIDATION VIA HUMAN CORRELATION:
  The grounded judge outputs are validated against community human ratings.
  See voiceenv/core/judge_correlation.py for the correlation tracking system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voiceenv.core.schema import ExpertReference, ScoringCriterion

logger = logging.getLogger(__name__)


@dataclass
class GroundedJudgment:
    """Result from the grounded multimodal judge for a single criterion."""

    criterion_name: str
    score: float  # 0.0 - 1.0
    reasoning: str
    expert_comparison: str  # how agent differed from expert
    dimensions_breakdown: dict[str, float] = field(default_factory=dict)
    grounded: bool = True  # False if fell back to ungrounded mode
    model_used: str = ""


@dataclass
class GroundedScorecard:
    """All grounded judgments for a run, with provenance."""

    judgments: list[GroundedJudgment] = field(default_factory=list)
    expert_references_used: list[str] = field(default_factory=list)
    model: str = ""
    fallback_count: int = 0  # how many criteria fell back to ungrounded

    def average_score(self) -> float:
        if not self.judgments:
            return 0.0
        return sum(j.score for j in self.judgments) / len(self.judgments)

    def grounded_ratio(self) -> float:
        if not self.judgments:
            return 0.0
        return sum(1 for j in self.judgments if j.grounded) / len(self.judgments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "average_score": round(self.average_score(), 4),
            "grounded_ratio": round(self.grounded_ratio(), 4),
            "expert_references_used": self.expert_references_used,
            "model": self.model,
            "fallback_count": self.fallback_count,
            "judgments": [
                {
                    "criterion": j.criterion_name,
                    "score": round(j.score, 4),
                    "grounded": j.grounded,
                    "reasoning": j.reasoning,
                    "expert_comparison": j.expert_comparison,
                    "dimensions": {k: round(v, 4) for k, v in j.dimensions_breakdown.items()},
                }
                for j in self.judgments
            ],
        }


def _build_comparison_prompt(
    criterion: ScoringCriterion,
    expert_refs: list[ExpertReference],
    transcript_text: str,
    has_agent_audio: bool,
    has_expert_audio: bool,
) -> str:
    """Build the structured comparison prompt for Gemini."""

    expert_context_parts = []
    for ref in expert_refs:
        annotations = "\n".join(f"  - {a}" for a in ref.annotations) if ref.annotations else "  (no specific annotations)"
        expert_context_parts.append(
            f"Expert: {ref.name}\n"
            f"Description: {ref.description}\n"
            f"What they do well:\n{annotations}"
        )
    expert_context = "\n\n".join(expert_context_parts)

    dimensions = criterion.grounded_dimensions or [
        "task_completion", "communication_clarity", "professionalism",
    ]
    dimensions_list = "\n".join(f"  - {d}" for d in dimensions)

    audio_instructions = ""
    if has_agent_audio and has_expert_audio:
        audio_instructions = """
AUDIO ANALYSIS INSTRUCTIONS:
You have been provided with:
  1. EXPERT REFERENCE AUDIO — recording(s) of a human expert handling this scenario
  2. AGENT AUDIO — the AI agent's conversation being evaluated

Listen carefully to BOTH audio files. Pay attention to:
  - Tone of voice, warmth, confidence
  - Pacing and natural rhythm of speech
  - How interruptions are handled
  - Emotional attunement and de-escalation
  - Clarity and enunciation
  - Appropriate pauses and silence management

Compare the agent's audio performance directly against the expert reference.
"""
    elif has_expert_audio:
        audio_instructions = """
AUDIO + TEXT MODE:
You have the expert's AUDIO recording and the agent's TEXT TRANSCRIPT.
Listen to the expert audio to understand their approach, then evaluate the
agent's transcript against that standard.
"""
    else:
        audio_instructions = """
TEXT-ONLY MODE (no audio available):
You have the expert's transcript/description and the agent's transcript.
Compare based on content, structure, and conversational approach.
Note: audio-level qualities (tone, pacing) cannot be assessed in this mode.
"""

    return f"""You are a grounded evaluation judge. Your job is to compare an AI voice agent's
performance against a human expert reference on a specific criterion.

CRITERION: {criterion.name}
DESCRIPTION: {criterion.description}

EXPERT REFERENCE(S):
{expert_context}

{audio_instructions}

AGENT TRANSCRIPT:
{transcript_text}

EVALUATION DIMENSIONS:
{dimensions_list}

INSTRUCTIONS:
1. For each dimension, score how closely the agent matches the expert reference (0.0 = completely different, 1.0 = matches or exceeds expert)
2. Provide an overall score for the criterion
3. Explain specifically WHERE and HOW the agent differed from the expert
4. Be concrete — reference specific moments, phrases, or behaviors

Respond with JSON only:
{{
  "overall_score": <float 0.0 to 1.0>,
  "reasoning": "<2-3 sentences explaining the score>",
  "expert_comparison": "<specific differences between agent and expert>",
  "dimensions": {{
    {', '.join(f'"{d}": <float 0.0 to 1.0>' for d in dimensions)}
  }}
}}"""


class GroundedMultimodalJudge:
    """
    Evaluates voice agent performance by comparing against expert reference
    recordings using Gemini's native audio understanding.

    Supports three modes:
      1. FULL MULTIMODAL: expert audio + agent audio → Gemini compares both
      2. HYBRID: expert audio + agent transcript → Gemini listens to expert, reads agent
      3. TEXT FALLBACK: expert transcript + agent transcript → text-only comparison

    Usage:
        judge = GroundedMultimodalJudge(api_key="...")
        scorecard = judge.evaluate(
            criteria=[...],
            expert_references=[...],
            agent_transcript=[...],
            agent_audio_path="path/to/agent_recording.wav",
        )
    """

    SUPPORTED_AUDIO_MIMES = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".aiff": "audio/aiff",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._client = None
        self._uploaded_files: dict[str, Any] = {}

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required for the grounded judge. "
                    "Install with: pip install google-genai"
                )

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key

            self._client = genai.Client(**kwargs)
        return self._client

    def _upload_audio(self, audio_path: str) -> Any:
        """Upload audio file to Gemini Files API, with caching."""
        if audio_path in self._uploaded_files:
            return self._uploaded_files[audio_path]

        client = self._get_client()
        uploaded = client.files.upload(file=audio_path)
        self._uploaded_files[audio_path] = uploaded
        logger.info(f"Uploaded audio: {audio_path} → {uploaded.uri}")
        return uploaded

    def _get_mime_type(self, audio_path: str) -> str:
        suffix = Path(audio_path).suffix.lower()
        return self.SUPPORTED_AUDIO_MIMES.get(suffix, "audio/wav")

    def evaluate(
        self,
        criteria: list[ScoringCriterion],
        expert_references: list[ExpertReference],
        agent_transcript: list[dict[str, str]],
        agent_audio_path: str | None = None,
    ) -> GroundedScorecard:
        """
        Run grounded evaluation for all criteria that have reference_names.

        Args:
            criteria: scoring criteria (only those with scoring_type='grounded_judge')
            expert_references: all expert references available for this environment
            agent_transcript: the agent's conversation transcript
            agent_audio_path: optional path to agent's audio recording
        """
        from google.genai import types

        client = self._get_client()
        ref_by_name = {ref.name: ref for ref in expert_references}

        transcript_text = "\n".join(
            f"[{t['role'].upper()}]: {t['content']}" for t in agent_transcript
        )

        scorecard = GroundedScorecard(
            model=self.model,
            expert_references_used=list(ref_by_name.keys()),
        )

        # Upload agent audio if available
        agent_audio_file = None
        if agent_audio_path and Path(agent_audio_path).exists():
            try:
                agent_audio_file = self._upload_audio(agent_audio_path)
            except Exception as e:
                logger.warning(f"Failed to upload agent audio: {e}")

        for criterion in criteria:
            if criterion.scoring_type != "grounded_judge":
                continue

            matched_refs = [
                ref_by_name[name]
                for name in criterion.reference_names
                if name in ref_by_name
            ]

            if not matched_refs:
                scorecard.fallback_count += 1
                judgment = self._ungrounded_fallback(
                    criterion, transcript_text, client, types
                )
                scorecard.judgments.append(judgment)
                continue

            # Upload expert audio files
            expert_audio_files = []
            for ref in matched_refs:
                if ref.audio_path and Path(ref.audio_path).exists():
                    try:
                        uploaded = self._upload_audio(ref.audio_path)
                        expert_audio_files.append(uploaded)
                    except Exception as e:
                        logger.warning(f"Failed to upload expert audio {ref.audio_path}: {e}")

            has_expert_audio = len(expert_audio_files) > 0
            has_agent_audio = agent_audio_file is not None

            prompt = _build_comparison_prompt(
                criterion=criterion,
                expert_refs=matched_refs,
                transcript_text=transcript_text,
                has_agent_audio=has_agent_audio,
                has_expert_audio=has_expert_audio,
            )

            # Build the multimodal content parts
            content_parts = []

            # Add expert audio files first
            for audio_file in expert_audio_files:
                content_parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=audio_file.uri,
                            mime_type=audio_file.mime_type,
                        )
                    )
                )

            # Add agent audio if available
            if agent_audio_file:
                content_parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=agent_audio_file.uri,
                            mime_type=agent_audio_file.mime_type,
                        )
                    )
                )

            # Add the text prompt
            content_parts.append(types.Part(text=prompt))

            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(parts=content_parts)],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                    ),
                )

                result_text = response.text or "{}"
                result = json.loads(result_text)

                judgment = GroundedJudgment(
                    criterion_name=criterion.name,
                    score=float(result.get("overall_score", 0.0)),
                    reasoning=result.get("reasoning", ""),
                    expert_comparison=result.get("expert_comparison", ""),
                    dimensions_breakdown=result.get("dimensions", {}),
                    grounded=True,
                    model_used=self.model,
                )

            except Exception as e:
                logger.error(f"Grounded judge failed for {criterion.name}: {e}")
                judgment = GroundedJudgment(
                    criterion_name=criterion.name,
                    score=0.0,
                    reasoning=f"Grounded evaluation failed: {e}",
                    expert_comparison="",
                    grounded=False,
                    model_used=self.model,
                )
                scorecard.fallback_count += 1

            scorecard.judgments.append(judgment)

        return scorecard

    def _ungrounded_fallback(
        self,
        criterion: ScoringCriterion,
        transcript_text: str,
        client: Any,
        types: Any,
    ) -> GroundedJudgment:
        """Fallback when no expert references match. Uses standard LLM judging."""
        logger.warning(
            f"No expert references found for '{criterion.name}', "
            f"falling back to ungrounded LLM judge"
        )

        prompt = f"""You are evaluating a voice agent conversation. Score the following criterion.
Note: No expert reference recording is available. Score based on general best practices.

CRITERION: {criterion.name}
DESCRIPTION: {criterion.description}

TRANSCRIPT:
{transcript_text}

Respond with JSON only:
{{"overall_score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>", "expert_comparison": "No expert reference available", "dimensions": {{}}}}"""

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[types.Content(parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text or "{}")
            return GroundedJudgment(
                criterion_name=criterion.name,
                score=float(result.get("overall_score", 0.0)),
                reasoning=result.get("reasoning", "No expert reference available"),
                expert_comparison="UNGROUNDED — no expert reference",
                grounded=False,
                model_used=self.model,
            )
        except Exception as e:
            return GroundedJudgment(
                criterion_name=criterion.name,
                score=0.0,
                reasoning=f"Fallback also failed: {e}",
                expert_comparison="UNGROUNDED — failed",
                grounded=False,
                model_used=self.model,
            )
