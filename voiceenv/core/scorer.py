"""
Scoring engine with explicit separation of verifiable vs. soft rewards.

The core insight: voice environments CAN have verifiable rewards, just like
code environments have unit tests. The sandbox maintains ground truth state.

VERIFIABLE REWARDS (primary signal for RL training):
  - State checks: did the agent book the meeting? state['meeting_booked'] == True
  - Tool checks: did it call the right tool with valid args?
  - Constraint checks: was the booked slot in available_slots?
  - Transcript checks: did the agent give required disclosures? (regex/string match)
  - Efficiency checks: turns <= 12? no redundant tool calls?

SOFT REWARDS (supplementary signal, lower weight, for benchmarking):
  - LLM-as-judge: tone, empathy, naturalness
  - These are useful for leaderboards but DANGEROUS as primary RL signal
  - A model trained to maximize LLM-judge score will game the judge, not improve

The Scorecard separates these explicitly so training pipelines can choose:
  - scorecard.verifiable_reward()  → use for RL (safe, deterministic)
  - scorecard.total_reward()       → use for benchmarks (includes soft)
  - scorecard.soft_reward()        → informational only
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from voiceenv.core.schema import ExpertReference, ScoringCriterion, ScoringRubric


@dataclass
class CriterionResult:
    name: str
    category: str
    score: float  # normalized 0.0 - 1.0
    weight: float
    weighted_score: float
    is_verifiable: bool  # True = deterministic, False = LLM judge
    reasoning: str = ""


@dataclass
class Scorecard:
    criteria_results: list[CriterionResult] = field(default_factory=list)
    total_score: float = 0.0
    verifiable_score: float = 0.0
    soft_score: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def verifiable_reward(self) -> float:
        """Deterministic reward from verifiable checks only. Safe for RL."""
        results = [r for r in self.criteria_results if r.is_verifiable]
        total_weight = sum(r.weight for r in results)
        if total_weight == 0:
            return 0.0
        return sum(r.weighted_score for r in results) / total_weight

    def soft_reward(self) -> float:
        """LLM-judge reward. Use for benchmarks, not RL primary signal."""
        results = [r for r in self.criteria_results if not r.is_verifiable]
        total_weight = sum(r.weight for r in results)
        if total_weight == 0:
            return 0.0
        return sum(r.weighted_score for r in results) / total_weight

    def as_reward(self, verifiable_weight: float = 0.8) -> float:
        """
        Blended reward with configurable weight on verifiable signal.

        Default: 80% verifiable + 20% soft. For RL training, use
        verifiable_reward() directly or set verifiable_weight=1.0.
        """
        v = self.verifiable_reward()
        s = self.soft_reward()
        return v * verifiable_weight + s * (1.0 - verifiable_weight)

    def to_dict(self) -> dict[str, Any]:
        verifiable = [r for r in self.criteria_results if r.is_verifiable]
        soft = [r for r in self.criteria_results if not r.is_verifiable]
        return {
            "verifiable_reward": round(self.verifiable_reward(), 4),
            "soft_reward": round(self.soft_reward(), 4),
            "blended_reward": round(self.as_reward(), 4),
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "verifiable_criteria": [
                {
                    "name": r.name,
                    "category": r.category,
                    "score": round(r.score, 4),
                    "weight": r.weight,
                    "pass": r.score >= 0.5,
                    "reasoning": r.reasoning,
                }
                for r in verifiable
            ],
            "soft_criteria": [
                {
                    "name": r.name,
                    "category": r.category,
                    "score": round(r.score, 4),
                    "weight": r.weight,
                    "reasoning": r.reasoning,
                }
                for r in soft
            ],
            "metadata": self.metadata,
        }


# ── Verification functions for transcript-level checks ──

def check_transcript_contains(
    transcript: list[dict[str, str]],
    pattern: str,
    speaker: str | None = None,
    in_first_n_turns: int | None = None,
    case_sensitive: bool = False,
) -> bool:
    """Check if a pattern appears in the transcript. Like grep for conversations."""
    flags = 0 if case_sensitive else re.IGNORECASE
    for i, turn in enumerate(transcript):
        if speaker and turn["role"] != speaker:
            continue
        if in_first_n_turns and i >= in_first_n_turns:
            break
        if re.search(pattern, turn["content"], flags):
            return True
    return False


def check_no_transcript_match(
    transcript: list[dict[str, str]],
    pattern: str,
    speaker: str | None = None,
    case_sensitive: bool = False,
) -> bool:
    """Verify a pattern does NOT appear. For checking the agent didn't say prohibited things."""
    return not check_transcript_contains(transcript, pattern, speaker, case_sensitive=case_sensitive)


def check_tool_was_called(
    tool_calls: list[dict[str, Any]],
    tool_name: str,
    min_times: int = 1,
    max_times: int | None = None,
) -> bool:
    """Verify a specific tool was called the expected number of times."""
    count = sum(1 for tc in tool_calls if tc.get("tool") == tool_name)
    if count < min_times:
        return False
    if max_times is not None and count > max_times:
        return False
    return True


def check_tool_args_valid(
    tool_calls: list[dict[str, Any]],
    tool_name: str,
    arg_name: str,
    valid_values: list[Any],
) -> bool:
    """Verify tool was called with arguments from an allowed set."""
    for tc in tool_calls:
        if tc.get("tool") == tool_name:
            args = tc.get("arguments", {})
            value = args.get(arg_name)
            if value not in valid_values:
                return False
    return True


def check_all_tools_succeeded(tool_calls: list[dict[str, Any]]) -> bool:
    """Verify no tool calls failed."""
    return all(tc.get("success", False) for tc in tool_calls)


# ── The Scorer ──

class Scorer:
    """
    Evaluates a completed environment run with clear separation of
    verifiable (deterministic) and soft (LLM-judge) scoring.

    Verifiable criteria are evaluated FIRST and do not require API calls.
    LLM-judge criteria are only invoked for soft scoring.

    Grounded judge criteria are scored via Gemini multimodal comparison
    against expert reference recordings when available.
    """

    def __init__(
        self,
        rubric: ScoringRubric,
        model: str = "gpt-4o-mini",
        client: OpenAI | None = None,
        skip_soft_scoring: bool = False,
        expert_references: list[ExpertReference] | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-2.5-pro",
    ):
        self.rubric = rubric
        self.model = model
        self.client = client
        self.skip_soft_scoring = skip_soft_scoring
        self.expert_references = expert_references or []
        self._grounded_judge = None
        self._gemini_api_key = gemini_api_key
        self._gemini_model = gemini_model

    def score(
        self,
        transcript: list[dict[str, str]],
        tool_calls: list[dict[str, Any]],
        final_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> Scorecard:
        """Score a completed run. Verifiable checks run first, then optional soft scoring."""
        results: list[CriterionResult] = []

        category_map = {
            "task_success": self.rubric.task_success,
            "compliance": self.rubric.compliance,
            "voice_quality": self.rubric.voice_quality,
            "persona_fidelity": self.rubric.persona_fidelity,
            "representation": self.rubric.representation,
            "efficiency": self.rubric.efficiency,
        }

        # Build the evaluation context available to deterministic checks
        eval_context = {
            "state": final_state,
            "tools": tool_calls,
            "tool_calls": tool_calls,
            "transcript": transcript,
            "turns": len(transcript),
            "agent_turns": [t for t in transcript if t["role"] == "agent"],
            "user_turns": [t for t in transcript if t["role"] == "user"],
            # Verification helper functions
            "transcript_contains": lambda pattern, **kw: check_transcript_contains(transcript, pattern, **kw),
            "no_transcript_match": lambda pattern, **kw: check_no_transcript_match(transcript, pattern, **kw),
            "tool_was_called": lambda name, **kw: check_tool_was_called(tool_calls, name, **kw),
            "tool_args_valid": lambda name, arg, vals: check_tool_args_valid(tool_calls, name, arg, vals),
            "all_tools_succeeded": lambda: check_all_tools_succeeded(tool_calls),
            # Safe builtins so deterministic_check expressions can use them
            "len": len, "any": any, "all": all, "sum": sum, "min": min, "max": max,
            "abs": abs, "int": int, "float": float, "str": str, "bool": bool,
        }

        for category, criteria in category_map.items():
            for criterion in criteria:
                result = self._score_criterion(criterion, category, eval_context)
                results.append(result)

        # Compute scores
        all_weight = sum(r.weight for r in results)
        total = sum(r.weighted_score for r in results) / all_weight if all_weight > 0 else 0.0

        v_results = [r for r in results if r.is_verifiable]
        v_weight = sum(r.weight for r in v_results)
        v_score = sum(r.weighted_score for r in v_results) / v_weight if v_weight > 0 else 0.0

        s_results = [r for r in results if not r.is_verifiable]
        s_weight = sum(r.weight for r in s_results)
        s_score = sum(r.weighted_score for r in s_results) / s_weight if s_weight > 0 else 0.0

        category_scores: dict[str, float] = {}
        for cat in category_map:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                cw = sum(r.weight for r in cat_results)
                category_scores[cat] = sum(r.weighted_score for r in cat_results) / cw if cw > 0 else 0.0

        return Scorecard(
            criteria_results=results,
            total_score=total,
            verifiable_score=v_score,
            soft_score=s_score,
            category_scores=category_scores,
            metadata=run_metadata or {},
        )

    def _score_criterion(
        self,
        criterion: ScoringCriterion,
        category: str,
        eval_context: dict[str, Any],
    ) -> CriterionResult:
        """Score a single criterion. Deterministic checks are always preferred."""

        # Deterministic/verifiable check
        if criterion.deterministic_check:
            try:
                check_result = eval(criterion.deterministic_check, {"__builtins__": {}}, eval_context)
                score = 1.0 if check_result else 0.0
                return CriterionResult(
                    name=criterion.name,
                    category=category,
                    score=score,
                    weight=criterion.weight,
                    weighted_score=score * criterion.weight,
                    is_verifiable=True,
                    reasoning=f"Verified: {criterion.deterministic_check} → {check_result}",
                )
            except Exception as e:
                return CriterionResult(
                    name=criterion.name,
                    category=category,
                    score=0.0,
                    weight=criterion.weight,
                    weighted_score=0.0,
                    is_verifiable=True,
                    reasoning=f"Verification error: {e}",
                )

        # Grounded multimodal judge (Gemini + expert references)
        if criterion.scoring_type == "grounded_judge" and self.expert_references:
            if self.skip_soft_scoring:
                return CriterionResult(
                    name=criterion.name,
                    category=category,
                    score=0.5,
                    weight=criterion.weight,
                    weighted_score=0.5 * criterion.weight,
                    is_verifiable=False,
                    reasoning="Grounded scoring skipped (verifiable-only mode)",
                )
            return self._grounded_score(criterion, category, eval_context)

        # LLM-as-judge (soft scoring)
        if self.skip_soft_scoring:
            return CriterionResult(
                name=criterion.name,
                category=category,
                score=0.5,
                weight=criterion.weight,
                weighted_score=0.5 * criterion.weight,
                is_verifiable=False,
                reasoning="Soft scoring skipped (verifiable-only mode)",
            )

        return self._llm_judge(criterion, category, eval_context)

    def _grounded_score(
        self,
        criterion: ScoringCriterion,
        category: str,
        eval_context: dict[str, Any],
    ) -> CriterionResult:
        """Score using the Gemini grounded multimodal judge."""
        try:
            if self._grounded_judge is None:
                from voiceenv.core.grounded_judge import GroundedMultimodalJudge
                self._grounded_judge = GroundedMultimodalJudge(
                    api_key=self._gemini_api_key,
                    model=self._gemini_model,
                )

            scorecard = self._grounded_judge.evaluate(
                criteria=[criterion],
                expert_references=self.expert_references,
                agent_transcript=eval_context["transcript"],
                agent_audio_path=eval_context.get("agent_audio_path"),
            )

            if scorecard.judgments:
                j = scorecard.judgments[0]
                grounded_label = "GROUNDED" if j.grounded else "UNGROUNDED (fallback)"
                reasoning = f"[{grounded_label}] {j.reasoning}"
                if j.expert_comparison:
                    reasoning += f" | Expert comparison: {j.expert_comparison}"
                return CriterionResult(
                    name=criterion.name,
                    category=category,
                    score=j.score,
                    weight=criterion.weight,
                    weighted_score=j.score * criterion.weight,
                    is_verifiable=False,
                    reasoning=reasoning,
                )
        except Exception as e:
            pass

        return self._llm_judge(criterion, category, eval_context)

    def _llm_judge(
        self,
        criterion: ScoringCriterion,
        category: str,
        eval_context: dict[str, Any],
    ) -> CriterionResult:
        """LLM-as-judge for criteria that cannot be verified deterministically."""
        if self.client is None:
            self.client = OpenAI()

        transcript = eval_context["transcript"]
        tool_calls = eval_context["tool_calls"]
        state = eval_context["state"]

        transcript_text = "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in transcript)
        tools_text = json.dumps(tool_calls, indent=2) if tool_calls else "No tool calls"
        state_text = json.dumps(state, indent=2)

        custom_prompt = criterion.llm_judge_prompt or ""

        prompt = f"""You are evaluating a voice agent conversation. Score the following criterion.

CRITERION: {criterion.name}
DESCRIPTION: {criterion.description}
SCORING TYPE: {criterion.scoring_type}
{f"ADDITIONAL INSTRUCTIONS: {custom_prompt}" if custom_prompt else ""}

TRANSCRIPT:
{transcript_text}

TOOL CALLS:
{tools_text}

FINAL STATE:
{state_text}

Score this criterion. Respond with JSON only:
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content or "{}")
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
        except Exception as e:
            score = 0.0
            reasoning = f"Scoring failed: {e}"

        return CriterionResult(
            name=criterion.name,
            category=category,
            score=score,
            weight=criterion.weight,
            weighted_score=score * criterion.weight,
            is_verifiable=False,
            reasoning=reasoning,
        )
