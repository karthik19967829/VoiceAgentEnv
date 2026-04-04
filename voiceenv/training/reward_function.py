"""
VoiceEnv reward function — the bridge between our environments and VERL/TRL.

This file is designed to be passed directly to VERL as a custom reward function:

  python3 -m verl.trainer.main_ppo \\
      custom_reward_function.path=voiceenv/training/reward_function.py \\
      custom_reward_function.name=voiceenv_reward \\
      ...

It can also be imported by ms-swift or TRL training scripts.

WHAT IT DOES:
  Takes a model completion, runs it through the VoiceEnv scorer with
  verifiable checks (state, tool calls, transcript patterns), and returns
  a scalar reward. No LLM-as-judge calls — purely deterministic, fast,
  and safe for RL training.
"""

from __future__ import annotations

import json
import re
from typing import Any


def voiceenv_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """
    VERL-compatible reward function.

    Args:
        data_source: dataset name (maps to environment name)
        solution_str: model's completion (the conversation)
        ground_truth: expected criteria / environment config (JSON string)
        extra_info: additional context from VERL

    Returns:
        float reward in [0.0, 1.0]
    """
    try:
        config = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (json.JSONDecodeError, TypeError):
        config = {}

    transcript = _parse_transcript(solution_str)
    tool_calls = config.get("tool_calls", [])
    state = config.get("expected_state", {})
    checks = config.get("verifiable_checks", [])

    if not checks:
        return _simple_heuristic_reward(transcript, tool_calls)

    passed = 0
    total_weight = 0.0

    for check in checks:
        weight = check.get("weight", 1.0)
        total_weight += weight
        expr = check.get("check", "")

        eval_context = {
            "state": state,
            "tools": tool_calls,
            "tool_calls": tool_calls,
            "transcript": transcript,
            "turns": len(transcript),
            "agent_turns": [t for t in transcript if t["role"] == "agent"],
            "user_turns": [t for t in transcript if t["role"] == "user"],
            "transcript_contains": lambda pattern, **kw: _transcript_contains(transcript, pattern, **kw),
            "no_transcript_match": lambda pattern, **kw: not _transcript_contains(transcript, pattern, **kw),
            "tool_was_called": lambda name, **kw: _tool_was_called(tool_calls, name, **kw),
            "tool_args_valid": lambda name, arg, vals: _tool_args_valid(tool_calls, name, arg, vals),
            "all_tools_succeeded": lambda: all(tc.get("success", False) for tc in tool_calls),
        }

        try:
            result = eval(expr, {"__builtins__": {}}, eval_context)
            if result:
                passed += weight
        except Exception:
            pass

    return passed / total_weight if total_weight > 0 else 0.0


def compute_reward(
    completion: str,
    environment_config: dict,
) -> float:
    """
    Simplified reward interface for TRL / ms-swift integration.

    Usage in TRL GRPOTrainer:
        def reward_fn(completions, **kwargs):
            return [compute_reward(c, env_config) for c in completions]
    """
    return voiceenv_reward(
        data_source="voiceenv",
        solution_str=completion,
        ground_truth=environment_config,
    )


def _parse_transcript(text: str) -> list[dict[str, str]]:
    """Parse a conversation string back into transcript format."""
    transcript = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\[?(AGENT|USER|ASSISTANT|SYSTEM)\]?:\s*(.*)", line, re.IGNORECASE)
        if match:
            role = match.group(1).lower()
            if role == "assistant":
                role = "agent"
            transcript.append({"role": role, "content": match.group(2)})
    return transcript


def _transcript_contains(
    transcript: list[dict[str, str]],
    pattern: str,
    speaker: str | None = None,
    in_first_n_turns: int | None = None,
    case_sensitive: bool = False,
) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    for i, turn in enumerate(transcript):
        if speaker and turn["role"] != speaker:
            continue
        if in_first_n_turns and i >= in_first_n_turns:
            break
        if re.search(pattern, turn["content"], flags):
            return True
    return False


def _tool_was_called(
    tool_calls: list[dict[str, Any]],
    tool_name: str,
    min_times: int = 1,
    max_times: int | None = None,
) -> bool:
    count = sum(1 for tc in tool_calls if tc.get("tool") == tool_name)
    if count < min_times:
        return False
    if max_times is not None and count > max_times:
        return False
    return True


def _tool_args_valid(
    tool_calls: list[dict[str, Any]],
    tool_name: str,
    arg_name: str,
    valid_values: list[Any],
) -> bool:
    for tc in tool_calls:
        if tc.get("tool") == tool_name:
            value = tc.get("arguments", {}).get(arg_name)
            if value not in valid_values:
                return False
    return True


def _simple_heuristic_reward(
    transcript: list[dict[str, str]],
    tool_calls: list[dict[str, Any]],
) -> float:
    """Fallback when no verifiable checks are configured."""
    score = 0.0
    n = 0

    if transcript:
        agent_turns = [t for t in transcript if t["role"] == "agent"]
        if agent_turns:
            score += 0.3
        if len(transcript) >= 4:
            score += 0.2
        if len(transcript) <= 20:
            score += 0.1
        n = 3

    if tool_calls:
        succeeded = sum(1 for tc in tool_calls if tc.get("success", False))
        if tool_calls:
            score += 0.4 * (succeeded / len(tool_calls))
        n += 1

    return score / max(n * 0.25, 1.0)
