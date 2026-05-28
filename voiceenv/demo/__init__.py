"""End-to-end demo: ingest a WAV, run a stateless turn-level eval against a speech
LLM, score against the auto-extracted rubric. One command for the talk."""

from voiceenv.demo.turn_eval import (
    TurnResult,
    DemoResult,
    slice_caller_turns,
    slice_human_responses,
    run_stateless_eval,
    run_grounded_eval,
    score_eval,
    improve_agent_prompt,
)

__all__ = [
    "TurnResult",
    "DemoResult",
    "slice_caller_turns",
    "slice_human_responses",
    "run_stateless_eval",
    "run_grounded_eval",
    "score_eval",
    "improve_agent_prompt",
]
