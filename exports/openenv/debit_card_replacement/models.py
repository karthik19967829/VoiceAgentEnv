"""Auto-generated models for OpenEnv environment: debit_card_replacement"""

from dataclasses import dataclass, field
from typing import Any

from openenv.core.env_server import Action, Observation, State


@dataclass
class VoiceAction(Action):
    """Agent action in the voice environment."""
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VoiceObservation(Observation):
    """Observation returned from the voice environment."""
    speaker: str = ""
    content: str = ""
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    world_state: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class VoiceState(State):
    """Episode state for the voice environment."""
    turn_count: int = 0
    transcript: list[dict[str, str]] = field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    world_state: dict[str, Any] = field(default_factory=dict)
    done_reason: str = ""