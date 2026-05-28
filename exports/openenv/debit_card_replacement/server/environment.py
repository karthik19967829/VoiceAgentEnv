"""Auto-generated OpenEnv server for: debit_card_replacement"""

import json
import uuid
from pathlib import Path

import yaml
from openenv.core.env_server import Environment

from ..models import VoiceAction, VoiceObservation, VoiceState

ENV_YAML = Path(__file__).parent.parent / "environment.yaml"


class VoiceAgentEnvironment(Environment):
    """
    OpenEnv-compatible wrapper around a VoiceEnv environment.
    Implements step/reset/state for the voice conversation loop.
    """

    def __init__(self):
        super().__init__()
        from voiceenv.core.schema import VoiceEnvironment as VE
        from voiceenv.core.simulator import UserSimulator
        from voiceenv.core.sandbox import Sandbox

        self._env_spec = VE.from_yaml(ENV_YAML)
        self._simulator = None
        self._sandbox = None
        self._state = VoiceState()

    def reset(self) -> VoiceObservation:
        from voiceenv.core.simulator import UserSimulator
        from voiceenv.core.sandbox import Sandbox

        self._simulator = UserSimulator(
            profile=self._env_spec.simulator,
            task=self._env_spec.task,
            world_state=self._env_spec.world_state,
        )
        self._sandbox = Sandbox(
            tools=self._env_spec.tools,
            initial_state=self._env_spec.world_state,
        )
        self._state = VoiceState(
            episode_id=str(uuid.uuid4()),
            world_state=self._env_spec.world_state.fields.copy(),
        )

        opening = self._simulator.reset()
        self._state.transcript.append({"role": "user", "content": opening})
        self._state.turn_count = 1

        return VoiceObservation(
            speaker="user",
            content=opening,
            world_state=self._sandbox.get_state(),
        )

    def step(self, action: VoiceAction) -> VoiceObservation:
        # Handle tool calls
        tool_results = []
        if action.tool_calls:
            for tc in action.tool_calls:
                result = self._sandbox.execute(tc["name"], tc.get("arguments", {}))
                tool_results.append({"tool": tc["name"], "result": result})
                self._state.tool_call_log.append({"tool": tc["name"], "result": result})

        # Record agent turn
        if action.content:
            self._state.transcript.append({"role": "agent", "content": action.content})

        # Check if conversation is done
        if self._simulator.state.done:
            self._state.done_reason = self._simulator.state.done_reason
            return VoiceObservation(
                done=True,
                reward=self._compute_reward(),
                speaker="system",
                content="[Call ended]",
                tool_results=tool_results,
                world_state=self._sandbox.get_state(),
            )

        # Get simulator response
        if self._state.turn_count >= self._env_spec.voice.max_turns:
            self._state.done_reason = "max_turns_exceeded"
            return VoiceObservation(
                done=True,
                reward=self._compute_reward(),
                speaker="system",
                content="[Max turns exceeded]",
                tool_results=tool_results,
                world_state=self._sandbox.get_state(),
            )

        caller_response = self._simulator.respond(action.content)
        self._state.transcript.append({"role": "user", "content": caller_response})
        self._state.turn_count += 1

        done = self._simulator.state.done
        if done:
            self._state.done_reason = self._simulator.state.done_reason

        return VoiceObservation(
            done=done,
            reward=self._compute_reward() if done else None,
            speaker="user",
            content=caller_response,
            tool_results=tool_results,
            world_state=self._sandbox.get_state(),
        )

    def _compute_reward(self) -> float:
        """Quick deterministic reward from rubric checks."""
        score = 0.0
        total_weight = 0.0
        state = self._sandbox.get_state() if self._sandbox else {}
        for criterion in self._env_spec.rubric.all_criteria():
            total_weight += criterion.weight
            if criterion.deterministic_check:
                try:
                    result = eval(
                        criterion.deterministic_check,
                        {"state": state, "tools": self._state.tool_call_log,
                         "turns": self._state.turn_count},
                    )
                    if result:
                        score += criterion.weight
                except Exception:
                    pass
        return score / total_weight if total_weight > 0 else 0.0

    @property
    def state(self) -> VoiceState:
        return self._state