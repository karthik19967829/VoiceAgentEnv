"""
Environment runner — the orchestrator that ties simulator, agent, sandbox,
and scorer together into a complete evaluation run.

A run produces:
  - Full transcript (agent + user turns)
  - Tool call log with state mutations
  - Scorecard with per-criterion results
  - A single reward float for RL training

Supports pluggable agent backends: OpenAI API, custom endpoints,
or local speech LLMs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from voiceenv.core.sandbox import Sandbox
from voiceenv.core.schema import VoiceEnvironment
from voiceenv.core.scorer import Scorecard, Scorer
from voiceenv.core.simulator import UserSimulator


@dataclass
class RunResult:
    """Complete output of a single environment run."""

    environment_name: str
    transcript: list[dict[str, Any]]  # each turn: {role, content, audio_path?}
    tool_calls: list[dict[str, Any]]
    final_state: dict[str, Any]
    scorecard: Scorecard
    reward: float
    turn_count: int
    duration_seconds: float
    agent_model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment_name,
            "transcript": self.transcript,
            "tool_calls": self.tool_calls,
            "final_state": self.final_state,
            "scorecard": self.scorecard.to_dict(),
            "reward": round(self.reward, 4),
            "turn_count": self.turn_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "agent_model": self.agent_model,
            "metadata": self.metadata,
        }

    @property
    def verifiable_reward(self) -> float:
        """Deterministic reward only. Safe for RL training."""
        return self.scorecard.verifiable_reward()

    @property
    def soft_reward(self) -> float:
        """LLM-judge reward only. For benchmarks, not RL primary signal."""
        return self.scorecard.soft_reward()

    def to_training_example(self) -> dict[str, Any]:
        """Format as an RL training example with separated reward signals."""
        messages = []
        for turn in self.transcript:
            messages.append({"role": turn["role"], "content": turn["content"]})
        return {
            "messages": messages,
            "reward": self.verifiable_reward,
            "verifiable_reward": self.verifiable_reward,
            "soft_reward": self.soft_reward,
            "blended_reward": self.reward,
            "environment": self.environment_name,
            "tool_calls": self.tool_calls,
            "scores": self.scorecard.to_dict(),
        }


class AgentBackend(Protocol):
    """Protocol for pluggable agent backends."""

    def respond(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return {"content": str, "tool_calls": [{"name": str, "arguments": dict}] | None}"""
        ...


class OpenAIAgentBackend:
    """Agent backend using any OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: OpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        if client:
            self.client = client
        elif base_url:
            self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        else:
            self.client = OpenAI()

    def respond(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        result: dict[str, Any] = {"content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
        return result


class EnvironmentRunner:
    """
    Runs a VoiceEnvironment end-to-end:
      1. Initializes simulator and sandbox
      2. Loops: simulator speaks → agent responds (may call tools) → repeat
      3. Scores the completed run
      4. Returns RunResult with transcript, scores, and reward
    """

    def __init__(
        self,
        env: VoiceEnvironment,
        agent: AgentBackend | None = None,
        agent_model: str = "gpt-4o-mini",
        simulator_model: str = "gpt-4o-mini",
        scorer_model: str = "gpt-4o-mini",
        openai_client: OpenAI | None = None,
    ):
        self.env = env
        self.agent_model = agent_model
        self.client = openai_client or OpenAI()
        self.agent = agent or OpenAIAgentBackend(model=agent_model, client=self.client)
        self.simulator = UserSimulator(
            profile=env.simulator,
            task=env.task,
            world_state=env.world_state,
            model=simulator_model,
            client=self.client,
        )
        self.sandbox = Sandbox(tools=env.tools, initial_state=env.world_state)
        self.scorer = Scorer(rubric=env.rubric, model=scorer_model, client=self.client)

    def run(self) -> RunResult:
        """Execute a complete environment run and return results."""
        start_time = time.time()

        # Build agent system prompt with world state
        system_prompt = self.env.agent_system_prompt
        if self.env.world_state.fields:
            for key, value in self.env.world_state.fields.items():
                system_prompt = system_prompt.replace(f"{{{key}}}", str(value))

        agent_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        transcript: list[dict[str, str]] = []

        # Simulator opens the call
        caller_opening = self.simulator.reset()
        transcript.append({"role": "user", "content": caller_opening})
        agent_messages.append({"role": "user", "content": caller_opening})

        turn_count = 0
        tool_schemas = self.sandbox.get_tool_schemas() if self.env.tools else None

        while turn_count < self.env.voice.max_turns:
            turn_count += 1

            # Agent responds
            agent_response = self.agent.respond(agent_messages, tools=tool_schemas)

            # Handle tool calls
            if agent_response.get("tool_calls"):
                for tc in agent_response["tool_calls"]:
                    tool_result = self.sandbox.execute(tc["name"], tc["arguments"])
                    # Feed tool result back to agent
                    agent_messages.append({
                        "role": "assistant",
                        "content": f"[Calling tool: {tc['name']}({json.dumps(tc['arguments'])})]",
                    })
                    agent_messages.append({
                        "role": "user",
                        "content": f"[Tool result: {json.dumps(tool_result)}]",
                    })

                # Get agent's spoken response after tool calls
                agent_response = self.agent.respond(agent_messages, tools=tool_schemas)

            agent_text = agent_response.get("content", "")
            if agent_text:
                transcript.append({"role": "agent", "content": agent_text})
                agent_messages.append({"role": "assistant", "content": agent_text})

            # Check terminal conditions
            if self.simulator.state.done:
                break

            # Simulator responds
            caller_response = self.simulator.respond(agent_text)
            if not caller_response:
                break
            transcript.append({"role": "user", "content": caller_response})
            agent_messages.append({"role": "user", "content": caller_response})

            if self.simulator.state.done:
                break

        duration = time.time() - start_time

        # Score the run
        scorecard = self.scorer.score(
            transcript=transcript,
            tool_calls=self.sandbox.get_call_log(),
            final_state=self.sandbox.get_state(),
            run_metadata={"turns": turn_count, "duration": duration, "model": self.agent_model},
        )

        return RunResult(
            environment_name=self.env.name,
            transcript=transcript,
            tool_calls=self.sandbox.get_call_log(),
            final_state=self.sandbox.get_state(),
            scorecard=scorecard,
            reward=scorecard.as_reward(),
            turn_count=turn_count,
            duration_seconds=duration,
            agent_model=self.agent_model,
        )
