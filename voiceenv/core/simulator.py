"""
Text-first user simulator that plays the human side of a voice conversation.

The simulator is driven by an LLM, conditioned on the SimulatorProfile from
the environment spec. It generates realistic caller behavior: objections,
interruptions, topic drift, hidden goals, and emotional dynamics.

Voice-native simulation (TTS/STT) layers on top of this text loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from voiceenv.core.schema import SimulatorProfile, TaskDefinition, WorldState


@dataclass
class SimulatorTurn:
    role: str  # "user" or "agent"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorState:
    turns: list[SimulatorTurn] = field(default_factory=list)
    hidden_state: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    done_reason: str = ""
    turn_count: int = 0


class UserSimulator:
    """
    LLM-backed user simulator. Generates the 'caller' side of a voice
    conversation based on the SimulatorProfile's behavioral dimensions.
    """

    def __init__(
        self,
        profile: SimulatorProfile,
        task: TaskDefinition,
        world_state: WorldState,
        model: str = "gpt-4o-mini",
        client: OpenAI | None = None,
    ):
        self.profile = profile
        self.task = task
        self.world_state = world_state
        self.model = model
        self.client = client or OpenAI()
        self.state = SimulatorState()
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        behavioral = []
        if self.profile.patience < 0.3:
            behavioral.append("You are impatient and want things resolved quickly.")
        elif self.profile.patience > 0.7:
            behavioral.append("You are patient and willing to go through longer processes.")

        if self.profile.cooperativeness < 0.3:
            behavioral.append("You are uncooperative and resistant to suggestions.")
        elif self.profile.cooperativeness > 0.7:
            behavioral.append("You are cooperative and open to the agent's proposals.")

        if self.profile.skepticism > 0.6:
            behavioral.append("You are skeptical and challenge claims aggressively.")

        if self.profile.emotional_volatility > 0.6:
            behavioral.append("Your emotions can shift quickly — you may become frustrated or excited.")

        if self.profile.dominance > 0.6:
            behavioral.append("You like to control the conversation and don't let the agent lead easily.")

        if self.profile.verbosity > 0.7:
            behavioral.append("You tend to give long, detailed answers and go on tangents.")
        elif self.profile.verbosity < 0.3:
            behavioral.append("You give short, terse answers.")

        if self.profile.interrupt_probability > 0.3:
            behavioral.append("You sometimes interrupt the agent mid-sentence.")

        if self.profile.code_switching_probability > 0.2 and self.profile.secondary_languages:
            langs = ", ".join(self.profile.secondary_languages)
            behavioral.append(f"You occasionally switch to {langs} mid-conversation.")

        hidden_goals_text = ""
        if self.profile.hidden_goals:
            goals = "\n".join(f"  - {g}" for g in self.profile.hidden_goals)
            hidden_goals_text = f"\n\nHIDDEN GOALS (pursue these naturally, don't reveal them directly):\n{goals}"

        triggers_text = ""
        if self.profile.scripted_triggers:
            trigs = "\n".join(f"  - When {k}: respond with \"{v}\"" for k, v in self.profile.scripted_triggers.items())
            triggers_text = f"\n\nSCRIPTED TRIGGERS:\n{trigs}"

        world_context = ""
        if self.world_state.fields:
            world_context = f"\n\nWORLD CONTEXT:\n{json.dumps(self.world_state.fields, indent=2)}"

        return f"""You are a simulated human caller in a voice conversation. You are NOT the AI agent — you are the human the agent is talking to.

PERSONA: {self.profile.persona_description}

BEHAVIORAL TRAITS:
{chr(10).join(f'- {b}' for b in behavioral)}

LANGUAGE:
- Primary: {self.profile.primary_language}
- Formality: {"formal" if self.profile.formality > 0.6 else "casual" if self.profile.formality < 0.4 else "neutral"}
- Use filler words {"frequently" if self.profile.filler_word_frequency > 0.5 else "occasionally" if self.profile.filler_word_frequency > 0.2 else "rarely"}

SITUATION: {self.task.goal}
{self.task.context}{world_context}{hidden_goals_text}{triggers_text}

RULES:
- Respond as the human caller, not as an AI
- Stay in character at all times
- Keep responses conversational and natural (like actual speech, not written text)
- If the conversation reaches a natural conclusion, end your message with [END_CALL]
- Never break character or acknowledge you are a simulator"""

    def reset(self) -> str:
        """Reset simulator and return the opening line from the caller."""
        self.state = SimulatorState()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": "The call has just connected. You called in. Say your opening line."},
            ],
            max_tokens=300,
            temperature=0.8,
        )
        content = response.choices[0].message.content or ""
        self.state.turns.append(SimulatorTurn(role="user", content=content))
        self.state.turn_count += 1
        if "[END_CALL]" in content:
            self.state.done = True
            self.state.done_reason = "caller_ended"
        return content.replace("[END_CALL]", "").strip()

    def respond(self, agent_message: str) -> str:
        """Given the agent's latest message, generate the caller's next response."""
        self.state.turns.append(SimulatorTurn(role="agent", content=agent_message))

        messages = [{"role": "system", "content": self._system_prompt}]
        for turn in self.state.turns:
            if turn.role == "user":
                messages.append({"role": "assistant", "content": turn.content})
            else:
                messages.append({"role": "user", "content": turn.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300,
            temperature=0.8,
        )
        content = response.choices[0].message.content or ""
        self.state.turns.append(SimulatorTurn(role="user", content=content))
        self.state.turn_count += 1

        if "[END_CALL]" in content:
            self.state.done = True
            self.state.done_reason = "caller_ended"

        return content.replace("[END_CALL]", "").strip()

    def get_transcript(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.state.turns]
