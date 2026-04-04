"""
VoiceEnv environment specification schema.

This is the canonical format for defining voice agent environments.
Environments defined in this schema can be:
  - Run locally against any speech LLM or voice agent API
  - Exported to OpenEnv (Meta + HuggingFace) as Docker environments
  - Exported to Prime Intellect Environments Hub as verifiers modules
  - Used to generate RL training data for speech model post-training
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ── Enums ──


class Vertical(str, Enum):
    SALES = "sales"
    SUPPORT = "support"
    HEALTHCARE = "healthcare"
    COLLECTIONS = "collections"
    SCHEDULING = "scheduling"
    RECRUITING = "recruiting"
    ONBOARDING = "onboarding"
    EMERGENCY = "emergency"
    CUSTOM = "custom"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class InteractionMode(str, Enum):
    TURN_BASED = "turn_based"
    FULL_DUPLEX = "full_duplex"


# ── Tool definitions ──


class ToolParameter(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


class ToolDefinition(BaseModel):
    """A tool the agent can invoke inside the sandbox."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    success_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of tool succeeding (for stress testing)",
    )
    latency_ms: int = Field(
        default=0,
        description="Simulated latency in milliseconds",
    )
    side_effects: dict[str, Any] = Field(
        default_factory=dict,
        description="State mutations this tool causes on success",
    )


# ── World state ──


class WorldState(BaseModel):
    """Initial world state for the environment."""

    fields: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


# ── Simulator (the synthetic caller/user) ──


class SimulatorProfile(BaseModel):
    """Defines how the simulated human behaves."""

    persona_description: str = Field(
        description="Natural language description of who this person is"
    )

    # Behavioral dimensions (0.0 = low, 1.0 = high)
    patience: float = Field(default=0.5, ge=0.0, le=1.0)
    cooperativeness: float = Field(default=0.5, ge=0.0, le=1.0)
    skepticism: float = Field(default=0.5, ge=0.0, le=1.0)
    verbosity: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_volatility: float = Field(default=0.3, ge=0.0, le=1.0)
    dominance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Linguistic dimensions
    primary_language: str = "en"
    secondary_languages: list[str] = Field(default_factory=list)
    code_switching_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    formality: float = Field(default=0.5, ge=0.0, le=1.0)
    filler_word_frequency: float = Field(default=0.3, ge=0.0, le=1.0)

    # Voice interaction behavior
    interrupt_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    backchannel_frequency: float = Field(default=0.3, ge=0.0, le=1.0)
    pause_tolerance_ms: int = Field(default=2000, ge=0)
    topic_drift_probability: float = Field(default=0.1, ge=0.0, le=1.0)

    # Hidden objectives the simulator pursues
    hidden_goals: list[str] = Field(default_factory=list)

    # Specific phrases or behaviors to exhibit
    scripted_triggers: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of trigger conditions to scripted responses",
    )


# ── Expert references ──


class ExpertReference(BaseModel):
    """
    A recording of a human expert handling this environment's task.

    This is what makes the judge grounded. Instead of asking an LLM
    "was the agent empathetic?", we ask "how does the agent compare
    to this expert nurse handling the same triage scenario?"

    The reference provides:
      - The audio of a real expert doing the task well
      - Annotations highlighting what makes it good
      - The transcript (for text-mode fallback)
    """

    name: str = Field(description="Short label, e.g. 'expert_nurse_triage_calm'")
    description: str = Field(
        default="",
        description="What makes this reference recording a good example",
    )
    audio_path: str = Field(
        description="Path to expert audio file (wav/mp3/ogg/flac)"
    )
    transcript: str = Field(
        default="",
        description="Transcript of the expert recording (for text-mode fallback)",
    )
    annotations: list[str] = Field(
        default_factory=list,
        description="Specific things the expert does well in this recording",
    )
    segment_start_ms: int | None = Field(
        default=None,
        description="If only a segment is relevant, start time in ms",
    )
    segment_end_ms: int | None = Field(
        default=None,
        description="If only a segment is relevant, end time in ms",
    )


# ── Scoring ──


class ScoringCriterion(BaseModel):
    name: str
    description: str
    weight: float = Field(default=1.0, ge=0.0)
    scoring_type: str = Field(
        default="binary",
        description="binary, scale_1_5, scale_1_10, float_0_1, llm_judge, or grounded_judge",
    )
    llm_judge_prompt: str | None = Field(
        default=None,
        description="Prompt template for LLM-as-judge scoring",
    )
    deterministic_check: str | None = Field(
        default=None,
        description="Python expression evaluated against run state for deterministic scoring",
    )
    reference_names: list[str] = Field(
        default_factory=list,
        description="Names of ExpertReferences to ground this criterion on (for grounded_judge)",
    )
    grounded_dimensions: list[str] = Field(
        default_factory=list,
        description="Specific audio dimensions to compare against reference "
                    "(e.g. 'tone', 'pacing', 'interruption_handling', 'empathy', 'de_escalation')",
    )


class ScoringRubric(BaseModel):
    """Complete scoring rubric for evaluating a run."""

    task_success: list[ScoringCriterion] = Field(default_factory=list)
    compliance: list[ScoringCriterion] = Field(default_factory=list)
    voice_quality: list[ScoringCriterion] = Field(default_factory=list)
    persona_fidelity: list[ScoringCriterion] = Field(default_factory=list)
    representation: list[ScoringCriterion] = Field(default_factory=list)
    efficiency: list[ScoringCriterion] = Field(default_factory=list)

    def all_criteria(self) -> list[ScoringCriterion]:
        return (
            self.task_success
            + self.compliance
            + self.voice_quality
            + self.persona_fidelity
            + self.representation
            + self.efficiency
        )

    def total_weight(self) -> float:
        return sum(c.weight for c in self.all_criteria())


# ── Voice-specific configuration ──


class VoiceConfig(BaseModel):
    """Voice-specific interaction parameters."""

    interaction_mode: InteractionMode = InteractionMode.TURN_BASED
    max_duration_seconds: int = 300
    max_turns: int = 50

    # Audio parameters
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    silence_timeout_ms: int = 3000

    # Latency thresholds for scoring
    acceptable_response_latency_ms: int = 1500
    good_response_latency_ms: int = 800

    # Interruption policy
    allow_agent_interrupts: bool = False
    allow_user_interrupts: bool = True


# ── Task definition ──


class TaskDefinition(BaseModel):
    """What the agent must accomplish."""

    goal: str = Field(description="What the agent is trying to achieve")
    context: str = Field(
        default="", description="Background context given to the agent"
    )
    success_criteria: list[str] = Field(
        description="Conditions that constitute task success"
    )
    failure_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that constitute task failure",
    )
    terminal_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that end the run (success or failure)",
    )


# ── Top-level environment ──


class VoiceEnvironment(BaseModel):
    """
    A complete voice agent environment definition.

    This is the atomic unit of the VoiceEnv platform. Everything else —
    benchmarks, training data, leaderboards — is built on top of environments.
    """

    # Metadata
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = Field(default_factory=list)
    vertical: Vertical = Vertical.CUSTOM
    difficulty: Difficulty = Difficulty.MEDIUM
    languages: list[str] = Field(default_factory=lambda: ["en"])

    # Core components
    task: TaskDefinition
    world_state: WorldState = Field(default_factory=WorldState)
    simulator: SimulatorProfile
    tools: list[ToolDefinition] = Field(default_factory=list)
    rubric: ScoringRubric
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    # Expert references for grounded judging
    expert_references: list[ExpertReference] = Field(
        default_factory=list,
        description="Expert human recordings that ground the LLM judge. "
                    "Without these, soft scoring falls back to ungrounded LLM-as-judge.",
    )

    # Agent system prompt (injected into the agent under test)
    agent_system_prompt: str = Field(
        default="",
        description="System prompt given to the agent. Can reference {world_state} fields.",
    )

    # Contribution metadata
    license: str = "Apache-2.0"
    allow_benchmark: bool = True
    allow_training: bool = True

    # ── Serialization ──

    def to_yaml(self, path: str | Path | None = None) -> str:
        data = self.model_dump(mode="json")
        content = yaml.dump(data, default_flow_style=False, sort_keys=False, width=120)
        if path:
            Path(path).write_text(content)
        return content

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VoiceEnvironment":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)

    def to_json(self, path: str | Path | None = None) -> str:
        content = self.model_dump_json(indent=2)
        if path:
            Path(path).write_text(content)
        return content

    @classmethod
    def from_json(cls, path: str | Path) -> "VoiceEnvironment":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def generate_json_schema(self) -> dict:
        return self.model_json_schema()
