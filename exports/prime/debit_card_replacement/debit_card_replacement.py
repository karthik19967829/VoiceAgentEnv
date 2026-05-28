"""
VoiceEnv environment: debit_card_replacement
Auto-exported from VoiceEnv spec for Prime Intellect Environments Hub.

A call for replacing a lost debit card.
"""

import json
from typing import Any

ENV_YAML = """
name: debit_card_replacement
description: A call for replacing a lost debit card.
version: 1.0.0
author: voiceenv-ingest
tags:
- auto-ingested
- real-call
- 0002f70f
vertical: support
difficulty: easy
languages:
- en
task:
  goal: Assist the caller in replacing their lost debit card and ensure they have no further requests.
  context: The caller has reported a lost debit card and is requesting a replacement.
  success_criteria:
  - The caller confirms the card to be replaced.
  - The agent successfully closes the call without any further issues.
  failure_conditions: []
  terminal_conditions: []
world_state:
  fields:
    caller_name: Patricia Brown
    issue: Lost debit card
    replacement_requested: true
  description: The environment simulates a bank support call regarding debit card issues.
simulator:
  persona_description: The caller is a customer who is slightly frustrated about losing their debit card but is cooperative
    and straightforward in their requests.
  patience: 0.7
  cooperativeness: 0.9
  skepticism: 0.2
  verbosity: 0.5
  emotional_volatility: 0.4
  dominance: 0.5
  primary_language: en
  secondary_languages: []
  code_switching_probability: 0.0
  formality: 0.5
  filler_word_frequency: 0.3
  interrupt_probability: 0.1
  backchannel_frequency: 0.3
  pause_tolerance_ms: 2000
  topic_drift_probability: 0.1
  hidden_goals: []
  scripted_triggers: {}
tools:
- name: replace_card
  description: Initiates the process to replace a lost debit card.
  parameters:
  - name: card_type
    type: string
    description: Type of card to be replaced.
    required: true
    enum: null
    default: null
  success_rate: 1.0
  latency_ms: 0
  side_effects:
    world_field_to_change: replacement_requested
rubric:
  task_success:
  - name: confirmed_card_replacement
    description: The caller confirms the card to be replaced.
    weight: 1.0
    scoring_type: binary
    llm_judge_prompt: null
    deterministic_check: transcript_contains('my debit card', speaker='user')
    reference_names: []
    grounded_dimensions: []
  - name: successful_call_closure
    description: The agent successfully closes the call without further issues.
    weight: 1.0
    scoring_type: binary
    llm_judge_prompt: null
    deterministic_check: transcript_contains('thank you', speaker='agent') and len(agent_turns) >= 2
    reference_names: []
    grounded_dimensions: []
  compliance:
  - name: greeting_provided
    description: The agent greets the caller at the beginning of the call.
    weight: 1.0
    scoring_type: binary
    llm_judge_prompt: null
    deterministic_check: transcript_contains('hello', speaker='agent')
    reference_names: []
    grounded_dimensions: []
  voice_quality: []
  persona_fidelity: []
  representation: []
  efficiency:
  - name: turns_limit
    description: The call is handled within a reasonable number of turns.
    weight: 1.0
    scoring_type: binary
    llm_judge_prompt: null
    deterministic_check: turns <= 8
    reference_names: []
    grounded_dimensions: []
voice:
  interaction_mode: turn_based
  max_duration_seconds: 300
  max_turns: 50
  sample_rate: 16000
  vad_threshold: 0.5
  silence_timeout_ms: 3000
  acceptable_response_latency_ms: 1500
  good_response_latency_ms: 800
  allow_agent_interrupts: false
  allow_user_interrupts: true
expert_references:
- name: source_call_0002f70f7386445b
  description: The original human-human call from which this environment was auto-extracted. Use this to ground LLM judges
    on real human behavior.
  audio_path: expert_reference/source_call.wav
  transcript: 'agent: hello this is harper valley national bank my name is elizabeth how can i help you today

    caller: hi my name is patricia brown

    caller: i lost my debit card

    caller: can you send me a new one

    agent: which card would you like to replace

    caller: my debit card

    agent: can you repeat that please

    caller: yes my debit card

    agent: is there anything else i can help you with today

    caller: no that was going to be it

    agent: thank you for calling have a great day

    caller: bye'
  annotations:
  - 'Ground truth: this is how a real human agent handled this exact task.'
  - Use for grounded comparison of tone, pacing, and de-escalation.
  segment_start_ms: null
  segment_end_ms: null
agent_system_prompt: You are a bank support agent. Assist the caller with their request to replace a lost debit card. Use
  the 'replace_card' tool if necessary.
license: Apache-2.0
allow_benchmark: true
allow_training: true

"""


def load_environment(
    simulator_model: str = "gpt-4o-mini",
    max_turns: int = 50,
):
    """
    Load this voice environment for evaluation or RL training.

    Returns a verifiers Environment with:
      - A dataset of conversation prompts from the environment spec
      - A rubric that scores agent responses against the voice environment's criteria
    """
    try:
        import verifiers as vf
    except ImportError:
        raise ImportError(
            "verifiers is required for Prime Intellect environments. "
            "Install with: pip install verifiers"
        )

    import yaml
    from voiceenv.core.schema import VoiceEnvironment as VE
    from voiceenv.core.runner import EnvironmentRunner, OpenAIAgentBackend

    env_spec = VE(**yaml.safe_load(ENV_YAML))

    # Build dataset: each item is a conversation starter from the simulator
    dataset = [
        {
            "question": _build_agent_prompt(env_spec),
            "answer": json.dumps(env_spec.task.success_criteria),
            "environment_name": env_spec.name,
        }
    ]

    async def voice_env_reward(completion: list[dict], answer: str, **kwargs) -> float:
        """
        Run the full voice environment simulation and return a reward.

        The completion is the agent's system prompt / behavior. We run a full
        simulated conversation and score it against the environment rubric.
        """
        try:
            agent_content = completion[-1].get("content", "") if completion else ""

            runner = EnvironmentRunner(
                env=env_spec,
                agent_model=kwargs.get("model", "gpt-4o-mini"),
                simulator_model=simulator_model,
            )
            result = runner.run()
            return result.reward
        except Exception as e:
            return 0.0

    rubric = vf.Rubric(funcs=[voice_env_reward])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
    return env


def _build_agent_prompt(env_spec) -> str:
    """Build the agent's initial prompt from the environment spec."""
    prompt = env_spec.agent_system_prompt
    if env_spec.world_state.fields:
        for key, value in env_spec.world_state.fields.items():
            prompt = prompt.replace("{" + key + "}", str(value))

    tools_desc = ""
    if env_spec.tools:
        tool_names = [t.name for t in env_spec.tools]
        tools_desc = f"\n\nAvailable tools: {', '.join(tool_names)}"

    return prompt + tools_desc


# Metadata for the Environments Hub
ENVIRONMENT_METADATA = {
    "name": "debit_card_replacement",
    "description": "A call for replacing a lost debit card.",
    "version": "1.0.0",
    "vertical": "support",
    "difficulty": "easy",
    "tags": ['auto-ingested', 'real-call', '0002f70f'],
    "source": "voiceenv",
}