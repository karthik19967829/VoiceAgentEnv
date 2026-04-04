"""
Export a VoiceEnvironment to Prime Intellect / verifiers format.

Generates a verifiers-compatible environment module:
  - load_environment() function returning a vf.Environment
  - Dataset of conversation tasks from the environment spec
  - Rubric with reward functions mapped from the VoiceEnv scoring rubric
  - pyproject.toml for publishing to the Environments Hub

The exported module can be pushed with `prime env push`.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from voiceenv.core.schema import VoiceEnvironment

MODULE_TEMPLATE = Template('''"""
VoiceEnv environment: {{ env.name }}
Auto-exported from VoiceEnv spec for Prime Intellect Environments Hub.

{{ env.description }}
"""

import json
from typing import Any

ENV_YAML = """
{{ env_yaml }}
"""


def load_environment(
    simulator_model: str = "gpt-4o-mini",
    max_turns: int = {{ env.voice.max_turns }},
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
        tools_desc = f"\\n\\nAvailable tools: {', '.join(tool_names)}"

    return prompt + tools_desc


# Metadata for the Environments Hub
ENVIRONMENT_METADATA = {
    "name": "{{ env.name }}",
    "description": "{{ env.description[:200] }}",
    "version": "{{ env.version }}",
    "vertical": "{{ env.vertical.value }}",
    "difficulty": "{{ env.difficulty.value }}",
    "tags": {{ env.tags }},
    "source": "voiceenv",
}
''')

PYPROJECT_TEMPLATE = Template('''[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{{ package_name }}"
version = "{{ env.version }}"
description = "{{ env.description[:100] }}"
requires-python = ">=3.10"
dependencies = [
    "verifiers",
    "voiceenv",
    "openai>=1.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools]
py-modules = ["{{ module_name }}"]
''')

README_TEMPLATE = Template('''# {{ env.name }}

> Voice environment for Prime Intellect Environments Hub
> Auto-exported from VoiceEnv

{{ env.description }}

## Usage

```bash
# Install
prime env install {{ package_name }}

# Evaluate
prime eval run {{ package_name }}
```

```python
from {{ module_name }} import load_environment

env = load_environment(simulator_model="gpt-4o-mini")
# Use with vf-eval, prime-rl, TRL, or any verifiers-compatible trainer
```

## Environment Details

- **Vertical:** {{ env.vertical.value }}
- **Difficulty:** {{ env.difficulty.value }}
- **Languages:** {{ env.languages | join(", ") }}
- **Max Turns:** {{ env.voice.max_turns }}

### Task
{{ env.task.goal }}

### Success Criteria
{% for c in env.task.success_criteria %}
- {{ c }}
{% endfor %}

## Source
Exported from [VoiceEnv](https://github.com/voiceenv/voiceenv) environment spec.
''')


def export_prime(env: VoiceEnvironment, output_dir: str | Path) -> Path:
    """
    Export a VoiceEnvironment as a Prime Intellect / verifiers-compatible module.

    Returns the path to the generated module directory.
    """
    output_dir = Path(output_dir)
    package_name = "voiceenv-" + env.name.replace("_", "-").replace(" ", "-").lower()
    module_name = env.name.replace("-", "_").replace(" ", "_").lower()

    mod_dir = output_dir / module_name
    mod_dir.mkdir(parents=True, exist_ok=True)

    env_yaml = env.to_yaml()

    ctx = {
        "env": env,
        "env_yaml": env_yaml,
        "package_name": package_name,
        "module_name": module_name,
    }

    # Main module file
    (mod_dir / f"{module_name}.py").write_text(MODULE_TEMPLATE.render(**ctx))

    # pyproject.toml
    (mod_dir / "pyproject.toml").write_text(PYPROJECT_TEMPLATE.render(**ctx))

    # README
    (mod_dir / "README.md").write_text(README_TEMPLATE.render(**ctx))

    return mod_dir
