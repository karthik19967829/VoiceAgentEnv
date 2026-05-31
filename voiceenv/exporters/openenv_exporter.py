"""
Export a VoiceEnvironment to OpenEnv (Meta + HuggingFace) format.

Generates a complete OpenEnv-compatible environment package:
  - Environment server (FastAPI with step/reset/state)
  - Client class (EnvClient subclass)
  - Models (Action, Observation, State dataclasses)
  - Dockerfile for containerized deployment
  - openenv.yaml manifest
  - pyproject.toml

The exported package can be pushed to HuggingFace Spaces with `openenv push`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from jinja2 import Template

from voiceenv.core.schema import VoiceEnvironment

# ── Templates ──

MODELS_TEMPLATE = Template('''"""Auto-generated models for OpenEnv environment: {{ env.name }}"""

from typing import Any
from pydantic import Field

from openenv.core.env_server import Action, Observation, State


# openenv-core >= 0.3 uses Pydantic BaseModel for Action/Observation/State,
# so subclasses must also be Pydantic models (not @dataclass).
class VoiceAction(Action):
    """Agent action in the voice environment."""
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class VoiceObservation(Observation):
    """Observation returned from the voice environment."""
    speaker: str = ""
    content: str = ""
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    world_state: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)


class VoiceState(State):
    """Episode state for the voice environment."""
    turn_count: int = 0
    transcript: list[dict[str, str]] = Field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = Field(default_factory=list)
    world_state: dict[str, Any] = Field(default_factory=dict)
    done_reason: str = ""
''')

SERVER_TEMPLATE = Template('''"""Auto-generated OpenEnv server for: {{ env.name }}"""

import os
import json
import uuid
from pathlib import Path

import yaml
from openenv.core.env_server import Environment

from ..models import VoiceAction, VoiceObservation, VoiceState

ENV_YAML = Path(__file__).parent.parent / "environment.yaml"


class _ReplaySimulator:
    """Deterministic simulator that replays the caller's turns from the
    expert reference transcript stored in env.yaml. Requires no API key
    and produces identical rollouts every time — ideal for the public
    Space and benchmarking. Use the LLM-backed UserSimulator only when
    you explicitly want diverse / OOD caller behavior at training time."""

    def __init__(self, transcript: str):
        self._caller_turns: list[str] = []
        for raw in (transcript or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            for prefix in ("caller:", "user:", "customer:"):
                if line.lower().startswith(prefix):
                    text = line[len(prefix):].strip()
                    if text:
                        self._caller_turns.append(text)
                    break
        self._idx = 0

        class _S:
            done = False
            done_reason = ""
        self.state = _S()

    def reset(self) -> str:
        self._idx = 0
        self.state.done = False
        self.state.done_reason = ""
        if not self._caller_turns:
            self.state.done = True
            self.state.done_reason = "no_caller_turns_in_transcript"
            return "[no caller turns]"
        first = self._caller_turns[0]
        self._idx = 1
        return first

    def respond(self, agent_text: str) -> str:
        if self._idx >= len(self._caller_turns):
            self.state.done = True
            self.state.done_reason = "transcript_exhausted"
            return "[end of call]"
        out = self._caller_turns[self._idx]
        self._idx += 1
        if self._idx >= len(self._caller_turns):
            self.state.done = True
            self.state.done_reason = "transcript_exhausted"
        return out


def _build_simulator(env_spec):
    """Pick a simulator. Default is deterministic replay (no API keys
    required). Set VOICEENV_SIMULATOR=generative to opt into the
    LLM-driven UserSimulator (then OPENAI_API_KEY must also be set)."""
    mode = os.environ.get("VOICEENV_SIMULATOR", "replay").lower()
    if mode == "generative" and os.environ.get("OPENAI_API_KEY"):
        from voiceenv.core.simulator import UserSimulator
        return UserSimulator(
            profile=env_spec.simulator,
            task=env_spec.task,
            world_state=env_spec.world_state,
        )
    transcript = ""
    if env_spec.expert_references:
        transcript = env_spec.expert_references[0].transcript or ""
    return _ReplaySimulator(transcript)


class VoiceAgentEnvironment(Environment):
    """
    OpenEnv-compatible wrapper around a VoiceEnv environment.
    Implements step/reset/state for the voice conversation loop.
    """

    def __init__(self):
        super().__init__()
        from voiceenv.core.schema import VoiceEnvironment as VE

        self._env_spec = VE.from_yaml(ENV_YAML)
        self._simulator = None
        self._sandbox = None
        self._state = VoiceState()

    def reset(self) -> VoiceObservation:
        from voiceenv.core.sandbox import Sandbox

        self._simulator = _build_simulator(self._env_spec)
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
''')

APP_TEMPLATE = Template('''"""FastAPI app for OpenEnv environment: {{ env.name }}"""

from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app

from ..models import VoiceAction, VoiceObservation
from .environment import VoiceAgentEnvironment

# openenv-core 0.3 calls the factory on every HTTP /reset and /step, which
# would erase conversation state between requests. For a single-tenant
# demo Space we hold one persistent env so /reset → /step → /step works
# the way users (and the SDK over WebSocket) expect.
_singleton = None
def _env_factory():
    global _singleton
    if _singleton is None:
        _singleton = VoiceAgentEnvironment()
    return _singleton

app = create_fastapi_app(_env_factory, VoiceAction, VoiceObservation)


_LANDING = """<!doctype html><html><head><meta charset='utf-8'>
<title>VoiceEnv \xb7 {{ env.name }}</title>
<style>body{font-family:-apple-system,system-ui,sans-serif;background:#0d1117;
color:#e6edf3;margin:0;display:flex;align-items:center;justify-content:center;
min-height:100vh;padding:28px}.c{max-width:720px;background:#161b22;
border:1px solid #21262d;border-radius:12px;padding:28px 32px;line-height:1.55}
h1{margin:0 0 8px;font-size:22px}.s{color:#8b949e;font-size:13px;margin-bottom:20px}
a{color:#58a6ff;text-decoration:none;font-weight:500}a:hover{text-decoration:underline}
code{background:#0d1117;border:1px solid #21262d;padding:2px 6px;border-radius:4px;
font-size:12px;color:#d2a8ff}ul{padding-left:18px}li{margin:6px 0}
.b{display:inline-block;padding:3px 10px;border-radius:999px;
background:rgba(63,185,80,0.15);color:#3fb950;border:1px solid #3fb950;
font-size:11px;margin-left:8px}pre{background:#0d1117;border:1px solid #21262d;
border-radius:6px;padding:12px;font-size:12px;overflow-x:auto;color:#e6edf3}
</style></head><body><div class='c'>
<h1>VoiceEnv \xb7 {{ env.name }} <span class='b'>RUNNING</span></h1>
<div class='s'>{{ env.description }}</div>
<p><strong>Live API endpoints:</strong></p><ul>
<li><a href='/docs'>/docs</a> \u2014 interactive Swagger UI</li>
<li><a href='/health'>/health</a>, <a href='/metadata'>/metadata</a>,
<a href='/schema'>/schema</a></li>
<li><code>POST /reset</code>, <code>POST /step</code>, <code>GET /state</code>
\u2014 OpenEnv RL loop</li></ul>
<p><strong>Source:</strong>
<a href='https://github.com/karthik19967829/VoiceAgentEnv'>github.com/karthik19967829/VoiceAgentEnv</a></p>
</div></body></html>"""


@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse(_LANDING)
''')

CLIENT_TEMPLATE = Template('''"""Client for OpenEnv environment: {{ env.name }}"""

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import VoiceAction, VoiceObservation, VoiceState


class {{ client_class }}(EnvClient[VoiceAction, VoiceObservation, VoiceState]):
    """OpenEnv client for the {{ env.name }} voice environment."""

    def _step_payload(self, action: VoiceAction) -> dict:
        return {"content": action.content, "tool_calls": action.tool_calls}

    def _parse_result(self, payload: dict) -> StepResult[VoiceObservation]:
        obs = VoiceObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> VoiceState:
        return VoiceState(**payload)
''')

DOCKERFILE_TEMPLATE = Template('''FROM python:3.11-slim

WORKDIR /app

# git → pip git+https installs (voiceenv from GitHub)
# ffmpeg → audio slicing at runtime
# curl → Docker HEALTHCHECK probe (slim image doesn't ship with it)
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . /app/{{ package_name }}/

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "{{ package_name }}.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
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
    "openenv-core",
    "voiceenv",
    "openai>=1.0",
    "pyyaml>=6.0",
]
''')

OPENENV_YAML_TEMPLATE = Template('''name: {{ env.name }}
description: {{ env.description }}
version: {{ env.version }}
tags: {{ env.tags }}
''')

README_TEMPLATE = Template('''---
title: {{ env.name }}
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
tags:
{% for t in env.tags %}  - {{ t }}
{% endfor %}---

# {{ env.name }}

{{ env.description }}

## Quick Start

```python
from {{ package_name }} import VoiceAction, {{ client_class }}

async with {{ client_class }}(base_url="...") as client:
    result = await client.reset()
    print(result.observation.content)  # Caller's opening line

    result = await client.step(VoiceAction(content="Hello, how can I help you?"))
    print(result.observation.content)  # Caller's response
```

## Environment Details

- **Vertical:** {{ env.vertical.value }}
- **Difficulty:** {{ env.difficulty.value }}
- **Languages:** {{ env.languages | join(", ") }}
- **Max Turns:** {{ env.voice.max_turns }}

### Task
{{ env.task.goal }}

### Scoring Categories
{% for cat in ["task_success", "compliance", "voice_quality", "persona_fidelity", "representation", "efficiency"] %}
{% set criteria = env.rubric[cat] if env.rubric[cat] is defined else [] %}
- **{{ cat }}**: {{ env.rubric.__getattribute__(cat) | length }} criteria
{% endfor %}

## License
{{ env.license }}
''')


def export_openenv(env: VoiceEnvironment, output_dir: str | Path) -> Path:
    """
    Export a VoiceEnvironment as an OpenEnv-compatible package.

    Returns the path to the generated package directory.
    """
    output_dir = Path(output_dir)
    package_name = env.name.replace("-", "_").replace(" ", "_").lower()
    client_class = "".join(word.capitalize() for word in env.name.replace("-", "_").split("_")) + "Env"

    pkg_dir = output_dir / package_name
    server_dir = pkg_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)

    ctx = {"env": env, "package_name": package_name, "client_class": client_class}

    # __init__.py
    init_content = f'"""OpenEnv package for {env.name}"""\n\n'
    init_content += f"from .models import VoiceAction, VoiceObservation, VoiceState\n"
    init_content += f"from .client import {client_class}\n\n"
    init_content += f'__all__ = ["VoiceAction", "VoiceObservation", "VoiceState", "{client_class}"]\n'
    (pkg_dir / "__init__.py").write_text(init_content)

    # Models
    (pkg_dir / "models.py").write_text(MODELS_TEMPLATE.render(**ctx))

    # Client
    (pkg_dir / "client.py").write_text(CLIENT_TEMPLATE.render(**ctx))

    # Server
    (server_dir / "__init__.py").write_text("")
    (server_dir / "environment.py").write_text(SERVER_TEMPLATE.render(**ctx))
    (server_dir / "app.py").write_text(APP_TEMPLATE.render(**ctx))

    # Environment YAML (the original spec)
    env.to_yaml(pkg_dir / "environment.yaml")

    # Dockerfile (must live at repo ROOT so HuggingFace Space picks it up
    # and so the `COPY requirements.txt` resolves against the build context root)
    (pkg_dir / "Dockerfile").write_text(DOCKERFILE_TEMPLATE.render(**ctx))

    # requirements.txt — also at repo ROOT for the same reason
    # `voiceenv` isn't on PyPI yet; install from the public GitHub source so
    # the HuggingFace Space build can resolve it.
    requirements = (
        "openenv-core\n"
        "voiceenv @ git+https://github.com/karthik19967829/VoiceAgentEnv.git@main\n"
        "openai>=1.0\npyyaml>=6.0\nhttpx>=0.25\n"
        "uvicorn[standard]>=0.27\nfastapi>=0.110\n"
    )
    (pkg_dir / "requirements.txt").write_text(requirements)
    # Keep a copy under server/ for local `docker build server/` workflows too.
    (server_dir / "requirements.txt").write_text(requirements)

    # pyproject.toml
    (pkg_dir / "pyproject.toml").write_text(PYPROJECT_TEMPLATE.render(**ctx))

    # openenv.yaml
    (pkg_dir / "openenv.yaml").write_text(OPENENV_YAML_TEMPLATE.render(**ctx))

    # README
    (pkg_dir / "README.md").write_text(README_TEMPLATE.render(**ctx))

    return pkg_dir
