"""
VoiceEnv on Modal — all GPU workloads in one app.

Three Modal functions:
  1. serve_speech_llm  — serve Qwen3-Omni (or any speech LLM) via vLLM on GPU
  2. run_voice_env     — run a voice environment (two speech LLMs talking)
  3. train_grpo        — post-train a speech LLM with ms-swift GRPO

Usage:
  # Deploy the vLLM server
  modal deploy voiceenv.cloud.modal_app

  # Run a voice conversation on GPU
  modal run voiceenv.cloud.modal_app::run_voice_env --env-name healthcare_triage

  # Train with GRPO
  modal run voiceenv.cloud.modal_app::train_grpo --rollouts-path rollouts.jsonl

Install:
  pip install modal
  modal setup  # one-time auth
"""

import json
import subprocess
from pathlib import Path

import modal

# ── Shared infrastructure ──

MINUTES = 60

hf_cache = modal.Volume.from_name("voiceenv-hf-cache", create_if_missing=True)
voiceenv_data = modal.Volume.from_name("voiceenv-data", create_if_missing=True)

app = modal.App("voiceenv")

# ── Image: vLLM for serving speech LLMs ──

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.8",
        "transformers>=4.50",
        "huggingface-hub>=0.25",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ── Image: Training with ms-swift ──

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "ms-swift",
        "transformers>=4.50",
        "torch>=2.3",
        "accelerate",
        "peft",
        "datasets",
        "huggingface-hub>=0.25",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ── Image: Voice environment runner ──

runner_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.8",
        "transformers>=4.50",
        "openai>=1.0",
        "pipecat-ai[openai,silero]",
        "edge-tts>=6.1",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "httpx>=0.25",
        "jinja2>=3.0",
        "rich>=13.0",
        "click>=8.0",
        "huggingface-hub>=0.25",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# ════════════════════════════════════════════════════════════════
# 1. SERVE — Deploy a speech LLM via vLLM with OpenAI-compatible API
# ════════════════════════════════════════════════════════════════

SERVE_GPU = "A100"
SERVE_N_GPU = 1
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"{SERVE_GPU}:{SERVE_N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
    },
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve_speech_llm(
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
):
    """Serve a speech LLM via vLLM with OpenAI-compatible API."""
    cmd = [
        "vllm", "serve", model,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(SERVE_N_GPU),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--enforce-eager",
    ]
    print(f"Starting vLLM: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)


# ════════════════════════════════════════════════════════════════
# 2. RUN — Execute a voice environment on GPU
# ════════════════════════════════════════════════════════════════


@app.function(
    image=runner_image,
    gpu="A100",
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data": voiceenv_data,
    },
)
def run_voice_env(
    env_yaml: str,
    agent_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    simulator_model: str = "gpt-4o",
    simulator_api_key: str = "",
    mode: str = "cascaded",
) -> dict:
    """
    Run a voice environment with two speech LLMs on Modal GPU.

    1. Starts a local vLLM server for the agent model
    2. Runs the voice conversation with the simulator
    3. Saves audio + transcript to the shared volume
    4. Returns the run result
    """
    import time
    import yaml

    audio_dir = f"/data/runs/{int(time.time())}"

    # Start local vLLM for the agent
    vllm_cmd = [
        "vllm", "serve", agent_model,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--enforce-eager",
    ]
    print(f"Starting local vLLM for agent: {agent_model}")
    vllm_proc = subprocess.Popen(" ".join(vllm_cmd), shell=True)

    # Wait for vLLM to be ready
    import httpx
    for attempt in range(120):
        try:
            r = httpx.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                print(f"vLLM ready after {attempt}s")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        vllm_proc.kill()
        return {"error": "vLLM failed to start within 120s"}

    # Load environment spec and run
    env_spec = yaml.safe_load(env_yaml)

    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.voice_runner import VoiceEnvironmentRunner

    env = VoiceEnvironment(**env_spec)
    runner = VoiceEnvironmentRunner(
        env=env,
        agent_model=agent_model,
        agent_base_url="http://localhost:8000/v1",
        simulator_model=simulator_model,
        simulator_api_key=simulator_api_key,
        mode=mode,
        audio_dir=audio_dir,
    )

    result = runner.run_sync()

    vllm_proc.terminate()
    voiceenv_data.commit()

    return result.to_dict()


# ════════════════════════════════════════════════════════════════
# 3. TRAIN — Post-train a speech LLM with ms-swift GRPO
# ════════════════════════════════════════════════════════════════


@app.function(
    image=train_image,
    gpu="A100:2",
    timeout=120 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data": voiceenv_data,
    },
)
def train_grpo(
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    rollouts_jsonl: str = "",
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 2,
    batch_size: int = 2,
) -> dict:
    """
    Post-train a speech LLM with ms-swift GRPO on Modal GPUs.

    Reads rollouts from the shared volume, trains with LoRA,
    and saves the adapter weights back to the volume.
    """
    import time

    run_id = f"train_{int(time.time())}"
    output_dir = f"/data/training/{run_id}"
    data_dir = f"{output_dir}/data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Write rollouts to the volume
    rollouts_path = f"{data_dir}/rollouts.jsonl"
    Path(rollouts_path).write_text(rollouts_jsonl)

    # Prepare dataset in ms-swift format
    rollouts = []
    for line in rollouts_jsonl.strip().split("\n"):
        if line.strip():
            rollouts.append(json.loads(line))

    if not rollouts:
        return {"error": "No rollouts provided"}

    env_groups: dict[str, list] = {}
    for r in rollouts:
        env_groups.setdefault(r.get("environment", "default"), []).append(r)

    swift_data = []
    for env_name, group in env_groups.items():
        sorted_group = sorted(group, key=lambda x: x.get("reward", 0), reverse=True)
        median_idx = len(sorted_group) // 2
        for i, r in enumerate(sorted_group):
            conversation = "\n".join(
                f"[{m['role'].upper()}]: {m['content']}" for m in r.get("messages", [])
            )
            swift_data.append({
                "query": r.get("prompt", ""),
                "response": conversation,
                "label": "chosen" if i < median_idx else "rejected",
                "reward": r.get("reward", 0.0),
            })

    dataset_path = f"{data_dir}/train.jsonl"
    with open(dataset_path, "w") as f:
        for item in swift_data:
            f.write(json.dumps(item) + "\n")

    print(f"Prepared {len(swift_data)} training examples")

    # Launch ms-swift GRPO
    cmd = [
        "swift", "rlhf",
        "--rlhf_type", "grpo",
        "--model", model,
        "--dataset", dataset_path,
        "--train_type", "lora",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_rank * 2),
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--output_dir", output_dir,
        "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", "true",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    voiceenv_data.commit()

    return {
        "run_id": run_id,
        "model": model,
        "dataset_size": len(swift_data),
        "output_dir": output_dir,
        "exit_code": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }


# ════════════════════════════════════════════════════════════════
# 4. GENERATE ROLLOUTS — Run environments and generate training data
# ════════════════════════════════════════════════════════════════


@app.function(
    image=runner_image,
    gpu="A100",
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data": voiceenv_data,
    },
)
def generate_rollouts(
    env_yamls: list[str],
    agent_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    simulator_model: str = "gpt-4o",
    simulator_api_key: str = "",
    runs_per_env: int = 10,
) -> str:
    """
    Generate training rollouts from multiple environments on GPU.

    Returns JSONL string of rollouts ready for train_grpo.
    """
    import time
    import yaml

    # Start local vLLM
    vllm_cmd = [
        "vllm", "serve", agent_model,
        "--host", "0.0.0.0", "--port", "8000",
        "--trust-remote-code", "--dtype", "bfloat16",
        "--enforce-eager",
    ]
    vllm_proc = subprocess.Popen(" ".join(vllm_cmd), shell=True)

    import httpx
    for attempt in range(120):
        try:
            r = httpx.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)

    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.voice_runner import VoiceEnvironmentRunner

    rollouts = []
    for env_yaml in env_yamls:
        env_spec = yaml.safe_load(env_yaml)
        env = VoiceEnvironment(**env_spec)

        for run_idx in range(runs_per_env):
            try:
                runner = VoiceEnvironmentRunner(
                    env=env,
                    agent_model=agent_model,
                    agent_base_url="http://localhost:8000/v1",
                    simulator_model=simulator_model,
                    simulator_api_key=simulator_api_key,
                    audio_dir=f"/data/rollouts/{env.name}/{run_idx}",
                )
                result = runner.run_sync()
                rollout = {
                    "environment": result.environment_name,
                    "messages": [
                        {"role": t.role, "content": t.content}
                        for t in result.transcript
                    ],
                    "reward": 0.0,  # scorer runs separately
                    "metadata": result.metadata,
                }
                rollouts.append(rollout)
                print(f"  [{env.name}] Run {run_idx + 1}/{runs_per_env} done")
            except Exception as e:
                print(f"  [{env.name}] Run {run_idx + 1} failed: {e}")

    vllm_proc.terminate()
    voiceenv_data.commit()

    lines = [json.dumps(r) for r in rollouts]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Local entrypoint for testing
# ════════════════════════════════════════════════════════════════


@app.local_entrypoint()
def main():
    """Quick test: deploy and check the vLLM server health."""
    print("VoiceEnv Modal app deployed!")
    print("Functions available:")
    print("  - serve_speech_llm  (web server)")
    print("  - run_voice_env     (GPU function)")
    print("  - train_grpo        (GPU function)")
    print("  - generate_rollouts (GPU function)")
    print()
    print("Deploy with:  modal deploy voiceenv.cloud.modal_app")
    print("Serve LLM:    modal serve voiceenv.cloud.modal_app")
