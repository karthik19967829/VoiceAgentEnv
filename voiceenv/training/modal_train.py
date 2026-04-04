"""
Fine-tune Qwen3-Omni on VoiceEnv environments using Modal serverless GPUs.

Modal provides on-demand H100/A100 GPUs without managing infrastructure.
This script handles the full pipeline:
  1. Generate rollouts from VoiceEnv environments on a GPU
  2. GRPO fine-tune Qwen3-Omni-30B-A3B using those rollouts
  3. Save trained LoRA adapter to Modal Volume for download

The end-to-end flow:
  # Generate rollouts + train in one shot
  modal run voiceenv/training/modal_train.py

  # Or step by step
  modal run voiceenv/training/modal_train.py::generate
  modal run voiceenv/training/modal_train.py::train --rollouts /data/rollouts.jsonl

  # Compare base vs trained
  modal run voiceenv/training/modal_train.py::evaluate

Requirements:
  pip install modal
  modal setup  # one-time auth
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import modal

# ── Modal app + images ──

app = modal.App("voiceenv-qwen3-omni-training")

# Image for rollout generation (needs openai + voiceenv but NOT heavy ML libs)
rollout_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai>=1.0",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "httpx>=0.25",
        "rich>=13.0",
        "click>=8.0",
        "jinja2>=3.0",
    )
)

# Image for GRPO training (heavy ML dependencies)
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.50",
        "trl>=0.15",
        "peft>=0.14",
        "datasets>=3.0",
        "accelerate>=1.2",
        "bitsandbytes>=0.45",
        "flash-attn>=2.7",
        "openai>=1.0",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "httpx>=0.25",
        "rich>=13.0",
        "click>=8.0",
        "jinja2>=3.0",
        "wandb>=0.19",
    )
    .env({"HF_HOME": "/model_cache"})
)

# ── Volumes ──

model_cache = modal.Volume.from_name("voiceenv-model-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("voiceenv-data", create_if_missing=True)
output_volume = modal.Volume.from_name("voiceenv-output", create_if_missing=True)


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    rollouts_path: str = "/data/rollouts.jsonl"
    output_dir: str = "/output/voiceenv-qwen3-omni"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_completion_length: int = 2048
    max_steps: int = -1  # -1 = use num_epochs
    warmup_ratio: float = 0.1

    # Logging
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50


# ── Step 1: Generate rollouts ──

@app.function(
    image=rollout_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=3600,
)
def generate(
    runs_per_env: int = 20,
    agent_model: str = "gpt-4o-mini",
    simulator_model: str = "gpt-4o-mini",
):
    """Generate training rollouts from VoiceEnv environments."""
    import json
    import sys

    sys.path.insert(0, "/data")

    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.runner import EnvironmentRunner

    # Copy built-in environments to data volume if not present
    env_dir = pathlib.Path("/data/environments")
    if not env_dir.exists():
        env_dir.mkdir(parents=True, exist_ok=True)
        # Load from package
        import voiceenv.environments as envmod
        src_dir = pathlib.Path(envmod.__file__).parent
        for yaml_file in src_dir.glob("*.yaml"):
            dest = env_dir / yaml_file.name
            dest.write_text(yaml_file.read_text())
        print(f"Copied {len(list(env_dir.glob('*.yaml')))} environments to {env_dir}")

    yaml_files = sorted(env_dir.glob("*.yaml"))
    print(f"Running {len(yaml_files)} environments × {runs_per_env} runs each")

    rollouts = []
    for yaml_file in yaml_files:
        env = VoiceEnvironment.from_yaml(yaml_file)
        print(f"\n--- {env.name} ---")

        for run_idx in range(runs_per_env):
            try:
                runner = EnvironmentRunner(
                    env=env,
                    agent_model=agent_model,
                    simulator_model=simulator_model,
                )
                result = runner.run()

                rollout = {
                    "prompt": env.agent_system_prompt,
                    "messages": result.transcript,
                    "reward": result.reward,
                    "environment": env.name,
                    "vertical": env.vertical.value,
                    "difficulty": env.difficulty.value,
                    "turn_count": result.turn_count,
                    "tool_calls": result.tool_calls,
                    "category_scores": result.scorecard.category_scores,
                }
                rollouts.append(rollout)
                print(f"  Run {run_idx+1}/{runs_per_env}: reward={result.reward:.3f} turns={result.turn_count}")

            except Exception as e:
                print(f"  Run {run_idx+1}/{runs_per_env}: FAILED - {e}")

    output_path = pathlib.Path("/data/rollouts.jsonl")
    with output_path.open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    data_volume.commit()

    avg_reward = sum(r["reward"] for r in rollouts) / len(rollouts) if rollouts else 0
    print(f"\nGenerated {len(rollouts)} rollouts, avg reward: {avg_reward:.4f}")
    print(f"Saved to: {output_path}")
    return len(rollouts)


# ── Step 2: GRPO fine-tuning on GPU ──

@app.function(
    image=train_image,
    gpu="H100",
    volumes={
        "/model_cache": model_cache,
        "/data": data_volume,
        "/output": output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 3600,
    retries=modal.Retries(initial_delay=0.0, max_retries=2),
)
def train(config: Optional[TrainingConfig] = None):
    """GRPO fine-tune Qwen3-Omni on VoiceEnv rollouts using H100."""
    import json
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, get_peft_model

    if config is None:
        config = TrainingConfig()

    print(f"=== VoiceEnv GRPO Training ===")
    print(f"Model: {config.model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"LoRA rank: {config.lora_rank}, 4-bit: {config.use_4bit}")

    # Load rollouts
    rollouts = []
    with open(config.rollouts_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    print(f"Loaded {len(rollouts)} rollouts")
    avg_reward = sum(r["reward"] for r in rollouts) / len(rollouts)
    print(f"Average reward: {avg_reward:.4f}")

    # Prepare dataset
    env_groups: dict[str, list] = {}
    for r in rollouts:
        env_groups.setdefault(r["environment"], []).append(r)

    training_examples = []
    for env_name, group in env_groups.items():
        sorted_group = sorted(group, key=lambda x: x["reward"], reverse=True)
        median_reward = sorted_group[len(sorted_group) // 2]["reward"]

        for r in sorted_group:
            conversation = ""
            for msg in r["messages"]:
                role = msg["role"].upper()
                conversation += f"[{role}]: {msg['content']}\n"

            training_examples.append({
                "prompt": r["prompt"],
                "completion": conversation,
                "reward": r["reward"],
                "environment": r["environment"],
            })

    dataset = Dataset.from_list(training_examples)
    print(f"Training dataset: {len(dataset)} examples")

    # Load model with quantization
    print(f"\nLoading {config.model_name}...")

    quant_config = None
    if config.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # GRPO training config
    output_dir = config.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_completion_length=config.max_completion_length,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    if config.max_steps > 0:
        grpo_config.max_steps = config.max_steps

    def reward_fn(completions, **kwargs):
        return [ex.get("reward", 0.0) for ex in completions]

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    print(f"\nStarting GRPO training...")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    output_volume.commit()
    print(f"\nTraining complete! Model saved to {output_dir}")
    print(f"Download with: modal volume get voiceenv-output {output_dir}")

    return output_dir


# ── Step 3: Evaluate base vs trained ──

@app.function(
    image=rollout_image,
    volumes={"/data": data_volume, "/output": output_volume},
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=3600,
)
def evaluate(
    base_model: str = "gpt-4o-mini",
    trained_base_url: Optional[str] = None,
    runs_per_env: int = 5,
):
    """Run environments against base and trained models, compare scores."""
    import json

    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.runner import EnvironmentRunner, OpenAIAgentBackend

    env_dir = pathlib.Path("/data/environments")
    yaml_files = sorted(env_dir.glob("*.yaml"))

    results = {"base": [], "trained": []}

    for yaml_file in yaml_files:
        env = VoiceEnvironment.from_yaml(yaml_file)
        print(f"\n--- {env.name} ---")

        for run_idx in range(runs_per_env):
            # Base model
            try:
                runner = EnvironmentRunner(env=env, agent_model=base_model)
                result = runner.run()
                results["base"].append({
                    "environment": env.name,
                    "reward": result.reward,
                    "scores": result.scorecard.category_scores,
                })
                print(f"  Base run {run_idx+1}: reward={result.reward:.3f}")
            except Exception as e:
                print(f"  Base run {run_idx+1}: FAILED - {e}")

            # Trained model (if endpoint provided)
            if trained_base_url:
                try:
                    agent = OpenAIAgentBackend(
                        model="trained",
                        base_url=trained_base_url,
                    )
                    runner = EnvironmentRunner(env=env, agent=agent, agent_model="trained")
                    result = runner.run()
                    results["trained"].append({
                        "environment": env.name,
                        "reward": result.reward,
                        "scores": result.scorecard.category_scores,
                    })
                    print(f"  Trained run {run_idx+1}: reward={result.reward:.3f}")
                except Exception as e:
                    print(f"  Trained run {run_idx+1}: FAILED - {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")

    base_avg = sum(r["reward"] for r in results["base"]) / len(results["base"]) if results["base"] else 0
    print(f"Base model ({base_model}): avg reward = {base_avg:.4f} ({len(results['base'])} runs)")

    if results["trained"]:
        trained_avg = sum(r["reward"] for r in results["trained"]) / len(results["trained"])
        improvement = ((trained_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0
        print(f"Trained model: avg reward = {trained_avg:.4f} ({len(results['trained'])} runs)")
        print(f"Improvement: {improvement:+.1f}%")

    # Save
    eval_path = pathlib.Path("/data/eval_results.json")
    eval_path.write_text(json.dumps(results, indent=2))
    data_volume.commit()
    print(f"\nResults saved to {eval_path}")


# ── Entrypoint: full pipeline ──

@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    runs_per_env: int = 20,
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 2,
    max_steps: int = -1,
    generate_only: bool = False,
    train_only: bool = False,
    eval_only: bool = False,
):
    if eval_only:
        evaluate.remote()
        return

    if not train_only:
        print("Step 1: Generating rollouts from VoiceEnv environments...")
        n_rollouts = generate.remote(runs_per_env=runs_per_env)
        print(f"Generated {n_rollouts} rollouts")

    if generate_only:
        return

    print(f"\nStep 2: GRPO fine-tuning {model}...")
    config = TrainingConfig(
        model_name=model,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        num_epochs=epochs,
        max_steps=max_steps,
    )
    output_dir = train.remote(config)
    print(f"Training complete! Output: {output_dir}")
    print(f"\nTo download the trained model:")
    print(f"  modal volume get voiceenv-output {output_dir} ./voiceenv-trained")
    print(f"\nTo evaluate:")
    print(f"  modal run voiceenv/training/modal_train.py --eval-only")
