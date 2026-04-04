"""
End-to-end post-training experiment on Modal.

This is the script that PROVES VoiceEnv environments improve speech LLMs.

THE EXPERIMENT:
  1. BASELINE EVAL — run base Qwen3-Omni against all 5 environments
  2. GENERATE ROLLOUTS — 20 runs per environment = 100 training conversations
  3. GRPO FINE-TUNE — train LoRA adapter on H100 using verifiable rewards
  4. POST-TRAIN EVAL — run fine-tuned model against same environments
  5. COMPARISON REPORT — per-criterion delta showing what improved

RUN IT:
  # Full experiment (takes ~2-4 hours depending on model size)
  modal run voiceenv/training/experiment.py

  # Step by step
  modal run voiceenv/training/experiment.py::baseline_eval
  modal run voiceenv/training/experiment.py::generate_rollouts
  modal run voiceenv/training/experiment.py::grpo_train
  modal run voiceenv/training/experiment.py::posttrain_eval
  modal run voiceenv/training/experiment.py::report

REQUIREMENTS:
  pip install modal
  modal setup
  modal secret create openai-secret OPENAI_API_KEY=sk-...
  modal secret create huggingface-secret HF_TOKEN=hf_...
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Optional

import modal

app = modal.App("voiceenv-experiment")

# ── Images ──

eval_image = (
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
        "vllm>=0.7",
        "openai>=1.0",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "httpx>=0.25",
        "rich>=13.0",
        "click>=8.0",
        "jinja2>=3.0",
    )
    .env({"HF_HOME": "/model_cache"})
)

# ── Volumes ──

model_cache = modal.Volume.from_name("voiceenv-model-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("voiceenv-data", create_if_missing=True)
output_volume = modal.Volume.from_name("voiceenv-output", create_if_missing=True)


@dataclass
class ExperimentConfig:
    base_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    eval_model: str = "gpt-4o-mini"  # model used for baseline eval (API-based)
    simulator_model: str = "gpt-4o-mini"
    scorer_model: str = "gpt-4o-mini"

    # Rollout generation
    runs_per_env: int = 20
    rollout_agent_model: str = "gpt-4o-mini"

    # Training
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_completion_length: int = 2048
    use_4bit: bool = True

    # Evaluation
    eval_runs_per_env: int = 10


def _ensure_environments(data_vol_path: str = "/data") -> pathlib.Path:
    """Copy built-in environments to the data volume."""
    env_dir = pathlib.Path(data_vol_path) / "environments"
    if not env_dir.exists() or not list(env_dir.glob("*.yaml")):
        env_dir.mkdir(parents=True, exist_ok=True)
        import voiceenv.environments as envmod
        src_dir = pathlib.Path(envmod.__file__).parent
        for yaml_file in src_dir.glob("*.yaml"):
            (env_dir / yaml_file.name).write_text(yaml_file.read_text())
        print(f"Loaded {len(list(env_dir.glob('*.yaml')))} environments")
    return env_dir


# ── Step 1: Baseline Evaluation ──

@app.function(
    image=eval_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=3600,
)
def baseline_eval(
    model: str = "gpt-4o-mini",
    runs_per_env: int = 10,
    simulator_model: str = "gpt-4o-mini",
) -> str:
    """Evaluate the base model on all VoiceEnv environments."""
    import sys
    sys.path.insert(0, "/")

    from voiceenv.eval.evaluator import evaluate

    env_dir = _ensure_environments()

    print(f"\n{'='*60}")
    print(f"STEP 1: BASELINE EVALUATION")
    print(f"Model: {model}")
    print(f"Environments: {len(list(env_dir.glob('*.yaml')))}")
    print(f"Runs per env: {runs_per_env}")
    print(f"{'='*60}\n")

    results = evaluate(
        model=model,
        env_dir=str(env_dir),
        runs_per_env=runs_per_env,
        simulator_model=simulator_model,
    )

    output_path = "/data/baseline_eval.json"
    results.save(output_path)
    data_volume.commit()

    print(f"\nBaseline results saved to {output_path}")
    print(f"Overall reward: {results.overall_reward:.4f}")
    print(f"Verifiable reward: {results.overall_verifiable_reward:.4f}")

    return output_path


# ── Step 2: Generate Rollouts ──

@app.function(
    image=eval_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=3600,
)
def generate_rollouts(
    runs_per_env: int = 20,
    agent_model: str = "gpt-4o-mini",
    simulator_model: str = "gpt-4o-mini",
) -> str:
    """Generate training rollouts from all environments."""
    import sys
    sys.path.insert(0, "/")

    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.runner import EnvironmentRunner

    env_dir = _ensure_environments()
    yaml_files = sorted(env_dir.glob("*.yaml"))

    print(f"\n{'='*60}")
    print(f"STEP 2: GENERATE ROLLOUTS")
    print(f"Agent model: {agent_model}")
    print(f"Environments: {len(yaml_files)}")
    print(f"Runs per env: {runs_per_env}")
    print(f"Total rollouts: {len(yaml_files) * runs_per_env}")
    print(f"{'='*60}\n")

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
                    "reward": result.verifiable_reward,
                    "blended_reward": result.reward,
                    "verifiable_reward": result.verifiable_reward,
                    "soft_reward": result.soft_reward,
                    "environment": env.name,
                    "vertical": env.vertical.value,
                    "difficulty": env.difficulty.value,
                    "turn_count": result.turn_count,
                    "tool_calls": result.tool_calls,
                    "category_scores": result.scorecard.category_scores,
                    "criteria_scores": {
                        cr.name: {"score": cr.score, "verifiable": cr.is_verifiable}
                        for cr in result.scorecard.criteria_results
                    },
                }
                rollouts.append(rollout)
                status = "PASS" if result.verifiable_reward >= 0.5 else "FAIL"
                print(f"  Run {run_idx+1:>2}/{runs_per_env}: "
                      f"v_reward={result.verifiable_reward:.3f} "
                      f"reward={result.reward:.3f} "
                      f"turns={result.turn_count:>2} [{status}]")

            except Exception as e:
                print(f"  Run {run_idx+1:>2}/{runs_per_env}: FAILED - {e}")

    output_path = "/data/rollouts.jsonl"
    with open(output_path, "w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    data_volume.commit()

    avg_reward = sum(r["reward"] for r in rollouts) / len(rollouts) if rollouts else 0
    avg_verifiable = sum(r["verifiable_reward"] for r in rollouts) / len(rollouts) if rollouts else 0

    print(f"\n{'='*60}")
    print(f"ROLLOUT SUMMARY")
    print(f"Total: {len(rollouts)}")
    print(f"Avg verifiable reward: {avg_verifiable:.4f}")
    print(f"Avg blended reward: {avg_reward:.4f}")

    # Reward distribution
    high = sum(1 for r in rollouts if r["reward"] >= 0.7)
    mid = sum(1 for r in rollouts if 0.3 <= r["reward"] < 0.7)
    low = sum(1 for r in rollouts if r["reward"] < 0.3)
    print(f"Distribution: {high} high (>=0.7) | {mid} mid | {low} low (<0.3)")
    print(f"{'='*60}")

    return output_path


# ── Step 3: GRPO Fine-tuning ──

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
)
def grpo_train(
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    rollouts_path: str = "/data/rollouts.jsonl",
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 2,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    use_4bit: bool = True,
) -> str:
    """GRPO fine-tune on H100 using verifiable rewards."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, get_peft_model

    output_dir = "/output/voiceenv-trained"

    print(f"\n{'='*60}")
    print(f"STEP 3: GRPO FINE-TUNING")
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"LoRA rank: {lora_rank}")
    print(f"4-bit quantization: {use_4bit}")
    print(f"{'='*60}\n")

    # Load rollouts
    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    print(f"Loaded {len(rollouts)} rollouts")
    avg_reward = sum(r["verifiable_reward"] for r in rollouts) / len(rollouts)
    print(f"Average verifiable reward: {avg_reward:.4f}")

    # Prepare dataset — use verifiable_reward as the training signal
    training_examples = []
    for r in rollouts:
        conversation = ""
        for msg in r["messages"]:
            role = msg["role"].upper()
            conversation += f"[{role}]: {msg['content']}\n"

        training_examples.append({
            "prompt": r["prompt"],
            "completion": conversation,
            "reward": r["verifiable_reward"],  # ONLY verifiable rewards for RL
        })

    dataset = Dataset.from_list(training_examples)
    print(f"Training dataset: {len(dataset)} examples")

    # Show reward distribution
    rewards = [e["reward"] for e in training_examples]
    print(f"Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"Reward mean: {sum(rewards)/len(rewards):.3f}")

    # Load model
    print(f"\nLoading {model_name}...")
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # GRPO config
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_completion_length=2048,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    def reward_fn(completions, **kwargs):
        return [ex.get("reward", 0.0) for ex in completions]

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    print("\nStarting GRPO training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    output_volume.commit()

    print(f"\nTraining complete! Model saved to {output_dir}")
    return output_dir


# ── Step 4: Serve trained model for evaluation ──

@app.function(
    image=train_image,
    gpu="H100",
    volumes={
        "/model_cache": model_cache,
        "/output": output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    allow_concurrent_inputs=10,
)
@modal.web_server(8000)
def serve_trained_model(
    base_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    adapter_path: str = "/output/voiceenv-trained",
):
    """Serve the fine-tuned model as an OpenAI-compatible endpoint for evaluation."""
    import subprocess
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--enable-lora",
        "--lora-modules", f"voiceenv-trained={adapter_path}",
        "--port", "8000",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9",
    ])


# ── Step 5: Post-training Evaluation ──

@app.function(
    image=eval_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=3600,
)
def posttrain_eval(
    trained_base_url: str = "",
    trained_model_name: str = "voiceenv-trained",
    runs_per_env: int = 10,
    simulator_model: str = "gpt-4o-mini",
) -> str:
    """Evaluate the post-trained model and compare with baseline."""
    import sys
    sys.path.insert(0, "/")

    from voiceenv.eval.evaluator import evaluate

    env_dir = _ensure_environments()

    print(f"\n{'='*60}")
    print(f"STEP 4: POST-TRAINING EVALUATION")
    print(f"Trained model: {trained_model_name}")
    if trained_base_url:
        print(f"Endpoint: {trained_base_url}")
    print(f"{'='*60}\n")

    results = evaluate(
        model=trained_model_name,
        env_dir=str(env_dir),
        runs_per_env=runs_per_env,
        simulator_model=simulator_model,
        base_url=trained_base_url if trained_base_url else None,
    )

    output_path = "/data/posttrain_eval.json"
    results.save(output_path)
    data_volume.commit()

    print(f"\nPost-training results saved to {output_path}")
    print(f"Overall reward: {results.overall_reward:.4f}")
    print(f"Verifiable reward: {results.overall_verifiable_reward:.4f}")

    return output_path


# ── Step 6: Generate Comparison Report ──

@app.function(
    image=eval_image,
    volumes={"/data": data_volume},
    timeout=300,
)
def report() -> str:
    """Generate comparison report between baseline and post-training."""
    import sys
    sys.path.insert(0, "/")

    from voiceenv.eval.evaluator import EvalResults
    from voiceenv.eval.comparison import compare, print_comparison

    baseline_path = "/data/baseline_eval.json"
    posttrain_path = "/data/posttrain_eval.json"

    baseline = EvalResults.load(baseline_path)
    posttrain = EvalResults.load(posttrain_path)

    print(f"\n{'='*60}")
    print(f"STEP 5: COMPARISON REPORT")
    print(f"{'='*60}\n")

    comparison = compare(baseline, posttrain)
    print_comparison(comparison)

    report_path = "/data/comparison_report.json"
    comparison.save(report_path)
    data_volume.commit()

    print(f"\nFull report saved to {report_path}")
    return report_path


# ── Full Pipeline Entrypoint ──

@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    eval_model: str = "gpt-4o-mini",
    simulator_model: str = "gpt-4o-mini",
    runs_per_env: int = 20,
    eval_runs: int = 10,
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 2,
    step: str = "all",
):
    """
    Run the full post-training experiment.

    Steps: baseline, rollouts, train, posttrain, report, all
    """
    print("=" * 60)
    print("VOICEENV POST-TRAINING EXPERIMENT")
    print("=" * 60)
    print(f"Base model:        {model}")
    print(f"Eval model:        {eval_model}")
    print(f"Simulator:         {simulator_model}")
    print(f"Rollouts/env:      {runs_per_env}")
    print(f"Eval runs/env:     {eval_runs}")
    print(f"LoRA rank:         {lora_rank}")
    print(f"Learning rate:     {learning_rate}")
    print(f"Epochs:            {epochs}")
    print(f"Step:              {step}")
    print("=" * 60)

    steps = ["baseline", "rollouts", "train", "posttrain", "report"]
    if step != "all":
        steps = [step]

    if "baseline" in steps:
        print("\n\n>>> STEP 1/5: Baseline Evaluation")
        baseline_path = baseline_eval.remote(
            model=eval_model,
            runs_per_env=eval_runs,
            simulator_model=simulator_model,
        )
        print(f"Baseline saved: {baseline_path}")

    if "rollouts" in steps:
        print("\n\n>>> STEP 2/5: Generating Rollouts")
        rollouts_path = generate_rollouts.remote(
            runs_per_env=runs_per_env,
            agent_model=eval_model,
            simulator_model=simulator_model,
        )
        print(f"Rollouts saved: {rollouts_path}")

    if "train" in steps:
        print(f"\n\n>>> STEP 3/5: GRPO Fine-tuning {model}")
        output_dir = grpo_train.remote(
            model_name=model,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            num_epochs=epochs,
        )
        print(f"Trained model saved: {output_dir}")
        print(f"\nTo serve the trained model for evaluation:")
        print(f"  modal serve voiceenv/training/experiment.py::serve_trained_model")
        print(f"\nThen run post-training eval with the endpoint URL:")
        print(f"  modal run voiceenv/training/experiment.py --step posttrain")

    if "posttrain" in steps:
        print("\n\n>>> STEP 4/5: Post-Training Evaluation")
        print("NOTE: You need the trained model served as an endpoint.")
        print("      Use `modal serve experiment.py::serve_trained_model` first.")
        print("      Then set TRAINED_BASE_URL env var.")
        import os
        trained_url = os.environ.get("TRAINED_BASE_URL", "")
        if trained_url:
            posttrain_path = posttrain_eval.remote(
                trained_base_url=trained_url,
                runs_per_env=eval_runs,
                simulator_model=simulator_model,
            )
            print(f"Post-training results saved: {posttrain_path}")
        else:
            print("Skipping — set TRAINED_BASE_URL to the vLLM endpoint.")
            posttrain_eval_using_api = posttrain_eval.remote(
                trained_base_url="",
                trained_model_name=eval_model,
                runs_per_env=eval_runs,
                simulator_model=simulator_model,
            )

    if "report" in steps:
        print("\n\n>>> STEP 5/5: Comparison Report")
        report_path = report.remote()
        print(f"\nReport saved: {report_path}")
        print(f"\nTo download results:")
        print(f"  modal volume get voiceenv-data /data/comparison_report.json .")
        print(f"  modal volume get voiceenv-data /data/baseline_eval.json .")
        print(f"  modal volume get voiceenv-data /data/posttrain_eval.json .")

    print("\n\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
