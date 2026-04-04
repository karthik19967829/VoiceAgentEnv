"""
Training launcher — one-click post-training using external frameworks.

We generate the data, they do the training. Supported frameworks:
  - verl:     pip install verl
  - ms-swift: pip install ms-swift
  - trl:      pip install trl

Usage:
  voiceenv train run --framework verl --model Qwen/Qwen2.5-3B-Instruct
  voiceenv train run --framework ms-swift --model Qwen/Qwen3-Omni-30B-A3B-Instruct
  voiceenv train run --framework trl --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def prepare_verl_dataset(rollouts_path: str, output_dir: str) -> Path:
    """
    Convert VoiceEnv rollouts to VERL's expected parquet format.

    VERL expects a parquet dataset with columns:
      - prompt: the system prompt
      - data_source: environment name
      - ground_truth: JSON with verifiable checks
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    # Convert to VERL format
    verl_data = []
    for r in rollouts:
        verl_data.append({
            "prompt": r["prompt"],
            "data_source": r.get("environment", "voiceenv"),
            "ground_truth": json.dumps({
                "tool_calls": r.get("tool_calls", []),
                "expected_state": {},
                "verifiable_checks": [],
            }),
        })

    dataset_path = out / "train.jsonl"
    with open(dataset_path, "w") as f:
        for item in verl_data:
            f.write(json.dumps(item) + "\n")

    console.print(f"[green]Prepared {len(verl_data)} examples for VERL[/green] → {dataset_path}")
    return dataset_path


def prepare_swift_dataset(rollouts_path: str, output_dir: str) -> Path:
    """
    Convert VoiceEnv rollouts to ms-swift's expected JSONL format.

    ms-swift GRPO expects:
      - query: the prompt
      - response: the completion
      - rejected_response: (optional) low-reward completion
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    # Group by environment, create preference pairs
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

    dataset_path = out / "train.jsonl"
    with open(dataset_path, "w") as f:
        for item in swift_data:
            f.write(json.dumps(item) + "\n")

    console.print(f"[green]Prepared {len(swift_data)} examples for ms-swift[/green] → {dataset_path}")
    return dataset_path


def prepare_trl_dataset(rollouts_path: str, output_dir: str) -> Path:
    """
    Convert VoiceEnv rollouts to TRL GRPOTrainer's expected format.

    TRL expects a HuggingFace Dataset with:
      - prompt: str
      - completion: str
    Plus a reward function that scores completions.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    trl_data = []
    for r in rollouts:
        conversation = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in r.get("messages", [])
        )
        trl_data.append({
            "prompt": r.get("prompt", ""),
            "completion": conversation,
            "reward": r.get("reward", 0.0),
        })

    dataset_path = out / "train.jsonl"
    with open(dataset_path, "w") as f:
        for item in trl_data:
            f.write(json.dumps(item) + "\n")

    console.print(f"[green]Prepared {len(trl_data)} examples for TRL[/green] → {dataset_path}")
    return dataset_path


def _check_installed(package: str) -> bool:
    return shutil.which(package) is not None or _can_import(package)


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def launch_training(
    framework: str,
    model: str,
    rollouts_path: str,
    output_dir: str = "voiceenv_training",
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 2,
    batch_size: int = 2,
    num_gpus: int = 1,
    extra_args: list[str] | None = None,
) -> None:
    """
    Launch post-training on the specified framework.

    All the heavy lifting is done by the external framework.
    We just format data and construct the right command.
    """
    data_dir = Path(output_dir) / "data"

    if framework == "verl":
        _launch_verl(model, rollouts_path, output_dir, data_dir, lora_rank,
                     learning_rate, epochs, batch_size, num_gpus, extra_args)
    elif framework == "ms-swift":
        _launch_swift(model, rollouts_path, output_dir, data_dir, lora_rank,
                      learning_rate, epochs, batch_size, num_gpus, extra_args)
    elif framework == "trl":
        _launch_trl(model, rollouts_path, output_dir, data_dir, lora_rank,
                    learning_rate, epochs, batch_size, extra_args)
    else:
        console.print(f"[red]Unknown framework: {framework}[/red]")
        console.print("Supported: verl, ms-swift, trl")
        sys.exit(1)


def _launch_verl(model, rollouts_path, output_dir, data_dir, lora_rank,
                 lr, epochs, batch_size, num_gpus, extra_args):
    """Launch GRPO training via VERL."""
    if not _can_import("verl"):
        console.print(Panel(
            "[bold red]VERL not installed[/bold red]\n\n"
            "Install with:\n"
            "  [cyan]pip install verl[/cyan]\n\n"
            "See: https://github.com/volcengine/verl",
            title="Missing Dependency",
        ))
        sys.exit(1)

    dataset_path = prepare_verl_dataset(rollouts_path, str(data_dir))
    reward_fn_path = Path(__file__).parent / "reward_function.py"

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator=grpo",
        f"data.train_files={dataset_path}",
        f"data.train_batch_size={batch_size}",
        f"actor_rollout_ref.model.path={model}",
        f"actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef=0.001",
        f"actor_rollout_ref.rollout.n=4",
        f"actor_rollout_ref.actor.ppo_epochs={epochs}",
        f"actor_rollout_ref.actor.lr={lr}",
        f"custom_reward_function.path={reward_fn_path}",
        f"custom_reward_function.name=voiceenv_reward",
        f"trainer.project_name=voiceenv",
        f"trainer.experiment_name=voiceenv-grpo",
        f"trainer.default_local_dir={output_dir}",
    ]

    if extra_args:
        cmd.extend(extra_args)

    console.print(Panel(
        f"[bold]Framework:[/bold] VERL (GRPO)\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Dataset:[/bold] {dataset_path}\n"
        f"[bold]Reward fn:[/bold] {reward_fn_path}\n"
        f"[bold]Output:[/bold] {output_dir}",
        title="Launching Training",
        border_style="green",
    ))

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]\n")
    subprocess.run(cmd, check=True)


def _launch_swift(model, rollouts_path, output_dir, data_dir, lora_rank,
                  lr, epochs, batch_size, num_gpus, extra_args):
    """Launch GRPO training via ms-swift."""
    if not shutil.which("swift"):
        console.print(Panel(
            "[bold red]ms-swift not installed[/bold red]\n\n"
            "Install with:\n"
            "  [cyan]pip install ms-swift[/cyan]\n\n"
            "See: https://github.com/modelscope/ms-swift",
            title="Missing Dependency",
        ))
        sys.exit(1)

    dataset_path = prepare_swift_dataset(rollouts_path, str(data_dir))

    cmd = [
        "swift", "rlhf",
        "--rlhf_type", "grpo",
        "--model", model,
        "--dataset", str(dataset_path),
        "--train_type", "lora",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_rank * 2),
        "--learning_rate", str(lr),
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--output_dir", output_dir,
        "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", "true",
    ]

    if extra_args:
        cmd.extend(extra_args)

    console.print(Panel(
        f"[bold]Framework:[/bold] ms-swift (GRPO)\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Dataset:[/bold] {dataset_path}\n"
        f"[bold]LoRA rank:[/bold] {lora_rank}\n"
        f"[bold]Output:[/bold] {output_dir}",
        title="Launching Training",
        border_style="green",
    ))

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]\n")
    subprocess.run(cmd, check=True)


def _launch_trl(model, rollouts_path, output_dir, data_dir, lora_rank,
                lr, epochs, batch_size, extra_args):
    """Generate a TRL training script and run it."""
    if not _can_import("trl"):
        console.print(Panel(
            "[bold red]TRL not installed[/bold red]\n\n"
            "Install with:\n"
            "  [cyan]pip install trl peft transformers[/cyan]\n\n"
            "See: https://github.com/huggingface/trl",
            title="Missing Dependency",
        ))
        sys.exit(1)

    dataset_path = prepare_trl_dataset(rollouts_path, str(data_dir))

    script = f'''"""Auto-generated TRL GRPO training script from VoiceEnv."""
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

# Load dataset
examples = []
with open("{dataset_path}") as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))
dataset = Dataset.from_list(examples)

# Load model
tokenizer = AutoTokenizer.from_pretrained("{model}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "{model}", trust_remote_code=True, torch_dtype="auto", device_map="auto",
)
model = get_peft_model(model, LoraConfig(
    r={lora_rank}, lora_alpha={lora_rank * 2},
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
))

# Import VoiceEnv reward function
from voiceenv.training.reward_function import compute_reward

def reward_fn(completions, **kwargs):
    return [compute_reward(c, {{}}) for c in completions]

trainer = GRPOTrainer(
    model=model,
    config=GRPOConfig(
        output_dir="{output_dir}",
        num_train_epochs={epochs},
        per_device_train_batch_size={batch_size},
        learning_rate={lr},
        bf16=True,
        gradient_checkpointing=True,
        save_total_limit=3,
        report_to="none",
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
)
trainer.train()
trainer.save_model("{output_dir}")
tokenizer.save_pretrained("{output_dir}")
print(f"\\nModel saved to {output_dir}")
'''

    script_path = Path(output_dir) / "train_trl.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)

    console.print(Panel(
        f"[bold]Framework:[/bold] TRL (GRPOTrainer)\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Dataset:[/bold] {dataset_path}\n"
        f"[bold]Script:[/bold] {script_path}\n"
        f"[bold]Output:[/bold] {output_dir}",
        title="Launching Training",
        border_style="green",
    ))

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]\n")
    subprocess.run(cmd, check=True)
