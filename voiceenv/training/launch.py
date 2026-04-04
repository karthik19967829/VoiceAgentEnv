"""
Training launcher — one-click post-training via ms-swift.

ms-swift (ModelScope) has native Qwen3-Omni GRPO support, handles
LoRA, multi-GPU, quantization, and all the training infrastructure.
We just format the data and call it.

Usage:
  voiceenv train run --model Qwen/Qwen3-Omni-30B-A3B-Instruct --rollouts rollouts.jsonl
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


def prepare_dataset(rollouts_path: str, output_dir: str) -> Path:
    """
    Convert VoiceEnv rollouts to ms-swift's expected JSONL format.

    ms-swift GRPO expects:
      - query: the prompt
      - response: the completion
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    if not rollouts:
        console.print(f"[red]No rollouts found in {rollouts_path}[/red]")
        sys.exit(1)

    # Group by environment, create preference pairs for GRPO
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

    console.print(f"[green]Prepared {len(swift_data)} examples[/green] → {dataset_path}")
    return dataset_path


def launch_training(
    model: str,
    rollouts_path: str,
    output_dir: str = "voiceenv_trained",
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 2,
    batch_size: int = 2,
    num_gpus: int = 1,
    extra_args: list[str] | None = None,
) -> None:
    """
    Launch GRPO post-training via ms-swift.

    Formats the rollout data and invokes `swift rlhf --rlhf_type grpo`.
    """
    if not shutil.which("swift"):
        console.print(Panel(
            "[bold red]ms-swift not installed[/bold red]\n\n"
            "Install with:\n"
            "  [cyan]pip install ms-swift[/cyan]\n\n"
            "Docs: https://github.com/modelscope/ms-swift",
            title="Missing Dependency",
        ))
        sys.exit(1)

    data_dir = Path(output_dir) / "data"
    dataset_path = prepare_dataset(rollouts_path, str(data_dir))

    cmd = [
        "swift", "rlhf",
        "--rlhf_type", "grpo",
        "--model", model,
        "--dataset", str(dataset_path),
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

    if extra_args:
        cmd.extend(extra_args)

    console.print(Panel(
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Dataset:[/bold] {dataset_path} ({len(list(open(dataset_path)))} examples)\n"
        f"[bold]LoRA rank:[/bold] {lora_rank}\n"
        f"[bold]LR:[/bold] {learning_rate}\n"
        f"[bold]Epochs:[/bold] {epochs}\n"
        f"[bold]Output:[/bold] {output_dir}",
        title="ms-swift GRPO Training",
        border_style="green",
    ))

    console.print(f"\n[dim]$ {' '.join(cmd)}[/dim]\n")
    subprocess.run(cmd, check=True)
