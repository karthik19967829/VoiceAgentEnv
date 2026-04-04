"""
GRPO post-training script for speech LLMs using VoiceEnv rollouts.

This trains a model (e.g., Qwen3-Omni or any HuggingFace model) using
Group Relative Policy Optimization (GRPO) on rollouts generated from
VoiceEnv environments.

For GPU access, prefer the Modal or Baseten wrappers:
  - modal_train.py  — serverless H100s, full pipeline in one command
  - baseten_train.py — managed GPUs with auto-deploy to endpoint

Local usage (requires local GPU):
  python -m voiceenv.training.grpo_train \
    --rollouts rollouts.jsonl \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --output ./voiceenv-trained \
    --epochs 2 \
    --lr 2e-5

Requirements: pip install voiceenv[training]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()


def prepare_dataset(rollouts_path: str):
    """
    Load rollouts JSONL and prepare a HuggingFace Dataset for GRPO training.

    Each example has:
      - prompt: the agent system prompt
      - chosen: high-reward conversations (reward > threshold)
      - rejected: low-reward conversations (reward <= threshold)
    """
    from datasets import Dataset

    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))

    if not rollouts:
        raise ValueError(f"No rollouts found in {rollouts_path}")

    # Group by environment and create preference pairs
    env_groups: dict[str, list] = {}
    for r in rollouts:
        env_groups.setdefault(r["environment"], []).append(r)

    preference_pairs = []
    for env_name, group in env_groups.items():
        sorted_group = sorted(group, key=lambda x: x["reward"], reverse=True)
        median_reward = sorted_group[len(sorted_group) // 2]["reward"]

        for r in sorted_group:
            # Format conversation as a single string
            conversation = ""
            for msg in r["messages"]:
                role = msg["role"].upper()
                conversation += f"[{role}]: {msg['content']}\n"

            preference_pairs.append({
                "prompt": r["prompt"],
                "completion": conversation,
                "reward": r["reward"],
                "environment": r["environment"],
                "is_chosen": r["reward"] > median_reward,
            })

    return Dataset.from_list(preference_pairs)


def train_grpo(
    rollouts_path: str,
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    output_dir: str = "./voiceenv-trained",
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    lora_rank: int = 16,
    use_lora: bool = True,
):
    """
    Fine-tune a model using GRPO on VoiceEnv rollouts.

    Uses LoRA by default for efficient training on consumer GPUs.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig, get_peft_model
    except ImportError:
        console.print("[red]Training dependencies required. Install with:[/red]")
        console.print("  pip install voiceenv[training]")
        console.print("  pip install peft")
        return

    console.print(f"[bold]GRPO Post-Training with VoiceEnv Environments[/bold]\n")
    console.print(f"Model: [cyan]{model_name}[/cyan]")
    console.print(f"Rollouts: [cyan]{rollouts_path}[/cyan]")
    console.print(f"Output: [cyan]{output_dir}[/cyan]")
    console.print(f"LoRA: [cyan]{use_lora} (rank={lora_rank})[/cyan]")

    # Prepare dataset
    console.print("\n[bold]Preparing dataset...[/bold]")
    dataset = prepare_dataset(rollouts_path)
    console.print(f"Dataset size: {len(dataset)} examples")

    reward_values = dataset["reward"]
    avg_reward = sum(reward_values) / len(reward_values)
    console.print(f"Average reward in dataset: {avg_reward:.4f}")

    # Load model and tokenizer
    console.print(f"\n[bold]Loading model: {model_name}...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    if use_lora:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # GRPO configuration
    training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_completion_length=max_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        report_to="none",
    )

    # Format dataset for GRPO
    def format_for_grpo(example):
        return {
            "prompt": example["prompt"],
            "completion": example["completion"],
        }

    formatted_dataset = dataset.map(format_for_grpo)

    # Define reward function from VoiceEnv scores
    def reward_fn(completions, **kwargs):
        """Use pre-computed rewards from rollouts."""
        return [example.get("reward", 0.0) for example in completions]

    console.print(f"\n[bold]Starting GRPO training...[/bold]")

    trainer = GRPOTrainer(
        model=model,
        config=training_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    console.print(f"\n[green]Model saved to: {output_dir}[/green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Evaluate: [cyan]voiceenv benchmark --model {output_dir}[/cyan]")
    console.print(f"  2. Compare: [cyan]voiceenv benchmark --model {model_name} --model {output_dir}[/cyan]")


def main():
    parser = argparse.ArgumentParser(description="GRPO post-training with VoiceEnv environments")
    parser.add_argument("--rollouts", required=True, help="Path to rollouts JSONL file")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--output", default="./voiceenv-trained", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune)")

    args = parser.parse_args()
    train_grpo(
        rollouts_path=args.rollouts,
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        use_lora=not args.no_lora,
    )


if __name__ == "__main__":
    main()
