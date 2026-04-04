"""
Fine-tune Qwen3-Omni on VoiceEnv environments using Baseten managed GPUs.

Baseten provides managed training infrastructure with automatic checkpointing
and one-command deployment of trained models as production endpoints.

The pipeline:
  1. Generate rollouts locally or on Modal (creates rollouts.jsonl)
  2. Submit training job to Baseten with `truss train push`
  3. Deploy trained checkpoint as an endpoint with `truss train deploy_checkpoints`
  4. Evaluate using the deployed endpoint

Setup:
  pip install truss
  truss login
  # Add HuggingFace token in Baseten Settings > Secrets as "hf_access_token"

Usage:
  # Generate the Baseten training project
  python -m voiceenv.training.baseten_train --rollouts rollouts.jsonl

  # Submit to Baseten
  cd baseten_voiceenv_training && uvx truss train push config.py

  # Monitor
  uvx truss train logs --job-id <job_id> --tail

  # Deploy best checkpoint
  uvx truss train deploy_checkpoints --job-id <job_id>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from textwrap import dedent


def generate_baseten_project(
    rollouts_path: str,
    output_dir: str = "baseten_voiceenv_training",
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    gpu_type: str = "H100",
    gpu_count: int = 1,
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 2,
    batch_size: int = 2,
    max_steps: int = 200,
):
    """Generate a complete Baseten training project directory."""

    project_dir = Path(output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Copy rollouts into project
    rollouts_src = Path(rollouts_path)
    if rollouts_src.exists():
        shutil.copy2(rollouts_src, project_dir / "rollouts.jsonl")
        print(f"Copied rollouts to {project_dir / 'rollouts.jsonl'}")
    else:
        print(f"Warning: {rollouts_path} not found. Generate rollouts first:")
        print(f"  python -m voiceenv.training.generate_rollouts --envs voiceenv/environments/ --output {rollouts_path}")

    # ── config.py ──
    (project_dir / "config.py").write_text(dedent(f'''\
        """Baseten training config for VoiceEnv Qwen3-Omni fine-tuning."""

        from truss_train import (
            TrainingProject,
            TrainingJob,
            Image,
            Compute,
            Runtime,
            CacheConfig,
            CheckpointingConfig,
        )
        from truss.base.truss_config import AcceleratorSpec

        BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

        training_runtime = Runtime(
            start_commands=[
                "chmod +x ./run.sh && ./run.sh",
            ],
            cache_config=CacheConfig(enabled=True),
            checkpointing_config=CheckpointingConfig(enabled=True),
        )

        training_compute = Compute(
            accelerator=AcceleratorSpec(accelerator="{gpu_type}", count={gpu_count}),
        )

        training_job = TrainingJob(
            image=Image(base_image=BASE_IMAGE),
            compute=training_compute,
            runtime=training_runtime,
        )

        training_project = TrainingProject(
            name="voiceenv-qwen3-omni-grpo",
            job=training_job,
        )
    '''))

    # ── run.sh ──
    (project_dir / "run.sh").write_text(dedent(f'''\
        #!/bin/bash
        set -eux

        pip install \\
            "transformers>=4.50" \\
            "trl>=0.15" \\
            "peft>=0.14" \\
            "datasets>=3.0" \\
            "accelerate>=1.2" \\
            "bitsandbytes>=0.45" \\
            "flash-attn>=2.7"

        python train.py \\
            --model "{model_name}" \\
            --rollouts ./rollouts.jsonl \\
            --output "${{BT_CHECKPOINT_DIR:-./checkpoints}}" \\
            --lora-rank {lora_rank} \\
            --lr {learning_rate} \\
            --epochs {num_epochs} \\
            --batch-size {batch_size} \\
            --max-steps {max_steps}
    '''))
    (project_dir / "run.sh").chmod(0o755)

    # ── train.py ──
    (project_dir / "train.py").write_text(dedent('''\
        """
        GRPO fine-tuning of Qwen3-Omni on VoiceEnv rollouts.
        Runs on Baseten managed GPU infrastructure.
        """

        import argparse
        import json
        import os
        from pathlib import Path

        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig, get_peft_model


        def load_rollouts(path: str) -> list[dict]:
            rollouts = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rollouts.append(json.loads(line))
            return rollouts


        def prepare_dataset(rollouts: list[dict]) -> Dataset:
            env_groups: dict[str, list] = {}
            for r in rollouts:
                env_groups.setdefault(r["environment"], []).append(r)

            examples = []
            for env_name, group in env_groups.items():
                for r in group:
                    conversation = ""
                    for msg in r["messages"]:
                        conversation += f"[{msg['role'].upper()}]: {msg['content']}\\n"
                    examples.append({
                        "prompt": r["prompt"],
                        "completion": conversation,
                        "reward": r["reward"],
                        "environment": r["environment"],
                    })
            return Dataset.from_list(examples)


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
            parser.add_argument("--rollouts", required=True)
            parser.add_argument("--output", default="./checkpoints")
            parser.add_argument("--lora-rank", type=int, default=16)
            parser.add_argument("--lr", type=float, default=2e-5)
            parser.add_argument("--epochs", type=int, default=2)
            parser.add_argument("--batch-size", type=int, default=2)
            parser.add_argument("--max-steps", type=int, default=200)
            args = parser.parse_args()

            print(f"=== VoiceEnv GRPO Training on Baseten ===")
            print(f"Model: {args.model}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

            # Load data
            rollouts = load_rollouts(args.rollouts)
            print(f"Loaded {len(rollouts)} rollouts")
            dataset = prepare_dataset(rollouts)
            print(f"Dataset: {len(dataset)} training examples")

            avg_reward = sum(r["reward"] for r in rollouts) / len(rollouts)
            print(f"Average pre-training reward: {avg_reward:.4f}")

            # Load model with 4-bit quantization
            print(f"\\nLoading {args.model}...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

            # LoRA
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
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
            output_dir = args.output
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            grpo_config = GRPOConfig(
                output_dir=output_dir,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps if args.max_steps > 0 else -1,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=8,
                learning_rate=args.lr,
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

            print(f"\\nStarting GRPO training (max_steps={args.max_steps})...")
            trainer.train()

            # Save final model
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"\\nTraining complete! Model saved to {output_dir}")
            print(f"Deploy with: uvx truss train deploy_checkpoints")


        if __name__ == "__main__":
            main()
    '''))

    # ── README ──
    (project_dir / "README.md").write_text(dedent(f'''\
        # VoiceEnv Qwen3-Omni Training (Baseten)

        Fine-tunes `{model_name}` on VoiceEnv environment rollouts using
        GRPO on Baseten managed {gpu_type} GPUs.

        ## Prerequisites

        ```bash
        pip install truss
        truss login
        ```

        Add your HuggingFace token in Baseten Settings > Secrets as `hf_access_token`.

        ## Generate rollouts (if not already done)

        ```bash
        pip install voiceenv
        python -m voiceenv.training.generate_rollouts \\
            --envs voiceenv/environments/ \\
            --runs-per-env 20 \\
            --output rollouts.jsonl
        cp rollouts.jsonl {output_dir}/
        ```

        ## Submit training job

        ```bash
        cd {output_dir}
        uvx truss train push config.py
        ```

        ## Monitor

        ```bash
        uvx truss train logs --job-id <job_id> --tail
        uvx truss train metrics --job-id <job_id>
        ```

        ## Deploy trained model

        ```bash
        uvx truss train deploy_checkpoints --job-id <job_id>
        ```

        ## Evaluate

        ```bash
        voiceenv benchmark \\
            --model <base_url_from_deploy> \\
            voiceenv/environments/
        ```
    '''))

    print(f"\nBaseten training project created at: {project_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Ensure rollouts.jsonl exists in {project_dir}/")
    print(f"  2. cd {project_dir}")
    print(f"  3. uvx truss train push config.py")
    print(f"  4. uvx truss train logs --job-id <job_id> --tail")
    print(f"  5. uvx truss train deploy_checkpoints --job-id <job_id>")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Baseten training project for VoiceEnv Qwen3-Omni fine-tuning"
    )
    parser.add_argument("--rollouts", required=True, help="Path to rollouts JSONL file")
    parser.add_argument("--output", default="baseten_voiceenv_training", help="Output project directory")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model to fine-tune")
    parser.add_argument("--gpu", default="H100", choices=["H100", "H200", "A10G"], help="GPU type")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--max-steps", type=int, default=200, help="Max training steps")

    args = parser.parse_args()
    generate_baseten_project(
        rollouts_path=args.rollouts,
        output_dir=args.output,
        model_name=args.model,
        gpu_type=args.gpu,
        gpu_count=args.gpu_count,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
