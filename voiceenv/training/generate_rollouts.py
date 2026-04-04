"""
Generate RL training rollouts from VoiceEnv environments.

This script runs a target model against voice environments, collects
transcripts + rewards, and outputs a JSONL dataset suitable for GRPO / DPO
post-training of speech LLMs like Qwen3-Omni.

Usage:
  python -m voiceenv.training.generate_rollouts \
    --envs voiceenv/environments/ \
    --model gpt-4o-mini \
    --runs-per-env 10 \
    --output rollouts.jsonl

The output format is compatible with TRL, verifiers, and HuggingFace datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()


def generate_rollouts(
    env_dir: str,
    model: str = "gpt-4o-mini",
    simulator_model: str = "gpt-4o-mini",
    runs_per_env: int = 10,
    output_path: str = "rollouts.jsonl",
    base_url: str | None = None,
    api_key: str | None = None,
):
    """
    Generate training rollouts from all environments in a directory.

    Each rollout is a complete conversation with a reward signal, formatted
    for RL post-training.
    """
    from openai import OpenAI
    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.runner import EnvironmentRunner, OpenAIAgentBackend

    env_path = Path(env_dir)
    yaml_files = sorted(env_path.glob("*.yaml"))

    if not yaml_files:
        console.print(f"[red]No environment YAML files found in {env_dir}[/red]")
        sys.exit(1)

    # Set up the agent backend
    client = None
    agent_backend = None
    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        agent_backend = OpenAIAgentBackend(model=model, client=client)

    total_runs = len(yaml_files) * runs_per_env
    console.print(f"[bold]Generating {total_runs} rollouts from {len(yaml_files)} environments[/bold]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    if base_url:
        console.print(f"Base URL: [cyan]{base_url}[/cyan]")

    rollouts = []
    stats = {"total": 0, "success": 0, "failed": 0, "avg_reward": 0.0}

    with Progress() as progress:
        task = progress.add_task("Generating rollouts...", total=total_runs)

        for yaml_file in yaml_files:
            try:
                env = VoiceEnvironment.from_yaml(yaml_file)
            except Exception as e:
                console.print(f"[red]Failed to load {yaml_file}: {e}[/red]")
                continue

            for run_idx in range(runs_per_env):
                try:
                    runner = EnvironmentRunner(
                        env=env,
                        agent=agent_backend,
                        agent_model=model,
                        simulator_model=simulator_model,
                        openai_client=client,
                    )
                    result = runner.run()

                    # Format for GRPO training
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
                        "metadata": {
                            "model": model,
                            "simulator_model": simulator_model,
                            "run_index": run_idx,
                            "duration_seconds": result.duration_seconds,
                        },
                    }
                    rollouts.append(rollout)
                    stats["total"] += 1
                    stats["success"] += 1
                    stats["avg_reward"] += result.reward

                except Exception as e:
                    console.print(f"[yellow]Run failed for {env.name} (run {run_idx}): {e}[/yellow]")
                    stats["total"] += 1
                    stats["failed"] += 1

                progress.update(task, advance=1)

    # Write output
    output = Path(output_path)
    with output.open("w") as f:
        for rollout in rollouts:
            f.write(json.dumps(rollout) + "\n")

    if stats["success"] > 0:
        stats["avg_reward"] /= stats["success"]

    console.print(f"\n[green]Generated {stats['success']} rollouts[/green]")
    console.print(f"Failed: {stats['failed']}")
    console.print(f"Average reward: {stats['avg_reward']:.4f}")
    console.print(f"Output: [cyan]{output}[/cyan]")

    return rollouts


def main():
    parser = argparse.ArgumentParser(description="Generate RL training rollouts from VoiceEnv environments")
    parser.add_argument("--envs", required=True, help="Directory containing environment YAML files")
    parser.add_argument("--model", default="gpt-4o-mini", help="Agent model to generate rollouts for")
    parser.add_argument("--simulator-model", default="gpt-4o-mini", help="Simulator model")
    parser.add_argument("--runs-per-env", type=int, default=10, help="Number of runs per environment")
    parser.add_argument("--output", default="rollouts.jsonl", help="Output JSONL file")
    parser.add_argument("--base-url", default=None, help="Custom OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key for custom endpoint")

    args = parser.parse_args()
    generate_rollouts(
        env_dir=args.envs,
        model=args.model,
        simulator_model=args.simulator_model,
        runs_per_env=args.runs_per_env,
        output_path=args.output,
        base_url=args.base_url,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
