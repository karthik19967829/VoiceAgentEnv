"""
Systematic evaluation harness for comparing speech LLMs across VoiceEnv environments.

This is the measurement instrument. It runs a model against every environment
multiple times, collects per-criterion scores, and produces structured results
that can be compared across models.

The output format is designed for before/after comparison:
  1. Run `evaluate()` on base model → baseline results
  2. Post-train the model
  3. Run `evaluate()` on trained model → post-train results
  4. Compare with `compare_results()` → delta report

Every score is broken down by:
  - Environment (which scenario improved?)
  - Category (task_success, compliance, voice_quality, etc.)
  - Criterion (exactly which check improved?)
  - Verifiable vs soft (is the improvement real or gamed?)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

console = Console()


@dataclass
class CriterionScore:
    """Score for a single criterion in a single run."""
    name: str
    category: str
    score: float
    weight: float
    is_verifiable: bool
    reasoning: str = ""


@dataclass
class EnvironmentEvalResult:
    """Aggregated evaluation results for a single environment."""
    environment_name: str
    n_runs: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_verifiable_reward: float = 0.0
    mean_soft_reward: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    criterion_scores: dict[str, float] = field(default_factory=dict)
    criterion_pass_rates: dict[str, float] = field(default_factory=dict)
    mean_turns: float = 0.0
    mean_duration: float = 0.0
    raw_runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment_name,
            "n_runs": self.n_runs,
            "mean_reward": round(self.mean_reward, 4),
            "std_reward": round(self.std_reward, 4),
            "mean_verifiable_reward": round(self.mean_verifiable_reward, 4),
            "mean_soft_reward": round(self.mean_soft_reward, 4),
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "criterion_scores": {k: round(v, 4) for k, v in self.criterion_scores.items()},
            "criterion_pass_rates": {k: round(v, 4) for k, v in self.criterion_pass_rates.items()},
            "mean_turns": round(self.mean_turns, 1),
            "mean_duration": round(self.mean_duration, 2),
        }


@dataclass
class EvalResults:
    """Complete evaluation results for a model across all environments."""
    model_name: str
    timestamp: float = field(default_factory=time.time)
    environments: list[EnvironmentEvalResult] = field(default_factory=list)
    overall_reward: float = 0.0
    overall_verifiable_reward: float = 0.0
    overall_soft_reward: float = 0.0
    total_runs: int = 0
    total_duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "overall_reward": round(self.overall_reward, 4),
            "overall_verifiable_reward": round(self.overall_verifiable_reward, 4),
            "overall_soft_reward": round(self.overall_soft_reward, 4),
            "total_runs": self.total_runs,
            "total_duration_seconds": round(self.total_duration, 2),
            "environments": [e.to_dict() for e in self.environments],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EvalResults:
        data = json.loads(Path(path).read_text())
        envs = [EnvironmentEvalResult(**e) for e in data.pop("environments", [])]
        results = cls(**{k: v for k, v in data.items() if k != "total_duration_seconds"})
        results.environments = envs
        results.total_duration = data.get("total_duration_seconds", 0)
        return results


def evaluate(
    model: str = "gpt-4o-mini",
    env_dir: str | None = None,
    runs_per_env: int = 5,
    simulator_model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    skip_soft_scoring: bool = False,
    verbose: bool = True,
) -> EvalResults:
    """
    Run systematic evaluation of a model across all VoiceEnv environments.

    Args:
        model: Model to evaluate (OpenAI model name or local endpoint)
        env_dir: Directory with environment YAMLs (None = use built-in)
        runs_per_env: Number of runs per environment for statistical significance
        simulator_model: Model for the user simulator
        base_url: Custom OpenAI-compatible API endpoint
        api_key: API key for custom endpoint
        skip_soft_scoring: Skip LLM-judge scoring (faster, verifiable-only)
        verbose: Print progress

    Returns:
        EvalResults with per-environment, per-criterion breakdown
    """
    import math

    from openai import OpenAI
    from voiceenv.core.runner import EnvironmentRunner, OpenAIAgentBackend
    from voiceenv.core.schema import VoiceEnvironment

    # Load environments
    if env_dir:
        env_path = Path(env_dir)
        yaml_files = sorted(env_path.glob("*.yaml"))
        envs = [VoiceEnvironment.from_yaml(f) for f in yaml_files]
    else:
        from voiceenv.environments import list_environments, load_environment
        env_names = list_environments()
        envs = [load_environment(name) for name in sorted(env_names)]

    if not envs:
        console.print("[red]No environments found[/red]")
        return EvalResults(model_name=model)

    # Set up client
    client = None
    agent_backend = None
    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        agent_backend = OpenAIAgentBackend(model=model, client=client)
    else:
        client = OpenAI()

    total_runs = len(envs) * runs_per_env
    if verbose:
        console.print(f"\n[bold]Evaluating [cyan]{model}[/cyan] across {len(envs)} environments × {runs_per_env} runs[/bold]")

    eval_start = time.time()
    env_results: list[EnvironmentEvalResult] = []

    progress_ctx = Progress() if verbose else None
    if progress_ctx:
        progress_ctx.start()
        task = progress_ctx.add_task("Evaluating...", total=total_runs)

    for env in envs:
        run_rewards = []
        run_verifiable = []
        run_soft = []
        run_turns = []
        run_durations = []
        criterion_scores_agg: dict[str, list[float]] = defaultdict(list)
        criterion_pass_agg: dict[str, list[float]] = defaultdict(list)
        category_scores_agg: dict[str, list[float]] = defaultdict(list)
        raw_runs = []

        for run_idx in range(runs_per_env):
            try:
                runner = EnvironmentRunner(
                    env=env,
                    agent=agent_backend,
                    agent_model=model,
                    simulator_model=simulator_model,
                    openai_client=client,
                )
                # Override scorer if skip_soft_scoring
                if skip_soft_scoring:
                    runner.scorer.skip_soft_scoring = True

                result = runner.run()

                run_rewards.append(result.reward)
                run_verifiable.append(result.verifiable_reward)
                run_soft.append(result.soft_reward)
                run_turns.append(result.turn_count)
                run_durations.append(result.duration_seconds)

                for cr in result.scorecard.criteria_results:
                    criterion_scores_agg[cr.name].append(cr.score)
                    criterion_pass_agg[cr.name].append(1.0 if cr.score >= 0.5 else 0.0)

                for cat, score in result.scorecard.category_scores.items():
                    category_scores_agg[cat].append(score)

                raw_runs.append({
                    "run_index": run_idx,
                    "reward": result.reward,
                    "verifiable_reward": result.verifiable_reward,
                    "soft_reward": result.soft_reward,
                    "turn_count": result.turn_count,
                    "duration": result.duration_seconds,
                    "category_scores": result.scorecard.category_scores,
                })

            except Exception as e:
                if verbose:
                    console.print(f"  [yellow]{env.name} run {run_idx+1}: {e}[/yellow]")

            if progress_ctx:
                progress_ctx.update(task, advance=1)

        # Aggregate
        n = len(run_rewards)
        if n == 0:
            env_results.append(EnvironmentEvalResult(
                environment_name=env.name, n_runs=0,
            ))
            continue

        mean_r = sum(run_rewards) / n
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in run_rewards) / n) if n > 1 else 0.0

        env_result = EnvironmentEvalResult(
            environment_name=env.name,
            n_runs=n,
            mean_reward=mean_r,
            std_reward=std_r,
            mean_verifiable_reward=sum(run_verifiable) / n,
            mean_soft_reward=sum(run_soft) / n,
            category_scores={k: sum(v) / len(v) for k, v in category_scores_agg.items()},
            criterion_scores={k: sum(v) / len(v) for k, v in criterion_scores_agg.items()},
            criterion_pass_rates={k: sum(v) / len(v) for k, v in criterion_pass_agg.items()},
            mean_turns=sum(run_turns) / n,
            mean_duration=sum(run_durations) / n,
            raw_runs=raw_runs,
        )
        env_results.append(env_result)

    if progress_ctx:
        progress_ctx.stop()

    # Overall aggregation
    all_rewards = []
    all_verifiable = []
    all_soft = []
    for er in env_results:
        if er.n_runs > 0:
            all_rewards.extend([er.mean_reward] * er.n_runs)
            all_verifiable.extend([er.mean_verifiable_reward] * er.n_runs)
            all_soft.extend([er.mean_soft_reward] * er.n_runs)

    total_n = sum(er.n_runs for er in env_results)
    total_dur = time.time() - eval_start

    results = EvalResults(
        model_name=model,
        environments=env_results,
        overall_reward=sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        overall_verifiable_reward=sum(all_verifiable) / len(all_verifiable) if all_verifiable else 0,
        overall_soft_reward=sum(all_soft) / len(all_soft) if all_soft else 0,
        total_runs=total_n,
        total_duration=total_dur,
    )

    if verbose:
        _print_eval_summary(results)

    return results


def _print_eval_summary(results: EvalResults) -> None:
    """Print a formatted evaluation summary."""
    console.print(f"\n{'='*70}")
    console.print(f"[bold]EVALUATION: {results.model_name}[/bold]")
    console.print(f"{'='*70}")

    table = Table(show_header=True)
    table.add_column("Environment", style="cyan", min_width=30)
    table.add_column("Runs", justify="right", width=5)
    table.add_column("Reward", justify="right", width=8)
    table.add_column("Verifiable", justify="right", width=10)
    table.add_column("Soft", justify="right", width=8)
    table.add_column("Turns", justify="right", width=6)

    for er in results.environments:
        if er.n_runs == 0:
            table.add_row(er.environment_name, "0", "-", "-", "-", "-")
            continue
        r_color = "green" if er.mean_reward >= 0.7 else "yellow" if er.mean_reward >= 0.4 else "red"
        v_color = "green" if er.mean_verifiable_reward >= 0.7 else "yellow" if er.mean_verifiable_reward >= 0.4 else "red"
        table.add_row(
            er.environment_name,
            str(er.n_runs),
            f"[{r_color}]{er.mean_reward:.3f}[/{r_color}]",
            f"[{v_color}]{er.mean_verifiable_reward:.3f}[/{v_color}]",
            f"{er.mean_soft_reward:.3f}",
            f"{er.mean_turns:.0f}",
        )

    console.print(table)

    o_color = "green" if results.overall_reward >= 0.7 else "yellow" if results.overall_reward >= 0.4 else "red"
    console.print(f"\n[bold]Overall:[/bold] [{o_color}]{results.overall_reward:.4f}[/{o_color}]"
                  f"  (verifiable: {results.overall_verifiable_reward:.4f}"
                  f"  soft: {results.overall_soft_reward:.4f})")
    console.print(f"Total runs: {results.total_runs}  Duration: {results.total_duration:.1f}s")
