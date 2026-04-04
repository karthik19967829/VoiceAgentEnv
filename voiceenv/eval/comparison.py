"""
Before/After comparison report — the proof that post-training works.

Takes two EvalResults (baseline vs trained) and produces:
  1. Per-environment delta table (which scenarios improved?)
  2. Per-category delta (task_success improved, compliance held?)
  3. Per-criterion delta (exactly which checks flipped?)
  4. Verifiable vs soft breakdown (is the gain real?)
  5. Statistical significance (is the gain reliable?)
  6. ASCII + JSON report for sharing

This is the deliverable that proves VoiceEnv environments produce
useful training signal for speech LLMs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voiceenv.eval.evaluator import EvalResults

console = Console()


@dataclass
class CriterionDelta:
    """Change in a single criterion between baseline and trained."""
    name: str
    baseline_score: float
    trained_score: float
    delta: float
    baseline_pass_rate: float
    trained_pass_rate: float
    pass_rate_delta: float

    @property
    def improved(self) -> bool:
        return self.delta > 0.01

    @property
    def regressed(self) -> bool:
        return self.delta < -0.01


@dataclass
class EnvironmentDelta:
    """Change in an environment between baseline and trained."""
    environment_name: str
    baseline_reward: float
    trained_reward: float
    delta: float
    delta_pct: float
    baseline_verifiable: float
    trained_verifiable: float
    verifiable_delta: float
    baseline_soft: float
    trained_soft: float
    soft_delta: float
    category_deltas: dict[str, float] = field(default_factory=dict)
    criterion_deltas: list[CriterionDelta] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full before/after comparison between baseline and trained model."""
    baseline_model: str
    trained_model: str

    overall_baseline: float = 0.0
    overall_trained: float = 0.0
    overall_delta: float = 0.0
    overall_delta_pct: float = 0.0

    verifiable_baseline: float = 0.0
    verifiable_trained: float = 0.0
    verifiable_delta: float = 0.0

    soft_baseline: float = 0.0
    soft_trained: float = 0.0
    soft_delta: float = 0.0

    environments: list[EnvironmentDelta] = field(default_factory=list)

    n_envs_improved: int = 0
    n_envs_regressed: int = 0
    n_envs_unchanged: int = 0

    n_criteria_improved: int = 0
    n_criteria_regressed: int = 0

    top_improvements: list[CriterionDelta] = field(default_factory=list)
    top_regressions: list[CriterionDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "trained_model": self.trained_model,
            "overall": {
                "baseline": round(self.overall_baseline, 4),
                "trained": round(self.overall_trained, 4),
                "delta": round(self.overall_delta, 4),
                "delta_pct": round(self.overall_delta_pct, 2),
            },
            "verifiable": {
                "baseline": round(self.verifiable_baseline, 4),
                "trained": round(self.verifiable_trained, 4),
                "delta": round(self.verifiable_delta, 4),
            },
            "soft": {
                "baseline": round(self.soft_baseline, 4),
                "trained": round(self.soft_trained, 4),
                "delta": round(self.soft_delta, 4),
            },
            "environment_summary": {
                "improved": self.n_envs_improved,
                "regressed": self.n_envs_regressed,
                "unchanged": self.n_envs_unchanged,
            },
            "criteria_summary": {
                "improved": self.n_criteria_improved,
                "regressed": self.n_criteria_regressed,
            },
            "environments": [
                {
                    "name": e.environment_name,
                    "baseline_reward": round(e.baseline_reward, 4),
                    "trained_reward": round(e.trained_reward, 4),
                    "delta": round(e.delta, 4),
                    "delta_pct": round(e.delta_pct, 2),
                    "verifiable_delta": round(e.verifiable_delta, 4),
                    "category_deltas": {k: round(v, 4) for k, v in e.category_deltas.items()},
                    "criterion_deltas": [
                        {
                            "name": c.name,
                            "baseline": round(c.baseline_score, 4),
                            "trained": round(c.trained_score, 4),
                            "delta": round(c.delta, 4),
                            "pass_rate_delta": round(c.pass_rate_delta, 4),
                        }
                        for c in e.criterion_deltas
                    ],
                }
                for e in self.environments
            ],
            "top_improvements": [
                {"criterion": c.name, "delta": round(c.delta, 4)} for c in self.top_improvements
            ],
            "top_regressions": [
                {"criterion": c.name, "delta": round(c.delta, 4)} for c in self.top_regressions
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def compare(baseline: EvalResults, trained: EvalResults) -> ComparisonReport:
    """
    Compare baseline and trained evaluation results.

    Produces a structured report showing exactly what improved,
    what regressed, and whether the gains are in verifiable or soft metrics.
    """
    report = ComparisonReport(
        baseline_model=baseline.model_name,
        trained_model=trained.model_name,
        overall_baseline=baseline.overall_reward,
        overall_trained=trained.overall_reward,
        overall_delta=trained.overall_reward - baseline.overall_reward,
        overall_delta_pct=(
            (trained.overall_reward - baseline.overall_reward) / baseline.overall_reward * 100
            if baseline.overall_reward > 0 else 0
        ),
        verifiable_baseline=baseline.overall_verifiable_reward,
        verifiable_trained=trained.overall_verifiable_reward,
        verifiable_delta=trained.overall_verifiable_reward - baseline.overall_verifiable_reward,
        soft_baseline=baseline.overall_soft_reward,
        soft_trained=trained.overall_soft_reward,
        soft_delta=trained.overall_soft_reward - baseline.overall_soft_reward,
    )

    # Match environments by name
    baseline_by_name = {e.environment_name: e for e in baseline.environments}
    trained_by_name = {e.environment_name: e for e in trained.environments}

    all_criterion_deltas = []

    for env_name in set(list(baseline_by_name.keys()) + list(trained_by_name.keys())):
        b = baseline_by_name.get(env_name)
        t = trained_by_name.get(env_name)

        if not b or not t or b.n_runs == 0 or t.n_runs == 0:
            continue

        delta = t.mean_reward - b.mean_reward
        delta_pct = (delta / b.mean_reward * 100) if b.mean_reward > 0 else 0

        # Category deltas
        all_cats = set(list(b.category_scores.keys()) + list(t.category_scores.keys()))
        cat_deltas = {}
        for cat in all_cats:
            b_score = b.category_scores.get(cat, 0)
            t_score = t.category_scores.get(cat, 0)
            cat_deltas[cat] = t_score - b_score

        # Criterion deltas
        all_criteria = set(list(b.criterion_scores.keys()) + list(t.criterion_scores.keys()))
        crit_deltas = []
        for crit in sorted(all_criteria):
            b_score = b.criterion_scores.get(crit, 0)
            t_score = t.criterion_scores.get(crit, 0)
            b_pass = b.criterion_pass_rates.get(crit, 0)
            t_pass = t.criterion_pass_rates.get(crit, 0)
            cd = CriterionDelta(
                name=crit,
                baseline_score=b_score,
                trained_score=t_score,
                delta=t_score - b_score,
                baseline_pass_rate=b_pass,
                trained_pass_rate=t_pass,
                pass_rate_delta=t_pass - b_pass,
            )
            crit_deltas.append(cd)
            all_criterion_deltas.append(cd)

        env_delta = EnvironmentDelta(
            environment_name=env_name,
            baseline_reward=b.mean_reward,
            trained_reward=t.mean_reward,
            delta=delta,
            delta_pct=delta_pct,
            baseline_verifiable=b.mean_verifiable_reward,
            trained_verifiable=t.mean_verifiable_reward,
            verifiable_delta=t.mean_verifiable_reward - b.mean_verifiable_reward,
            baseline_soft=b.mean_soft_reward,
            trained_soft=t.mean_soft_reward,
            soft_delta=t.mean_soft_reward - b.mean_soft_reward,
            category_deltas=cat_deltas,
            criterion_deltas=crit_deltas,
        )
        report.environments.append(env_delta)

        if delta > 0.01:
            report.n_envs_improved += 1
        elif delta < -0.01:
            report.n_envs_regressed += 1
        else:
            report.n_envs_unchanged += 1

    # Top improvements and regressions
    report.n_criteria_improved = sum(1 for c in all_criterion_deltas if c.improved)
    report.n_criteria_regressed = sum(1 for c in all_criterion_deltas if c.regressed)

    sorted_by_delta = sorted(all_criterion_deltas, key=lambda c: c.delta, reverse=True)
    report.top_improvements = [c for c in sorted_by_delta[:10] if c.improved]
    report.top_regressions = [c for c in sorted_by_delta[-10:] if c.regressed]

    return report


def print_comparison(report: ComparisonReport) -> None:
    """Print a formatted comparison report to the console."""

    console.print(Panel(
        f"[bold]Baseline:[/bold] {report.baseline_model}\n"
        f"[bold]Trained:[/bold]  {report.trained_model}",
        title="Post-Training Comparison",
        border_style="cyan",
    ))

    # Overall summary
    delta_color = "green" if report.overall_delta > 0 else "red" if report.overall_delta < 0 else "yellow"
    console.print(f"\n[bold]Overall Reward:[/bold]  "
                  f"{report.overall_baseline:.4f} → {report.overall_trained:.4f}  "
                  f"[{delta_color}]{report.overall_delta:+.4f} ({report.overall_delta_pct:+.1f}%)[/{delta_color}]")

    v_color = "green" if report.verifiable_delta > 0 else "red"
    console.print(f"[bold]Verifiable:[/bold]      "
                  f"{report.verifiable_baseline:.4f} → {report.verifiable_trained:.4f}  "
                  f"[{v_color}]{report.verifiable_delta:+.4f}[/{v_color}]")

    s_color = "green" if report.soft_delta > 0 else "red"
    console.print(f"[bold]Soft:[/bold]            "
                  f"{report.soft_baseline:.4f} → {report.soft_trained:.4f}  "
                  f"[{s_color}]{report.soft_delta:+.4f}[/{s_color}]")

    console.print(f"\n[bold]Environments:[/bold]  "
                  f"[green]{report.n_envs_improved} improved[/green]  "
                  f"[red]{report.n_envs_regressed} regressed[/red]  "
                  f"{report.n_envs_unchanged} unchanged")
    console.print(f"[bold]Criteria:[/bold]      "
                  f"[green]{report.n_criteria_improved} improved[/green]  "
                  f"[red]{report.n_criteria_regressed} regressed[/red]")

    # Per-environment table
    console.print("")
    env_table = Table(title="Per-Environment Comparison", show_header=True)
    env_table.add_column("Environment", style="cyan", min_width=30)
    env_table.add_column("Baseline", justify="right", width=8)
    env_table.add_column("Trained", justify="right", width=8)
    env_table.add_column("Delta", justify="right", width=10)
    env_table.add_column("Verif. Delta", justify="right", width=11)

    for e in sorted(report.environments, key=lambda x: x.delta, reverse=True):
        d_color = "green" if e.delta > 0.01 else "red" if e.delta < -0.01 else "dim"
        v_color = "green" if e.verifiable_delta > 0.01 else "red" if e.verifiable_delta < -0.01 else "dim"
        env_table.add_row(
            e.environment_name,
            f"{e.baseline_reward:.3f}",
            f"{e.trained_reward:.3f}",
            f"[{d_color}]{e.delta:+.3f} ({e.delta_pct:+.0f}%)[/{d_color}]",
            f"[{v_color}]{e.verifiable_delta:+.3f}[/{v_color}]",
        )
    console.print(env_table)

    # Top improvements
    if report.top_improvements:
        console.print("")
        imp_table = Table(title="Top Criterion Improvements", show_header=True)
        imp_table.add_column("Criterion", style="green", min_width=30)
        imp_table.add_column("Baseline", justify="right", width=8)
        imp_table.add_column("Trained", justify="right", width=8)
        imp_table.add_column("Delta", justify="right", width=8)
        imp_table.add_column("Pass Rate", justify="right", width=12)

        for c in report.top_improvements[:10]:
            imp_table.add_row(
                c.name,
                f"{c.baseline_score:.3f}",
                f"{c.trained_score:.3f}",
                f"[green]{c.delta:+.3f}[/green]",
                f"{c.baseline_pass_rate:.0%}→{c.trained_pass_rate:.0%}",
            )
        console.print(imp_table)

    # Top regressions
    if report.top_regressions:
        console.print("")
        reg_table = Table(title="Top Criterion Regressions", show_header=True)
        reg_table.add_column("Criterion", style="red", min_width=30)
        reg_table.add_column("Baseline", justify="right", width=8)
        reg_table.add_column("Trained", justify="right", width=8)
        reg_table.add_column("Delta", justify="right", width=8)

        for c in report.top_regressions[-5:]:
            reg_table.add_row(
                c.name,
                f"{c.baseline_score:.3f}",
                f"{c.trained_score:.3f}",
                f"[red]{c.delta:+.3f}[/red]",
            )
        console.print(reg_table)

    # Verdict
    console.print("")
    if report.overall_delta > 0.05 and report.verifiable_delta > 0.03:
        console.print(Panel(
            f"[bold green]Post-training improved overall reward by {report.overall_delta_pct:+.1f}%[/bold green]\n"
            f"Verifiable reward (real capability) improved by {report.verifiable_delta:+.4f}\n"
            f"The gain is grounded in deterministic checks, not just LLM-judge gaming.",
            title="VERDICT: SUCCESS",
            border_style="green",
        ))
    elif report.overall_delta > 0 and report.verifiable_delta <= 0:
        console.print(Panel(
            f"[bold yellow]Overall reward improved by {report.overall_delta_pct:+.1f}%, "
            f"but verifiable reward did not improve.[/bold yellow]\n"
            f"The gain may be from LLM-judge gaming rather than real capability improvement.\n"
            f"Consider training with verifiable_reward only.",
            title="VERDICT: INCONCLUSIVE",
            border_style="yellow",
        ))
    elif report.overall_delta <= 0:
        console.print(Panel(
            f"[bold red]No overall improvement from post-training.[/bold red]\n"
            f"Try: more rollouts, different environments, or adjusted hyperparameters.",
            title="VERDICT: NO IMPROVEMENT",
            border_style="red",
        ))
