"""
VoiceEnv CLI — create, run, export, and publish voice agent environments.

Commands:
  voiceenv init <name>            Create a new environment from template
  voiceenv list                   List built-in environments
  voiceenv run <env> [--model]    Run an environment against a speech LLM
  voiceenv score <run-file>       Score a completed run
  voiceenv export <env> --target  Export to OpenEnv or Prime Intellect format
  voiceenv publish <env> --target Push to HuggingFace or Prime Intellect hub
  voiceenv benchmark <dir>        Run all environments and produce leaderboard
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """VoiceEnv — Community-driven voice agent environments for speech LLMs."""
    pass


@cli.command()
def list():
    """List all built-in environments."""
    from voiceenv.environments import list_environments, load_environment

    envs = list_environments()
    table = Table(title="Built-in Voice Environments", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Vertical", style="green")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Languages")
    table.add_column("Description", max_width=50)

    for name in sorted(envs):
        env = load_environment(name)
        table.add_row(
            env.name,
            env.vertical.value,
            env.difficulty.value,
            ", ".join(env.languages),
            env.description[:80] + "..." if len(env.description) > 80 else env.description,
        )

    console.print(table)


@cli.command()
@click.argument("name")
@click.option("--output", "-o", default=".", help="Output directory")
def init(name: str, output: str):
    """Create a new environment from a starter template."""
    from voiceenv.core.schema import (
        VoiceEnvironment, TaskDefinition, SimulatorProfile,
        ScoringRubric, ScoringCriterion, WorldState, VoiceConfig,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = VoiceEnvironment(
        name=name,
        description=f"A custom voice environment: {name}",
        author="your-name",
        tags=["custom"],
        task=TaskDefinition(
            goal="Define the agent's goal here",
            context="Provide context for the agent",
            success_criteria=["Criterion 1", "Criterion 2"],
            failure_conditions=["Failure condition 1"],
        ),
        world_state=WorldState(
            description="Initial world state",
            fields={"example_field": "example_value"},
        ),
        simulator=SimulatorProfile(
            persona_description="Describe the simulated caller here",
        ),
        rubric=ScoringRubric(
            task_success=[
                ScoringCriterion(
                    name="goal_achieved",
                    description="The agent achieved the primary goal",
                    weight=2.0,
                    scoring_type="llm_judge",
                ),
            ],
        ),
        voice=VoiceConfig(),
        agent_system_prompt="You are a voice agent. Define your system prompt here.",
    )

    filepath = output_dir / f"{name}.yaml"
    env.to_yaml(filepath)
    console.print(f"[green]Created environment template:[/green] {filepath}")
    console.print(f"Edit the YAML file to define your environment, then run:")
    console.print(f"  [cyan]voiceenv run {filepath}[/cyan]")


@cli.command()
@click.argument("env_path")
@click.option("--model", "-m", default="gpt-4o-mini", help="Agent model to evaluate")
@click.option("--simulator-model", default="gpt-4o-mini", help="Simulator model")
@click.option("--runs", "-n", default=1, help="Number of runs")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def run(env_path: str, model: str, simulator_model: str, runs: int, output: str | None):
    """Run an environment against a voice agent model."""
    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.runner import EnvironmentRunner
    from voiceenv.environments import load_environment

    # Load environment (from file or built-in)
    path = Path(env_path)
    if path.exists():
        env = VoiceEnvironment.from_yaml(path)
    else:
        try:
            env = load_environment(env_path)
        except FileNotFoundError:
            console.print(f"[red]Environment not found:[/red] {env_path}")
            console.print("Use [cyan]voiceenv list[/cyan] to see built-in environments.")
            sys.exit(1)

    console.print(Panel(
        f"[bold]{env.name}[/bold]\n{env.description[:120]}",
        title="Voice Environment",
        border_style="cyan",
    ))
    console.print(f"Agent model: [cyan]{model}[/cyan]")
    console.print(f"Simulator model: [cyan]{simulator_model}[/cyan]")
    console.print(f"Runs: [cyan]{runs}[/cyan]\n")

    all_results = []

    with Progress() as progress:
        task = progress.add_task("Running environments...", total=runs)

        for i in range(runs):
            runner = EnvironmentRunner(
                env=env,
                agent_model=model,
                simulator_model=simulator_model,
            )
            result = runner.run()
            all_results.append(result)
            progress.update(task, advance=1)

    # Display results
    for i, result in enumerate(all_results):
        console.print(f"\n[bold]Run {i+1}/{runs}[/bold]")

        # Transcript
        table = Table(title="Transcript", show_header=True)
        table.add_column("Speaker", style="bold", width=8)
        table.add_column("Content", max_width=90)
        for turn in result.transcript:
            style = "cyan" if turn["role"] == "agent" else "yellow"
            table.add_row(turn["role"].upper(), turn["content"][:200])
        console.print(table)

        # Tool calls
        if result.tool_calls:
            tool_table = Table(title="Tool Calls", show_header=True)
            tool_table.add_column("Tool", style="green")
            tool_table.add_column("Args", max_width=40)
            tool_table.add_column("Success", width=8)
            for tc in result.tool_calls:
                tool_table.add_row(
                    tc["tool"],
                    str(tc["arguments"])[:60],
                    "[green]Yes[/green]" if tc["success"] else "[red]No[/red]",
                )
            console.print(tool_table)

        # Scorecard
        score_table = Table(title="Scorecard", show_header=True)
        score_table.add_column("Category", style="bold")
        score_table.add_column("Score", justify="right")
        for cat, score in result.scorecard.category_scores.items():
            color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
            score_table.add_row(cat, f"[{color}]{score:.2%}[/{color}]")
        score_table.add_row("", "")
        color = "green" if result.reward >= 0.7 else "yellow" if result.reward >= 0.4 else "red"
        score_table.add_row("[bold]TOTAL REWARD[/bold]", f"[bold {color}]{result.reward:.2%}[/bold {color}]")
        console.print(score_table)

        console.print(f"Turns: {result.turn_count} | Duration: {result.duration_seconds:.1f}s")

    # Save results
    if output:
        output_path = Path(output)
        results_data = [r.to_dict() for r in all_results]
        output_path.write_text(json.dumps(results_data, indent=2))
        console.print(f"\n[green]Results saved to:[/green] {output_path}")

    # Also output training examples
    if runs > 0:
        training_data = [r.to_training_example() for r in all_results]
        console.print(f"\n[dim]Training examples generated: {len(training_data)} "
                      f"(use --output to save)[/dim]")


@cli.command()
@click.argument("env_path")
@click.option("--target", "-t", required=True,
              type=click.Choice(["openenv", "prime", "both"]),
              help="Target hub to export to")
@click.option("--output", "-o", default="./exports", help="Output directory")
def export(env_path: str, target: str, output: str):
    """Export an environment to OpenEnv or Prime Intellect format."""
    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.environments import load_environment

    path = Path(env_path)
    if path.exists():
        env = VoiceEnvironment.from_yaml(path)
    else:
        env = load_environment(env_path)

    output_dir = Path(output)

    if target in ("openenv", "both"):
        from voiceenv.exporters.openenv_exporter import export_openenv
        pkg_path = export_openenv(env, output_dir / "openenv")
        console.print(f"[green]OpenEnv package exported to:[/green] {pkg_path}")
        console.print(f"  Push with: [cyan]cd {pkg_path} && openenv push[/cyan]")

    if target in ("prime", "both"):
        from voiceenv.exporters.prime_exporter import export_prime
        mod_path = export_prime(env, output_dir / "prime")
        console.print(f"[green]Prime Intellect module exported to:[/green] {mod_path}")
        console.print(f"  Push with: [cyan]cd {mod_path} && prime env push[/cyan]")


@cli.command()
@click.argument("env_path")
@click.option("--target", "-t", required=True,
              type=click.Choice(["openenv", "prime", "both"]),
              help="Target hub to publish to")
@click.option("--repo-id", default=None, help="HuggingFace repo ID (for OpenEnv)")
@click.option("--team", default=None, help="Team name (for Prime Intellect)")
def publish(env_path: str, target: str, repo_id: str | None, team: str | None):
    """Export and publish an environment to a hub in one step."""
    import tempfile

    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.environments import load_environment

    path = Path(env_path)
    if path.exists():
        env = VoiceEnvironment.from_yaml(path)
    else:
        env = load_environment(env_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        if target in ("openenv", "both"):
            from voiceenv.exporters.openenv_exporter import export_openenv
            pkg_path = export_openenv(env, Path(tmpdir) / "openenv")
            console.print(f"[green]OpenEnv package created[/green]")

            cmd = ["openenv", "push"]
            if repo_id:
                cmd.extend(["--repo-id", repo_id])

            console.print(f"[cyan]Running: {' '.join(cmd)}[/cyan]")
            try:
                result = subprocess.run(cmd, cwd=str(pkg_path), capture_output=True, text=True)
                if result.returncode == 0:
                    console.print(f"[green]Published to OpenEnv / HuggingFace![/green]")
                else:
                    console.print(f"[yellow]openenv push output:[/yellow]\n{result.stderr or result.stdout}")
                    console.print(f"[dim]Package is at {pkg_path} — you can push manually.[/dim]")
            except FileNotFoundError:
                console.print(f"[yellow]openenv CLI not found. Install with: pip install openenv-core[/yellow]")
                console.print(f"Package exported to: {pkg_path}")

        if target in ("prime", "both"):
            from voiceenv.exporters.prime_exporter import export_prime
            mod_path = export_prime(env, Path(tmpdir) / "prime")
            console.print(f"[green]Prime Intellect module created[/green]")

            cmd = ["prime", "env", "push"]
            if team:
                cmd.extend(["--team", team])

            console.print(f"[cyan]Running: {' '.join(cmd)}[/cyan]")
            try:
                result = subprocess.run(cmd, cwd=str(mod_path), capture_output=True, text=True)
                if result.returncode == 0:
                    console.print(f"[green]Published to Prime Intellect Environments Hub![/green]")
                else:
                    console.print(f"[yellow]prime env push output:[/yellow]\n{result.stderr or result.stdout}")
                    console.print(f"[dim]Module is at {mod_path} — you can push manually.[/dim]")
            except FileNotFoundError:
                console.print(f"[yellow]prime CLI not found. Install with: uv tool install prime[/yellow]")
                console.print(f"Module exported to: {mod_path}")


@cli.command()
@click.argument("env_dir", default=".")
@click.option("--model", "-m", multiple=True, default=["gpt-4o-mini"],
              help="Models to benchmark (can specify multiple)")
@click.option("--runs", "-n", default=3, help="Runs per environment per model")
@click.option("--output", "-o", default="benchmark_results.json", help="Output file")
def benchmark(env_dir: str, model: tuple[str, ...], runs: int, output: str):
    """Run all environments in a directory and produce benchmark results."""
    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.core.runner import EnvironmentRunner

    env_path = Path(env_dir)
    yaml_files = sorted(env_path.glob("*.yaml"))

    if not yaml_files:
        console.print(f"[red]No .yaml environment files found in {env_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Benchmarking {len(yaml_files)} environments × {len(model)} models × {runs} runs[/bold]\n")

    results = []

    with Progress() as progress:
        total = len(yaml_files) * len(model) * runs
        task = progress.add_task("Running benchmark...", total=total)

        for yaml_file in yaml_files:
            env = VoiceEnvironment.from_yaml(yaml_file)
            for m in model:
                for run_idx in range(runs):
                    try:
                        runner = EnvironmentRunner(env=env, agent_model=m)
                        result = runner.run()
                        results.append({
                            "environment": env.name,
                            "model": m,
                            "run": run_idx + 1,
                            "reward": result.reward,
                            "category_scores": result.scorecard.category_scores,
                            "turns": result.turn_count,
                            "duration": result.duration_seconds,
                        })
                    except Exception as e:
                        results.append({
                            "environment": env.name,
                            "model": m,
                            "run": run_idx + 1,
                            "reward": 0.0,
                            "error": str(e),
                        })
                    progress.update(task, advance=1)

    # Display leaderboard
    leaderboard: dict[str, list[float]] = {}
    for r in results:
        key = r["model"]
        leaderboard.setdefault(key, []).append(r["reward"])

    table = Table(title="Benchmark Leaderboard", show_header=True)
    table.add_column("Rank", width=6)
    table.add_column("Model", style="cyan")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Runs", justify="right")

    sorted_models = sorted(leaderboard.items(), key=lambda x: -sum(x[1]) / len(x[1]))
    for rank, (m, rewards) in enumerate(sorted_models, 1):
        avg = sum(rewards) / len(rewards)
        color = "green" if avg >= 0.7 else "yellow" if avg >= 0.4 else "red"
        table.add_row(str(rank), m, f"[{color}]{avg:.2%}[/{color}]", str(len(rewards)))

    console.print(table)

    # Save
    Path(output).write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]Results saved to:[/green] {output}")


@cli.group()
def train():
    """Training commands — generate rollouts and fine-tune speech LLMs."""
    pass


@train.command("rollouts")
@click.argument("env_dir", default="voiceenv/environments")
@click.option("--model", "-m", default="gpt-4o-mini", help="Agent model for rollouts")
@click.option("--simulator-model", default="gpt-4o-mini", help="Simulator model")
@click.option("--runs-per-env", "-n", default=10, help="Runs per environment")
@click.option("--output", "-o", default="rollouts.jsonl", help="Output JSONL file")
@click.option("--base-url", default=None, help="Custom OpenAI-compatible API URL")
@click.option("--api-key", default=None, help="API key for custom endpoint")
def train_rollouts(env_dir, model, simulator_model, runs_per_env, output, base_url, api_key):
    """Generate training rollouts from environments."""
    from voiceenv.training.generate_rollouts import generate_rollouts
    generate_rollouts(
        env_dir=env_dir,
        model=model,
        simulator_model=simulator_model,
        runs_per_env=runs_per_env,
        output_path=output,
        base_url=base_url,
        api_key=api_key,
    )


@train.command("modal")
@click.option("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model to fine-tune")
@click.option("--runs-per-env", default=20, help="Rollout runs per environment")
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--lr", default=2e-5, help="Learning rate")
@click.option("--epochs", default=2, help="Training epochs")
def train_modal(model, runs_per_env, lora_rank, lr, epochs):
    """Fine-tune on Modal serverless GPUs (H100)."""
    console.print("[bold]Modal Training Pipeline[/bold]\n")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"GPU: [cyan]H100 (serverless)[/cyan]")
    console.print(f"\nTo run the full pipeline:")
    console.print(f"  [cyan]modal run voiceenv/training/modal_train.py "
                  f"--model {model} --runs-per-env {runs_per_env} "
                  f"--lora-rank {lora_rank} --learning-rate {lr} --epochs {epochs}[/cyan]")
    console.print(f"\nOr step by step:")
    console.print(f"  [cyan]modal run voiceenv/training/modal_train.py --generate-only[/cyan]")
    console.print(f"  [cyan]modal run voiceenv/training/modal_train.py --train-only[/cyan]")
    console.print(f"  [cyan]modal run voiceenv/training/modal_train.py --eval-only[/cyan]")


@train.command("baseten")
@click.option("--rollouts", required=True, help="Path to rollouts JSONL")
@click.option("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model to fine-tune")
@click.option("--output", "-o", default="baseten_voiceenv_training", help="Output project dir")
@click.option("--gpu", default="H100", type=click.Choice(["H100", "H200", "A10G"]))
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--max-steps", default=200, help="Max training steps")
def train_baseten(rollouts, model, output, gpu, lora_rank, max_steps):
    """Generate a Baseten training project for managed GPU fine-tuning."""
    from voiceenv.training.baseten_train import generate_baseten_project
    generate_baseten_project(
        rollouts_path=rollouts,
        output_dir=output,
        model_name=model,
        gpu_type=gpu,
        lora_rank=lora_rank,
        max_steps=max_steps,
    )


@cli.group()
def judge():
    """Judge validation — rate runs, compute correlation, build trust."""
    pass


@judge.command("save-run")
@click.argument("results_json")
@click.option("--ratings-dir", default="ratings", help="Directory for ratings data")
def judge_save_run(results_json: str, ratings_dir: str):
    """Save a completed run for community rating."""
    from voiceenv.core.human_ratings import RatingStore, package_run_for_rating

    results_path = Path(results_json)
    if not results_path.exists():
        console.print(f"[red]Results file not found:[/red] {results_json}")
        sys.exit(1)

    data = json.loads(results_path.read_text())
    runs = data if isinstance(data, list) else [data]

    store = RatingStore(ratings_dir)
    saved = 0
    for run in runs:
        transcript = run.get("transcript", [])
        env_name = run.get("environment", "unknown")

        criteria = []
        for sc in run.get("soft_criteria", []):
            criteria.append({"name": sc["name"], "description": sc.get("category", "")})

        llm_scores = {}
        for sc in run.get("soft_criteria", []):
            llm_scores[sc["name"]] = sc.get("score", 0.0)

        pkg = package_run_for_rating(
            env_name=env_name,
            transcript=transcript,
            criteria=criteria,
            tool_calls=run.get("tool_calls", []),
            llm_scores=llm_scores,
        )
        path = store.save_run_for_rating(pkg)
        saved += 1
        console.print(f"[green]Saved run for rating:[/green] {pkg.run_id} → {path}")

    console.print(f"\n[bold]Saved {saved} run(s) for community rating.[/bold]")
    console.print(f"Rate them with: [cyan]voiceenv judge rate --ratings-dir {ratings_dir}[/cyan]")


@judge.command("rate")
@click.option("--run-id", default=None, help="Specific run ID to rate (omit for next unrated)")
@click.option("--rater-id", prompt="Your rater ID (name/handle)", help="Your identifier")
@click.option("--ratings-dir", default="ratings", help="Directory for ratings data")
def judge_rate(run_id: str | None, rater_id: str, ratings_dir: str):
    """Rate a run as a human judge (interactive)."""
    from voiceenv.core.human_ratings import HumanRating, RatingStore

    store = RatingStore(ratings_dir)

    if run_id is None:
        available = store.list_runs()
        if not available:
            console.print("[yellow]No runs available for rating.[/yellow]")
            console.print("First save some runs: [cyan]voiceenv judge save-run results.json[/cyan]")
            return
        run_id = available[0]
        console.print(f"[dim]Auto-selected run: {run_id}[/dim]")

    try:
        run_data = store.load_run_for_rating(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run not found:[/red] {run_id}")
        return

    console.print(Panel(
        f"[bold]Environment:[/bold] {run_data.environment_name}\n"
        f"[bold]Run ID:[/bold] {run_data.run_id}",
        title="Rating Session",
        border_style="cyan",
    ))

    # Show transcript
    table = Table(title="Conversation Transcript", show_header=True)
    table.add_column("Speaker", style="bold", width=8)
    table.add_column("Content", max_width=90)
    for turn in run_data.transcript:
        style = "cyan" if turn["role"] == "agent" else "yellow"
        table.add_row(turn["role"].upper(), turn["content"][:200])
    console.print(table)

    if run_data.audio_path:
        console.print(f"\n[dim]Audio available at: {run_data.audio_path}[/dim]")

    console.print(f"\n[bold]Rate each criterion (0.0 = terrible, 1.0 = perfect):[/bold]\n")

    ratings = []
    for crit in run_data.criteria_to_rate:
        console.print(f"[bold cyan]{crit['name']}[/bold cyan]: {crit.get('description', '')}")

        while True:
            score_str = click.prompt("  Score (0.0-1.0)", type=str)
            try:
                score = float(score_str)
                if 0.0 <= score <= 1.0:
                    break
                console.print("  [red]Score must be between 0.0 and 1.0[/red]")
            except ValueError:
                console.print("  [red]Please enter a number[/red]")

        reasoning = click.prompt("  Brief reasoning (optional)", default="", show_default=False)
        audio = click.confirm("  Did you listen to audio?", default=False) if run_data.audio_path else False

        ratings.append(HumanRating(
            run_id=run_id,
            criterion_name=crit["name"],
            rater_id=rater_id,
            score=score,
            reasoning=reasoning,
            audio_listened=audio,
        ))

    store.submit_ratings(ratings)
    console.print(f"\n[green]Submitted {len(ratings)} ratings. Thank you![/green]")

    # Show comparison with LLM scores if available
    if run_data.llm_scores:
        compare_table = Table(title="Your Ratings vs LLM Judge", show_header=True)
        compare_table.add_column("Criterion", style="cyan")
        compare_table.add_column("Your Score", justify="right")
        compare_table.add_column("LLM Score", justify="right")
        compare_table.add_column("Delta", justify="right")

        for rating in ratings:
            llm = run_data.llm_scores.get(rating.criterion_name)
            if llm is not None:
                delta = rating.score - llm
                color = "green" if abs(delta) < 0.2 else "yellow" if abs(delta) < 0.4 else "red"
                compare_table.add_row(
                    rating.criterion_name,
                    f"{rating.score:.2f}",
                    f"{llm:.2f}",
                    f"[{color}]{delta:+.2f}[/{color}]",
                )
        console.print(compare_table)


@judge.command("correlation")
@click.option("--ratings-dir", default="ratings", help="Directory for ratings data")
@click.option("--output", "-o", default=None, help="Save report as JSON")
def judge_correlation(ratings_dir: str, output: str | None):
    """Compute correlation between LLM judge and human ratings."""
    from voiceenv.core.human_ratings import RatingStore
    from voiceenv.core.judge_correlation import (
        compute_correlation,
        format_correlation_report,
    )

    store = RatingStore(ratings_dir)
    all_ratings = store.load_all_ratings()

    if not all_ratings:
        console.print("[yellow]No human ratings found.[/yellow]")
        console.print(f"Start rating: [cyan]voiceenv judge rate --ratings-dir {ratings_dir}[/cyan]")
        return

    # Collect LLM scores from saved runs
    llm_scores: dict[str, dict[str, float]] = {}
    for run_id in store.list_runs():
        run_data = store.load_run_for_rating(run_id)
        if run_data.llm_scores:
            llm_scores[run_id] = run_data.llm_scores

    if not llm_scores:
        console.print("[yellow]No LLM scores found in saved runs.[/yellow]")
        return

    report = compute_correlation(all_ratings, llm_scores)

    # Display
    console.print(format_correlation_report(report))

    # Health summary
    console.print("")
    stats = store.get_rating_stats()
    console.print(Panel(
        f"[bold]Total ratings:[/bold] {stats['total_ratings']}\n"
        f"[bold]Unique raters:[/bold] {stats['unique_raters']}\n"
        f"[bold]Runs rated:[/bold] {stats['unique_runs_rated']}\n"
        f"[bold]Audio ratings:[/bold] {stats['audio_ratings_pct']:.0%}",
        title="Community Rating Stats",
        border_style="green",
    ))

    if report.flagged_criteria:
        console.print("\n[bold red]Action needed:[/bold red] These criteria have low "
                      "correlation with human judges:")
        for c in report.flagged_criteria:
            console.print(f"  [red]•[/red] {c} — consider adding better expert references")

    if output:
        Path(output).write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"\n[green]Report saved to:[/green] {output}")


@judge.command("stats")
@click.option("--ratings-dir", default="ratings", help="Directory for ratings data")
def judge_stats(ratings_dir: str):
    """Show statistics about collected human ratings."""
    from voiceenv.core.human_ratings import RatingStore

    store = RatingStore(ratings_dir)
    stats = store.get_rating_stats()

    if stats["total_ratings"] == 0:
        console.print("[yellow]No ratings collected yet.[/yellow]")
        console.print("Get started: [cyan]voiceenv judge save-run results.json[/cyan]")
        return

    table = Table(title="Human Rating Statistics", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total ratings", str(stats["total_ratings"]))
    table.add_row("Unique runs rated", str(stats["unique_runs_rated"]))
    table.add_row("Unique criteria", str(stats["unique_criteria_rated"]))
    table.add_row("Unique raters", str(stats["unique_raters"]))
    table.add_row("Audio rating %", f"{stats['audio_ratings_pct']:.0%}")
    table.add_row("Avg confidence", f"{stats['avg_confidence']:.2f}")

    console.print(table)
    console.print(f"\nCompute correlation: [cyan]voiceenv judge correlation --ratings-dir {ratings_dir}[/cyan]")


if __name__ == "__main__":
    cli()
