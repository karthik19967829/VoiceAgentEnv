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
import os
import subprocess
import sys
import time
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


def _load_dotenv():
    """Light .env loader so commands work without manually `source`-ing."""
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@cli.command()
@click.argument("wav_path", type=click.Path(exists=True))
@click.option("--env-dir", default=None,
              help="Reuse an already-ingested env dir (skips ingest). Default: ingest fresh.")
@click.option("--model", "-m", default="gpt-audio-mini",
              help="Speech LLM under test (gpt-audio, gpt-audio-mini, etc.)")
@click.option("--max-turns", default=8, help="Cap on caller turns to evaluate")
@click.option("--parallelism", default=4, help="Concurrent API calls")
@click.option("--grounded/--no-grounded", default=False,
              help="Also run multimodal grounded judge (Gemini, ~+5s, ~$0.01)")
@click.option("--judge-model", default="gemini-2.5-flash", help="Gemini model for grounded judge")
@click.option("--output", "-o", default=None, help="Save JSON results here")
def demo(wav_path: str, env_dir: str | None, model: str, max_turns: int,
         parallelism: int, grounded: bool, judge_model: str, output: str | None):
    """End-to-end speech-LLM demo from a single WAV.

    Pipeline:
      1. Auto-ingest the WAV into a VoiceEnv (or reuse --env-dir)
      2. Slice the caller channel into per-turn audio clips
      3. Run a STATELESS turn-level evaluation against the speech LLM
         (each caller turn is an independent test case, parallelized)
      4. Score AI responses with the env's verifiable rubric
      5. Print a side-by-side scorecard (human vs AI per turn)
    """
    _load_dotenv()
    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.demo import slice_caller_turns, run_stateless_eval, run_grounded_eval, score_eval
    from voiceenv.ingest.from_call import _load_hvb_transcript, _whisper_transcribe, _merge_consecutive

    wav = Path(wav_path)

    # ── 1. ingest (or load existing) ──
    if env_dir:
        env = VoiceEnvironment.from_yaml(Path(env_dir) / "env.yaml")
        console.print(f"[cyan]Reusing existing env:[/cyan] {env.name}")
    else:
        from voiceenv.ingest import ingest_call
        ingest_out = Path(f"environments/auto_{wav.stem[:8]}")
        console.print(Panel.fit(
            f"[bold cyan]VoiceEnv demo[/bold cyan]   "
            f"[white]{wav.name}[/white]   →   model: [white]{model}[/white]",
            border_style="cyan",
        ))
        console.print("[bold]Stage 1: autonomous ingest[/bold]")
        result = ingest_call(
            wav_path=wav, output_dir=ingest_out,
            extraction_model="gpt-4o-mini",
            on_log=lambda m: console.print(f"  [dim]{m}[/dim]"),
        )
        env = result.env
        env_dir = str(ingest_out)
        console.print(f"  [green]✓ env extracted:[/green] {env.name} "
                      f"({len(env.tools)} tools, {len(env.rubric.all_criteria())} rubric criteria)")

    # ── 2. rebuild turn list with timing (need it for slicing) ──
    console.print("\n[bold]Stage 2: slice caller channel into per-turn clips[/bold]")
    transcript_json = None
    for up in (1, 2, 3):
        try:
            cand = wav.parents[up] / "transcript" / (wav.stem + ".json")
            if cand.exists():
                transcript_json = cand
                break
        except IndexError:
            break

    if transcript_json:
        all_turns = _load_hvb_transcript(transcript_json)
    else:
        all_turns = _whisper_transcribe(wav)
    all_turns = _merge_consecutive(all_turns)

    clips_dir = Path(env_dir) / "caller_clips"
    clips = slice_caller_turns(wav, all_turns, clips_dir, max_turns=max_turns)
    console.print(f"  [green]✓ {len(clips)} caller turns sliced[/green] → {clips_dir}")

    # ── 3. stateless eval ──
    console.print(f"\n[bold]Stage 3: stateless eval against {model}[/bold]   "
                  f"(parallel={parallelism})")
    t0 = time.time()
    ai_audio_dir = Path(env_dir) / "ai_clips"
    results, cost = run_stateless_eval(
        env, all_turns, clips, model=model, parallelism=parallelism,
        on_log=lambda m: console.print(f"  [dim]{m}[/dim]"),
        capture_audio=True,
        audio_out_dir=ai_audio_dir,
    )
    wall = time.time() - t0

    # ── 4a. score with verifiable rubric ──
    console.print("\n[bold]Stage 4a: score with verifiable rubric (deterministic)[/bold]")
    scorecard = score_eval(env, results)

    # ── 4b. grounded multimodal judge (optional) ──
    grounded_result = None
    if grounded:
        console.print(f"\n[bold]Stage 4b: grounded multimodal judge ({judge_model})[/bold]")
        if not env.expert_references or not Path(env_dir, env.expert_references[0].audio_path).exists():
            console.print("  [yellow]No expert reference audio found — skipping grounded judge.[/yellow]")
        else:
            expert_wav = str(Path(env_dir) / env.expert_references[0].audio_path)
            console.print(f"  [dim]anchor: {expert_wav}[/dim]")
            try:
                grounded_result = run_grounded_eval(env, results, expert_wav, model=judge_model, all_turns=all_turns)
                console.print(f"  [green]✓ avg score: {grounded_result['average_score_1_5']}/5[/green]")
            except Exception as e:
                console.print(f"  [red]✗ grounded judge failed: {e}[/red]")

    # ── 5. report ──
    summary = Table(title="Demo summary", show_header=True, header_style="bold green")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value")
    summary.add_row("Source WAV", wav.name)
    summary.add_row("Env (auto-ingested)", env.name)
    summary.add_row("Speech LLM", model)
    summary.add_row("Turns evaluated", str(len(results)))
    summary.add_row("Wall time", f"{wall:.1f}s ({len(results)} turns in parallel)")
    summary.add_row("LLM cost", f"${cost:.4f}")
    summary.add_row("Verifiable score", f"{scorecard.verifiable_score:.0%}")
    summary.add_row("Total reward", f"{scorecard.total_score:.0%}")
    if grounded_result:
        summary.add_row("Grounded judge avg", f"{grounded_result['average_score_1_5']}/5")
    console.print(summary)

    if grounded_result:
        gtable = Table(title=f"Grounded judge ({grounded_result['model']}) — anchored on real human call",
                       show_header=True, header_style="bold magenta")
        gtable.add_column("Dimension", style="cyan")
        gtable.add_column("Score", justify="center")
        gtable.add_column("Reasoning (vs human expert)", max_width=80, style="dim")
        for dim, d in grounded_result["dimensions"].items():
            score = d.get("score", 0)
            color = "green" if score >= 4 else ("yellow" if score >= 3 else "red")
            gtable.add_row(dim, f"[{color}]{score}/5[/{color}]", d.get("reasoning", ""))
        console.print(gtable)

    side = Table(title="Per-turn comparison: real human caller / human agent / AI agent",
                 show_header=True, show_lines=True, header_style="bold magenta")
    side.add_column("#", style="dim", width=3)
    side.add_column("Caller (real audio in)", max_width=34, style="white")
    side.add_column("Human agent baseline", max_width=34, style="green")
    side.add_column(f"{model}", max_width=34, style="yellow")
    for r in results:
        ai = r.ai_response or ("✗ " + (r.error or "no response"))
        if r.ai_tool_calls:
            ai = ai + f"\n[bold cyan]→ {r.ai_tool_calls[0]['tool']}({json.dumps(r.ai_tool_calls[0]['args'])})[/bold cyan]"
        side.add_row(str(r.turn_idx), r.caller_text or "(audio)",
                     r.human_response or "—", ai)
    console.print(side)

    crit_table = Table(title="Verifiable rubric breakdown", show_header=True, header_style="bold blue")
    crit_table.add_column("Criterion", style="cyan")
    crit_table.add_column("Category")
    crit_table.add_column("Pass", justify="center")
    crit_table.add_column("Reasoning", max_width=60, style="dim")
    for cr in scorecard.criteria_results:
        crit_table.add_row(cr.name, cr.category,
                           "[green]✓[/green]" if cr.score >= 0.5 else "[red]✗[/red]",
                           cr.reasoning or "")
    console.print(crit_table)

    if output:
        out_data = {
            "env_name": env.name,
            "model": model,
            "wall_time_seconds": wall,
            "cost_usd": cost,
            "scorecard": {
                "total_score": scorecard.total_score,
                "verifiable_score": scorecard.verifiable_score,
                "category_scores": scorecard.category_scores,
            },
            "grounded_judge": grounded_result,
            "turns": [
                {
                    "turn_idx": r.turn_idx,
                    "caller_text": r.caller_text,
                    "caller_audio_path": r.caller_audio_path,
                    "human_response": r.human_response,
                    "ai_response": r.ai_response,
                    "ai_tool_calls": r.ai_tool_calls,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in results
            ],
        }
        Path(output).write_text(json.dumps(out_data, indent=2))
        console.print(f"\n[green]Saved results to:[/green] {output}")


@cli.command()
@click.argument("wav_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None,
              help="Output directory (default: environments/<auto_name>)")
@click.option("--transcript", "-t", default=None,
              help="Optional sibling transcript JSON (HVB format). Auto-detected if absent.")
@click.option("--model", "-m", default="gpt-4o-mini",
              help="LLM used to extract task / persona / tools / rubric")
def ingest(wav_path: str, output: str | None, transcript: str | None, model: str):
    """Autonomously turn a real call recording (WAV) into a publishable VoiceEnv.

    Pipeline: transcribe → segment → LLM extraction → schema build → emit.
    Output is a self-contained directory with env.yaml + expert_reference/ + rollouts/
    that can be passed straight to `voiceenv export` and `voiceenv publish`.
    """
    from voiceenv.ingest import ingest_call

    # Auto-load .env if present
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    if output is None:
        output = f"environments/auto_{Path(wav_path).stem[:8]}"

    console.print(Panel.fit(
        f"[bold cyan]VoiceEnv autonomous ingest[/bold cyan]\n"
        f"  source: [white]{wav_path}[/white]\n"
        f"  output: [white]{output}[/white]\n"
        f"  model:  [white]{model}[/white]",
        border_style="cyan",
    ))

    result = ingest_call(
        wav_path=wav_path,
        output_dir=output,
        transcript_path=transcript,
        extraction_model=model,
        on_log=lambda m: console.print(f"[dim]{m}[/dim]"),
    )

    total_ms = sum(result.timings_ms.values())

    table = Table(title="Ingest result", show_header=True, header_style="bold green")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Environment name", result.env.name)
    table.add_row("Vertical", result.env.vertical.value)
    table.add_row("Difficulty", result.env.difficulty.value)
    table.add_row("Tools", str(len(result.env.tools)))
    table.add_row("Rubric criteria", str(len(result.env.rubric.all_criteria())))
    table.add_row("Source turns", str(result.n_turns))
    table.add_row("Source duration", f"{result.duration_seconds:.1f}s")
    table.add_row("Total wall time", f"{total_ms / 1000:.1f}s")
    table.add_row("LLM cost", f"${result.cost_usd:.4f}")
    table.add_row("Output dir", str(result.output_dir))
    console.print(table)

    console.print(
        f"\n[bold green]✓ Ready to publish.[/bold green] Next:\n"
        f"  [cyan]voiceenv run {result.output_dir}/env.yaml -m gpt-4o-mini -n 5[/cyan]\n"
        f"  [cyan]voiceenv export {result.output_dir}/env.yaml --target both[/cyan]\n"
        f"  [cyan]voiceenv publish {result.output_dir}/env.yaml --target both[/cyan]\n"
    )


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


@cli.command("run-voice")
@click.argument("env_path")
@click.option("--agent-model", "-m", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
              help="Speech LLM for the agent (the model being evaluated)")
@click.option("--agent-base-url", default=None,
              help="API endpoint for agent model (e.g. http://localhost:8000/v1)")
@click.option("--agent-api-key", default=None, help="API key for agent endpoint")
@click.option("--simulator-model", default="gpt-4o",
              help="Speech LLM for the caller/simulator")
@click.option("--simulator-api-key", default=None, help="API key for simulator model")
@click.option("--mode", type=click.Choice(["cascaded", "realtime"]), default="cascaded",
              help="Pipeline mode: cascaded (STT→LLM→TTS) or realtime (speech-to-speech)")
@click.option("--audio-dir", default="run_audio", help="Directory to save per-turn audio")
@click.option("--output", "-o", default=None, help="Save results JSON")
@click.option("--save-for-rating", is_flag=True, help="Also save to ratings store for community review")
@click.option("--ratings-dir", default="ratings", help="Ratings directory (with --save-for-rating)")
def run_voice(env_path, agent_model, agent_base_url, agent_api_key,
              simulator_model, simulator_api_key, mode, audio_dir, output,
              save_for_rating, ratings_dir):
    """Run an environment with speech LLMs (requires: pip install voiceenv[voice]).

    \b
    Both sides of the conversation are real speech LLMs with full audio capture.
    Pipecat handles VAD, interruptions, and turn management.

    \b
    Examples:
      # Agent = locally-served Qwen3-Omni, Simulator = GPT-4o
      voiceenv run-voice healthcare_triage \\
        --agent-base-url http://localhost:8000/v1 \\
        --simulator-model gpt-4o

      # Save audio for community rating
      voiceenv run-voice healthcare_triage --save-for-rating
    """
    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.environments import load_environment

    path = Path(env_path)
    if path.exists():
        env = VoiceEnvironment.from_yaml(path)
    else:
        try:
            env = load_environment(env_path)
        except FileNotFoundError:
            console.print(f"[red]Environment not found:[/red] {env_path}")
            sys.exit(1)

    console.print(Panel(
        f"[bold]{env.name}[/bold]\n{env.description[:120]}",
        title="Voice Environment (Speech Mode)",
        border_style="green",
    ))
    console.print(f"Agent:     [cyan]{agent_model}[/cyan]")
    console.print(f"Simulator: [cyan]{simulator_model}[/cyan]")
    console.print(f"Mode:      [cyan]{mode}[/cyan]")
    console.print(f"Audio dir: [cyan]{audio_dir}[/cyan]\n")

    try:
        from voiceenv.core.voice_runner import VoiceEnvironmentRunner
    except ImportError:
        console.print("[red]Voice mode requires pipecat-ai.[/red]")
        console.print("Install with: [cyan]pip install voiceenv[voice][/cyan]")
        sys.exit(1)

    runner = VoiceEnvironmentRunner(
        env=env,
        agent_model=agent_model,
        agent_base_url=agent_base_url,
        agent_api_key=agent_api_key,
        simulator_model=simulator_model,
        simulator_api_key=simulator_api_key,
        mode=mode,
        audio_dir=audio_dir,
    )

    result = runner.run_sync()

    # Display transcript
    table = Table(title="Voice Conversation", show_header=True)
    table.add_column("Speaker", style="bold", width=8)
    table.add_column("Content", max_width=80)
    table.add_column("Audio", width=12)
    table.add_column("Info", width=15)

    for turn in result.transcript:
        role = turn.role.upper()
        style = "cyan" if turn.role == "agent" else "yellow"
        audio_status = "[green]recorded[/green]" if turn.audio_path else "[dim]none[/dim]"
        info_parts = []
        if turn.duration_ms:
            info_parts.append(f"{turn.duration_ms / 1000:.1f}s")
        if turn.interrupted:
            info_parts.append("[yellow]INTERRUPTED[/yellow]")
        table.add_row(role, turn.content[:200], audio_status, " ".join(info_parts))

    console.print(table)
    console.print(f"\nTurns: {result.turn_count} | "
                  f"Interruptions: {result.interruption_count} | "
                  f"Duration: {result.duration_seconds:.1f}s")
    console.print(f"Audio saved to: [cyan]{result.audio_dir}[/cyan]")

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"[green]Results saved to:[/green] {output}")

    if save_for_rating:
        from voiceenv.core.human_ratings import RatingStore, RunForRating, generate_run_id

        transcript_dicts = [t.to_dict() for t in result.transcript]
        run_id = generate_run_id(result.environment_name, transcript_dicts)

        criteria = []
        for sc in env.rubric.all_criteria():
            criteria.append({"name": sc.name, "description": sc.description})

        run_for_rating = RunForRating(
            run_id=run_id,
            environment_name=result.environment_name,
            transcript=transcript_dicts,
            criteria_to_rate=criteria,
            audio_dir=result.audio_dir,
        )

        store = RatingStore(ratings_dir)
        store.save_run_for_rating(run_for_rating)
        console.print(f"\n[green]Saved for community rating:[/green] {run_id}")
        console.print(f"Launch rating UI: [cyan]voiceenv judge serve --ratings-dir {ratings_dir}[/cyan]")


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


def _load_dotenv_for_publish() -> None:
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@cli.command()
@click.argument("env_path")
@click.option("--target", "-t", default="openenv",
              type=click.Choice(["openenv", "prime", "both"]),
              help="Target hub (default: openenv → HuggingFace Spaces)")
@click.option("--repo-id", default=None,
              help="HF Space repo id, e.g. yourname/voiceenv-my-env "
                   "(default: {you}/voiceenv-{env_name})")
@click.option("--namespace", "-n", default=None,
              help="HF namespace/org for default repo id (default: your username)")
@click.option("--no-register", is_flag=True,
              help="Skip adding this Space to the VoiceEnv hub collection")
@click.option("--team", default=None, help="Team name (for Prime Intellect)")
def publish(
    env_path: str,
    target: str,
    repo_id: str | None,
    namespace: str | None,
    no_register: bool,
    team: str | None,
):
    """Publish an environment to the VoiceEnv hub on HuggingFace (one command, no PR).

    Creates an OpenEnv-compatible Docker Space and registers it in the public
    VoiceEnv Environments collection.

    Example:
      voiceenv publish environments/auto_00d676d7/env.yaml
    """
    import tempfile

    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.environments import load_environment

    _load_dotenv_for_publish()

    path = Path(env_path)
    if path.exists():
        env = VoiceEnvironment.from_yaml(path)
    else:
        env = load_environment(env_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        if target in ("openenv", "both"):
            from voiceenv.exporters.openenv_exporter import export_openenv
            from voiceenv.exporters.hf_hub import push_openenv_space

            pkg_path = export_openenv(env, Path(tmpdir) / "openenv")
            console.print("[green]OpenEnv package created[/green]")

            try:
                result = push_openenv_space(
                    pkg_path,
                    env,
                    repo_id=repo_id,
                    register=not no_register,
                    namespace=namespace,
                )
                console.print(f"[bold green]✓ Live Space:[/bold green] {result['space_url']}")
                console.print(f"[dim]  App URL: {result['app_url']}[/dim]")
                console.print(f"[dim]  Try: {result['app_url']}/docs[/dim]")
                if result.get("hub_collection_url"):
                    console.print(
                        f"[bold green]✓ Listed in VoiceEnv hub:[/bold green] "
                        f"{result['hub_collection_url']}"
                    )
            except Exception as e:
                console.print(f"[red]Publish failed:[/red] {e}")
                console.print(f"[dim]Package exported to {pkg_path} for manual push.[/dim]")
                raise SystemExit(1) from e

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
def eval():
    """Evaluation commands — measure and compare model performance."""
    pass


@eval.command("run")
@click.option("--model", "-m", default="gpt-4o-mini", help="Model to evaluate")
@click.option("--env-dir", default=None, help="Environment directory (default: built-in)")
@click.option("--runs", "-n", default=5, help="Runs per environment")
@click.option("--simulator-model", default="gpt-4o-mini", help="Simulator model")
@click.option("--base-url", default=None, help="Custom API endpoint for model")
@click.option("--api-key", default=None, help="API key for custom endpoint")
@click.option("--output", "-o", required=True, help="Save results JSON to this path")
@click.option("--verifiable-only", is_flag=True, help="Skip soft scoring (faster)")
def eval_run(model, env_dir, runs, simulator_model, base_url, api_key, output, verifiable_only):
    """Evaluate a model across all VoiceEnv environments."""
    from voiceenv.eval.evaluator import evaluate

    results = evaluate(
        model=model,
        env_dir=env_dir,
        runs_per_env=runs,
        simulator_model=simulator_model,
        base_url=base_url,
        api_key=api_key,
        skip_soft_scoring=verifiable_only,
    )
    results.save(output)
    console.print(f"\n[green]Results saved to:[/green] {output}")


@eval.command("compare")
@click.argument("baseline_path")
@click.argument("trained_path")
@click.option("--output", "-o", default=None, help="Save report JSON")
def eval_compare(baseline_path, trained_path, output):
    """Compare baseline vs trained model evaluation results."""
    from voiceenv.eval.evaluator import EvalResults
    from voiceenv.eval.comparison import compare, print_comparison

    baseline = EvalResults.load(baseline_path)
    trained = EvalResults.load(trained_path)

    report = compare(baseline, trained)
    print_comparison(report)

    if output:
        report.save(output)
        console.print(f"\n[green]Report saved to:[/green] {output}")


@eval.command("experiment")
@click.option("--eval-model", default="gpt-4o-mini", help="Model for eval (API-based)")
@click.option("--runs", "-n", default=5, help="Eval runs per environment")
@click.option("--rollout-runs", default=20, help="Rollout runs per environment")
@click.option("--output-dir", "-o", default="experiment_results", help="Output directory")
def eval_experiment(eval_model, runs, rollout_runs, output_dir):
    """Run a local baseline eval + rollout generation experiment."""
    from voiceenv.eval.evaluator import evaluate
    from voiceenv.training.generate_rollouts import generate_rollouts

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Baseline eval
    console.print(Panel("[bold]Step 1/3: Baseline Evaluation[/bold]", border_style="cyan"))
    baseline = evaluate(model=eval_model, runs_per_env=runs)
    baseline.save(out / "baseline_eval.json")
    console.print(f"[green]Baseline saved:[/green] {out / 'baseline_eval.json'}")

    # Step 2: Generate rollouts
    console.print(Panel("[bold]Step 2/3: Generating Rollouts[/bold]", border_style="cyan"))
    generate_rollouts(
        env_dir="voiceenv/environments",
        model=eval_model,
        runs_per_env=rollout_runs,
        output_path=str(out / "rollouts.jsonl"),
    )
    console.print(f"[green]Rollouts saved:[/green] {out / 'rollouts.jsonl'}")

    # Step 3: Instructions for training
    console.print(Panel(
        f"[bold]Step 3/3: Post-train with one command[/bold]\n\n"
        f"  [cyan]voiceenv train run -m Qwen/Qwen3-Omni-30B-A3B-Instruct "
        f"-r {out / 'rollouts.jsonl'}[/cyan]\n\n"
        f"After training, compare:\n"
        f"  [cyan]voiceenv eval compare {out / 'baseline_eval.json'} posttrain_eval.json[/cyan]",
        border_style="green",
    ))


@cli.group()
def train():
    """Training commands — generate rollouts and post-train speech LLMs."""
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


@train.command("run")
@click.option("--model", "-m", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model to fine-tune")
@click.option("--rollouts", "-r", required=True, help="Path to rollouts JSONL")
@click.option("--output", "-o", default="voiceenv_trained", help="Output directory")
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--lr", default=2e-5, help="Learning rate")
@click.option("--epochs", default=2, help="Training epochs")
@click.option("--batch-size", default=2, help="Batch size per device")
def train_run(model, rollouts, output, lora_rank, lr, epochs, batch_size):
    """Post-train a model via ms-swift GRPO. Requires: pip install ms-swift

    \b
    Examples:
      voiceenv train run -m Qwen/Qwen3-Omni-30B-A3B-Instruct -r rollouts.jsonl
      voiceenv train run -m Qwen/Qwen2.5-3B-Instruct -r rollouts.jsonl --lora-rank 8
    """
    from voiceenv.training.launch import launch_training
    launch_training(
        model=model,
        rollouts_path=rollouts,
        output_dir=output,
        lora_rank=lora_rank,
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
    )


@cli.group()
def cloud():
    """Cloud GPU commands via Modal — serve, run, and train on remote GPUs."""
    pass


@cloud.command("serve")
@click.option("--model", "-m", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
              help="Speech LLM to serve")
@click.option("--gpu", default="A100", help="GPU type (A100, H100, A10G, L4)")
def cloud_serve(model, gpu):
    """Deploy a speech LLM on Modal GPU via vLLM.

    \b
    Examples:
      voiceenv cloud serve
      voiceenv cloud serve -m Qwen/Qwen2.5-7B-Instruct --gpu L4
    """
    console.print(Panel(
        f"[bold]Deploying speech LLM on Modal[/bold]\n\n"
        f"Model: [cyan]{model}[/cyan]\n"
        f"GPU:   [cyan]{gpu}[/cyan]\n\n"
        f"This will deploy a vLLM server with an OpenAI-compatible API.\n"
        f"The URL will be printed once the server is ready.",
        title="Modal Deploy",
        border_style="green",
    ))
    import subprocess as sp
    sp.run(["modal", "deploy", "voiceenv.cloud.modal_app"], check=True)


@cloud.command("run")
@click.argument("env_path")
@click.option("--agent-model", "-m", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
              help="Agent speech LLM")
@click.option("--simulator-model", default="gpt-4o", help="Simulator speech LLM")
@click.option("--simulator-api-key", default=None, help="API key for simulator model")
@click.option("--save-for-rating", is_flag=True, help="Save to ratings store for community review")
@click.option("--ratings-dir", default="ratings", help="Ratings directory")
def cloud_run(env_path, agent_model, simulator_model, simulator_api_key,
              save_for_rating, ratings_dir):
    """Run a voice environment on Modal GPU (two speech LLMs talking).

    \b
    Examples:
      voiceenv cloud run healthcare_triage
      voiceenv cloud run my_env.yaml --save-for-rating
    """
    from voiceenv.core.schema import VoiceEnvironment
    from voiceenv.environments import load_environment

    path = Path(env_path)
    if path.exists():
        env = VoiceEnvironment.from_yaml(path)
    else:
        try:
            env = load_environment(env_path)
        except FileNotFoundError:
            console.print(f"[red]Environment not found:[/red] {env_path}")
            sys.exit(1)

    env_yaml = env.to_yaml()

    console.print(Panel(
        f"[bold]{env.name}[/bold]\n"
        f"Agent: [cyan]{agent_model}[/cyan]\n"
        f"Simulator: [cyan]{simulator_model}[/cyan]\n"
        f"Running on Modal GPU...",
        title="Cloud Voice Run",
        border_style="green",
    ))

    import modal
    app_ref = modal.App.lookup("voiceenv")
    run_fn = app_ref.run_voice_env

    result = run_fn.remote(
        env_yaml=env_yaml,
        agent_model=agent_model,
        simulator_model=simulator_model,
        simulator_api_key=simulator_api_key or "",
    )

    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    console.print(f"\n[green]Run completed![/green]")
    console.print(f"Turns: {result.get('turn_count', 0)} | "
                  f"Duration: {result.get('duration_seconds', 0):.1f}s | "
                  f"Interruptions: {result.get('interruption_count', 0)}")

    output_path = Path(f"cloud_run_{env.name}.json")
    output_path.write_text(json.dumps(result, indent=2))
    console.print(f"Results saved to: [cyan]{output_path}[/cyan]")

    if save_for_rating:
        from voiceenv.core.human_ratings import RatingStore, RunForRating, generate_run_id

        transcript = result.get("transcript", [])
        run_id = generate_run_id(env.name, transcript)
        criteria = [{"name": sc.name, "description": sc.description} for sc in env.rubric.all_criteria()]

        run_for_rating = RunForRating(
            run_id=run_id,
            environment_name=env.name,
            transcript=transcript,
            criteria_to_rate=criteria,
            audio_dir=result.get("audio_dir"),
        )
        store = RatingStore(ratings_dir)
        store.save_run_for_rating(run_for_rating)
        console.print(f"[green]Saved for community rating:[/green] {run_id}")


@cloud.command("train")
@click.option("--model", "-m", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
              help="Model to fine-tune")
@click.option("--rollouts", "-r", required=True, help="Path to rollouts JSONL")
@click.option("--gpu", default="A100:2", help="GPU config (e.g. A100:2, H100:4)")
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--lr", default=2e-5, help="Learning rate")
@click.option("--epochs", default=2, help="Training epochs")
@click.option("--batch-size", default=2, help="Batch size per device")
def cloud_train(model, rollouts, gpu, lora_rank, lr, epochs, batch_size):
    """Post-train a speech LLM with ms-swift GRPO on Modal GPUs.

    \b
    Examples:
      voiceenv cloud train -r rollouts.jsonl
      voiceenv cloud train -m Qwen/Qwen2.5-7B-Instruct -r rollouts.jsonl --gpu H100:4
    """
    rollouts_path = Path(rollouts)
    if not rollouts_path.exists():
        console.print(f"[red]Rollouts file not found:[/red] {rollouts}")
        sys.exit(1)

    rollouts_jsonl = rollouts_path.read_text()
    n_examples = sum(1 for line in rollouts_jsonl.strip().split("\n") if line.strip())

    console.print(Panel(
        f"[bold]ms-swift GRPO Training on Modal[/bold]\n\n"
        f"Model:    [cyan]{model}[/cyan]\n"
        f"GPU:      [cyan]{gpu}[/cyan]\n"
        f"Examples: [cyan]{n_examples}[/cyan]\n"
        f"LoRA:     [cyan]rank={lora_rank}[/cyan]\n"
        f"LR:       [cyan]{lr}[/cyan]\n"
        f"Epochs:   [cyan]{epochs}[/cyan]",
        title="Cloud Training",
        border_style="green",
    ))

    import modal
    app_ref = modal.App.lookup("voiceenv")
    train_fn = app_ref.train_grpo

    result = train_fn.remote(
        model=model,
        rollouts_jsonl=rollouts_jsonl,
        lora_rank=lora_rank,
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
    )

    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    exit_code = result.get("exit_code", -1)
    if exit_code == 0:
        console.print(f"\n[green]Training completed successfully![/green]")
        console.print(f"Output: [cyan]{result.get('output_dir')}[/cyan]")
    else:
        console.print(f"\n[red]Training failed (exit code {exit_code})[/red]")
        if result.get("stderr_tail"):
            console.print(f"[dim]{result['stderr_tail'][-500:]}[/dim]")


@cloud.command("rollouts")
@click.argument("env_dir", default="voiceenv/environments")
@click.option("--agent-model", "-m", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
              help="Agent speech LLM")
@click.option("--simulator-model", default="gpt-4o", help="Simulator speech LLM")
@click.option("--simulator-api-key", default=None, help="API key for simulator")
@click.option("--runs-per-env", "-n", default=10, help="Runs per environment")
@click.option("--output", "-o", default="rollouts.jsonl", help="Output file")
def cloud_rollouts(env_dir, agent_model, simulator_model, simulator_api_key,
                   runs_per_env, output):
    """Generate training rollouts on Modal GPU.

    \b
    Examples:
      voiceenv cloud rollouts voiceenv/environments/ --runs-per-env 20
    """
    from voiceenv.core.schema import VoiceEnvironment

    env_path = Path(env_dir)
    yaml_files = sorted(env_path.glob("*.yaml"))
    if not yaml_files:
        console.print(f"[red]No .yaml files found in {env_dir}[/red]")
        sys.exit(1)

    env_yamls = []
    for yf in yaml_files:
        env = VoiceEnvironment.from_yaml(yf)
        env_yamls.append(env.to_yaml())

    console.print(Panel(
        f"[bold]Generating rollouts on Modal GPU[/bold]\n\n"
        f"Environments: [cyan]{len(env_yamls)}[/cyan]\n"
        f"Runs/env:     [cyan]{runs_per_env}[/cyan]\n"
        f"Agent:        [cyan]{agent_model}[/cyan]\n"
        f"Simulator:    [cyan]{simulator_model}[/cyan]",
        title="Cloud Rollouts",
        border_style="green",
    ))

    import modal
    app_ref = modal.App.lookup("voiceenv")
    rollouts_fn = app_ref.generate_rollouts

    rollouts_jsonl = rollouts_fn.remote(
        env_yamls=env_yamls,
        agent_model=agent_model,
        simulator_model=simulator_model,
        simulator_api_key=simulator_api_key or "",
        runs_per_env=runs_per_env,
    )

    Path(output).write_text(rollouts_jsonl)
    n_lines = sum(1 for line in rollouts_jsonl.strip().split("\n") if line.strip())
    console.print(f"\n[green]Generated {n_lines} rollouts[/green] → {output}")


@cli.command("ui")
@click.option("--port", "-p", default=8911, type=int, help="Port to bind to")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def ui_cmd(port: int, host: str):
    """Launch the live demo UI in a browser (single-page, streams pipeline)."""
    _load_dotenv()
    from voiceenv.ui.demo_app import run_demo_ui
    console.print(Panel.fit(
        f"[bold cyan]VoiceEnv demo UI[/bold cyan]\n"
        f"  open: [white]http://{host}:{port}/[/white]\n"
        f"  pick a sample, click Run, watch the pipeline stream live",
        border_style="cyan",
    ))
    run_demo_ui(host=host, port=port)


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

    has_audio = any(t.get("audio_path") for t in run_data.transcript)
    if run_data.audio_dir:
        console.print(f"\n[dim]Audio directory: {run_data.audio_dir}[/dim]")
    elif has_audio:
        console.print(f"\n[dim]Per-turn audio available[/dim]")

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
        audio = click.confirm("  Did you listen to audio?", default=False) if has_audio else False

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


@judge.command("serve")
@click.option("--port", "-p", default=8910, help="Port to serve on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--ratings-dir", default="ratings", help="Directory for ratings data")
@click.option("--demo", is_flag=True, help="Seed demo conversations for testing")
def judge_serve(port: int, host: str, ratings_dir: str, demo: bool):
    """Launch a web UI for community rating (grandma-friendly)."""
    if demo:
        from voiceenv.ui.demo_data import seed_demo_data
        n = seed_demo_data(ratings_dir)
        console.print(f"[green]Seeded {n} demo conversations.[/green]")

    console.print(Panel(
        f"[bold]Rating UI is starting![/bold]\n\n"
        f"Open your browser to: [cyan underline]http://localhost:{port}[/cyan underline]\n\n"
        f"Share this link with anyone — no login needed.\n"
        f"Press Ctrl+C to stop.",
        title="VoiceEnv Community Rating",
        border_style="green",
    ))

    from voiceenv.ui.app import run_server
    run_server(host=host, port=port, ratings_dir=ratings_dir)


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
