"""
VoiceEnv Evaluation on Modal — Run all speech RL environments against a model on GPU.

This script:
  1. Serves a model via vLLM on A100
  2. Runs all 5 built-in environments (agent = model under test, simulator = same model)
  3. Scores with verifiable rewards (deterministic, no external API needed)
  4. Reports per-environment and aggregate results

Usage:
  modal run eval_on_modal.py
"""

import json
import subprocess
import time
from pathlib import Path

import modal

MINUTES = 60

app = modal.App("voiceenv-eval")

hf_cache = modal.Volume.from_name("voiceenv-hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
VLLM_PORT = 8000

eval_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "vllm>=0.8",
        "transformers>=4.50",
        "huggingface-hub[hf_transfer]>=0.25",
        "openai>=1.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "jinja2>=3.0",
        "rich>=13.0",
        "click>=8.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')\""
    )
    .add_local_dir("voiceenv", remote_path="/app/voiceenv", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    .run_commands("cd /app && pip install -e . --no-deps")
)

ENVIRONMENT_YAMLS = [
    "voiceenv/environments/healthcare_triage.yaml",
    "voiceenv/environments/founder_sales.yaml",
    "voiceenv/environments/support_escalation.yaml",
    "voiceenv/environments/collections_call.yaml",
    "voiceenv/environments/appointment_scheduling.yaml",
    "voiceenv/environments/banking_fraud_investigation.yaml",
]


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=45 * MINUTES,
)
def run_evaluation(
    model: str = MODEL,
    runs_per_env: int = 3,
    skip_soft_scoring: bool = False,
) -> dict:
    """
    Full evaluation pipeline on a single A100.
    Serves the model, runs environments, scores, returns results.
    """
    import sys
    sys.path.insert(0, "/app")

    print(f"{'='*70}")
    print(f"  VOICEENV EVALUATION")
    print(f"  Model: {model}")
    print(f"  Environments: {len(ENVIRONMENT_YAMLS)}")
    print(f"  Runs per env: {runs_per_env}")
    print(f"{'='*70}\n")

    # Step 1: Start vLLM
    print("[1/4] Starting vLLM server...")
    vllm_log = Path("/tmp/vllm.log")
    vllm_cmd = (
        f"vllm serve {model} "
        f"--host 0.0.0.0 --port {VLLM_PORT} "
        f"--trust-remote-code --dtype bfloat16 --enforce-eager "
        f"--max-model-len 4096 "
        f"--gpu-memory-utilization 0.9 "
        f"--enable-auto-tool-choice --tool-call-parser hermes"
    )
    log_fh = open(vllm_log, "w")
    vllm_proc = subprocess.Popen(
        vllm_cmd, shell=True,
        stdout=log_fh, stderr=subprocess.STDOUT,
    )

    # Wait for vLLM to be ready (up to 5 min for first-time model download)
    import httpx
    ready = False
    for attempt in range(300):
        try:
            r = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"       vLLM ready after {attempt}s")
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)
        if attempt % 30 == 0 and attempt > 0:
            print(f"       Still waiting... ({attempt}s)")

    if not ready:
        log_fh.close()
        logs = vllm_log.read_text()[-3000:]
        vllm_proc.kill()
        return {"error": "vLLM failed to start within 300s", "logs": logs}

    # Step 2: Load environments
    print("\n[2/4] Loading environments...")
    from voiceenv.core.schema import VoiceEnvironment

    environments = []
    for yaml_path in ENVIRONMENT_YAMLS:
        path = Path("/app") / yaml_path
        if path.exists():
            env = VoiceEnvironment.from_yaml(path)
            environments.append(env)
            print(f"       ✓ {env.name} ({env.difficulty.value}, {len(env.rubric.all_criteria())} criteria)")
        else:
            print(f"       ✗ {yaml_path} not found")

    if not environments:
        vllm_proc.kill()
        return {"error": "No environments loaded"}

    # Step 3: Run evaluations
    print(f"\n[3/4] Running evaluations ({len(environments)} envs × {runs_per_env} runs)...")

    from openai import OpenAI
    from voiceenv.core.runner import EnvironmentRunner, OpenAIAgentBackend, RunResult

    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="not-needed")

    all_results: dict[str, list[dict]] = {}

    for env in environments:
        env_name = env.name
        all_results[env_name] = []
        print(f"\n  ┌─ {env_name} ({env.vertical.value}/{env.difficulty.value})")

        for run_idx in range(runs_per_env):
            try:
                agent = OpenAIAgentBackend(
                    model=model,
                    base_url=f"http://localhost:{VLLM_PORT}/v1",
                    api_key="not-needed",
                )
                runner = EnvironmentRunner(
                    env=env,
                    agent=agent,
                    agent_model=model,
                    simulator_model=model,
                    scorer_model=model,
                    openai_client=client,
                )
                runner.scorer.skip_soft_scoring = skip_soft_scoring

                result = runner.run()

                run_data = {
                    "run_idx": run_idx,
                    "reward": round(result.reward, 4),
                    "verifiable_reward": round(result.verifiable_reward, 4),
                    "soft_reward": round(result.soft_reward, 4),
                    "turn_count": result.turn_count,
                    "duration_seconds": round(result.duration_seconds, 2),
                    "scorecard": result.scorecard.to_dict(),
                    "transcript_preview": [
                        {"role": t["role"], "content": t["content"][:100]}
                        for t in result.transcript[:4]
                    ],
                }
                all_results[env_name].append(run_data)

                v = run_data["verifiable_reward"]
                s = run_data["soft_reward"]
                turns = run_data["turn_count"]
                dur = run_data["duration_seconds"]
                status = "✓" if v > 0.5 else "✗"
                print(f"  │  Run {run_idx+1}/{runs_per_env}: "
                      f"V={v:.2f} S={s:.2f} turns={turns} "
                      f"time={dur:.1f}s {status}")

            except Exception as e:
                print(f"  │  Run {run_idx+1}/{runs_per_env}: FAILED - {e}")
                all_results[env_name].append({
                    "run_idx": run_idx,
                    "error": str(e),
                    "reward": 0.0,
                    "verifiable_reward": 0.0,
                })

        # Summary for this env
        successful = [r for r in all_results[env_name] if "error" not in r]
        if successful:
            avg_v = sum(r["verifiable_reward"] for r in successful) / len(successful)
            avg_s = sum(r.get("soft_reward", 0) for r in successful) / len(successful)
            print(f"  └─ Avg: V={avg_v:.3f} S={avg_s:.3f} ({len(successful)}/{runs_per_env} succeeded)")
        else:
            print(f"  └─ All runs failed")

    # Step 4: Aggregate report
    print(f"\n[4/4] Computing aggregate report...")
    vllm_proc.terminate()

    report = build_report(all_results, model, runs_per_env)
    print_report(report)

    return report


def build_report(all_results: dict, model: str, runs_per_env: int) -> dict:
    """Build a structured evaluation report."""
    env_summaries = {}
    total_v_rewards = []
    total_s_rewards = []

    for env_name, runs in all_results.items():
        successful = [r for r in runs if "error" not in r]
        failed = [r for r in runs if "error" in r]

        if successful:
            v_rewards = [r["verifiable_reward"] for r in successful]
            s_rewards = [r.get("soft_reward", 0) for r in successful]
            total_v_rewards.extend(v_rewards)
            total_s_rewards.extend(s_rewards)

            # Per-criterion breakdown
            criteria_scores: dict[str, list[float]] = {}
            for r in successful:
                sc = r.get("scorecard", {})
                for c in sc.get("verifiable_criteria", []):
                    criteria_scores.setdefault(c["name"], []).append(c["score"])
                for c in sc.get("soft_criteria", []):
                    criteria_scores.setdefault(c["name"], []).append(c["score"])

            criteria_avgs = {
                name: round(sum(scores) / len(scores), 3)
                for name, scores in criteria_scores.items()
            }

            env_summaries[env_name] = {
                "runs_succeeded": len(successful),
                "runs_failed": len(failed),
                "avg_verifiable_reward": round(sum(v_rewards) / len(v_rewards), 4),
                "avg_soft_reward": round(sum(s_rewards) / len(s_rewards), 4),
                "avg_blended_reward": round(
                    sum(r["reward"] for r in successful) / len(successful), 4
                ),
                "avg_turns": round(
                    sum(r["turn_count"] for r in successful) / len(successful), 1
                ),
                "avg_duration_s": round(
                    sum(r["duration_seconds"] for r in successful) / len(successful), 1
                ),
                "criteria_pass_rates": criteria_avgs,
            }
        else:
            env_summaries[env_name] = {
                "runs_succeeded": 0,
                "runs_failed": len(failed),
                "errors": [r.get("error", "unknown") for r in failed[:3]],
            }

    overall = {}
    if total_v_rewards:
        overall = {
            "total_runs": sum(len(r) for r in all_results.values()),
            "successful_runs": len(total_v_rewards),
            "avg_verifiable_reward": round(sum(total_v_rewards) / len(total_v_rewards), 4),
            "avg_soft_reward": round(sum(total_s_rewards) / len(total_s_rewards), 4) if total_s_rewards else 0.0,
        }

    return {
        "model": model,
        "runs_per_env": runs_per_env,
        "environments_evaluated": len(all_results),
        "overall": overall,
        "per_environment": env_summaries,
        "raw_results": all_results,
    }


def print_report(report: dict):
    """Print a nice summary to stdout."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT")
    print(f"  Model: {report['model']}")
    print(f"  Environments: {report['environments_evaluated']}")
    print(f"{'='*70}")

    overall = report.get("overall", {})
    if overall:
        print(f"\n  OVERALL:")
        print(f"    Successful runs: {overall['successful_runs']}/{overall['total_runs']}")
        print(f"    Avg Verifiable Reward: {overall['avg_verifiable_reward']:.4f}")
        print(f"    Avg Soft Reward:       {overall['avg_soft_reward']:.4f}")

    print(f"\n  PER-ENVIRONMENT:")
    print(f"  {'Environment':<35} {'V-Reward':<10} {'S-Reward':<10} {'Turns':<8} {'Pass'}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    for env_name, summary in report.get("per_environment", {}).items():
        if summary.get("runs_succeeded", 0) > 0:
            v = summary["avg_verifiable_reward"]
            s = summary["avg_soft_reward"]
            t = summary["avg_turns"]
            status = "✓" if v > 0.5 else "△" if v > 0.3 else "✗"
            short_name = env_name[:33]
            print(f"  {short_name:<35} {v:<10.4f} {s:<10.4f} {t:<8.1f} {status}")
        else:
            short_name = env_name[:33]
            print(f"  {short_name:<35} {'FAILED':<10} {'—':<10} {'—':<8} ✗")

    # Criterion-level breakdown
    print(f"\n  CRITERION PASS RATES (across all environments):")
    all_criteria: dict[str, list[float]] = {}
    for summary in report.get("per_environment", {}).values():
        for name, score in summary.get("criteria_pass_rates", {}).items():
            all_criteria.setdefault(name, []).append(score)

    if all_criteria:
        sorted_criteria = sorted(all_criteria.items(), key=lambda x: sum(x[1])/len(x[1]))
        for name, scores in sorted_criteria:
            avg = sum(scores) / len(scores)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            status = "✓" if avg >= 0.7 else "△" if avg >= 0.4 else "✗"
            print(f"    {name:<30} {bar} {avg:.2f} {status}")

    print(f"\n{'='*70}\n")


@app.local_entrypoint()
def main():
    print("Launching VoiceEnv evaluation on Modal A100...")
    print(f"Model: {MODEL}")
    print(f"This will take ~10-15 minutes (model download + 5 envs × 3 runs)\n")

    result = run_evaluation.remote(
        model=MODEL,
        runs_per_env=3,
        skip_soft_scoring=False,
    )

    # Save results locally
    output_path = Path("eval_results.json")
    with open(output_path, "w") as f:
        # Remove raw_results for cleaner output file (they can be large)
        clean = {k: v for k, v in result.items() if k != "raw_results"}
        json.dump(clean, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    if "error" in result:
        print(f"\n⚠ Error: {result['error']}")
        if "logs" in result:
            print(f"Logs:\n{result['logs'][-1000:]}")
