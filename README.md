# VoiceEnv

**Community-driven voice agent environments for speech LLM training and evaluation.**

VoiceEnv is an open platform where the community creates voice agent environments — realistic conversational scenarios with verifiable rewards — that directly drive the post-training of speech LLMs like Qwen3-Omni. Think of it as the voice-native equivalent of code execution environments for RL, but for spoken conversations.

## Why VoiceEnv?

Code LLMs have unit tests. Math LLMs have verifiable proofs. Voice LLMs have... vibes?

The gap: there is no consolidated, community-contributed environment hub specifically for voice agents. Existing voice testing tools are closed, single-tenant, and don't produce training signal. General environment hubs (OpenEnv, Prime Intellect) lack voice-native primitives.

VoiceEnv fills this by providing:

- **Verifiable rewards** for voice conversations (not just LLM-as-judge)
- **Expert-grounded multimodal judging** via Gemini (compare agent audio against real human expert recordings)
- **Community validation** of the judge through human-LLM correlation tracking
- **One-click export** to OpenEnv (HuggingFace) and Prime Intellect Environments Hub
- **End-to-end training pipeline** from environments → rollouts → GRPO fine-tuning

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT SPEC                         │
│  YAML/JSON defining: task, simulator, tools, rubric, voice   │
│  + expert reference recordings for grounded judging          │
└──────────┬──────────────────────────────────┬────────────────┘
           │                                  │
     ┌─────▼─────┐                    ┌───────▼────────┐
     │  RUNNER    │                    │   EXPORTERS    │
     │ simulator  │                    │ ┌────────────┐ │
     │ sandbox    │                    │ │  OpenEnv    │ │
     │ scorer     │                    │ │  (Docker)   │ │
     │            │                    │ ├────────────┤ │
     │            │                    │ │  Prime     │ │
     │            │                    │ │  Intellect │ │
     └─────┬──────┘                    │ └────────────┘ │
           │                           └────────────────┘
     ┌─────▼──────────────────────────────────┐
     │          THREE-LAYER SCORING            │
     │                                         │
     │  Layer 1: VERIFIABLE REWARDS            │
     │    state checks, tool call validation,  │
     │    transcript pattern matching           │
     │    → safe for RL training               │
     │                                         │
     │  Layer 2: GROUNDED JUDGE (Gemini)       │
     │    compare agent audio vs expert refs   │
     │    → tone, pacing, empathy, etc.        │
     │                                         │
     │  Layer 3: LLM-AS-JUDGE (fallback)       │
     │    ungrounded soft scoring              │
     │    → benchmarks only, not RL signal     │
     └─────┬──────────────────────────────────┘
           │
     ┌─────▼──────────────────────────────────┐
     │     COMMUNITY VALIDATION LOOP           │
     │                                         │
     │  humans rate same runs → correlation    │
     │  tracked per criterion → low-corr       │
     │  criteria flagged → better expert refs  │
     │  added → judge improves → repeat        │
     └────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install -e .

# With grounded judge (Gemini)
pip install -e ".[grounded-judge]"

# With training pipeline
pip install -e ".[training]"

# Everything
pip install -e ".[all]"
```

### List built-in environments

```bash
voiceenv list
```

```
┌──────────────────────────────────────────────────────────────────┐
│                    Built-in Voice Environments                    │
├──────────────────────────────┬───────────┬────────────┬──────────┤
│ Name                         │ Vertical  │ Difficulty │ Languages│
├──────────────────────────────┼───────────┼────────────┼──────────┤
│ founder_sales_skeptical_vp   │ sales     │ hard       │ en       │
│ support_escalation_frustrated│ support   │ medium     │ en       │
│ collections_call_hardship    │ collections│ hard      │ en       │
│ appointment_scheduling_complex│ healthcare│ medium    │ en       │
│ healthcare_triage_anxious    │ healthcare│ adversarial│ en       │
└──────────────────────────────┴───────────┴────────────┴──────────┘
```

### Create a new environment

```bash
voiceenv init my_insurance_claim
# Edit my_insurance_claim.yaml, then:
voiceenv run my_insurance_claim.yaml
```

### Run an environment

```bash
# Against GPT-4o
voiceenv run founder_sales_skeptical_vp --model gpt-4o

# Multiple runs for statistical significance
voiceenv run healthcare_triage --model gpt-4o -n 10 --output results.json
```

### Benchmark across models

```bash
voiceenv benchmark voiceenv/environments/ \
  --model gpt-4o --model gpt-4o-mini --model claude-sonnet-4-20250514 \
  -n 5 --output leaderboard.json
```

### Export to environment hubs

```bash
# Export to OpenEnv (HuggingFace)
voiceenv export healthcare_triage --target openenv --output ./exports

# Export to Prime Intellect
voiceenv export healthcare_triage --target prime --output ./exports

# One-click publish to both
voiceenv publish healthcare_triage --target both
```

## Environment Spec

Each environment is a YAML file defining a complete voice agent scenario:

```yaml
name: insurance_claim_dispute
description: Policyholder disputes a denied claim for water damage
vertical: insurance
difficulty: hard

task:
  goal: Resolve the claim dispute while following company policy
  success_criteria:
    - Verified policy coverage details
    - Explained denial reason clearly
    - Offered valid resolution path

simulator:
  persona_description: >
    Frustrated homeowner whose basement flooded. Already spent $8,000 on
    repairs expecting insurance to cover it. Increasingly upset but not
    abusive. Will escalate if they feel dismissed.
  patience: 0.3
  cooperativeness: 0.5
  emotional_volatility: 0.7
  hidden_goals:
    - Get the claim approved or find an appeal process
    - Feel heard and respected

tools:
  - name: lookup_policy
    description: Look up policy coverage details
    parameters:
      - name: policy_number
        type: string
        required: true
    side_effects:
      policy_looked_up: true

rubric:
  task_success:
    # VERIFIABLE — checked deterministically
    - name: policy_looked_up
      description: Agent looked up the actual policy before responding
      weight: 2.0
      scoring_type: binary
      deterministic_check: "state.get('policy_looked_up', False)"

  voice_quality:
    # GROUNDED — compared against expert recording via Gemini
    - name: de_escalation_skill
      description: How well agent de-escalated vs. expert claims handler
      weight: 1.0
      scoring_type: grounded_judge
      reference_names: [expert_claims_handler]
      grounded_dimensions: [empathy, de_escalation, pacing, tone]

expert_references:
  - name: expert_claims_handler
    audio_path: expert_recordings/claims_de_escalation.wav
    transcript: "..."
    annotations:
      - Acknowledged frustration before explaining policy
      - Used collaborative language ("let's look at this together")
      - Provided concrete next steps
```

## Three-Layer Scoring

### Layer 1: Verifiable Rewards (Primary RL Signal)

Deterministic checks against sandbox state, tool calls, and transcript patterns. These are **safe for RL training** — no reward hacking possible.

```yaml
- name: booking_tool_called
  deterministic_check: "tool_was_called('book_meeting', min_times=1, max_times=1)"

- name: hipaa_compliant
  deterministic_check: "no_transcript_match(r'social security|SSN|date of birth', speaker='agent')"

- name: asked_about_symptoms
  deterministic_check: "transcript_contains(r'(pain|symptom|feel)', speaker='agent')"
```

Available verification functions:
- `transcript_contains(pattern, speaker=, in_first_n_turns=)` — regex search in transcript
- `no_transcript_match(pattern, speaker=)` — verify pattern does NOT appear
- `tool_was_called(name, min_times=, max_times=)` — verify tool usage
- `tool_args_valid(name, arg, valid_values)` — verify tool arguments
- `all_tools_succeeded()` — no tool calls failed
- Direct state access: `state.get('field')`, `turns <= 12`, etc.

### Layer 2: Grounded Multimodal Judge (Gemini)

Instead of asking *"rate empathy 1-5"*, the judge gets expert reference recordings and compares:

*"Here is a recording of an expert nurse handling this exact triage scenario. Here is the agent's attempt. Compare them on tone, pacing, empathy, de-escalation, and clarity."*

This makes scoring:
- **Grounded** — anchored to concrete expert behavior
- **Multimodal** — Gemini listens to actual audio (tone, pacing, interruptions)
- **Defensible** — "the score is based on comparison to this specific expert recording"
- **Improvable** — better expert recordings → better judge

```bash
# Requires Gemini API key
export GEMINI_API_KEY=your-key
pip install -e ".[grounded-judge]"
```

### Layer 3: LLM-as-Judge (Fallback)

Standard text-based LLM scoring for criteria without expert references. Used for benchmarking only — **not recommended as primary RL signal** due to reward hacking risk.

## Community Judge Validation

The critical question: *"Does the LLM judge agree with humans?"*

VoiceEnv includes a full human-LLM correlation tracking system. Community members rate runs on the same criteria the judge uses, and we measure alignment.

### Rating workflow

```bash
# 1. Save completed runs for community rating
voiceenv judge save-run results.json

# 2. Rate a run interactively
voiceenv judge rate --rater-id alice
# Shows transcript, asks for scores on each criterion,
# then displays comparison with LLM judge scores

# 3. Check how many ratings have been collected
voiceenv judge stats

# 4. Compute correlation between human and LLM scores
voiceenv judge correlation --output correlation_report.json
```

### Correlation report

```
========================================================================
JUDGE-HUMAN CORRELATION REPORT
========================================================================

Total comparisons:  247
Total raters:       18
Overall Pearson:    0.743
Overall Spearman:   0.718

FLAGGED CRITERIA (low correlation, need better references):
  ⚠ culturally_aware

------------------------------------------------------------------------
Criterion                      N   Pearson  Spearman    MAE       Status
------------------------------------------------------------------------
calm_reassuring_tone          42     0.812     0.798  0.091       [HIGH]
de_escalation_skill           38     0.756     0.731  0.124       [HIGH]
anxiety_management            35     0.689     0.654  0.142   [MODERATE]
culturally_aware              19     0.312     0.289  0.287        [LOW]
------------------------------------------------------------------------
```

Low-correlation criteria get flagged — the community knows exactly where to add better expert references to improve the judge.

## Post-Training Experiment

The end-to-end proof that VoiceEnv environments produce useful training signal for speech LLMs.

### The experiment

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   BASELINE   │     │   GENERATE   │     │    GRPO      │     │  POST-TRAIN  │     │  COMPARISON  │
│   EVAL       │────▶│   ROLLOUTS   │────▶│  FINE-TUNE   │────▶│    EVAL      │────▶│   REPORT     │
│              │     │              │     │              │     │              │     │              │
│  5 envs ×10  │     │  5 envs ×20  │     │  Qwen3-Omni  │     │  5 envs ×10  │     │  per-criterion│
│  = baseline  │     │  = 100 convos│     │  LoRA + H100 │     │  = trained   │     │  delta table │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Run the full experiment on Modal

```bash
# Prerequisites
pip install modal
modal setup
modal secret create openai-secret OPENAI_API_KEY=sk-...
modal secret create huggingface-secret HF_TOKEN=hf_...

# Full pipeline (baseline → rollouts → train → eval → report)
modal run voiceenv/training/experiment.py

# Or step by step
modal run voiceenv/training/experiment.py --step baseline
modal run voiceenv/training/experiment.py --step rollouts
modal run voiceenv/training/experiment.py --step train
modal run voiceenv/training/experiment.py --step posttrain
modal run voiceenv/training/experiment.py --step report
```

### Run locally (eval + rollouts, then train on cloud)

```bash
# Step 1: Baseline eval + generate rollouts
voiceenv eval experiment --eval-model gpt-4o-mini --runs 5 --rollout-runs 20

# Step 2: Train on Modal
modal run voiceenv/training/experiment.py --step train

# Step 3: Compare results
voiceenv eval compare experiment_results/baseline_eval.json posttrain_eval.json
```

### What the comparison report looks like

```
┌──────────────────────────────────────────────────────────┐
│ Post-Training Comparison                                  │
│ Baseline: gpt-4o-mini                                     │
│ Trained:  voiceenv-qwen3-omni-lora                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│ Overall Reward:   0.4821 → 0.6437  +0.1616 (+33.5%)     │
│ Verifiable:       0.5103 → 0.7241  +0.2138              │
│ Soft:             0.4012 → 0.4650  +0.0638              │
│                                                           │
│ Environments:  4 improved  0 regressed  1 unchanged       │
│ Criteria:     18 improved  2 regressed                    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ VERDICT: SUCCESS                                          │
│                                                           │
│ Post-training improved overall reward by +33.5%           │
│ Verifiable reward (real capability) improved by +0.2138   │
│ The gain is grounded in deterministic checks, not just    │
│ LLM-judge gaming.                                         │
└──────────────────────────────────────────────────────────┘
```

The report explicitly separates verifiable vs soft improvements — if verifiable reward improves, the model genuinely got better at the task. If only soft reward improves, it may be gaming the LLM judge.

### Training pipeline details

**Generate rollouts:**

```bash
voiceenv train rollouts voiceenv/environments/ \
  --model gpt-4o-mini \
  --runs-per-env 20 \
  --output rollouts.jsonl
```

**Fine-tune on Modal (serverless H100s):**

```bash
modal run voiceenv/training/modal_train.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --runs-per-env 20 \
  --lora-rank 16 \
  --epochs 2
```

**Fine-tune on Baseten (managed GPUs):**

```bash
voiceenv train baseten \
  --rollouts rollouts.jsonl \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu H100

cd baseten_voiceenv_training && bash run.sh
```

## Project Structure

```
voiceenv/
├── core/
│   ├── schema.py              # Environment spec (Pydantic models)
│   ├── simulator.py           # LLM-backed user simulator
│   ├── sandbox.py             # Tool execution & world state
│   ├── scorer.py              # Verifiable + soft + grounded scoring
│   ├── runner.py              # End-to-end environment runner
│   ├── grounded_judge.py      # Gemini multimodal judge
│   ├── human_ratings.py       # Community rating collection
│   └── judge_correlation.py   # Human-LLM correlation tracking
├── eval/
│   ├── evaluator.py           # Systematic model evaluation harness
│   └── comparison.py          # Before/after comparison reports
├── environments/
│   ├── founder_sales.yaml
│   ├── support_escalation.yaml
│   ├── collections_call.yaml
│   ├── appointment_scheduling.yaml
│   └── healthcare_triage.yaml
├── exporters/
│   ├── openenv_exporter.py    # Export to OpenEnv (HuggingFace)
│   └── prime_exporter.py      # Export to Prime Intellect
├── training/
│   ├── experiment.py          # Full post-training experiment (Modal)
│   ├── generate_rollouts.py   # Rollout generation
│   ├── grpo_train.py          # Local GRPO fine-tuning
│   ├── modal_train.py         # Modal serverless training
│   └── baseten_train.py       # Baseten managed training
├── hackathon/                 # Hackathon templates & guides
└── cli/
    └── main.py                # CLI entry point
```

## Contributing Environments

The whole point of VoiceEnv is community contributions. Here's how:

### 1. Create an environment

```bash
voiceenv init my_environment
# Edit the generated YAML
```

### 2. Maximize verifiable rewards

The more criteria you can make deterministic, the more useful the environment is for training. Use `deterministic_check` with the built-in verification functions instead of `llm_judge` wherever possible.

### 3. Add expert references

Record (or find) examples of humans doing the task well. Add them as `expert_references` with annotations explaining what makes them good. This is what makes the grounded judge work.

### 4. Rate other people's runs

```bash
voiceenv judge rate --rater-id your-name
```

Every human rating improves the correlation data and helps the entire platform.

### 5. Publish to hubs

```bash
voiceenv publish my_environment.yaml --target both
```

## License

Apache 2.0
