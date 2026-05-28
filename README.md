# VoiceEnv

**Community-driven voice agent environments for speech LLM training and evaluation.**

VoiceEnv is an open platform where the community creates voice agent environments — realistic conversational scenarios with verifiable rewards — that directly drive the post-training of speech LLMs like Qwen3-Omni. Think of it as the voice-native equivalent of code execution environments for RL, but for spoken conversations.

## Published artifacts

- **HF Space (OpenEnv-compatible Docker)**: [karthik/voiceenv-money-transfer](https://huggingface.co/spaces/karthik/voiceenv-money-transfer)
- **HF Dataset (env + expert audio + rollouts)**: [karthik/voiceenv-money-transfer](https://huggingface.co/datasets/karthik/voiceenv-money-transfer)

## 30-second demo

```bash
git clone https://github.com/karthik19967829/VoiceAgentEnv.git
cd VoiceAgentEnv && pip install -e .
cp .env.example .env  # then fill in your OPENAI_API_KEY, HUGGINGFACE_TOKEN, GEMINI_API_KEY

# Launch the live web demo
voiceenv ui --port 8911
# → open http://127.0.0.1:8911/ and click "Run demo"
```

The demo runs the full loop:
1. Autonomously extract an RL environment from a single real call WAV
2. Slice the caller's audio per-turn
3. Run stateless turn-level eval against gpt-audio-mini (parallel)
4. Grounded multimodal judging with Gemini (anchored on the real human recording)
5. **Env-driven improvement**: feed the judge's feedback back into the agent's
   prompt, re-run, and measure the lift (`+0.50` avg score, `+1` on human_likeness
   in our run)


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

# With voice mode (Pipecat + edge-tts for speech LLM conversations)
pip install -e ".[voice]"

# With grounded judge (Gemini)
pip install -e ".[grounded-judge]"

# With rating UI
pip install -e ".[ui]"

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

### Run with speech LLMs (voice mode)

Both sides of the conversation are real speech LLMs. Pipecat handles the audio pipeline, VAD, interruptions, and per-turn audio recording.

```bash
pip install -e ".[voice]"

# Serve Qwen3-Omni locally (via ms-swift)
swift deploy --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# Run: Agent = Qwen3-Omni, Simulator = GPT-4o
voiceenv run-voice healthcare_triage \
  --agent-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --agent-base-url http://localhost:8000/v1 \
  --simulator-model gpt-4o \
  --save-for-rating

# Audio from both speakers saved per-turn in run_audio/
# Then launch the rating UI so people can listen and rate:
voiceenv judge serve
```

The `--save-for-rating` flag packages the run with all per-turn audio for community review. Raters hear the **actual speech LLM output** — not synthetic TTS — so they're evaluating real model quality (tone, pacing, empathy, interruption handling).

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

### Web UI (grandma-friendly)

The easiest way to participate — no coding required:

```bash
pip install -e ".[ui]"

# Launch with demo conversations pre-loaded
voiceenv judge serve --demo

# Open http://localhost:8910 in your browser
```

The web UI lets anyone:
1. **Enter their name** and start rating
2. **Read conversations** between AI agents and callers
3. **Rate each criterion** with emoji buttons (Bad → Great)
4. **See comparison** between their rating and the AI judge
5. **View the leaderboard** showing human-AI agreement per criterion

Share the link with anyone — no login, no setup, no terminal needed.

### CLI workflow (for power users)

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

## Cloud GPUs via Modal

All GPU workloads run on [Modal](https://modal.com) — serverless A100/H100 GPUs with sub-second startup. No infra to manage.

```bash
pip install -e ".[cloud]"
modal setup  # one-time auth

# Deploy Qwen3-Omni as a vLLM server on A100
voiceenv cloud serve

# Run a voice conversation on GPU (agent + simulator are speech LLMs)
voiceenv cloud run healthcare_triage --save-for-rating

# Generate training rollouts on GPU
voiceenv cloud rollouts voiceenv/environments/ --runs-per-env 20 -o rollouts.jsonl

# Post-train with ms-swift GRPO on 2x A100
voiceenv cloud train -r rollouts.jsonl
```

Under the hood this uses a single Modal app (`voiceenv.cloud.modal_app`) with four functions:

| Function | GPU | What it does |
|----------|-----|--------------|
| `serve_speech_llm` | A100 | Serves Qwen3-Omni via vLLM with OpenAI-compatible API |
| `run_voice_env` | A100 | Spins up a local vLLM, runs a voice conversation, saves audio |
| `generate_rollouts` | A100 | Generates training data across all environments |
| `train_grpo` | A100x2 | Runs ms-swift GRPO fine-tuning with LoRA |

Model weights are cached in a Modal Volume (`voiceenv-hf-cache`) so they're only downloaded once. Training checkpoints and audio recordings persist in `voiceenv-data`.

## Post-Training: One Command

We don't implement training. We generate the data and reward signal, then hand off to [ms-swift](https://github.com/modelscope/ms-swift) which has native Qwen3-Omni GRPO support.

```bash
pip install ms-swift
```

### Step 1: Generate rollouts (our code)

```bash
voiceenv train rollouts voiceenv/environments/ \
  --model gpt-4o-mini \
  --runs-per-env 20 \
  --output rollouts.jsonl
```

### Step 2: Post-train (one command)

```bash
voiceenv train run -m Qwen/Qwen3-Omni-30B-A3B-Instruct -r rollouts.jsonl
```

This formats the data and calls `swift rlhf --rlhf_type grpo` under the hood.

### Step 3: Evaluate and compare

```bash
voiceenv eval run -m gpt-4o-mini -n 10 -o baseline.json
voiceenv eval run -m ./voiceenv_trained -n 10 -o trained.json --base-url http://localhost:8000/v1
voiceenv eval compare baseline.json trained.json
```

## Project Structure

```
voiceenv/
├── core/                          # THE PLATFORM
│   ├── schema.py                  #   Environment spec (Pydantic models)
│   ├── simulator.py               #   LLM-backed user simulator
│   ├── sandbox.py                 #   Tool execution & world state
│   ├── scorer.py                  #   Verifiable + soft + grounded scoring
│   ├── runner.py                  #   Text-mode environment runner
│   ├── voice_runner.py            #   Speech LLM runner (Pipecat)
│   ├── grounded_judge.py          #   Gemini multimodal judge
│   ├── human_ratings.py           #   Community rating collection
│   └── judge_correlation.py       #   Human-LLM correlation tracking
├── eval/                          # MEASUREMENT
│   ├── evaluator.py               #   Systematic model evaluation
│   └── comparison.py              #   Before/after delta reports
├── environments/                  # COMMUNITY ENVIRONMENTS
│   ├── founder_sales.yaml
│   ├── support_escalation.yaml
│   ├── collections_call.yaml
│   ├── appointment_scheduling.yaml
│   └── healthcare_triage.yaml
├── exporters/                     # HUB INTEGRATION
│   ├── openenv_exporter.py        #   → OpenEnv (HuggingFace)
│   └── prime_exporter.py          #   → Prime Intellect
├── training/                      # THIN TRAINING LAYER
│   ├── generate_rollouts.py       #   Generate training data (OUR code)
│   └── launch.py                  #   Format data + invoke ms-swift (THEIR code)
├── ui/                            # WEB UI FOR COMMUNITY RATING
│   ├── app.py                     #   FastAPI server + single-page app
│   └── demo_data.py               #   Demo conversations for first-time use
├── cloud/                         # MODAL GPU INFRASTRUCTURE
│   └── modal_app.py               #   Serve, run, train on serverless GPUs
└── cli/
    └── main.py                    #   CLI entry point
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
