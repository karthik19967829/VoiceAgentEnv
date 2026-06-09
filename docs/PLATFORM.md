# VoiceEnv platform guide

Architecture, scoring, training, and CLI reference. For the quick start, see [README](../README.md).

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

## Install & CLI

### Installation

```bash
pip install -e .

pip install -e ".[voice]"           # Pipecat speech LLM conversations
pip install -e ".[grounded-judge]"  # Gemini multimodal judge
pip install -e ".[ui]"              # rating UI
pip install -e ".[training]"        # rollouts + GRPO handoff
pip install -e ".[all]"
```

### Commands

```bash
voiceenv list                       # built-in environments
voiceenv init my_env                # scaffold YAML by hand
voiceenv run my_env.yaml --model gpt-4o -n 10 -o results.json
voiceenv benchmark voiceenv/environments/ --model gpt-4o -n 5 -o leaderboard.json

voiceenv export my_env --target openenv --output ./exports
voiceenv publish my_env/env.yaml    # OpenEnv Space + hub
voiceenv publish-demo               # hosted talk demo Space
```

Republish demo after re-recording:

```bash
voiceenv ui --port 8920
python3 scripts/record_showcase.py --base-url http://127.0.0.1:8920
voiceenv publish-demo
```

### Voice mode

Both sides are speech LLMs. Pipecat handles audio, VAD, interruptions, per-turn recording.

```bash
pip install -e ".[voice]"

swift deploy --model Qwen/Qwen3-Omni-30B-A3B-Instruct

voiceenv run-voice healthcare_triage \
  --agent-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --agent-base-url http://localhost:8000/v1 \
  --simulator-model gpt-4o \
  --save-for-rating

voiceenv judge serve
```

## Environment spec

Each environment is a YAML file:

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
    - name: policy_looked_up
      description: Agent looked up the actual policy before responding
      weight: 2.0
      scoring_type: binary
      deterministic_check: "state.get('policy_looked_up', False)"

  voice_quality:
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

## Three-layer scoring

### Layer 1: Verifiable rewards (primary RL signal)

Deterministic checks on state, tool calls, and transcript patterns. Safe for RL — no reward hacking.

```yaml
- name: booking_tool_called
  deterministic_check: "tool_was_called('book_meeting', min_times=1, max_times=1)"

- name: hipaa_compliant
  deterministic_check: "no_transcript_match(r'social security|SSN|date of birth', speaker='agent')"

- name: asked_about_symptoms
  deterministic_check: "transcript_contains(r'(pain|symptom|feel)', speaker='agent')"
```

Verification helpers:

- `transcript_contains(pattern, speaker=, in_first_n_turns=)`
- `no_transcript_match(pattern, speaker=)`
- `tool_was_called(name, min_times=, max_times=)`
- `tool_args_valid(name, arg, valid_values)`
- `all_tools_succeeded()`
- Direct state: `state.get('field')`, `turns <= 12`, etc.

### Layer 2: Grounded judge (Gemini)

Compare agent audio to expert reference recordings on tone, pacing, empathy, de-escalation, clarity.

```bash
export GEMINI_API_KEY=your-key
pip install -e ".[grounded-judge]"
```

### Layer 3: LLM-as-judge (fallback)

Text-only scoring for criteria without expert refs. Benchmarking only — not recommended as RL signal.

## Community judge validation

Human raters score the same runs the LLM judge scores; VoiceEnv tracks correlation per criterion.

```bash
pip install -e ".[ui]"
voiceenv judge serve --demo          # http://localhost:8910

voiceenv judge save-run results.json
voiceenv judge rate --rater-id alice
voiceenv judge stats
voiceenv judge correlation --output correlation_report.json
```

Low-correlation criteria get flagged → add better expert references → judge improves.

## Cloud GPUs (Modal)

```bash
pip install -e ".[cloud]"
modal setup

voiceenv cloud serve
voiceenv cloud run healthcare_triage --save-for-rating
voiceenv cloud rollouts voiceenv/environments/ --runs-per-env 20 -o rollouts.jsonl
voiceenv cloud train -r rollouts.jsonl
```

| Function | GPU | What it does |
|----------|-----|--------------|
| `serve_speech_llm` | A100 | Qwen3-Omni via vLLM |
| `run_voice_env` | A100 | Voice conversation + audio |
| `generate_rollouts` | A100 | Training data across envs |
| `train_grpo` | A100x2 | ms-swift GRPO + LoRA |

## Post-training

VoiceEnv generates rollouts and rewards; [ms-swift](https://github.com/modelscope/ms-swift) runs GRPO.

```bash
pip install ms-swift

voiceenv train rollouts voiceenv/environments/ \
  --model gpt-4o-mini --runs-per-env 20 --output rollouts.jsonl

voiceenv train run -m Qwen/Qwen3-Omni-30B-A3B-Instruct -r rollouts.jsonl

voiceenv eval run -m gpt-4o-mini -n 10 -o baseline.json
voiceenv eval run -m ./voiceenv_trained -n 10 -o trained.json --base-url http://localhost:8000/v1
voiceenv eval compare baseline.json trained.json
```

## Project structure

```
voiceenv/
├── core/           schema, simulator, sandbox, scorer, runner, grounded_judge
├── eval/           systematic evaluation + comparison
├── environments/   built-in YAML envs
├── exporters/      OpenEnv, HF hub, demo Space, Prime
├── training/       rollouts + ms-swift launch
├── ui/             judge UI, talk demo, showcase replay
├── cloud/          Modal GPU jobs
└── cli/main.py
```

## Contributing environments

1. `voiceenv init my_environment` or `voiceenv ingest call.wav -o environments/my_env/`
2. Prefer `deterministic_check` over `llm_judge` where possible
3. Add `expert_references` with annotations for grounded judging
4. `voiceenv judge rate --rater-id your-name`
5. `voiceenv publish my_environment/env.yaml`
