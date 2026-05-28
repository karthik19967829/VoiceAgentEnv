# VoiceEnv E2E Demo — Talk Runbook

The full story in 5 commands. Total wall time on stage: ~90 seconds.

## Pre-flight (do once, before the talk)

```bash
cd /Users/karthikganesan/Desktop/VoiceAgentEnv
set -a && source .env && set +a
python3 -c "import voiceenv; print('ok')"
ls hvb_samples/audio/agent/*.wav | head -3   # confirm sample audio is there
```

If you want a clean slate before the demo:

```bash
rm -rf environments/auto_replace_card exports
```

## Live demo

### Step 1 — show the input

```bash
ls -la hvb_samples/audio/agent/0002f70f7386445b.wav
```

> "One real banking call. 49 seconds. That's it."

### Step 2 — autonomous ingest (the headline moment)

```bash
voiceenv ingest hvb_samples/audio/agent/0002f70f7386445b.wav \
    -o environments/auto_replace_card
```

**Talk track while the spinner runs (~15s):**
> "Whisper transcribes the call. We segment turns. GPT-4o-mini extracts the
> task, the persona, the tools the agent could plausibly use, and a verifiable
> rubric. All of that drops into a VoiceEnv schema. Total cost: half a tenth
> of a cent."

Output shows: 12 turns, 1 tool, 4 rubric criteria, $0.0006, 16 seconds.

### Step 3 — show what we got

```bash
cat environments/auto_replace_card/env.yaml | head -50
```

> "This is now a trainable RL environment. Persona, tools, deterministic
> reward functions. From a WAV file."

### Step 4 — run any LLM against it (validation)

```bash
voiceenv run environments/auto_replace_card/env.yaml -m gpt-4o-mini -n 3
```

> "GPT-4o-mini gets ~30% on the rubric. The agent is calling the right tool,
> but missing the closing-courtesy criterion. *That* is signal an RL trainer
> can climb."

### Step 5 — export to BOTH hubs

```bash
voiceenv export environments/auto_replace_card/env.yaml --target both -o exports
ls exports/openenv/debit_card_replacement/
ls exports/prime/debit_card_replacement/
```

> "Two artifacts. OpenEnv: a Docker package implementing reset/step/state.
> Prime Intellect: a verifiers module. Same source spec, two ecosystems."

### Step 6 — publish live (THE MIC-DROP)

```bash
voiceenv publish environments/auto_replace_card/env.yaml --target openenv
```

**Then open in browser**: https://huggingface.co/spaces/karthik-anyreach/debit_card_replacement

> "Live. Anyone in the room can `pip install` this and train against it."

---

## Backup plan if WiFi dies

Pre-recorded everything is in:
- `environments/auto_replace_card/` — the auto-ingested env
- `exports/` — the exported OpenEnv + Prime packages
- The HF Space already exists; show the public URL even if you can't push

The *worst case* is that all you have is `cat` on the existing artifacts —
which is still the entire story.

---

## What to say if asked

**"How is this different from just an eval set?"**
> Each env has tools, a sandbox, a persona simulator that adapts to the
> agent's behavior, and verifiable rewards. It's a *gym*, not a benchmark.
> You can train against it.

**"Why is the LLM-extracted rubric trustworthy?"**
> It isn't, fully — and that's by design. We pair it with the human-rating
> + correlation system (`voiceenv ratings`) so the community can refine
> rubrics until LLM judges correlate with human judges. The autonomous
> ingest gets you to a starting point in 15 seconds; the community closes
> the gap.

**"What's the ceiling? Can this scale to thousands of calls?"**
> Linear in cost. $0.0006 per call → 100k real calls → $60 → 100k
> publishable RL environments. That's the data exhaust thesis.

---

## Artifact URLs

- HF Space (live): https://huggingface.co/spaces/karthik-anyreach/debit_card_replacement
- Github (private/local): /Users/karthikganesan/Desktop/VoiceAgentEnv
