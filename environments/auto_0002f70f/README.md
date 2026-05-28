# debit_card_replacement

> A call for replacing a lost debit card.

**Auto-ingested** from a real human-human call recording (`0002f70f7386445b.wav`,
10 turns) by `voiceenv ingest`. No manual annotation. No template.

## Use

```bash
# Run any speech LLM against this environment:
voiceenv run env.yaml --model gpt-4o-mini -n 10

# Generate RL training rollouts:
voiceenv train rollouts . --model gpt-4o-mini -n 50

# Publish to HuggingFace Spaces (OpenEnv) or Prime Intellect:
voiceenv export env.yaml --target both
voiceenv publish env.yaml --target both
```

## Task

Assist the user in replacing their lost debit card and ensure they have no further requests.

**Success criteria:**
- The user confirms their request for a debit card replacement.
- The agent successfully closes the call without any unresolved issues.

## Verifiable rubric (4 criteria)

- **confirmed_replacement** (binary): The user confirms their request for a debit card replacement.
- **successful_closing** (binary): The agent successfully closes the call without unresolved issues.
- **greeting_given** (binary): The agent greets the caller at the beginning of the call.
- **turns_limit** (binary): The call is handled within a reasonable number of turns.

## Tools available to the agent

- `replace_card` — Initiates the process to replace a lost debit card.

## Expert reference

The original call recording is provided in `expert_reference/source_call.wav`
and is used to ground LLM judges on real human behavior (tone, pacing,
de-escalation). This is what makes VoiceEnv judges *grounded* rather than
ungrounded LLM-as-judge.

## License

Apache-2.0
