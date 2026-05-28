# check_account_balance

> A call where the user checks their account balance with a bank agent.

**Auto-ingested** from a real human-human call recording (`0091a706bc604188.wav`,
8 turns) by `voiceenv ingest`. No manual annotation. No template.

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

The AI agent must assist the user in checking their account balance and ensure the user is satisfied with the response.

**Success criteria:**
- The user receives the correct account balance.
- The user expresses satisfaction with the interaction.

## Verifiable rubric (4 criteria)

- **balance_checked** (binary): The user received the correct account balance.
- **user_satisfaction** (binary): The user expressed satisfaction with the interaction.
- **greeting_provided** (binary): The agent provided a greeting at the start of the call.
- **turns_used** (binary): The interaction was completed in a reasonable number of turns.

## Tools available to the agent

- `check_balance` — Retrieves the user's account balance.

## Expert reference

The original call recording is provided in `expert_reference/source_call.wav`
and is used to ground LLM judges on real human behavior (tone, pacing,
de-escalation). This is what makes VoiceEnv judges *grounded* rather than
ungrounded LLM-as-judge.

## License

Apache-2.0
