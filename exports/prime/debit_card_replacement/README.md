# debit_card_replacement

> Voice environment for Prime Intellect Environments Hub
> Auto-exported from VoiceEnv

A call for replacing a lost debit card.

## Usage

```bash
# Install
prime env install voiceenv-debit-card-replacement

# Evaluate
prime eval run voiceenv-debit-card-replacement
```

```python
from debit_card_replacement import load_environment

env = load_environment(simulator_model="gpt-4o-mini")
# Use with vf-eval, prime-rl, TRL, or any verifiers-compatible trainer
```

## Environment Details

- **Vertical:** support
- **Difficulty:** easy
- **Languages:** en
- **Max Turns:** 50

### Task
Assist the caller in replacing their lost debit card and ensure they have no further requests.

### Success Criteria

- The caller confirms the card to be replaced.

- The agent successfully closes the call without any further issues.


## Source
Exported from [VoiceEnv](https://github.com/voiceenv/voiceenv) environment spec.