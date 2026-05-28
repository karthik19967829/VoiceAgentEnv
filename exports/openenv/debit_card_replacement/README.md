---
title: debit_card_replacement
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
tags:
  - auto-ingested
  - real-call
  - 0002f70f
---

# debit_card_replacement

A call for replacing a lost debit card.

## Quick Start

```python
from debit_card_replacement import VoiceAction, DebitCardReplacementEnv

async with DebitCardReplacementEnv(base_url="...") as client:
    result = await client.reset()
    print(result.observation.content)  # Caller's opening line

    result = await client.step(VoiceAction(content="Hello, how can I help you?"))
    print(result.observation.content)  # Caller's response
```

## Environment Details

- **Vertical:** support
- **Difficulty:** easy
- **Languages:** en
- **Max Turns:** 50

### Task
Assist the caller in replacing their lost debit card and ensure they have no further requests.

### Scoring Categories


- **task_success**: 2 criteria


- **compliance**: 1 criteria


- **voice_quality**: 0 criteria


- **persona_fidelity**: 0 criteria


- **representation**: 0 criteria


- **efficiency**: 1 criteria


## License
Apache-2.0