# VoiceEnv

Robert calls his bank. *"Transfer $135 from savings to checking."* Jennifer handles it in under a minute — asks the right questions, runs the transfer, closes politely. That conversation lives in a WAV file on someone's laptop. Useful for QA, useless for training an AI agent.

Until you run:

```bash
voiceenv ingest robert_transfer.wav -o environments/money_transfer/
voiceenv publish environments/money_transfer/env.yaml
```

One real call → task, persona, tools, rubric, expert reference, hosted OpenEnv gym. No YAML authoring. No scenario design doc. Two commands.

**[Try the demo →](https://huggingface.co/spaces/karthik/voiceenv-demo)** — same money-transfer call, end to end: auto-ingest, side-by-side human vs AI audio, grounded judge, prompt improvement. Click Run. No API keys.

## CLI

```bash
pip install -e .
export OPENAI_API_KEY=...   # ingest + eval
export HF_TOKEN=...         # publish

voiceenv ingest my_call.wav -o environments/my_env/
voiceenv publish environments/my_env/env.yaml
```

`ingest` writes `env.yaml`, caller clips, and an expert reference from the recording.  
`publish` puts a Docker Space on Hugging Face (`reset` / `step` / reward) and lists it in the [hub](https://huggingface.co/collections/karthik/voiceenv-environments-hub-6a1e3b812449af97bea61a9f).

Also:

```bash
voiceenv demo my_call.wav    # eval a speech LLM against the env
voiceenv ui                    # full pipeline in the browser
```

Example published env from the demo call: [voiceenv-money-transfer](https://huggingface.co/spaces/karthik/voiceenv-money-transfer) · [API docs](https://karthik-voiceenv-money-transfer.hf.space/docs)

## Use the hosted gym

```python
from money_transfer import MoneyTransferEnv, VoiceAction

async with MoneyTransferEnv(base_url="https://karthik-voiceenv-money-transfer.hf.space") as env:
    r = await env.reset()
    r = await env.step(VoiceAction(content="How much to transfer?", tool_calls=[]))
```

Your agent plays Jennifer. The env replays Robert. Episode ends with a verifiable reward — did the transfer run, was there a greeting, how many turns.

Publish flags: `--repo-id`, `--namespace`, `--no-register`.

## Platform guide

Architecture, scoring layers, voice mode, Modal GPUs, GRPO training, full CLI reference → **[docs/PLATFORM.md](docs/PLATFORM.md)**

## License

Apache 2.0
