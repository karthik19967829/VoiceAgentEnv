"""
VoiceEnv training integration.

We don't implement training — we generate the data and reward signal,
then hand off to battle-tested frameworks:
  - VERL (ByteDance) — production GRPO, custom reward functions
  - ms-swift (ModelScope) — Qwen3-Omni native GRPO support
  - TRL (HuggingFace) — general-purpose post-training
"""
