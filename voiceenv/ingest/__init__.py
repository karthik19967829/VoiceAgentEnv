"""
Autonomous ingest: turn a raw call recording into a publishable VoiceEnv.

Given a WAV file (and optionally a sibling transcript JSON), this module:
  1. Builds a turn-segmented transcript (HVB-format passthrough or Whisper).
  2. Uses an LLM to extract task / persona / tools / verifiable rubric.
  3. Emits a complete VoiceEnvironment YAML, expert-reference audio, and a
     scored rollout that's ready for export to OpenEnv / Prime Intellect.

This is the "data exhaust → trainable environment" pipeline.
"""

from voiceenv.ingest.from_call import ingest_call, IngestResult

__all__ = ["ingest_call", "IngestResult"]
