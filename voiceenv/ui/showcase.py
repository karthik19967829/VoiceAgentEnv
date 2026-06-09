"""Replay pre-recorded demo pipeline events (no API keys required)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

def _showcase_dir() -> Path:
    import os

    override = os.environ.get("VOICEENV_SHOWCASE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "showcase"


def showcase_enabled() -> bool:
    """Public deploy sets VOICEENV_SHOWCASE=1; local dev stays live unless forced."""
    import os

    flag = os.environ.get("VOICEENV_SHOWCASE", "").lower()
    if flag not in ("1", "true", "yes"):
        return False
    return (_showcase_dir() / "run_events.json").exists()


def _load_events(name: str) -> list[dict]:
    path = _showcase_dir() / name
    if not path.exists():
        return []
    return json.loads(path.read_text())


async def replay_events(
    filename: str,
    *,
    speed: float = 1.0,
) -> AsyncGenerator[tuple[str, dict], None]:
    """Yield (event_name, data) with optional delays between events."""
    events = _load_events(filename)
    for ev in events:
        delay = ev.get("delay_ms", 120)
        if delay and speed > 0:
            await asyncio.sleep(delay / 1000.0 / speed)
        yield ev["event"], ev["data"]


async def replay_sse(
    filename: str,
    *,
    speed: float = 1.0,
) -> AsyncGenerator[bytes, None]:
    from voiceenv.ui.demo_app import _sse

    async for event, data in replay_events(filename, speed=speed):
        yield _sse(event, data)
