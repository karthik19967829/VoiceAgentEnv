#!/usr/bin/env python3
"""Record SSE events from a live demo run into showcase JSON (for public HF deploy)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _parse_sse(raw: str) -> list[dict]:
    events: list[dict] = []
    for block in re.split(r"\n\n+", raw.strip()):
        if not block.strip():
            continue
        ev_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            events.append({"event": ev_name, "data": json.loads("\n".join(data_lines))})
    return events


def record(url: str, path: str, *, improve: bool = False) -> list[dict]:
    recorded: list[dict] = []
    t0 = time.time()
    last = t0

    def gap() -> int:
        nonlocal last
        now = time.time()
        ms = int((now - last) * 1000)
        last = now
        return min(ms, 800)

    endpoint = "/api/improve" if improve else "/api/run"
    params = {"wav": path} if not improve else {"wav": path}
    if not improve:
        params.update({"model": "gpt-audio-mini", "max_turns": "6", "grounded": "true"})

    with httpx.Client(timeout=600.0) as client:
        with client.stream("GET", f"{url.rstrip('/')}{endpoint}", params=params) as resp:
            resp.raise_for_status()
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    part, buf = buf.split("\n\n", 1)
                    part = part.strip()
                    if not part:
                        continue
                    for ev in _parse_sse(part + "\n\n"):
                        ev["delay_ms"] = gap()
                        recorded.append(ev)
                        print(f"  {ev['event']}", flush=True)
                        if ev["event"] == "done":
                            return recorded
    return recorded


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8911")
    p.add_argument("--wav", default="hvb_samples/audio/agent/00d676d7058c49bb.wav")
    p.add_argument("--out-dir", default=str(ROOT / "voiceenv" / "ui" / "showcase"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_tmp = out / "run_events.json.tmp"
    imp_tmp = out / "improve_events.json.tmp"

    print("Recording /api/run …")
    run_events = record(args.base_url, args.wav, improve=False)
    run_tmp.write_text(json.dumps(run_events, indent=2))
    print(f"  → {len(run_events)} events (staging)")

    print("Recording /api/improve …")
    imp_events = record(args.base_url, args.wav, improve=True)
    imp_tmp.write_text(json.dumps(imp_events, indent=2))
    print(f"  → {len(imp_events)} events (staging)")

    if not imp_events:
        print("[error] improve stream empty — re-run /api/run on the same server, then retry", file=sys.stderr)
        return 1

    run_tmp.replace(out / "run_events.json")
    imp_tmp.replace(out / "improve_events.json")
    print(f"  → wrote {out / 'run_events.json'} and {out / 'improve_events.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
