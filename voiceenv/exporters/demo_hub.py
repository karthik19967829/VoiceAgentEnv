"""Publish the shareable talk demo UI to Hugging Face Spaces."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_TEMPLATE = REPO_ROOT / "exports" / "demo_space"
DEFAULT_REPO_ID = "karthik/voiceenv-demo"


def _token() -> str:
    tok = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN (or HF_TOKEN) must be set. "
            "Get one at https://huggingface.co/settings/tokens"
        )
    return tok


def push_demo_space(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    private: bool = False,
) -> str:
    """Build and upload the showcase demo Space (Docker, port 7860)."""
    from huggingface_hub import HfApi

    showcase = REPO_ROOT / "voiceenv" / "ui" / "showcase"
    if not (showcase / "run_events.json").exists():
        raise RuntimeError(
            "Missing voiceenv/ui/showcase/run_events.json. "
            "Record first: start `voiceenv ui`, then "
            "`python scripts/record_showcase.py --base-url http://127.0.0.1:8911`"
        )

    api = HfApi(token=_token())
    owner, name = repo_id.split("/", 1)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for item in ("Dockerfile", "requirements.txt", "README.md"):
            shutil.copy2(DEMO_TEMPLATE / item, root / item)

        (root / "voiceenv").mkdir(parents=True, exist_ok=True)
        # Minimal stub — full voiceenv/__init__.py pulls core deps not in this image.
        (root / "voiceenv" / "__init__.py").write_text(
            '"""VoiceEnv demo Space (UI-only stub)."""\n', encoding="utf-8"
        )
        shutil.copytree(REPO_ROOT / "voiceenv" / "ui", root / "voiceenv" / "ui")
        (root / "hvb_samples" / "audio" / "agent").mkdir(parents=True, exist_ok=True)
        (root / "hvb_samples" / "transcript").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            REPO_ROOT / "hvb_samples/audio/agent/00d676d7058c49bb.wav",
            root / "hvb_samples/audio/agent/00d676d7058c49bb.wav",
        )
        shutil.copy2(
            REPO_ROOT / "hvb_samples/transcript/00d676d7058c49bb.json",
            root / "hvb_samples/transcript/00d676d7058c49bb.json",
        )
        shutil.copytree(
            REPO_ROOT / "environments/auto_00d676d7",
            root / "environments/auto_00d676d7",
        )

        api.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Update VoiceEnv interactive demo",
        )

    return f"https://huggingface.co/spaces/{repo_id}"
