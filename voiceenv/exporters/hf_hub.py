"""Frictionless publish to the VoiceEnv HuggingFace hub.

No PRs, no maintainer approval. One command:
  voiceenv publish environments/my_env/env.yaml

Creates an OpenEnv-compatible Docker Space and registers it in the
public VoiceEnv Environments collection on HuggingFace.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from voiceenv.core.schema import VoiceEnvironment

# Public index of all published voice environments (HF Collection).
HUB_COLLECTION_TITLE = "VoiceEnv Environments Hub"
HUB_CONFIG_PATH = Path.home() / ".voiceenv" / "hub.json"


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "env"


def default_repo_id(env: VoiceEnvironment, namespace: str | None = None) -> str:
    """Default Space repo: `{namespace}/voiceenv-{env_name}`."""
    ns = namespace or os.environ.get("VOICEENV_HUB_NAMESPACE")
    if not ns:
        from huggingface_hub import HfApi

        ns = HfApi(token=_token()).whoami()["name"]
    return f"{ns}/voiceenv-{_slugify(env.name)}"


def space_url(repo_id: str) -> str:
    owner, name = repo_id.split("/", 1)
    return f"https://{owner}-{name.replace('_', '-')}.hf.space"


def hub_page_url(repo_id: str) -> str:
    return f"https://huggingface.co/spaces/{repo_id}"


def _token() -> str:
    tok = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN (or HF_TOKEN) must be set. "
            "Get one at https://huggingface.co/settings/tokens"
        )
    return tok


def _load_hub_config() -> dict:
    if HUB_CONFIG_PATH.exists():
        return json.loads(HUB_CONFIG_PATH.read_text())
    return {}


def _save_hub_config(cfg: dict) -> None:
    HUB_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HUB_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_or_create_hub_collection(namespace: str | None = None) -> tuple[str, str]:
    """Return (collection_slug, collection_url), creating the hub index if needed."""
    cfg = _load_hub_config()
    ns = namespace or os.environ.get("VOICEENV_HUB_NAMESPACE")
    if not ns:
        from huggingface_hub import HfApi

        ns = HfApi(token=_token()).whoami()["name"]

    key = f"collection_slug:{ns}"
    url_key = f"collection_url:{ns}"
    if key in cfg:
        return cfg[key], cfg.get(url_key, f"https://huggingface.co/collections/{ns}")

    from huggingface_hub import create_collection

    collection = create_collection(
        title=HUB_COLLECTION_TITLE,
        description=(
            "Community voice agent RL environments (OpenEnv-compatible). "
            "Publish: voiceenv publish my_env/env.yaml"
        ),
        namespace=ns,
        token=_token(),
    )
    cfg[key] = collection.slug
    cfg[url_key] = collection.url
    _save_hub_config(cfg)
    return collection.slug, collection.url


def register_in_hub(
    repo_id: str,
    env: VoiceEnvironment,
    *,
    namespace: str | None = None,
) -> str:
    """Add a published Space to the VoiceEnv hub collection. Returns collection URL."""
    from huggingface_hub import add_collection_item

    slug, collection_url = get_or_create_hub_collection(namespace)
    note = (env.description or env.name)[:500]
    add_collection_item(
        slug,
        item_id=repo_id,
        item_type="space",
        note=note,
        exists_ok=True,
        token=_token(),
    )
    return collection_url


def push_openenv_space(
    pkg_path: Path,
    env: VoiceEnvironment,
    repo_id: str | None = None,
    *,
    register: bool = True,
    namespace: str | None = None,
) -> dict:
    """Upload an OpenEnv package as a HuggingFace Docker Space.

    Returns dict with repo_id, space_url, hub_url (if registered).
    """
    from huggingface_hub import HfApi, create_repo

    repo_id = repo_id or default_repo_id(env, namespace)
    token = _token()
    api = HfApi(token=token)

    create_repo(
        repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=token,
    )

    api.upload_folder(
        folder_path=str(pkg_path),
        repo_id=repo_id,
        repo_type="space",
        token=token,
        commit_message=f"VoiceEnv publish: {env.name}",
    )

    result = {
        "repo_id": repo_id,
        "space_url": hub_page_url(repo_id),
        "app_url": space_url(repo_id),
    }

    if register:
        result["hub_collection_url"] = register_in_hub(
            repo_id, env, namespace=namespace
        )

    return result
