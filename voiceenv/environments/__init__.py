"""Built-in seed environments for VoiceEnv."""

from pathlib import Path

ENVIRONMENTS_DIR = Path(__file__).parent


def list_environments() -> list[str]:
    """List all built-in environment YAML files."""
    return [f.stem for f in ENVIRONMENTS_DIR.glob("*.yaml")]


def load_environment(name: str):
    """Load a built-in environment by name."""
    from voiceenv.core.schema import VoiceEnvironment

    path = ENVIRONMENTS_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_environments()
        raise FileNotFoundError(
            f"Environment '{name}' not found. Available: {available}"
        )
    return VoiceEnvironment.from_yaml(path)
