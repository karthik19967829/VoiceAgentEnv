"""
Human Rating Collection & Storage — the community side of judge validation.

THE PROBLEM:
  LLM judges produce scores, but how do we know they're accurate?
  In code environments, we have unit tests. In voice environments,
  we need human ground truth.

THE SOLUTION:
  Community members listen to agent conversations and rate them on
  the same criteria the LLM judge uses. We then measure correlation
  between human and LLM scores to:
    1. VALIDATE the judge (high correlation = trustworthy)
    2. IDENTIFY weak spots (low correlation criteria need better references)
    3. IMPROVE over time (more human data → better judge prompts/references)
    4. BUILD TRUST (published correlation stats = transparent methodology)

WORKFLOW:
  1. A run produces a transcript (and optionally audio)
  2. Community members rate it via `voiceenv judge rate`
  3. Multiple humans rate the same run (inter-rater agreement)
  4. `voiceenv judge correlation` computes alignment with LLM scores
  5. Criteria with low correlation get flagged for improvement
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HumanRating:
    """A single human's rating of a single criterion for a single run."""

    run_id: str
    criterion_name: str
    rater_id: str
    score: float  # 0.0 - 1.0 (normalized)
    reasoning: str = ""
    confidence: float = 1.0  # rater's self-reported confidence (0-1)
    timestamp: float = field(default_factory=time.time)

    # Rater metadata (for weighting/filtering)
    rater_expertise: str = "general"  # general, domain_expert, linguist, etc.
    audio_listened: bool = False  # did the rater listen to audio or just read transcript

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "criterion_name": self.criterion_name,
            "rater_id": self.rater_id,
            "score": self.score,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "rater_expertise": self.rater_expertise,
            "audio_listened": self.audio_listened,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanRating:
        return cls(**data)


@dataclass
class RunForRating:
    """
    A completed run packaged for human rating.

    Transcript entries can include per-turn audio:
      {"role": "agent", "content": "...", "audio_path": "path/to.wav",
       "interrupted": true, "duration_ms": 2400}
    """

    run_id: str
    environment_name: str
    transcript: list[dict[str, Any]]  # per-turn: {role, content, audio_path?, interrupted?}
    criteria_to_rate: list[dict[str, str]]  # [{name, description}]
    audio_dir: str | None = None  # directory containing per-turn audio files
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_scores: dict[str, float] = field(default_factory=dict)  # criterion_name → LLM score

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "environment_name": self.environment_name,
            "transcript": self.transcript,
            "criteria_to_rate": self.criteria_to_rate,
            "audio_dir": self.audio_dir,
            "tool_calls": self.tool_calls,
            "llm_scores": self.llm_scores,
        }


class RatingStore:
    """
    Persistent storage for human ratings. Uses a simple JSONL file format
    so it's easy to version control, share, and aggregate.

    Directory structure:
      ratings/
        ratings.jsonl          ← all human ratings (append-only)
        runs/
          <run_id>.json        ← run data for rating
        correlation_cache.json ← cached correlation results
    """

    def __init__(self, ratings_dir: str | Path = "ratings"):
        self.ratings_dir = Path(ratings_dir)
        self.ratings_file = self.ratings_dir / "ratings.jsonl"
        self.runs_dir = self.ratings_dir / "runs"
        self.correlation_cache = self.ratings_dir / "correlation_cache.json"

    def _ensure_dirs(self):
        self.ratings_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run_for_rating(self, run_data: RunForRating) -> Path:
        """Save a completed run so community members can rate it."""
        self._ensure_dirs()
        run_path = self.runs_dir / f"{run_data.run_id}.json"
        run_path.write_text(json.dumps(run_data.to_dict(), indent=2))
        return run_path

    def load_run_for_rating(self, run_id: str) -> RunForRating:
        """Load a run for rating."""
        run_path = self.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            raise FileNotFoundError(f"No run found for rating: {run_id}")
        data = json.loads(run_path.read_text())
        return RunForRating(**data)

    def list_runs(self) -> list[str]:
        """List all run IDs available for rating."""
        self._ensure_dirs()
        return [
            p.stem for p in sorted(self.runs_dir.glob("*.json"))
        ]

    def submit_rating(self, rating: HumanRating) -> None:
        """Append a human rating. Thread-safe via append-only."""
        self._ensure_dirs()
        with open(self.ratings_file, "a") as f:
            f.write(json.dumps(rating.to_dict()) + "\n")

    def submit_ratings(self, ratings: list[HumanRating]) -> None:
        """Batch submit multiple ratings."""
        self._ensure_dirs()
        with open(self.ratings_file, "a") as f:
            for rating in ratings:
                f.write(json.dumps(rating.to_dict()) + "\n")

    def load_all_ratings(self) -> list[HumanRating]:
        """Load all human ratings."""
        if not self.ratings_file.exists():
            return []
        ratings = []
        for line in self.ratings_file.read_text().strip().split("\n"):
            if line.strip():
                ratings.append(HumanRating.from_dict(json.loads(line)))
        return ratings

    def get_ratings_for_run(self, run_id: str) -> list[HumanRating]:
        """Get all human ratings for a specific run."""
        return [r for r in self.load_all_ratings() if r.run_id == run_id]

    def get_ratings_for_criterion(self, criterion_name: str) -> list[HumanRating]:
        """Get all human ratings for a specific criterion across all runs."""
        return [r for r in self.load_all_ratings() if r.criterion_name == criterion_name]

    def get_rating_stats(self) -> dict[str, Any]:
        """Summary statistics about the human rating corpus."""
        ratings = self.load_all_ratings()
        if not ratings:
            return {"total_ratings": 0}

        runs = set(r.run_id for r in ratings)
        criteria = set(r.criterion_name for r in ratings)
        raters = set(r.rater_id for r in ratings)

        return {
            "total_ratings": len(ratings),
            "unique_runs_rated": len(runs),
            "unique_criteria_rated": len(criteria),
            "unique_raters": len(raters),
            "audio_ratings_pct": sum(1 for r in ratings if r.audio_listened) / len(ratings),
            "avg_confidence": sum(r.confidence for r in ratings) / len(ratings),
        }


def generate_run_id(env_name: str, transcript: list[dict[str, str]]) -> str:
    """Generate a stable, deterministic run ID from environment + transcript."""
    content = json.dumps({"env": env_name, "transcript": transcript}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def package_run_for_rating(
    env_name: str,
    transcript: list[dict[str, Any]],
    criteria: list[dict[str, str]],
    tool_calls: list[dict[str, Any]] | None = None,
    llm_scores: dict[str, float] | None = None,
    audio_dir: str | None = None,
) -> RunForRating:
    """Package a completed run into a RatingPackage for community review."""
    run_id = generate_run_id(env_name, transcript)
    return RunForRating(
        run_id=run_id,
        environment_name=env_name,
        transcript=transcript,
        criteria_to_rate=criteria,
        tool_calls=tool_calls or [],
        llm_scores=llm_scores or {},
        audio_dir=audio_dir,
    )
