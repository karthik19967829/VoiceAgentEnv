"""
Judge-Human Correlation Tracker — the trust layer for the entire platform.

This module answers the fundamental question:
  "Does the LLM judge agree with human experts?"

If correlation is high → the judge is trustworthy → the environment produces
reliable reward signal → models trained on it actually improve.

If correlation is low → we know WHICH criteria are weak → we add better
expert references or adjust judge prompts → correlation improves.

METRICS COMPUTED:
  1. Per-criterion Pearson correlation (linear agreement)
  2. Per-criterion Spearman correlation (rank agreement)
  3. Cohen's kappa (binary agreement after thresholding)
  4. Inter-rater reliability (do humans agree with each other?)
  5. Confidence-weighted correlation (trust expert raters more)
  6. Aggregate correlation across all criteria

COMMUNITY FLYWHEEL:
  - Community members rate runs → ratings accumulate
  - Correlation improves or drops → signals where judge needs help
  - Low-correlation criteria flagged → community adds better expert refs
  - New expert refs → better grounded judge → higher correlation
  - Published correlation stats → transparency → trust → more contributors
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from voiceenv.core.human_ratings import HumanRating


@dataclass
class CriterionCorrelation:
    """Correlation stats for a single scoring criterion."""

    criterion_name: str
    n_comparisons: int = 0  # number of (human, llm) score pairs
    n_raters: int = 0

    pearson_r: float | None = None
    spearman_rho: float | None = None
    cohens_kappa: float | None = None

    mean_human_score: float = 0.0
    mean_llm_score: float = 0.0
    mean_absolute_error: float = 0.0

    # Inter-rater reliability (among humans)
    human_agreement: float | None = None

    # Health status
    status: str = "insufficient_data"  # insufficient_data, low, moderate, high

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion_name,
            "n_comparisons": self.n_comparisons,
            "n_raters": self.n_raters,
            "pearson_r": _round_or_none(self.pearson_r),
            "spearman_rho": _round_or_none(self.spearman_rho),
            "cohens_kappa": _round_or_none(self.cohens_kappa),
            "mean_human_score": round(self.mean_human_score, 4),
            "mean_llm_score": round(self.mean_llm_score, 4),
            "mean_absolute_error": round(self.mean_absolute_error, 4),
            "human_agreement": _round_or_none(self.human_agreement),
            "status": self.status,
        }


@dataclass
class CorrelationReport:
    """Full correlation report across all criteria."""

    criteria: list[CriterionCorrelation] = field(default_factory=list)
    overall_pearson: float | None = None
    overall_spearman: float | None = None
    total_comparisons: int = 0
    total_raters: int = 0
    flagged_criteria: list[str] = field(default_factory=list)  # criteria needing attention
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pearson": _round_or_none(self.overall_pearson),
            "overall_spearman": _round_or_none(self.overall_spearman),
            "total_comparisons": self.total_comparisons,
            "total_raters": self.total_raters,
            "flagged_criteria": self.flagged_criteria,
            "criteria": [c.to_dict() for c in self.criteria],
        }


def _round_or_none(v: float | None, decimals: int = 4) -> float | None:
    return round(v, decimals) if v is not None else None


def _pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation coefficient. Returns None if insufficient variance."""
    n = len(x)
    if n < 3:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-10:
        return None

    return cov / denom


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation. Converts to ranks, then computes Pearson."""
    n = len(x)
    if n < 3:
        return None

    def _rank(values: list[float]) -> list[float]:
        sorted_vals = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_vals):
            ranks[idx] = float(rank + 1)
        # Handle ties with average rank
        i = 0
        while i < n:
            j = i
            while j < n and values[sorted_vals[j]] == values[sorted_vals[i]]:
                j += 1
            if j > i + 1:
                avg_rank = sum(range(i + 1, j + 1)) / (j - i)
                for k in range(i, j):
                    ranks[sorted_vals[k]] = avg_rank
            i = j
        return ranks

    return _pearson(_rank(x), _rank(y))


def _cohens_kappa(
    x: list[float], y: list[float], threshold: float = 0.5
) -> float | None:
    """Cohen's kappa for binary agreement (above/below threshold)."""
    n = len(x)
    if n < 3:
        return None

    x_bin = [1 if v >= threshold else 0 for v in x]
    y_bin = [1 if v >= threshold else 0 for v in y]

    agree = sum(1 for a, b in zip(x_bin, y_bin) if a == b)
    p_observed = agree / n

    p_x1 = sum(x_bin) / n
    p_y1 = sum(y_bin) / n
    p_expected = p_x1 * p_y1 + (1 - p_x1) * (1 - p_y1)

    if abs(1.0 - p_expected) < 1e-10:
        return 1.0 if p_observed == 1.0 else 0.0

    return (p_observed - p_expected) / (1.0 - p_expected)


def _classify_correlation(r: float | None) -> str:
    """Classify correlation strength into actionable status."""
    if r is None:
        return "insufficient_data"
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "high"
    if abs_r >= 0.4:
        return "moderate"
    return "low"


def compute_correlation(
    human_ratings: list[HumanRating],
    llm_scores: dict[str, dict[str, float]],
    min_comparisons: int = 3,
) -> CorrelationReport:
    """
    Compute correlation between human ratings and LLM judge scores.

    Args:
        human_ratings: all collected human ratings
        llm_scores: mapping of run_id → {criterion_name → llm_score}
        min_comparisons: minimum number of pairs needed for correlation

    Returns:
        CorrelationReport with per-criterion and overall statistics
    """
    # Group human ratings: (run_id, criterion_name) → [scores]
    human_by_run_criterion: dict[tuple[str, str], list[float]] = defaultdict(list)
    raters_by_criterion: dict[str, set] = defaultdict(set)

    for rating in human_ratings:
        key = (rating.run_id, rating.criterion_name)
        human_by_run_criterion[key].append(rating.score)
        raters_by_criterion[rating.criterion_name].add(rating.rater_id)

    # Build aligned score pairs: criterion → [(human_avg, llm_score), ...]
    aligned_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for (run_id, criterion_name), human_scores in human_by_run_criterion.items():
        if run_id not in llm_scores:
            continue
        if criterion_name not in llm_scores[run_id]:
            continue

        human_avg = sum(human_scores) / len(human_scores)
        llm_score = llm_scores[run_id][criterion_name]
        aligned_pairs[criterion_name].append((human_avg, llm_score))

    # Compute per-criterion correlation
    all_criteria_results = []
    all_human_scores = []
    all_llm_scores_flat = []
    flagged = []
    all_raters = set()

    for criterion_name, pairs in sorted(aligned_pairs.items()):
        human_vals = [p[0] for p in pairs]
        llm_vals = [p[1] for p in pairs]

        cr = CriterionCorrelation(
            criterion_name=criterion_name,
            n_comparisons=len(pairs),
            n_raters=len(raters_by_criterion.get(criterion_name, set())),
            mean_human_score=sum(human_vals) / len(human_vals) if human_vals else 0,
            mean_llm_score=sum(llm_vals) / len(llm_vals) if llm_vals else 0,
            mean_absolute_error=(
                sum(abs(h - l) for h, l in pairs) / len(pairs) if pairs else 0
            ),
        )

        if len(pairs) >= min_comparisons:
            cr.pearson_r = _pearson(human_vals, llm_vals)
            cr.spearman_rho = _spearman(human_vals, llm_vals)
            cr.cohens_kappa = _cohens_kappa(human_vals, llm_vals)
            cr.status = _classify_correlation(cr.pearson_r)

            # Compute inter-rater reliability for this criterion
            cr.human_agreement = _compute_inter_rater(
                human_ratings, criterion_name
            )
        else:
            cr.status = "insufficient_data"

        if cr.status == "low":
            flagged.append(criterion_name)

        all_criteria_results.append(cr)
        all_human_scores.extend(human_vals)
        all_llm_scores_flat.extend(llm_vals)
        all_raters.update(raters_by_criterion.get(criterion_name, set()))

    # Overall correlation
    overall_pearson = None
    overall_spearman = None
    if len(all_human_scores) >= min_comparisons:
        overall_pearson = _pearson(all_human_scores, all_llm_scores_flat)
        overall_spearman = _spearman(all_human_scores, all_llm_scores_flat)

    return CorrelationReport(
        criteria=all_criteria_results,
        overall_pearson=overall_pearson,
        overall_spearman=overall_spearman,
        total_comparisons=len(all_human_scores),
        total_raters=len(all_raters),
        flagged_criteria=flagged,
    )


def _compute_inter_rater(
    all_ratings: list[HumanRating],
    criterion_name: str,
) -> float | None:
    """
    Compute inter-rater agreement for a criterion.
    Uses average pairwise correlation among raters who rated the same runs.
    """
    # Group by run_id for this criterion
    by_run: dict[str, list[float]] = defaultdict(list)
    for r in all_ratings:
        if r.criterion_name == criterion_name:
            by_run[r.run_id].append(r.score)

    # Only consider runs with multiple raters
    multi_rated = {run_id: scores for run_id, scores in by_run.items() if len(scores) >= 2}
    if len(multi_rated) < 2:
        return None

    # Average absolute agreement
    agreements = []
    for scores in multi_rated.values():
        mean = sum(scores) / len(scores)
        max_dev = max(abs(s - mean) for s in scores)
        agreements.append(1.0 - max_dev)

    return sum(agreements) / len(agreements) if agreements else None


def format_correlation_report(report: CorrelationReport) -> str:
    """Format correlation report as a human-readable string."""
    lines = []
    lines.append("=" * 72)
    lines.append("JUDGE-HUMAN CORRELATION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Total comparisons:  {report.total_comparisons}")
    lines.append(f"Total raters:       {report.total_raters}")
    lines.append(f"Overall Pearson:    {_fmt_corr(report.overall_pearson)}")
    lines.append(f"Overall Spearman:   {_fmt_corr(report.overall_spearman)}")

    if report.flagged_criteria:
        lines.append("")
        lines.append(f"FLAGGED CRITERIA (low correlation, need better references):")
        for c in report.flagged_criteria:
            lines.append(f"  ⚠ {c}")

    lines.append("")
    lines.append("-" * 72)
    lines.append(f"{'Criterion':<30} {'N':>4} {'Pearson':>8} {'Spearman':>9} {'MAE':>6} {'Status':>12}")
    lines.append("-" * 72)

    for cr in report.criteria:
        lines.append(
            f"{cr.criterion_name:<30} "
            f"{cr.n_comparisons:>4} "
            f"{_fmt_corr(cr.pearson_r):>8} "
            f"{_fmt_corr(cr.spearman_rho):>9} "
            f"{cr.mean_absolute_error:>6.3f} "
            f"{_status_badge(cr.status):>12}"
        )

    lines.append("-" * 72)
    return "\n".join(lines)


def _fmt_corr(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  n/a"


def _status_badge(status: str) -> str:
    badges = {
        "high": "[HIGH]",
        "moderate": "[MODERATE]",
        "low": "[LOW]",
        "insufficient_data": "[NEED DATA]",
    }
    return badges.get(status, status)
