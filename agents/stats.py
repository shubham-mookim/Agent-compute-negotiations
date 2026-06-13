"""
Statistical analysis utilities for experiment results.

All claims should go through these functions to ensure
proper confidence intervals and significance testing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class StatResult:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int

    def __repr__(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f} (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}], n={self.n})"


def describe(values: list[float]) -> StatResult:
    """Compute mean, std, and 95% confidence interval."""
    n = len(values)
    if n == 0:
        return StatResult(0, 0, 0, 0, 0)
    mean = sum(values) / n
    if n == 1:
        return StatResult(mean, 0, mean, mean, 1)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    # t-distribution approximation for 95% CI (use 1.96 for large n)
    t_val = 1.96 if n >= 30 else _t_critical(n - 1)
    margin = t_val * std / math.sqrt(n)
    return StatResult(
        mean=mean,
        std=std,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        n=n,
    )


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval.
    Returns (mean, ci_lower, ci_upper).
    """
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1
    return (sum(values) / n, means[lo_idx], means[hi_idx])


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Effect size: Cohen's d between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    var1 = sum((v - m1) ** 2 for v in group1) / (n1 - 1)
    var2 = sum((v - m2) ** 2 for v in group2) / (n2 - 1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def welch_t_test(group1: list[float], group2: list[float]) -> tuple[float, float]:
    """
    Welch's t-test (unequal variance).
    Returns (t_statistic, approximate_p_value).
    p-value is two-tailed, approximated using normal distribution for large n.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return (0.0, 1.0)
    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    var1 = sum((v - m1) ** 2 for v in group1) / (n1 - 1)
    var2 = sum((v - m2) ** 2 for v in group2) / (n2 - 1)
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return (0.0, 1.0)
    t_stat = (m1 - m2) / se
    # Approximate p-value using normal CDF for large sample sizes
    p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
    return (t_stat, p_value)


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient measuring inequality of distribution. 0=equal, 1=unequal."""
    if not values or all(v == 0 for v in values):
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def market_efficiency(actual_surplus: float, optimal_surplus: float) -> float:
    """Ratio of actual total surplus to theoretically optimal surplus."""
    if optimal_surplus == 0:
        return 1.0 if actual_surplus == 0 else 0.0
    return actual_surplus / optimal_surplus


def _norm_cdf(x: float) -> float:
    """Approximation of standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _t_critical(df: int, alpha: float = 0.025) -> float:
    """Rough t-critical value approximation for small df."""
    # Lookup for common df values at alpha=0.025 (two-tailed 95%)
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045,
    }
    if df in table:
        return table[df]
    # Find closest
    closest = min(table.keys(), key=lambda k: abs(k - df))
    return table[closest]
