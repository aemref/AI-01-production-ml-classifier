"""Deterministic feature-shift diagnostics for the educational classifier."""

from __future__ import annotations

from math import isfinite
from typing import Sequence


def feature_shift(reference: Sequence[float], candidate: Sequence[float]) -> dict:
    """Measure signed mean change in units of the reference population SD.

    A constant reference cannot define a standardized change. Return ``None``
    in that case instead of emitting infinity into JSON reports.
    """
    if not reference or not candidate:
        raise ValueError("Feature samples must be nonempty")
    values = [*reference, *candidate]
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("Feature samples must contain only finite values")

    reference_mean = sum(reference) / len(reference)
    candidate_mean = sum(candidate) / len(candidate)
    reference_sd = (
        sum((value - reference_mean) ** 2 for value in reference) / len(reference)
    ) ** 0.5
    difference = candidate_mean - reference_mean
    standardized = difference / reference_sd if reference_sd else None
    return {
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "reference_sd": reference_sd,
        "mean_change": difference,
        "standardized_mean_change": standardized,
    }
