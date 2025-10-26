"""Lorentzian distance based KNN implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import Settings


def lorentzian_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    diff = np.abs(vec_a - vec_b)
    return float(np.sum(np.log1p(diff)))


@dataclass
class Neighbor:
    index: int
    distance: float
    label: float


def compute_knn_predictions(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    settings: Settings,
) -> pd.Series:
    """Compute KNN prediction score for each bar."""
    feature_cols = feature_matrix.columns[: settings.feature_count]
    predictions: List[float] = []
    max_bars = settings.max_bars_back
    neighbors = settings.neighbors_count

    feature_values = feature_matrix[feature_cols].to_numpy(dtype=float)
    label_values = labels.to_numpy(dtype=float)

    for idx in range(len(feature_matrix)):
        current_vec = feature_values[idx]
        if np.any(np.isnan(current_vec)):
            predictions.append(0.0)
            continue

        start = max(0, idx - max_bars)
        candidates: List[Neighbor] = []

        for j in range(start, idx):
            if (idx - j) % 4 != 0:
                continue
            past_vec = feature_values[j]
            if np.any(np.isnan(past_vec)):
                continue
            label = label_values[j]
            if math.isnan(label):
                continue
            distance = lorentzian_distance(current_vec, past_vec)
            candidates.append(Neighbor(index=j, distance=distance, label=label))

        if not candidates:
            predictions.append(0.0)
            continue

        candidates.sort(key=lambda item: item.distance)
        top_neighbors = candidates[:neighbors]
        score = sum(item.label for item in top_neighbors)
        predictions.append(score)

    return pd.Series(predictions, index=feature_matrix.index, name="prediction")
