"""
K-means hyperparameter tiers and paths for log.jsonl clustering.

Tier presets are selected by training sample count via resolve_hyperparameters().
"""

from __future__ import annotations

LOG_PATH = "log.jsonl"
MODEL_PATH = "artifacts/kmeans_model.joblib"
PREDICTIONS_PATH = "artifacts/log_clusters.jsonl"
HEATMAP_PATH = "artifacts/cluster_heatmap.png"

LEVEL_ORDINAL = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

KNOWN_EVENTS = [
    "validation_start",
    "validation_complete",
    "flux_calculation_start",
    "flux_calculation_progress",
    "error_occurred",
    "other",
]

NUMERIC_DATA_KEYS = [
    "temperature",
    "distance",
    "progress_pct",
    "classical_flux",
    "gpff_flux",
    "correction",
    "wavelength_nm",
    "solar_error_pct",
    "neutron_star_ratio",
]

KMEANS_FIXED = {
    "algorithm": "lloyd",
    "random_state": 42,
}

TIER_PRESETS = {
    "tiny": {"n_clusters": 2, "n_init": 3, "max_iter": 100},
    "small": {"n_clusters": 3, "n_init": 5, "max_iter": 200},
    "medium": {"n_clusters": 5, "n_init": 10, "max_iter": 300},
    "large": {"n_clusters": 8, "n_init": 10, "max_iter": 500},
}

TINY_MAX = 30
SMALL_MAX = 200
MEDIUM_MAX = 2000


def _tier_name(n_samples: int) -> str:
    if n_samples < TINY_MAX:
        return "tiny"
    if n_samples < SMALL_MAX:
        return "small"
    if n_samples < MEDIUM_MAX:
        return "medium"
    return "large"


def resolve_hyperparameters(n_samples: int) -> dict:
    """
    Return sklearn KMeans kwargs plus metadata for the given sample count.

    Parameters
    ----------
    n_samples : int
        Number of log lines used for training.

    Returns
    -------
    dict
        Keys: tier, use_scaler, n_clusters, n_init, max_iter, algorithm, random_state
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")

    tier = _tier_name(n_samples)
    preset = TIER_PRESETS[tier]
    k = preset["n_clusters"]

    if n_samples == 0:
        k = 1
    else:
        k = min(k, n_samples)
        k = max(1, k)

    return {
        "tier": tier,
        "use_scaler": True,
        "n_clusters": k,
        "n_init": preset["n_init"],
        "max_iter": preset["max_iter"],
        "algorithm": KMEANS_FIXED["algorithm"],
        "random_state": KMEANS_FIXED["random_state"],
    }
