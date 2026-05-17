"""Tests for K-means tier configuration."""

import pytest

from config.kmeans_config import resolve_hyperparameters


def test_tiny_tier_caps_k():
    hp = resolve_hyperparameters(10)
    assert hp["tier"] == "tiny"
    assert hp["n_clusters"] == 2
    assert hp["n_init"] == 3


def test_tiny_single_sample_k_is_one():
    hp = resolve_hyperparameters(1)
    assert hp["tier"] == "tiny"
    assert hp["n_clusters"] == 1


def test_small_tier():
    hp = resolve_hyperparameters(50)
    assert hp["tier"] == "small"
    assert hp["n_clusters"] == 3


def test_medium_tier():
    hp = resolve_hyperparameters(500)
    assert hp["tier"] == "medium"
    assert hp["n_clusters"] == 5
    assert hp["n_init"] == 10


def test_large_tier():
    hp = resolve_hyperparameters(5000)
    assert hp["tier"] == "large"
    assert hp["n_clusters"] == 8


def test_zero_samples_tiny_tier():
    hp = resolve_hyperparameters(0)
    assert hp["tier"] == "tiny"
    assert hp["n_clusters"] == 1


def test_negative_samples_raises():
    with pytest.raises(ValueError):
        resolve_hyperparameters(-1)


def test_use_scaler_and_fixed_algorithm():
    hp = resolve_hyperparameters(100)
    assert hp["use_scaler"] is True
    assert hp["algorithm"] == "lloyd"
    assert hp["random_state"] == 42
