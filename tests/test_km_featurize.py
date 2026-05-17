"""Smoke tests for km.py featurization."""

import numpy as np
import pytest

from km import featurize, feature_names


def test_featurize_shape():
    entries = [
        {"seq": 1, "level": "INFO", "t": 0.0, "event": "validation_start",
         "data": {"source": "gpffii_sim"}},
        {"seq": 2, "level": "DEBUG", "t": 1.5, "event": "flux_calculation_progress",
         "data": {"progress_pct": 50.0, "correction": 1.0}},
    ]
    X, meta = featurize(entries)
    assert X.shape == (2, len(feature_names()))
    assert meta[0]["event"] == "validation_start"
    assert X[1, 2] == pytest.approx(1.5)  # delta_t for second row
    assert np.isfinite(X).all()
