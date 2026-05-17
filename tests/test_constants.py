"""Tests for physical constants helpers."""

import pytest

from config.constants import as_float, PLANCK_CONSTANT, SPEED_OF_LIGHT


def test_as_float_from_quantity():
    assert as_float(PLANCK_CONSTANT) == pytest.approx(6.62607015e-34)


def test_as_float_passthrough():
    assert as_float(3.0) == 3.0


def test_speed_of_light_positive():
    assert as_float(SPEED_OF_LIGHT) > 0
