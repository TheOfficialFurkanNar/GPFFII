"""Unit tests for GPFF II physics core."""

import math

import pytest

from config.constants import AU, SOLAR_LUMINOSITY, SPEED_OF_LIGHT, as_float
from config.physics_func import (
    GPFFIIMode,
    ModifiedDispersionRelation,
    ObservedFlux,
    ClassicalFormulas,
    validate_mode_normalization,
    is_classical_flat_configuration,
)
from utils.exceptions import PhysicalParameterError


def test_flat_observed_flux_matches_geometric():
    classical = ClassicalFormulas()
    flux_calc = ObservedFlux()
    lum = as_float(SOLAR_LUMINOSITY)
    dist = as_float(AU)
    modes = [GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)]
    gf = flux_calc.observed_flux(lum, dist, modes)
    cf = classical.geometric_flux(lum, dist)
    assert gf == pytest.approx(cf, rel=1e-9)


def test_mode_normalization_rejects_invalid_sum():
    modes = [
        GPFFIIMode(a=0, i=0, w=0.5, lambda_i=0.0, kappa=0.0),
        GPFFIIMode(a=0, i=0, w=0.3, lambda_i=0.0, kappa=0.0),
    ]
    with pytest.raises(PhysicalParameterError):
        validate_mode_normalization(modes)


def test_liv_speed_branch_modes_normalize():
    w = 1.0 / 3.0
    modes = [
        GPFFIIMode(a=0, i=-1, w=w, lambda_i=-1.0, kappa=0.0),
        GPFFIIMode(a=0, i=0, w=w, lambda_i=0.0, kappa=0.0),
        GPFFIIMode(a=0, i=1, w=w, lambda_i=1.0, kappa=0.0),
        GPFFIIMode(a=-1, i=0, w=0.0, lambda_i=0.0, kappa=0.0),
    ]
    validate_mode_normalization(modes)


def test_is_classical_flat_configuration():
    flat = [
        GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0),
        GPFFIIMode(a=-1, i=0, w=0.0, lambda_i=0.0, kappa=0.0),
    ]
    assert is_classical_flat_configuration(flat) is True

    liv = [
        GPFFIIMode(a=0, i=-1, w=1 / 3, lambda_i=-1.0, kappa=0.0),
        GPFFIIMode(a=0, i=0, w=1 / 3, lambda_i=0.0, kappa=0.0),
        GPFFIIMode(a=0, i=1, w=1 / 3, lambda_i=1.0, kappa=0.0),
    ]
    assert is_classical_flat_configuration(liv) is False


def test_mdr_flat_mode_energy_is_pc():
    mdr = ModifiedDispersionRelation()
    classical = ClassicalFormulas()
    wl = 500e-9
    p = classical.photon_momentum_wavelength(wl)
    mode = GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)
    e = mdr.mode_energy(p, mode)
    c = as_float(SPEED_OF_LIGHT)
    assert e == pytest.approx(p * c, rel=1e-6)


def test_liv_mode_contribution_differs_from_luminal():
    """Subluminal branch with strong λ should deviate from pc² scaling at short λ."""
    mdr = ModifiedDispersionRelation()
    flux_calc = ObservedFlux(mdr)
    classical = ClassicalFormulas()
    wl = 1e-9  # 1 nm — MDR correction becomes visible
    p = classical.photon_momentum_wavelength(wl)
    c = as_float(SPEED_OF_LIGHT)

    luminal = GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)
    subluminal = GPFFIIMode(a=0, i=-1, w=1.0, lambda_i=2.0, kappa=0.0)

    luminal_contrib = flux_calc.mode_contribution(p, luminal)
    subluminal_contrib = flux_calc.mode_contribution(p, subluminal)
    assert luminal_contrib == pytest.approx(p * c ** 2, rel=1e-6)
    assert subluminal_contrib != luminal_contrib
    assert math.isfinite(subluminal_contrib)
