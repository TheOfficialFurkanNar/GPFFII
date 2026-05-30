# GPFF II Calculator

General Photon Flux Framework II (GPFF II) - A quantum-sector formulation for calculating observed astrophysical photon flux with Lorentz-invariance violation (LIV) corrections and curvature coupling terms.

## Overview

GPFF II is a theoretical physics framework that embeds photon propagation into a nine-component quantum state space. Modes are labeled by curvature sector `a ∈ {-1, 0, +1}` and speed branch `i ∈ {-1, 0, +1}`, with a modified dispersion relation (MDR) governing the energy of each mode.

This implementation provides an interactive visualization tool for exploring GPFF II calculations, validated against observational data from the Sun and the isolated neutron star RX J1856.5-3754.

## Features

- **Interactive Visualization**: Real-time flux calculations with matplotlib GUI
- **Modified Dispersion Relations**: Implements LIV corrections and curvature coupling
- **Multi-Mode Calculations**: Support for 9-component quantum state space
- **Validation Test Cases**: Solar constant and neutron star flux validation
- **Unit-Aware Calculations**: Dimensional analysis using pint (optional)
- **Structured Logging**: JSON-based logging with console and file output

## Physics Background

### Modified Dispersion Relation

The photon energy for mode (a, i) satisfies:

```
E²_{a,i} = p²c² + λ_i p^n c^n / M^(n-2) + a κ (ℏc)²
```

Where:
- `p` - Photon momentum
- `c` - Speed of light
- `λ_i` - LIV coupling coefficient (dimensionless for n=3)
- `n` - MDR power index (default 3 for quantum gravity)
- `M` - Quantum gravity mass scale (Planck mass)
- `κ` - Curvature coupling constant
- `ℏ` - Reduced Planck constant

### Observed Flux

The observed energy flux at a detector:

```
F_obs = cos(α) / (4π d_L²) × (1+z)² × Σ_{a,i} w_{a,i} · E_{a,i} · v_{a,i}
```

Where:
- `α` - Incidence angle (Lambert's law)
- `d_L` - Luminosity distance
- `z` - Cosmological redshift
- `w_{a,i}` - Occupation weight for mode (a, i)
- `E_{a,i}` - MDR energy of mode (a, i)
- `v_{a,i}` - Group velocity of mode (a, i)

## Installation

### Requirements

- Python 3.13+
- numpy
- matplotlib
- pint (optional, for unit support)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd GPFFIIIcalc

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib pint
```

## Usage

### Running the Interactive Visualization

```bash
python main.py
```

This launches an interactive matplotlib window with:
- **Sliders**: Temperature, Distance, LIV coupling
- **Radio Buttons**: Source type (Solar/Neutron Star), Curvature sector
- **Plots**: Flux comparison, Mode weights, Blackbody spectrum, Validation panel

### Validation Tests

The program automatically runs validation tests on startup:

1. **Solar Constant Test**
   - Predicted: ~1361 W/m²
   - Observed: 1360.8 ± 0.5 W/m²
   - Error: <0.1%

2. **RX J1856.5-3754 Neutron Star Test**
   - Predicted: ~2.6×10⁻¹⁴ W/m² (bolometric)
   - Observed: ~1.5×10⁻¹⁴ W/m² (band-limited)
   - Ratio: ~1.7 (expected due to band limitation)

## Key Classes

### GPFFIIVisualizer
Main visualization class handling UI, calculations, and plotting.

### GPFFIIMode
Represents a single quantum state mode with parameters:
- `a`: Curvature sector (-1, 0, +1)
- `i`: Speed branch (-1, 0, +1)
- `w`: Occupation weight [0, 1]
- `lambda_i`: LIV coupling coefficient
- `kappa`: Curvature coupling constant

### ModifiedDispersionRelation
Implements the MDR equation and group velocity calculations.

### ObservedFlux
Calculates observed flux with geometric prefactors and mode contributions.

### ClassicalFormulas
Classical physics formulas for validation and comparison.

## Configuration

### Physical Constants

Constants are defined in `config/constants.py` with unit support via pint. If pint is unavailable, constants fall back to raw float values.

### Mode Parameters

Mode parameters can be adjusted via the interactive GUI:
- Temperature: 3000 - 10000 K
- Distance: 0.1 - 100 AU (or 120 pc for neutron star)
- LIV Coupling (λ): -2 to +2
- Curvature Sector: a = -1, 0, or +1

## Logging

The framework uses structured JSON logging with dual transport:
- **Console**: Real-time output
- **File**: `log.jsonl` (append mode)

Log entries include:
- Sequence number
- Relative and absolute timestamps
- Device ID
- Log level
- Event type
- Optional data and error fields

## Limitations

1. **Monochromatic Formulation**: Core flux equation applies to single momentum mode; bolometric calculations require integration
2. **Superluminal Branch**: The i=+1 branch with λ₊₁>0 formally yields v > c, treated as perturbative correction
3. **Unconstrained Parameters**: LIV couplings λ±₁ and curvature coupling κ are free parameters
4. **Point Source Assumption**: Assumes isotropic point source with no interstellar absorption
5. **No Mode Mixing**: Occupation weights are static; dynamic mode mixing not implemented

## References

Based on the research paper:
> "The General Photon Flux Framework II (GPFF II): A Quantum-Sector Formulation for Observed Astrophysical Photon Flux"  
> Furkan Nar, April 2026

Key references:
- Amelino-Camelia, G. (2013). Quantum-Spacetime Phenomenology
- Liberati, S. (2013). Tests of Lorentz invariance: a 2013 update
- Kopp, G. & Lean, J. (2011). A new, lower value of total solar irradiance
- Walter, F. M. et al. (2010). Optical/UV counterpart of RX J1856.5-3754

## License

MIT

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include tests
- Documentation is updated accordingly

## Contact

Furkan Nar - furkannar168@hotmail.com
