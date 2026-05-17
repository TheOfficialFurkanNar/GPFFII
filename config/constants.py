"""
constants.py

Physical constants with dimensional analysis using pint.
Sources: CODATA 2022, NIST, IAU 2015, NASA
"""

__version__ = "0.1.1"

try:
    from pint import UnitRegistry
except ImportError:
    UnitRegistry = None

if UnitRegistry:
    ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    ureg.setup_matplotlib()  # Enable unit-aware plotting
    Q_ = ureg.Quantity

    # Additional unit definitions
    # Solar luminosity (L☉) - IAU 2015 nominal value
    ureg.define("solar_luminosity = 3.828e26 * watt = L_sun")
    # Parsec - IAU 2015 exact definition
    ureg.define("parsec = 3.0857e16 * meter = pc")
    # Light year - IAU 2015 exact definition
    ureg.define("light_year = 9.4607e15 * meter = ly")
    # Astronomical unit - IAU 2012 exact definition
    ureg.define("astronomical_unit = 1.496e11 * meter = au")

else:
    ureg = None

    def Q_(value, unit):
        return value


def as_float(quantity) -> float:
    """Extract a plain float from a pint Quantity or return numeric values unchanged."""
    if hasattr(quantity, "magnitude"):
        return float(quantity.magnitude)
    return float(quantity)

    import warnings
    warnings.warn(
        "pint library could not be loaded. Constants will be defined as raw floats.",
        ImportWarning,
        stacklevel=2
    )

# Fundamental constants - CODATA 2022
SPEED_OF_LIGHT            = Q_(2.99792458e8, "meter/second")  # Exact, CODATA 2018
PLANCK_CONSTANT           = Q_(6.62607015e-34, "joule*second")  # Exact, SI 2019
BOLTZMANN_CONSTANT        = Q_(1.380649e-23, "joule/kelvin")  # Exact, SI 2019
GRAVITATIONAL_CONSTANT    = Q_(6.67430e-11, "meter**3 / kilogram / second**2")  # CODATA 2022
STEFAN_BOLTZMANN_CONSTANT = Q_(5.670374419e-8, "watt / meter**2 / kelvin**4")  # CODATA 2017
WIEN_DISPLACEMENT         = Q_(2.897771955e-3, "meter*kelvin")  # CODATA 2018
ELECTRON_CHARGE           = Q_(1.602176634e-19, "coulomb")  # Exact, SI 2019
VACUUM_PERMITTIVITY       = Q_(8.854187817e-12, "farad/meter")  # CODATA 2018
GAS_CONSTANT              = Q_(8.314462618, "joule / mole / kelvin")  # CODATA 2018
THOMSON_CROSS_SECTION = Q_(6.6524587321e-29, "meter**2")  # CODATA 2018, σ_T

# Astronomical constants - IAU 2015
SOLAR_MASS                = Q_(1.98847e30, "kilogram")  # IAU 2015 nominal solar mass
SOLAR_RADIUS              = Q_(6.957e8, "meter")  # IAU 2015 nominal solar radius

# Particle masses - CODATA 2022
ELECTRON_MASS             = Q_(9.10938356e-31, "kilogram")  # CODATA 2022
PROTON_MASS               = Q_(1.6726219e-27, "kilogram")  # CODATA 2022
AVOGADRO_CONSTANT         = Q_(6.02214076e23, "1/mole")  # Exact, SI 2019

# GPFF II specific constants - Derived from fundamental constants
PLANCK_MASS = Q_(2.176434e-8, "kilogram")  # M_P = sqrt(ℏc/G), derived
REDUCED_PLANCK_CONSTANT = Q_(1.054571817e-34, "joule*second")  # ℏ = h/(2π), derived

# LIV coupling parameters (dimensionless, to be constrained by observations)
LIV_COUPLING_SUB = -1.0  # λ_{-1} < 0 (subluminal)
LIV_COUPLING_LUM = 0.0    # λ_0 = 0 (luminal)  
LIV_COUPLING_SUP = 1.0    # λ_{+1} > 0 (superluminal)

# MDR parameters
MDR_POWER_INDEX = 3  # n = 3 for quantum gravity

# Curvature coupling constant (to be constrained, m^-2)
CURVATURE_COUPLING = Q_(0.0, "1/meter**2")

# Astronomical units for convenience - IAU 2015
AU = Q_(1.496e11, "meter")  # IAU 2012 exact definition
PC = Q_(3.0857e16, "meter")  # IAU 2015 exact definition
SOLAR_LUMINOSITY = Q_(3.828e26, "watt")  # IAU 2015 nominal solar luminosity

# Mathematical constant
PI = ureg.pi if ureg else 3.141592653589793

__all__ = [
    "__version__",
    "ureg", "Q_", "as_float",
    # Fundamental constants
    "SPEED_OF_LIGHT",
    "PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "GRAVITATIONAL_CONSTANT",
    "STEFAN_BOLTZMANN_CONSTANT",
    "WIEN_DISPLACEMENT",
    "ELECTRON_CHARGE",
    "VACUUM_PERMITTIVITY",
    "GAS_CONSTANT",
    "THOMSON_CROSS_SECTION",
    # Astronomical constants
    "SOLAR_MASS",
    "SOLAR_RADIUS",
    # Particle masses
    "ELECTRON_MASS",
    "PROTON_MASS",
    "AVOGADRO_CONSTANT",
    # GPFF II specific constants
    "PLANCK_MASS",
    "REDUCED_PLANCK_CONSTANT",
    "LIV_COUPLING_SUB",
    "LIV_COUPLING_LUM", 
    "LIV_COUPLING_SUP",
    "MDR_POWER_INDEX",
    "CURVATURE_COUPLING",
    "AU",
    "PC",
    "SOLAR_LUMINOSITY",
    # Mathematical constant
    "PI",
]