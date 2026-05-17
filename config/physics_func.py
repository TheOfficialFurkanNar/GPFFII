"""GPFF II Physics Formulas

Implementation of the General Photon Flux Framework II formulas from the research paper.
Includes modified dispersion relations, group velocities, and observed flux calculations.
"""

import numpy as np
import math
from config.constants import (
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    GRAVITATIONAL_CONSTANT,
    STEFAN_BOLTZMANN_CONSTANT,
    WIEN_DISPLACEMENT,
    as_float,
)


class GPFFIIMode:
    """Single mode in the 9-component quantum state space"""

    def __init__(self, a: int, i: int, w: float, lambda_i: float = 0.0, kappa: float = 0.0):
        """
        Initialize a GPFF II mode

        Parameters
        ----------
        a : int - Curvature sector (-1, 0, +1)
        i : int - Speed branch (-1, 0, +1)
        w : float - Occupation weight [0, 1]
        lambda_i : float - LIV coupling coefficient
        kappa : float - Curvature coupling constant (m^-2)
        """
        self.a = a
        self.i = i
        self.w = w
        self.lambda_i = lambda_i
        self.kappa = kappa

    def validate(self):
        """Validate mode parameters"""
        from utils.exceptions import ModeValidationError, PhysicalParameterError
        
        if self.a not in [-1, 0, 1]:
            raise ModeValidationError(f"Curvature sector a must be -1, 0, or 1, got {self.a}")
        if self.i not in [-1, 0, 1]:
            raise ModeValidationError(f"Speed branch i must be -1, 0, or 1, got {self.i}")
        if not (0 <= self.w <= 1):
            raise PhysicalParameterError(f"Occupation weight w must be in [0, 1], got {self.w}")


def is_classical_flat_configuration(modes: list) -> bool:
    """
    True when a single flat luminal mode (a=0, i=0, λ=0, κ=0) carries all weight.
    Other modes may be present with w=0 for visualization bookkeeping.
    """
    active = [m for m in modes if m.w > 0]
    if len(active) != 1:
        return False
    m = active[0]
    return (
        m.a == 0
        and m.i == 0
        and m.lambda_i == 0.0
        and m.kappa == 0.0
        and abs(m.w - 1.0) < 1e-9
    )


def validate_mode_normalization(modes: list):
    """
    Validate that mode weights sum to 1 (normalization condition from paper).
    
    Parameters
    ----------
    modes : list[GPFFIIMode] - List of modes to validate
    
    Raises
    ------
    PhysicalParameterError - If weights do not sum to 1 (within tolerance)
    """
    from utils.exceptions import PhysicalParameterError
    
    total_weight = sum(m.w for m in modes)
    tolerance = 1e-6
    
    if abs(total_weight - 1.0) > tolerance:
        raise PhysicalParameterError(
            f"Mode weights must sum to 1.0 (normalization condition), got {total_weight:.6f}"
        )


class ModifiedDispersionRelation:
    """Modified Dispersion Relation for GPFF II"""

    def __init__(self, n: int = 3, M=None):
        """
        Initialize MDR parameters

        Parameters
        ----------
        n : int - MDR power index (default 3 for quantum gravity)
        M : Quantity - Quantum gravity mass scale (default Planck mass)
        """
        self.n = n
        self.M = M if M is not None else self._planck_mass()

    def _planck_mass(self):
        """Calculate Planck mass in kg"""
        # M_P = sqrt(ℏc/G) ≈ 1.22×10^19 GeV/c²
        hbar = as_float(PLANCK_CONSTANT) / (2 * math.pi)
        c = as_float(SPEED_OF_LIGHT)
        G = as_float(GRAVITATIONAL_CONSTANT)
        return math.sqrt(hbar * c / G)

    def mode_energy(self, p: float, mode: GPFFIIMode) -> float:
        """
        Calculate mode energy from modified dispersion relation

        E²_a,i = p²c² + λ_i p^n c^n / M^(n-2) + a κ (ℏc)²

        Parameters
        ----------
        p : float - Photon momentum (kg m/s)
        mode : GPFFIIMode - Mode parameters

        Returns
        -------
        float - Mode energy (J)
        """
        c = as_float(SPEED_OF_LIGHT)
        hbar = as_float(PLANCK_CONSTANT) / (2 * math.pi)
        M_val = as_float(self.M)

        # Standard term: p²c²
        standard_term = (p * c) ** 2

        # LIV correction: λ_i p^n c^n / M^(n-2)
        liv_term = mode.lambda_i * (p ** self.n) * (c ** self.n) / (M_val ** (self.n - 2))

        # Curvature term: a κ (ℏc)²
        curvature_term = mode.a * mode.kappa * (hbar * c) ** 2

        E_squared = standard_term + liv_term + curvature_term

        if E_squared < 0:
            raise ValueError(f"Negative E²: {E_squared}")

        return math.sqrt(E_squared)

    def group_velocity(self, p: float, mode: GPFFIIMode) -> float:
        """
        Calculate group velocity for a mode

        v_a,i = (2pc² + n λ_i p^(n-1) c^n / M^(n-2)) / (2 E_a,i)

        Parameters
        ----------
        p : float - Photon momentum (kg m/s)
        mode : GPFFIIMode - Mode parameters

        Returns
        -------
        float - Group velocity (m/s)
        """
        c = as_float(SPEED_OF_LIGHT)
        M_val = as_float(self.M)

        # Calculate energy first
        E = self.mode_energy(p, mode)

        # Numerator: 2pc² + n λ_i p^(n-1) c^n / M^(n-2)
        numerator = 2 * p * c ** 2 + self.n * mode.lambda_i * (p ** (self.n - 1)) * (c ** self.n) / (
                    M_val ** (self.n - 2))

        return numerator / (2 * E)


class ObservedFlux:
    """Calculate observed photon flux using GPFF II framework"""

    def __init__(self, mdr: ModifiedDispersionRelation = None):
        """
        Initialize flux calculator

        Parameters
        ----------
        mdr : ModifiedDispersionRelation - MDR instance
        """
        self.mdr = mdr if mdr is not None else ModifiedDispersionRelation()

    def angular_incidence(self, alpha: float) -> float:
        """
        Calculate angular incidence factor (Lambert's law)

        Parameters
        ----------
        alpha : float - Incidence angle (radians)

        Returns
        -------
        float - cos(α) factor
        """
        return math.cos(alpha)

    def luminosity_distance(self, distance: float, redshift: float = 0.0) -> float:
        """
        Calculate luminosity distance

        For small redshifts (z << 1): d_L ≈ d(1 + z)

        Parameters
        ----------
        distance : float - Comoving distance (m)
        redshift : float - Cosmological redshift

        Returns
        -------
        float - Luminosity distance (m)
        """
        return distance * (1 + redshift)

    def redshift_factor(self, redshift: float) -> float:
        """
        Calculate redshift correction factor

        Parameters
        ----------
        redshift : float - Cosmological redshift

        Returns
        -------
        float - (1 + z)² factor
        """
        return (1 + redshift) ** 2

    def mode_contribution(self, p: float, mode: GPFFIIMode) -> float:
        """
        Calculate single mode contribution to flux

        w_a,i · E_a,i · v_a,i   [units: J · m/s = W·m]

        Parameters
        ----------
        p : float - Photon momentum (kg m/s)
        mode : GPFFIIMode - Mode parameters

        Returns
        -------
        float - Mode contribution (W·m)
        """
        E = self.mdr.mode_energy(p, mode)
        v = self.mdr.group_velocity(p, mode)
        return mode.w * E * v

    def observed_flux(self, luminosity: float, distance: float, modes: list,
                      alpha: float = 0.0, redshift: float = 0.0,
                      p: float = None) -> float:
        """
        Calculate observed energy flux per Eq. 6 of the paper:

            F_obs = cos(α) / (4π d_L²) × (1+z)² × Σ_{a,i} w_{a,i} · E_{a,i} · v_{a,i}

        For the flat/luminal mode (a=0, i=0, w=1, λ=0, κ=0) this reduces to
        the standard geometric flux L/(4πd²), because:
            E_{0,0} = pc,  v_{0,0} = c  →  w·E·v = pc²
        and integrating pc² over the Planck spectrum gives L (bolometric).
        The shortcut below applies that identity directly so the caller can
        pass a bolometric luminosity without needing a monochromatic p.

        For all other mode configurations a characteristic photon momentum p
        must be supplied (or is estimated from a 100 nm reference wavelength).
        The sum Σ w·E·v has units W·m; multiplied by the prefactor (m⁻²) it
        gives W/m² with no further luminosity factor — matching Eq. 6 exactly.

        Parameters
        ----------
        luminosity : float - Source luminosity (W); used only for flat/luminal shortcut
        distance   : float - Comoving distance (m)
        modes      : list[GPFFIIMode]
        alpha      : float - Incidence angle (radians), default 0
        redshift   : float - Cosmological redshift, default 0
        p          : float - Photon momentum (kg·m/s); required for non-flat modes

        Returns
        -------
        float - Observed flux (W/m²)
        """
        # --- geometric prefactor: cos(α) / (4π d_L²) ---
        cos_alpha = self.angular_incidence(alpha)
        d_L = self.luminosity_distance(distance, redshift)
        prefactor = cos_alpha / (4 * math.pi * d_L ** 2)

        # --- redshift correction: (1+z)² ---
        z_factor = self.redshift_factor(redshift)

        # --- flat/luminal shortcut (a=0, i=0, w=1, λ=0, κ=0) ---
        # Eq. 6 collapses to the inverse-square law for bolometric luminosity.
        if (len(modes) == 1
                and modes[0].a == 0
                and modes[0].i == 0
                and modes[0].lambda_i == 0.0
                and modes[0].kappa == 0.0):
            return prefactor * z_factor * luminosity

        # --- general case: directly implement Eq. 6 ---
        # Σ w_{a,i} · E_{a,i} · v_{a,i}  [W·m]
        # A characteristic momentum is needed to evaluate E and v.
        if p is None:
            # Fall back to p = h/λ at λ = 100 nm as a reference scale.
            h = as_float(PLANCK_CONSTANT)
            c = as_float(SPEED_OF_LIGHT)
            p = h / 100e-9  # kg·m/s

        mode_sum = sum(self.mode_contribution(p, mode) for mode in modes)  # W·m

        # F = prefactor [m⁻²] × (1+z)² × Σ wEv [W·m]  →  W/m²
        return prefactor * z_factor * mode_sum


class ClassicalFormulas:
    """Classical physics formulas for validation and comparison"""

    @staticmethod
    def wien_peak_wavelength(temperature: float) -> float:
        """
        Calculate peak emission wavelength using Wien's displacement law

        λ_peak = b/T where b = 2.898×10^-3 m·K

        Parameters
        ----------
        temperature : float - Temperature (K)

        Returns
        -------
        float - Peak wavelength (m)
        """
        b = as_float(WIEN_DISPLACEMENT)
        return b / temperature

    @staticmethod
    def photon_energy_wavelength(wavelength: float) -> float:
        """
        Calculate photon energy from wavelength

        E = hc/λ

        Parameters
        ----------
        wavelength : float - Wavelength (m)

        Returns
        -------
        float - Photon energy (J)
        """
        h = as_float(PLANCK_CONSTANT)
        c = as_float(SPEED_OF_LIGHT)
        return h * c / wavelength

    @staticmethod
    def photon_momentum_wavelength(wavelength: float) -> float:
        """
        Calculate photon momentum from wavelength

        p = h/λ

        Parameters
        ----------
        wavelength : float - Wavelength (m)

        Returns
        -------
        float - Photon momentum (kg·m/s)
        """
        h = as_float(PLANCK_CONSTANT)
        return h / wavelength

    @staticmethod
    def stefan_boltzmann_luminosity(temperature: float, radius: float) -> float:
        """
        Calculate blackbody luminosity using Stefan-Boltzmann law

        L = 4πR²σT⁴

        Parameters
        ----------
        temperature : float - Temperature (K)
        radius : float - Radius (m)

        Returns
        -------
        float - Luminosity (W)
        """
        sigma = as_float(STEFAN_BOLTZMANN_CONSTANT)
        return 4 * math.pi * radius ** 2 * sigma * temperature ** 4

    @staticmethod
    def geometric_flux(luminosity: float, distance: float) -> float:
        """
        Calculate geometric flux (inverse square law)

        F = L/(4πd²)

        Parameters
        ----------
        luminosity : float - Luminosity (W)
        distance : float - Distance (m)

        Returns
        -------
        float - Flux (W/m²)
        """
        return luminosity / (4 * math.pi * distance ** 2)