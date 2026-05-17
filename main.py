"""Main script for General Photon Flux Framework II (GPFF II)

Interactive visualization and validation of the GPFF II framework for astrophysical photon flux calculations.
Includes solar constant and neutron star validation test cases with matplotlib visualization.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec

# Import GPFF II modules
from config.physics_func import (
    GPFFIIMode,
    ModifiedDispersionRelation,
    ObservedFlux,
    ClassicalFormulas,
    validate_mode_normalization,
    is_classical_flat_configuration,
)
from config.constants import (
    AU,
    PC,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    ELECTRON_CHARGE,
    SOLAR_LUMINOSITY,
    as_float,
)

# Import utility modules
from utils import (
    StructuredLogger,
    ErrorHandler,
)


class GPFFIIVisualizer:
    """Interactive visualization for GPFF II framework"""

    def __init__(self):
        self.fig = None
        self.mdr = ModifiedDispersionRelation()
        self.flux_calc = ObservedFlux(self.mdr)
        self.classical = ClassicalFormulas()

        # Initialize error handling
        self.logger = StructuredLogger(device_id="gpffii_sim", min_level="INFO")
        self.error_handler = ErrorHandler(self.logger)

        self.current_mode = GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)
        self.curvature_sector = 0  # a parameter: -1, 0, or 1
        self.source_type = "solar"
        self.temperature = 5778
        self.radius = 6.957e8
        self.distance = 1.496e11
        self.lambda_sub = 0.0
        self.lambda_sup = 0.0

        # Track colorbar so we can remove it on redraw
        self._cbar = None

        # Calculation guards
        self._calculating    = False
        self._last_calc_time = 0.0

    # ------------------------------------------------------------------
    # Figure / control setup
    # ------------------------------------------------------------------

    def setup_figure(self):
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('GPFF II: General Photon Flux Framework II',
                          fontsize=16, fontweight='bold')

        # Reserve bottom 28 % for controls, top 72 % for plots
        gs = GridSpec(2, 4, figure=self.fig,
                      top=0.90, bottom=0.32,
                      hspace=0.45, wspace=0.40)

        self.ax_flux       = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_modes      = self.fig.add_subplot(gs[0,   2])
        self.ax_spectrum   = self.fig.add_subplot(gs[1,   2])
        self.ax_validation = self.fig.add_subplot(gs[0:2, 3])

        self.ax_flux.set_xlabel('Wavelength (nm)')
        self.ax_flux.set_ylabel('Flux (W/m²)')
        self.ax_flux.set_title('Observed Flux Comparison')
        self.ax_flux.grid(True, alpha=0.3)

        self.ax_modes.set_title('Mode Occupation Weights')
        self.ax_modes.set_xlabel('Speed Branch (i)')
        self.ax_modes.set_ylabel('Curvature Sector (a)')

        self.ax_spectrum.set_xlabel('Photon Energy (eV)')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title('Blackbody Spectrum')
        self.ax_spectrum.grid(True, alpha=0.3)

        self.ax_validation.axis('off')

    def create_controls(self):
        # ---- sliders ----
        ax_temp = plt.axes([0.10, 0.20, 0.35, 0.025])
        self.slider_temp = Slider(ax_temp, 'Temperature (K)', 3000, 10000,
                                  valinit=self.temperature, valstep=100)
        self.slider_temp.on_changed(self.update_temperature)

        ax_dist = plt.axes([0.10, 0.14, 0.35, 0.025])
        distance_val = self.distance / self._au()
        self.slider_dist = Slider(ax_dist, 'Distance (AU)', 0.1, 100,
                                  valinit=distance_val, valstep=0.1,
                                  valfmt='%.1f AU')
        self.slider_dist.on_changed(self.update_distance)

        ax_liv = plt.axes([0.10, 0.08, 0.35, 0.025])
        self.slider_liv = Slider(ax_liv, 'LIV Coupling (λ)', -2, 2,
                                 valinit=0.0, valstep=0.1)
        self.slider_liv.on_changed(self.update_liv)

        # ---- radio buttons ----
        ax_source = plt.axes([0.52, 0.06, 0.22, 0.16])
        ax_source.set_title('Source', fontsize=9, pad=2)
        self.radio_source = RadioButtons(
            ax_source,
            ('Solar', 'RX J1856.5-3754 (Neutron Star)'),
            activecolor='steelblue'
        )
        for lbl in self.radio_source.labels:
            lbl.set_fontsize(9)
        self.radio_source.on_clicked(self.update_source_type)

        ax_curvature = plt.axes([0.10, 0.01, 0.35, 0.06])
        ax_curvature.set_title('Curvature Sector (a)', fontsize=9, pad=2)
        self.radio_curvature = RadioButtons(
            ax_curvature,
            ('a = -1 (Negative)', 'a = 0 (Flat)', 'a = +1 (Positive)'),
            activecolor='steelblue'
        )
        for lbl in self.radio_curvature.labels:
            lbl.set_fontsize(8)
        self.radio_curvature.on_clicked(self.update_curvature_sector)

        # ---- buttons ----
        ax_calc  = plt.axes([0.80, 0.15, 0.10, 0.04])
        ax_reset = plt.axes([0.80, 0.09, 0.10, 0.04])
        self.btn_calc  = Button(ax_calc,  'Calculate')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_calc.on_clicked(self.calculate_flux)
        self.btn_reset.on_clicked(self.reset_parameters)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _au(self):
        return as_float(AU)

    def _pc(self):
        return as_float(PC)

    def _const(self, c):
        return as_float(c)

    # ------------------------------------------------------------------
    # Slider / radio callbacks
    # ------------------------------------------------------------------

    def update_temperature(self, val):
        self.temperature = val

    def update_distance(self, val):
        self.distance = val * self._au()

    def update_liv(self, val):
        self.current_mode.lambda_i = val
        # Update subluminal and superluminal coupling values
        self.lambda_sub = -abs(val) if val != 0 else 0.0
        self.lambda_sup = abs(val) if val != 0 else 0.0

    def update_curvature_sector(self, label):
        """Update curvature sector based on user selection"""
        if label == 'a = -1 (Negative)':
            self.curvature_sector = -1
        elif label == 'a = 0 (Flat)':
            self.curvature_sector = 0
        elif label == 'a = +1 (Positive)':
            self.curvature_sector = 1

    def update_source_type(self, label):
        # Disconnect sliders before programmatic set_val() calls so they
        # don't fire their on_changed callbacks and cascade into a
        # spurious recalculation.
        self.slider_temp.disconnect_events()
        self.slider_dist.disconnect_events()

        if label == 'Solar':
            self.source_type = "solar"
            self.temperature  = 5778
            self.radius       = 6.957e8
            self.distance     = 1.496e11
        else:
            self.source_type = "neutron"
            self.temperature  = 7.08e5
            self.radius       = 5.0e3
            self.distance     = 120 * self._pc()

        self.slider_temp.set_val(self.temperature)
        self.slider_dist.set_val(self.distance / self._au())

        # Reconnect after the programmatic updates are done
        self.slider_temp.on_changed(self.update_temperature)
        self.slider_dist.on_changed(self.update_distance)

    def reset_parameters(self, event):
        # Same pattern: disconnect → mutate → reconnect
        self.slider_temp.disconnect_events()
        self.slider_dist.disconnect_events()
        self.slider_liv.disconnect_events()

        self.source_type  = "solar"
        self.temperature  = 5778
        self.radius       = 6.957e8
        self.distance     = 1.496e11
        self.current_mode = GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)
        self.curvature_sector = 0
        self.lambda_sub = 0.0
        self.lambda_sup = 0.0

        self.slider_temp.set_val(self.temperature)
        self.slider_dist.set_val(self.distance / self._au())
        self.slider_liv.set_val(0.0)
        self.radio_source.set_active(0)
        self.radio_curvature.set_active(1)  # a = 0 (Flat) is index 1

        self.slider_temp.on_changed(self.update_temperature)
        self.slider_dist.on_changed(self.update_distance)
        self.slider_liv.on_changed(self.update_liv)

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def _clear_axes(self):
        """Clear all axes and restore labels/titles"""
        self.ax_flux.cla()
        self.ax_spectrum.cla()
        self.ax_validation.cla()
        self.ax_validation.axis('off')

        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self.ax_modes.cla()

        # Restore labels/titles after cla()
        self.ax_flux.set_xlabel('Wavelength (nm)')
        self.ax_flux.set_ylabel('Flux (W/m²)')
        self.ax_flux.set_title(f'Flux Comparison: {self.source_type.capitalize()} Source')
        self.ax_flux.grid(True, alpha=0.3)

        self.ax_spectrum.set_xlabel('Photon Energy (eV)')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title('Blackbody Spectrum')
        self.ax_spectrum.grid(True, alpha=0.3)

        self.ax_modes.set_title('Mode Weights')
        self.ax_modes.set_xlabel('Speed (i)')
        self.ax_modes.set_ylabel('Curvature (a)')

    def _build_mode_list(self):
        """Build and validate the mode list based on current parameters"""
        modes = []
        sector = self.curvature_sector
        liv = self.current_mode.lambda_i

        if abs(liv) <= 0.01:
            modes.append(GPFFIIMode(a=sector, i=0, w=1.0, lambda_i=0.0, kappa=0.0))
        else:
            # Split occupation across speed branches; λ on sub/super from slider mapping
            w = 1.0 / 3.0
            modes.extend([
                GPFFIIMode(a=sector, i=-1, w=w, lambda_i=self.lambda_sub, kappa=0.0),
                GPFFIIMode(a=sector, i=0,  w=w, lambda_i=0.0, kappa=0.0),
                GPFFIIMode(a=sector, i=1,  w=w, lambda_i=self.lambda_sup, kappa=0.0),
            ])

        # Zero-weight placeholders for the other curvature sectors (mode-weight heatmap)
        for a in (-1, 0, 1):
            if a != sector:
                modes.append(GPFFIIMode(a=a, i=0, w=0.0, lambda_i=0.0, kappa=0.0))

        for mode in modes:
            mode.validate()
        validate_mode_normalization(modes)

        return modes

    def _calculate_flux_values(self, wavelengths, modes, h, c, k_B):
        """Calculate flux values for all wavelengths"""
        def planck(wl):
            exp = h * c / (wl * k_B * self.temperature)
            if exp > 709:       # prevent overflow; physically correct to return 0
                return 0.0
            result = (2 * h * c**2 / wl**5) / (np.exp(exp) - 1)
            return result if np.isfinite(result) else 0.0

        ang_size = (self.radius / self.distance) ** 2
        flux_classical, flux_gpff, energies, intensities = [], [], [], []

        for idx, wl in enumerate(wavelengths):
            B  = planck(wl)
            sf = B * ang_size
            flux_classical.append(sf)

            correction = 1.0
            if not is_classical_flat_configuration(modes):
                try:
                    p = self.classical.photon_momentum_wavelength(wl)
                    s = sum(self.flux_calc.mode_contribution(p, m) for m in modes)
                    correction = s / (p * c ** 2)
                except Exception as e:
                    self.error_handler.handle(
                        e, context={"wavelength": wl, "source": self.source_type}
                    )

            flux_gpff.append(sf * correction)
            energies.append(
                self.classical.photon_energy_wavelength(wl) / self._const(ELECTRON_CHARGE)
            )
            intensities.append(B)

            if idx % 10 == 0:
                self.logger.debug("flux_calculation_progress", {
                    "wavelength_nm":  wl * 1e9,
                    "progress_pct":   idx / len(wavelengths) * 100,
                    "classical_flux": sf,
                    "gpff_flux":      sf * correction,
                    "correction":     correction,
                })

        return flux_classical, flux_gpff, energies, intensities

    def _plot_results(self, wavelengths, flux_classical, flux_gpff, energies, intensities, modes):
        """Plot all results on the axes"""
        # Flux plot
        self.ax_flux.loglog(wavelengths * 1e9, flux_classical,
                            'b-',  label='Classical', linewidth=2)
        self.ax_flux.loglog(wavelengths * 1e9, flux_gpff,
                            'r--', label='GPFF II',   linewidth=2)
        self.ax_flux.legend()

        # Mode weight matrix
        mode_matrix = np.zeros((3, 3))
        for m in modes:
            if m.a in (-1, 0, 1) and m.i in (-1, 0, 1):
                mode_matrix[m.a + 1, m.i + 1] = m.w

        im = self.ax_modes.imshow(mode_matrix, cmap='viridis',
                                   aspect='auto', vmin=0, vmax=1)
        self.ax_modes.set_xticks([0, 1, 2])
        self.ax_modes.set_yticks([0, 1, 2])
        self.ax_modes.set_xticklabels(['-1', '0', '+1'])
        self.ax_modes.set_yticklabels(['-1', '0', '+1'])

        self._cbar = self.fig.colorbar(im, ax=self.ax_modes,
                                        fraction=0.046, pad=0.04)
        self._cbar.set_label('Weight', rotation=270, labelpad=12)

        # Blackbody spectrum
        self.ax_spectrum.loglog(energies, intensities, 'g-', linewidth=2)

        # Add flux comparison text on the right side
        peak_flux_classical = max(flux_classical)
        peak_flux_gpff = max(flux_gpff)
        flux_ratio = peak_flux_gpff / peak_flux_classical if peak_flux_classical > 0 else 0
        
        flux_text = (
            f"Flux Comparison\n"
            f"{'─'*20}\n"
            f"Classical Peak: {peak_flux_classical:.2e} W/m²\n"
            f"GPFF II Peak:   {peak_flux_gpff:.2e} W/m²\n"
            f"Ratio (GPFF/Classical): {flux_ratio:.4f}"
        )
        
        self.ax_spectrum.text(
            1.15, 0.95, flux_text,
            transform=self.ax_spectrum.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='lightblue', edgecolor='gray', alpha=0.9)
        )

        # Validation panel
        self._add_validation_text()

    def calculate_flux(self, event):
        # ── Guards ────────────────────────────────────────────────────────
        # Re-entrancy: ignore calls that arrive while a calculation is
        # already running (matplotlib can queue multiple button events
        # before a slow canvas.draw_idle() completes).
        if self._calculating:
            return

        # Debounce: ignore calls within 200 ms of the last completed one.
        # This catches the draw_idle() → slider callback → calculate
        # cascade that produces burst sequences in the logs.
        now = time.time()
        if now - self._last_calc_time < 0.2:
            return

        self._calculating    = True
        self._last_calc_time = now

        try:
            self.logger.info("flux_calculation_start", {
                "source":      self.source_type,
                "temperature": self.temperature,
                "distance":    self.distance,
            })

            # ── Clear axes ────────────────────────────────────────────────
            self._clear_axes()

            # ── Physical constants ────────────────────────────────────────
            h   = self._const(PLANCK_CONSTANT)
            c   = self._const(SPEED_OF_LIGHT)
            k_B = self._const(BOLTZMANN_CONSTANT)

            wavelengths = np.logspace(-9, -6, 100)   # 1 nm – 1000 nm

            # ── Build and validate mode list ─────────────────────────────
            try:
                modes = self._build_mode_list()
            except Exception as e:
                self.error_handler.handle(e, context={"source": self.source_type})
                return

            # ── Calculate flux values ─────────────────────────────────────
            flux_classical, flux_gpff, energies, intensities = self._calculate_flux_values(
                wavelengths, modes, h, c, k_B
            )

            # ── Plot results ─────────────────────────────────────────────
            self._plot_results(wavelengths, flux_classical, flux_gpff, energies, intensities, modes)

            self.fig.canvas.draw_idle()

        finally:
            # Always release the guard, even if an exception occurs,
            # so future Calculate clicks are never permanently locked out.
            self._calculating = False

    def _add_validation_text(self):
        """Write validation results; ax_validation was already cleared."""
        if self.source_type == "solar":
            sol_L   = self._const(SOLAR_LUMINOSITY)
            d_earth = self._au()
            cf = self.classical.geometric_flux(sol_L, d_earth)
            gf = self.flux_calc.observed_flux(sol_L, d_earth,
                    [GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)])
            obs = 1360.8
            txt = (f"Solar Constant Validation\n"
                   f"{'─'*32}\n"
                   f"Classical : {cf:.1f} W/m²\n"
                   f"GPFF II   : {gf:.1f} W/m²\n"
                   f"Observed  : {obs} ± 0.5 W/m²\n"
                   f"Error     : {abs(gf - obs):.1f} W/m²\n"
                   f"Agreement : {abs(gf-obs)/obs*100:.2f} %")
        else:
            ns_L  = self.classical.stefan_boltzmann_luminosity(self.temperature, self.radius)
            ns_d  = 120 * self._pc()
            cf = self.classical.geometric_flux(ns_L, ns_d)
            gf = self.flux_calc.observed_flux(ns_L, ns_d,
                    [GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)])
            obs = 1.5e-14
            txt = (f"RX J1856.5-3754 Validation\n"
                   f"{'─'*32}\n"
                   f"Classical : {cf:.2e} W/m²\n"
                   f"GPFF II   : {gf:.2e} W/m²\n"
                   f"Observed  : {obs:.2e} W/m²\n"
                   f"Ratio     : {gf/obs:.2f}")

        self.ax_validation.text(
            0.05, 0.95, txt,
            transform=self.ax_validation.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='lightyellow', edgecolor='gray', alpha=0.9)
        )

    # ------------------------------------------------------------------
    # Validation (console)
    # ------------------------------------------------------------------

    def run_validation_tests(self):
        self.logger.info("validation_start", {"source": "gpffii_sim"})
        print("=" * 60)
        print("GPFF II VALIDATION TESTS")
        print("=" * 60)

        print("\n1. SOLAR CONSTANT TEST")
        print("-" * 30)
        sol_L   = self._const(SOLAR_LUMINOSITY)
        d_earth = self._au()
        cf = self.classical.geometric_flux(sol_L, d_earth)
        gf = self.flux_calc.observed_flux(sol_L, d_earth,
                [GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)])
        obs = 1360.8
        print(f"Classical : {cf:.1f} W/m²")
        print(f"GPFF II   : {gf:.1f} W/m²")
        print(f"Observed  : {obs} ± 0.5 W/m²")
        print(f"Error     : {abs(gf - obs):.1f} W/m²  ({abs(gf-obs)/obs*100:.2f} %)")

        print("\n2. RX J1856.5-3754 NEUTRON STAR TEST")
        print("-" * 40)
        ns_T, ns_R = 7.08e5, 5.0e3
        ns_d = 120 * self._pc()
        ns_L = self.classical.stefan_boltzmann_luminosity(ns_T, ns_R)
        cf_ns = self.classical.geometric_flux(ns_L, ns_d)
        gf_ns = self.flux_calc.observed_flux(ns_L, ns_d,
                [GPFFIIMode(a=0, i=0, w=1.0, lambda_i=0.0, kappa=0.0)])
        obs_ns = 1.5e-14
        print(f"Luminosity: {ns_L:.2e} W")
        print(f"Classical : {cf_ns:.2e} W/m²")
        print(f"GPFF II   : {gf_ns:.2e} W/m²")
        print(f"Observed  : {obs_ns:.2e} W/m²")
        print(f"Ratio     : {gf_ns/obs_ns:.2f}")
        print(f"Peak lambda: {self.classical.wien_peak_wavelength(ns_T)*1e9:.2f} nm")

        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        self.logger.info("validation_complete", {
            "solar_error_pct":    abs(gf - obs) / obs * 100,
            "neutron_star_ratio": gf_ns / obs_ns,
        })

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        print("Starting GPFF II Interactive Visualization...")
        self.setup_figure()
        self.create_controls()
        self.run_validation_tests()
        self.calculate_flux(None)
        plt.show()


def main():
    print("General Photon Flux Framework II (GPFF II)")
    print("=" * 50)
    print("A quantum-sector formulation for astrophysical photon flux")
    print("Based on the research paper by Furkan Nar (April 2026)")
    print("=" * 50)
    GPFFIIVisualizer().run()


if __name__ == "__main__":
    main()