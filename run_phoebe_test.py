#!/usr/bin/env python3
"""
Test script to check PHOEBE 2 availability and run a transit model
on TIC 261136679.
"""

from __future__ import annotations
import sys
from tess_pipeline import TESSAnalysis
from tess_pipeline.transit.phoebe_model import phoebe_available, run_phoebe_fit

def main() -> None:
    print("Checking PHOEBE 2 backend availability...")
    if not phoebe_available():
        print("PHOEBE 2 is not installed in the current environment.")
        print("To install PHOEBE, run:")
        print("    pip install 'tess-pipeline[phoebe]'")
        print("Or install the conda/system dependencies for PHOEBE 2.")
        print("Skipping active PHOEBE fit.")
        sys.exit(0)
        
    print("PHOEBE 2 is available! Setting up analysis for TIC 261136679...")
    
    # Initialize analysis with PHOEBE backend
    analysis = TESSAnalysis(
        target="TIC 261136679",
        inference=True,
        inference_backend="phoebe",
        search_method="tls",
        output_dir="output_phoebe",
        verbose=True
    )
    
    print("1. Resolving target coordinates...")
    analysis.resolve_target()
    
    print("2. Downloading TESS light curves from MAST...")
    analysis.load_lightcurves()
    
    print("3. Preprocessing light curves...")
    analysis.preprocess()
    
    print("4. Querying Gaia DR3 for stellar properties...")
    analysis.query_gaia()
    
    print("5. Characterizing host star properties...")
    analysis.characterize_star()
    
    print("6. Searching for transit signals...")
    analysis.search_period()
    
    # Downsample lightcurve for the test run to speed up PHOEBE fitting
    if analysis.results.lightcurve is None:
        print("No light curves loaded; cannot proceed.")
        sys.exit(1)
        
    print("Downsampling light curve (every 150th point) for fast PHOEBE fit test...")
    analysis.results.lightcurve = analysis.results.lightcurve[::150]
    
    print("7. Running integrated PHOEBE 2 fit via fit_transit()...")
    analysis.fit_transit()
    phoebe_results = analysis.results.model
    
    if phoebe_results is None:
        print("PHOEBE fit did not return results.")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("PHOEBE 2 FIT RESULTS SUMMARY")
    print("="*50)
    params = phoebe_results.get("planet_params", {})
    print(f"  Period (P)         : {params.get('period'):.6f} d")
    print(f"  Epoch (T0)         : {params.get('t0'):.6f} BTJD")
    print(f"  Radius Ratio Rp/R* : {params.get('rp_r_star'):.6f}")
    print(f"  Radius (Rp)        : {params.get('rp_earth'):.6f} R_earth")
    print(f"  a/R*               : {params.get('a_r_star'):.6f}")
    print(f"  Inclination (i)    : {params.get('incl'):.6f} deg")
    print(f"  Residuals std      : {phoebe_results['residuals'].std():.6e}")
    print("-" * 50)

    # Generate and save diagnostic plots
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plots_dir = os.path.join(analysis.config.output_dir, f"TIC {analysis.results.target['tic_id']}", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    times = analysis.results.lightcurve.time.value
    fluxes = analysis.results.lightcurve.flux.value
    model_fluxes = phoebe_results["flux_model"]
    residuals = phoebe_results["residuals"]

    # 1. Full Light Curve Fit Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(times, fluxes, "k.", alpha=0.5, label="Observed")
    ax1.plot(times, model_fluxes, "r-", lw=2, label="PHOEBE 2 Model")
    ax1.set_ylabel("Relative Flux")
    ax1.legend(loc="best")
    ax1.set_title(f"PHOEBE 2 Transit Model for TIC {analysis.results.target['tic_id']}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, residuals, "g.", alpha=0.5)
    ax2.axhline(0.0, color="r", linestyle="--")
    ax2.set_xlabel("Time (BTJD)")
    ax2.set_ylabel("Residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "phoebe_fit.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved PHOEBE fit plot to: {plot_path}")

    # 2. Phase Folded Plot
    period = params.get("period", 5.0)
    t0_fit = params.get("t0", times[np.argmin(fluxes)])
    phase = ((times - t0_fit) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = fluxes[sort_idx]
    model_sorted = model_fluxes[sort_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # Plot zoomed to transit window +/- 6 hours
    ax1.plot(phase_sorted * period * 24.0, flux_sorted, "k.", alpha=0.5, label="Observed")
    ax1.plot(phase_sorted * period * 24.0, model_sorted, "r-", lw=2, label="PHOEBE 2 Model")
    ax1.set_ylabel("Relative Flux")
    ax1.set_xlim(-6.0, 6.0)
    ax1.legend(loc="best")
    ax1.set_title(f"PHOEBE 2 Phased Transit for TIC {analysis.results.target['tic_id']}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(phase_sorted * period * 24.0, residuals[sort_idx], "g.", alpha=0.5)
    ax2.axhline(0.0, color="r", linestyle="--")
    ax2.set_xlabel("Time from mid-transit (hours)")
    ax2.set_ylabel("Residuals")
    ax2.set_xlim(-6.0, 6.0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    phase_plot_path = os.path.join(plots_dir, "phoebe_phase.png")
    fig.savefig(phase_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved PHOEBE phase-folded plot to: {phase_plot_path}")

    print("Done! PHOEBE test completed successfully.")

if __name__ == "__main__":
    main()