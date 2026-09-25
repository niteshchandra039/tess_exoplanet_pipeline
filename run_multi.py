#!/usr/bin/env python3
"""
Run the TESS Exoplanet Pipeline.

Workflow
--------
1. Resolve target
2. Query archive
3. Query Gaia
4. Characterize host star
5. Download TESS light curves
6. Preprocess light curves
7. Search for transit signals
8. Run Bayesian transit fit
9. Derive planetary parameters
10. Check MCMC convergence
11. Generate figures
12. Save results
"""

from __future__ import annotations

import time
from pathlib import Path

from tess_pipeline import TESSAnalysis

MODE = "science"  # Options: "test", "development", "science"
SECTORS = "longest"

if MODE == "test":
    CHAINS = 2
    DRAWS = 2
    TUNE = 2

elif MODE == "development":
    CHAINS = 2
    DRAWS = 500
    TUNE = 500

elif MODE == "science":
    CHAINS = 4
    DRAWS = 3000
    TUNE = 3000

else:
    raise ValueError(f"Unknown MODE = {MODE!r}")


OUTPUT_DIR = Path(f"output_bayesian_sector_{SECTORS}_{DRAWS}_draws_{CHAINS}_chains")


# =============================================================================
# Helper
# =============================================================================

def run_step(number: int, title: str, func):
    """Run a pipeline step with timing."""
    print(f"\n{'='*70}")
    print(f"Step {number}: {title}")
    print("="*70)

    t0 = time.perf_counter()
    result = func()
    dt = time.perf_counter() - t0

    print(f"✓ Completed in {dt:.2f} s")

    return result



# =============================================================================
# Configuration
# =============================================================================


# =============================================================================
# Main
# =============================================================================

def main(TIC):

    print("\n")
    print("=" * 80)
    print("TESS EXOPLANET PIPELINE")
    print("=" * 80)
    print(f"Target      : TIC {TIC}")
    print(f"Mode        : {MODE}")
    print(f"Search      : TLS")
    print(f"Backend     : exoplanet/PyMC")
    print(f"Output      : {OUTPUT_DIR}")
    print("=" * 80)

    analysis = TESSAnalysis(
        target=f"TIC {TIC}",
        search_method="tls",
        inference=True,
        inference_backend="exoplanet",
        sectors=SECTORS,
        max_planets=2,
        chains=CHAINS,
        draws=DRAWS,
        tune=TUNE,
        plots=True,
        verbose=True,
        output_dir=str(OUTPUT_DIR),
    )

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    run_step(1, "Resolve Target", analysis.resolve_target)

    run_step(2, "Query NASA Exoplanet Archive", analysis.lookup_archive_period)

    #
    # These should ideally happen BEFORE TLS.
    #
    # run_step(3, "Query Gaia DR3", analysis.query_gaia)

    run_step(4, "Characterize Host Star", analysis.characterize_star)

    run_step(5, "Download TESS Light Curves", analysis.load_lightcurves)

    run_step(6, "Preprocess Light Curves", analysis.preprocess)

    run_step(7, "Search for Transit Signals", analysis.search_period)

    #
    # Only run Bayesian inference if at least one candidate exists.
    #
    if analysis.results.detection is None:
        print("\nNo transit candidates detected.")
        return

    run_step(8, "Bayesian Transit Fit", analysis.fit_transit)

    run_step(9, "Derive Planet Parameters", analysis.derive_planet_parameters)

    run_step(10, "Check MCMC Convergence", analysis.check_convergence)

    run_step(11, "Generate Figures", analysis.generate_figures)

    run_step(12, "Save Results", analysis.save)

    total_time = time.perf_counter() - total_start

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    det = analysis.results.detection

    if det:

        print(f"Detected Period : {det.get('period', 'N/A'):.6f} d")
        print(f"Epoch           : {det.get('epoch', 'N/A'):.6f}")
        print(f"Depth           : {det.get('depth', 'N/A'):.6f}")
        print(f"Duration        : {det.get('duration_hr', 'N/A'):.2f} hr")
        print(f"SDE             : {det.get('sde', 'N/A'):.2f}")
        print(f"SNR             : {det.get('snr', 'N/A'):.2f}")

    print(f"\nOutput Directory : {OUTPUT_DIR}")
    print(f"Total Runtime    : {total_time/60:.2f} minutes")

    print("\n✓ Pipeline completed successfully.")


# =============================================================================
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("/mnt/home/project/cnitesh/nitesh/tess_exoplanet_pipeline/apc_list/APC_list.csv")

    print(df.head())

    print(f"Total TICs in the list: {len(df)}")

    for TIC in df['tid']:
        tic_str = str(TIC).strip()
        tic_output_dir = OUTPUT_DIR / f"TIC {tic_str}"

        # Skip TICs that already have an output directory.
        if tic_output_dir.exists():
            print(f"\nSkipping TIC {tic_str}: output already exists at {tic_output_dir}")
            continue

        print(f"\n\nRunning pipeline for TIC {tic_str}...")
        main(tic_str)