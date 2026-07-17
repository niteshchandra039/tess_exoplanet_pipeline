"""
Script to run the TESS Exoplanet Pipeline with the modernized Bayesian (exoplanet/PyMC/GP)

Author: Nitesh Kumar
Date: 2026-07-17

"""


from __future__ import annotations
import os
import sys
from tess_pipeline import TESSAnalysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table 


table = Table.read("/mnt/home/project/cnitesh/nitesh/tess_exoplanet_pipeline/data/TOI_2026.06.28_09.42.00.csv", format='ascii', comment='#')

df = table.to_pandas()

# %%
df['tfopwg_disp'].unique()

# %% [markdown]
# | Value  | Meaning          | Description                                                                                                                                                                                             |
# | ------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **PC** | Planet Candidate | The object is still considered a promising exoplanet candidate. It has not yet been confirmed or statistically validated. More observations are needed.                                                 |
# | **CP** | Confirmed Planet | The candidate has been confirmed as a genuine exoplanet through follow-up observations (e.g., radial velocities, transit timing variations, high-resolution imaging, statistical validation, etc.).     |
# | **FP** | False Positive   | The transit signal is not caused by a planet. Common causes include eclipsing binaries, background eclipsing binaries, instrumental artifacts, stellar variability, or contamination from nearby stars. |
# | **KP** | Known Planet     | The signal corresponds to a planet that was already known before the TESS detection. TESS has observed the transit of an already discovered exoplanet.                                                  |
# 

# %%
df_1 = df[df['tfopwg_disp'] == 'CP']


# %%
# select 100 random TIC ids from the df_1 dataframe
tic_ids = df_1['tid'].sample(n=100, random_state=1, replace=False).tolist()

sectors = 'longest'  # Use 'longest' to automatically select the longest available sector for the target

def main(TIC) -> None:
    print("Initializing TESS Analysis for TIC 52368076 using Bayesian MCMC...")
    
    # Configure the analysis session with 500 tune and 500 draws for a fast test run
    analysis = TESSAnalysis(
        target=f"TIC {TIC}",
        inference=True,
        inference_backend="exoplanet",  # uses exoplanet/PyMC/celerite2
        search_method="tls",
        sectors=sectors,
        # max_planets=1, by default, the pipeline will search for multiple planets if present
        chains=5,
        draws=500,
        tune=500,
        plots=True,
        output_dir=f"output_bayesian_sector_{sectors}_500_draws_5_chains",
        verbose=True
    )
    
    # # Run the target characterization, download, detrending, and period search
    print("1. Resolving target coordinates...")
    analysis.resolve_target()
    
    print("2. Checking archival orbital period...")
    analysis.lookup_archive_period()
    
    print("3. Downloading TESS light curves from MAST...")
    analysis.load_lightcurves()
    
    print("4. Preprocessing light curves (stitching, cleaning, detrending)...")
    analysis.preprocess()
    
    print("5. Searching for transit signals...")
    analysis.search_period()
    
    print("6. Querying Gaia DR3 for stellar properties...")
    analysis.query_gaia()
    
    print("7. Characterizing host star properties...")
    analysis.characterize_star()
    
    # Disable SDSS checking since it is deprecated/disabled
    print("8. Skipping SDSS checking (disabled in this version)...")
    analysis.query_sdss()
    
    print("9. Running Bayesian MCMC fit...")
    analysis.fit_transit()
    
    print("10. Deriving physical planetary parameters with uncertainty propagation...")
    analysis.derive_planet_parameters()
    
    print("11. Checking MCMC sampler convergence...")
    analysis.check_convergence()
    
    print("12. Generating figures...")
    analysis.generate_figures()
    
    # Print results summary
    print("\n" + "="*50)
    print("BAYESIAN FIT RESULTS SUMMARY")
    print("="*50)
    analysis.results.summary()
    
    print("\nSaving results and diagnostic plots...")
    analysis.save()
    print("Done! Bayesian test completed successfully.\n")

if __name__ == "__main__":
    
    for TIC in tic_ids:
        try:
            main(TIC)
            print(f"Completed analysis for TIC {TIC}. Moving to the next target...\n")
        except Exception as e:
            print(f"Error while processing TIC {TIC}: {e}")
            print("Skipping to the next target...\n")
