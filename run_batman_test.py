#!/usr/bin/env python3
"""
Test script to run the TESS Exoplanet Pipeline with the batman-only (analytic) backend
on target star TIC 261136679.
"""

from __future__ import annotations
import os
import sys
from tess_pipeline import TESSAnalysis

def main() -> None:
    print("Initializing TESS Analysis for TIC 261136679 using batman...")
    
    # Configure the analysis session
    analysis = TESSAnalysis(
        target="TIC 261136679",
        inference=True,
        inference_backend="batman_only",
        search_method="tls",
        plots=True,
        output_dir="output_batman",
        verbose=True
    )
    
    # Run the target characterization, download, detrending, and period search
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
    # analysis.query_sdss()
    
    # print("9. Running analytic batman transit fit...")
    analysis.fit_transit()
    
    # Print results summary
    print("\n" + "="*50)
    print("BATMAN FIT RESULTS SUMMARY")
    print("="*50)
    analysis.results.summary()
    
    print("\nSaving results and diagnostic plots...")
    analysis.save()
    print("Done! Batman test completed successfully.")

if __name__ == "__main__":
    main()
