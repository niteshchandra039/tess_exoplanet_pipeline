#!/usr/bin/env python3
"""
Script to execute only the period search stage for TIC 52368076
and display the results.
"""

from tess_pipeline import TESSAnalysis

def main() -> None:
    print("Initializing TESS Analysis for TIC 52368076...")
    analysis = TESSAnalysis(target="TIC 52368076", verbose=True, max_planets=3, search_method="tls")
    
    print("\n1. Resolving target coordinates...")
    analysis.resolve_target()

    print("\n2. Checking archival orbital period...")
    analysis.lookup_archive_period()

    print("\n3. Downloading TESS light curves from MAST...")
    analysis.load_lightcurves()

    print("\n4. Preprocessing light curves (stitching, cleaning, detrending)...")
    analysis.preprocess()

    print("\n5. Searching for transit signals...")
    analysis.search_period()

    print("\n================== RESULTS ==================")
    detections = analysis.results.metadata.get("detections", [])
    if detections:
        for i, det in enumerate(detections):
            period = det.get("period", 0)
            epoch = det.get("epoch", 0)
            snr = max(det.get("sde", 0), det.get("snr", 0))
            print(f"Planet {i+1}: Period = {period:.5f} days | T0 = {epoch:.4f} | SNR/SDE = {snr:.2f}")
    elif analysis.results.period:
        p = analysis.results.period.get('value', 0)
        print(f"Archival/Fall-back Period: {p:.5f} days")
    else:
        print("No robust detections found.")
        
    print("=============================================\n")


    
  

if __name__ == "__main__":
    main()
