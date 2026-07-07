#!/usr/bin/env python3
"""
Script to execute only the stellar characterization stage for TIC 107782586
and display the results.
"""

from tess_pipeline import TESSAnalysis

def main() -> None:
    print("Initializing TESS Analysis for TIC 107782586...")
    analysis = TESSAnalysis(target="TIC 107782586", verbose=True)
    
    print("\n1. Resolving target coordinates...")
    analysis.resolve_target()
    
    print("\n2. Querying Gaia DR3 for stellar properties...")
    analysis.query_gaia()
    
    print("\n3. Characterizing host star properties...")
    analysis.characterize_star()
    
    print("\n=== Stellar Characterization Results ===")
    for k, v in sorted(analysis.results.stellar.items()):
        print(f"  {k:<20} : {v}")

if __name__ == "__main__":
    main()
