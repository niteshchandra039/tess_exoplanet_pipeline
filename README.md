# TESS Exoplanet Pipeline

A Python library for step-by-step TESS light curve analysis: exoplanet detection and characterization in notebooks or scripts.

## Features

- **Step-by-step API**: Run each stage independently with `TESSAnalysis`
- **Data acquisition**: Download TESS SPOC PDCSAP light curves or load local FITS
- **Period detection**: Transit Least Squares (primary) and Box Least Squares
- **Stellar characterization**: Gaia DR3 + isochrone fitting via `isoclassify`
- **Bayesian inference**: `exoplanet` + PyMC + NUTS with GP noise models
- **Export**: CSV, JSON, NetCDF, figures, and PDF reports

## Installation

```bash
pip install -e ".[inference,stellar]"
```

## Quick Start (Notebook)

```python
from tess_pipeline import TESSAnalysis

analysis = TESSAnalysis("TIC 307210830", inference=True)

analysis.resolve_target()
analysis.lookup_archive_period()
analysis.load_lightcurves()
analysis.preprocess()
analysis.search_period()
analysis.query_gaia()
analysis.characterize_star()
analysis.query_sdss()
analysis.fit_transit()
analysis.derive_planet_parameters()
analysis.check_convergence()
analysis.generate_figures()

analysis.results.summary()
analysis.save("output/")
```

Or run all stages at once:

```python
results = TESSAnalysis("TIC 307210830").run()
```

See `notebooks/pipeline_walkthrough.ipynb` for a full walkthrough.

## Project Structure

```
src/tess_pipeline/
├── analysis/            # TESSAnalysis session and stage modules
│   ├── session.py
│   └── stages/
│       ├── target.py
│       ├── lightcurve.py
│       ├── period.py
│       ├── stellar.py
│       ├── inference.py
│       └── visualization.py
├── config.py
├── results.py
├── data/                # Download and preprocessing
├── catalogs/            # Gaia, NASA Archive, SDSS
├── transit/             # Period search and transit models
├── inference/           # Bayesian fitting
├── visualization/       # Plotting
└── io/                  # Export and reporting
```

## License

MIT
