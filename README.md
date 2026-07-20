# TESS Exoplanet Pipeline

A Python library for step-by-step TESS light curve analysis: exoplanet detection and characterization in notebooks or scripts.

## Features

- **Step-by-step API**: Run each stage independently with `TESSAnalysis`
- **Data acquisition**: Download TESS SPOC PDCSAP light curves or load local FITS
- **Period detection**: Transit Least Squares (primary) and Box Least Squares
- **Stellar characterization**: Multi-catalog merging (VizieR TIC v8.2 + Gaia DR3 + SIMBAD)
- **Bayesian inference**: `exoplanet` + PyMC + NUTS with GP noise models
- **Export**: CSV, JSON, compressed posterior sample values, figures, and PDF reports

## Installation

```bash
pip install -e .
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

## Documentation

The Sphinx documentation scaffold lives in [`docs/index.md`](docs/index.md) and is configured from [`docs/conf.py`](docs/conf.py). It covers the public API, the pipeline overview, and notebook-backed walkthroughs.

## Scientific manuscript

The MNRAS-style manuscript scaffold is in [`Notes/mnras_manuscript.tex`](Notes/mnras_manuscript.tex), with its starter bibliography in [`Notes/references.bib`](Notes/references.bib).

## Project Structure

```
src/tess_pipeline/
56: ├── analysis/            # TESSAnalysis session and stage modules
57: │   ├── session.py
58: │   └── stages/
59: │       ├── target.py
60: │       ├── lightcurve.py
61: │       ├── period.py
62: │       ├── stellar.py
63: │       ├── inference.py
64: │       └── visualization.py
65: ├── config.py
66: ├── results.py
67: ├── data/                # Download and preprocessing
68: ├── catalogs/            # Gaia, NASA Archive, SDSS
69: ├── transit/             # Period search and transit models
70: ├── inference/           # Bayesian fitting
71: ├── visualization/       # Plotting
72: └── io/                  # Export and reporting
```

## License

MIT
