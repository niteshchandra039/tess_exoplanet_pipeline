# Installation

## Requirements

- Python 3.10 or later

## Install from source (development)

```bash
git clone https://github.com/your-org/exoplanet_modelling.git
cd exoplanet_modelling/tess_exoplanet_pipeline
pip install -e ".[inference,dev]"
```

## Install options

```bash
# Core only (detection, no MCMC)
pip install tess-pipeline

# With Bayesian inference (recommended)
pip install "tess-pipeline[inference]"

# With stellar isochrone fitting
pip install "tess-pipeline[inference,stellar]"

# With PHOEBE backend (advanced phase-curve modeling)
pip install "tess-pipeline[phoebe]"

# Full development install
pip install "tess-pipeline[all]"
```

## Conda environment (recommended)

```bash
conda create -n tess_pipeline python=3.12
conda activate tess_pipeline
pip install -e ".[inference,dev]"
```

## Verify installation

```python
import tess_pipeline
print(tess_pipeline.__version__)
```

## Dependency notes

- `exoplanet` requires a C compiler for the starry backend. On Linux, `gcc` is sufficient.
- `isoclassify` requires additional stellar model grids. See its documentation for grid download instructions.
- `phoebe` is a heavy dependency (~200 MB). Only install if you need phase-curve modeling.
