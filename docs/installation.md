# Installation

The pipeline installs all dependencies by default, ensuring all features (Bayesian MCMC inference, stellar characterization via isochrone fitting, and advanced PHOEBE modeling) are fully functional out of the box.

## Install from Source

```bash
git clone https://github.com/your-org/exoplanet_modelling.git
cd exoplanet_modelling/tess_exoplanet_pipeline
pip install -e .
```

## Conda Environment (Recommended)

To avoid library version conflicts, it is recommended to install the package inside a dedicated Conda environment:

```bash
conda create -n tess_pipeline python=3.10
conda activate tess_pipeline
pip install -e .
```

## Verify Installation

You can verify that the library and its key components are correctly installed with:

```python
import tess_pipeline
print("tess-pipeline version:", tess_pipeline.__version__)

from tess_pipeline.inference.deps import check_inference_installed
try:
    check_inference_installed()
    print("Bayesian inference backend: OK")
except Exception as e:
    print("Bayesian inference check failed:", e)

from tess_pipeline.transit.phoebe_model import phoebe_available
print("PHOEBE backend available:", phoebe_available())
```

## Dependency Notes

- **C Compiler**: The `exoplanet` backend requires a C compiler for compiling C++ extensions (e.g. for `starry` / `exoplanet-core`). On Linux systems, `gcc` and `g++` are sufficient.
- **isoclassify Grids**: `isoclassify` requires downloading stellar model grids to function. Please refer to the `isoclassify` documentation for downloading and configuring the required grids on your local machine.
- **PHOEBE 2**: PHOEBE is installed automatically as part of the full stack. It is a heavy scientific library (~200 MB) with deep physical capabilities for binary and phase-curve modeling.
