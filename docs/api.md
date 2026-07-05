# API Reference

## Pipeline

```python
from tess_pipeline import Pipeline

pipeline = Pipeline(
    target,                 # str or int: "TIC 307210830", 307210830
    inference=True,         # bool: run Bayesian MCMC fit
    period=None,            # float or None: override period (days)
    output_dir="output/",   # str: results directory
    plots=True,             # bool: generate diagnostic plots
    author="SPOC",          # str: MAST pipeline filter
    sectors="all",         # int|str: 1, 2, 3, or "all"
    lightcurve_source="download",  # "download" or "fits"
    lightcurve_fits=None,   # list[str|Path]: used when lightcurve_source="fits"
    cadence=120,            # int: cadence in seconds
    search="tls",           # str: "tls", "bls", or "both"
    chains=4,               # int: MCMC chains
    draws=2000,             # int: posterior draws per chain
    save_report=False,      # bool: generate PDF report
    force_download=False,   # bool: bypass local cache
)

results = pipeline.run()
```

## PipelineResults

```python
# Summary
results.summary()           # print parameters + uncertainties

# Plotting
results.plot_all()
results.plot("raw")
results.plot("flat")
results.plot("periodogram")
results.plot("phase")
results.plot("posterior")

# Export
results.save("output/")    # CSV, JSON, NetCDF, FITS, figures

# Data access
results.target             # TIC ID, coordinates
results.lightcurve         # preprocessed LightCurve
results.period             # {'value': float, 'source': 'tls'|'archive'|'bls'}
results.detection          # TLS/BLS result dict
results.stellar            # Catalog-resolved stellar parameters (VizieR/Gaia/SIMBAD)
results.rv                 # SDSS radial velocity or None
results.posterior          # ArviZ InferenceData
results.planet             # derived parameters with credible intervals
results.model              # best-fit light curve and residuals
results.diagnostics        # R-hat, ESS, divergences
results.figures            # dict of matplotlib figures
results.metadata           # run config, timestamps, version
```

## CLI

```bash
tess-pipeline TIC307210830 [OPTIONS]

Options:
  --period FLOAT          Override period (days)
  --output PATH           Output directory [default: ./output/]
  --plots / --no-plots    Generate diagnostic plots [default: --plots]
  --author TEXT           MAST author filter [default: SPOC]
  --sectors N             Number of sectors: 1, 2, 3, all [default: 1]
  --lightcurve-source SOURCE  Light curve input source: download or fits
  --fits PATH             Local FITS file/directory (repeatable)
  --cadence INT           Cadence in seconds [default: 120]
  --search TEXT           Period search: tls, bls, both [default: tls]
  --inference / --no-inference  Run Bayesian MCMC [default: --inference]
  --chains INT            MCMC chains [default: 4]
  --draws INT             Posterior draws per chain [default: 2000]
  --save-report           Generate PDF summary report
  --force-download        Bypass local cache
  --help                  Show this message and exit
```
