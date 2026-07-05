# Pipeline Overview

## Workflow

```
Target (normalize TIC ID)
↓
NASA Exoplanet Archive lookup
↓
Acquire light curves
  Option A: Download from MAST (SPOC, selected cadence/sectors)
  Option B: Load local FITS files
↓
Preprocess (stitch, mask, initial flatten)
↓
Known period from archive?
  YES → use archived period
  NO  → TLS (primary) or BLS (triage)
↓
Gaia DR3 query
↓
Stellar characterization (VizieR/Gaia/SIMBAD → Teff, logg, R★, ρ★)
↓
SDSS query (optional; may return None)
↓
Phase-fold at best period
↓
Bayesian transit fit (exoplanet + PyMC + GP)
  → posteriors for Rp/R★, b, T14, t0, limb darkening, systematics
↓
Derive physical parameters from posterior
  → Rp, a, a/R★, Teq with credible intervals
↓
Convergence diagnostics (R-hat, ESS, posterior predictive)
↓
Diagnostic plots (phase-fold, corner, residuals)
↓
Export results (CSV, JSON, NetCDF posterior, figures, PDF)
```

## Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `pipeline.py` | Orchestrates the full workflow |
| `data/download.py` | TESS light curve download via lightkurve |
| `data/preprocess.py` | Stitching, masking, flattening |
| `data/metadata.py` | TIC ID resolution and observation metadata |
| `catalogs/nasa_archive.py` | Published period / planet parameters |
| `catalogs/gaia.py` | Gaia DR3 stellar properties |
| `catalogs/stellar.py` | Multi-catalog merging & characterization |
| `catalogs/sdss.py` | SDSS radial velocity |
| `transit/detection.py` | Period search orchestrator |
| `transit/tls.py` | Transit Least Squares |
| `transit/bls.py` | Box Least Squares |
| `transit/batman_model.py` | Analytic model for plots / MAP init |
| `inference/bayesian.py` | PyMC + exoplanet NUTS fit |
| `inference/gp.py` | GP noise model (celerite2) |
| `inference/priors.py` | Prior definitions |
| `inference/diagnostics.py` | Convergence checks |
| `visualization/` | All plotting functions |
| `io/export.py` | CSV, JSON, NetCDF, FITS export |
| `io/report.py` | PDF report generation |

## Configuration

See `config.py` for all configurable parameters and their defaults.
