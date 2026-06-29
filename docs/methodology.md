# Methodology

## Literature Foundations

### Transit Detection

**Transit Least Squares (TLS)** — Hippke et al. 2019, A&A 623, A39
([doi:10.1051/0004-6361/201834672](https://doi.org/10.1051/0004-6361/201834672))

TLS uses realistic limb-darkened transit templates (generated with batman) rather than the
box-shaped templates used by BLS. This yields ~10–17% higher recovery rates for small planets.
TLS is the default period search in this pipeline.

**Box Least Squares (BLS)** — Kovács et al. 2002

Used as a fast triage method and cross-check. Less sensitive than TLS for small planets.

### Transit Modeling

**Mandel & Agol (2002)** analytic transit model via batman and exoplanet/starry.

**Kipping (2013) limb-darkening reparameterization** — quadratic LD coefficients
mapped to (q1, q2) with uniform priors; ensures physically valid LD across the full
prior volume.

### Bayesian Inference Workflow

Follows:

1. **TESS-Keck Survey XV** (Polanski et al. 2023, AJ 166, 45,
   [doi:10.3847/1538-3881/acd557](https://doi.org/10.3847/1538-3881/acd557)):
   - `exoplanet` + PyMC transit fits with GP noise
   - Gaia-informed stellar parameters via isochrone fitting
   - Fit **T14** (transit duration) directly rather than sampling circular stellar density

2. **Precise Transit Photometry Using TESS II** (Saha et al. 2024, ApJS 275, 45,
   [doi:10.3847/1538-4365/ad6a60](https://doi.org/10.3847/1538-4365/ad6a60)):
   - Simultaneous transit + GP (Matérn ν=3/2) modeling with MCMC

### Why Fit T14 Directly?

Parameterizing via circular stellar density introduces subtle bias (Gilbert et al. 2022;
TESS-Keck XV). Fitting T14 (total transit duration) directly avoids this and yields
more robust parameter estimates especially for low-impact-parameter transits.

### Stellar Characterization

**Gaia-Kepler-TESS Host Catalog** (Berger et al. 2023, ApJS 264, 30,
[doi:10.3847/1538-4365/acafbc](https://doi.org/10.3847/1538-4365/acafbc)):

- Gaia DR3 parallax + photometry + spectroscopic [Fe/H] → isoclassify
- Homogeneous Teff, logg, R★, M★, ρ★, L★ with full uncertainties
- ρ★ posterior used as a prior in the Bayesian transit fit

### GP Noise Model

Residual stellar variability and instrumental systematics are modeled with a
Gaussian Process using a Stochastically-driven Harmonic Oscillator (SHO) kernel
via `celerite2`. The Matérn ν=3/2 kernel is available as an alternative (following
Saha et al. 2024). The GP is marginalized over jointly with transit parameters in
the PyMC model.

## Modeling Backends

| Tool | Role |
|------|------|
| batman | Fast analytic light curves; TLS templates; MAP initialization; diagnostic plots |
| exoplanet + PyMC + starry | Primary Bayesian inference: NUTS/HMC, GP, Kipping LD |
| PHOEBE 2 | Optional: phase curves, reflected light, multi-body systems |

## Parameter Conventions

- All times in BTJD (Barycentric TESS Julian Date = BJD − 2457000)
- Periods in days
- Radii in solar radii (R★) and Earth radii (Rp)
- ρ★ in g/cm³
- Equilibrium temperature assumes zero albedo and full redistribution
