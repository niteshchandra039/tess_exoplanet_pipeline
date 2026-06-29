# Examples

## Basic usage

```python
from tess_pipeline import Pipeline

# Full pipeline: download, detect, characterize
pipeline = Pipeline("TIC 307210830", inference=True)
results = pipeline.run()

results.summary()
results.plot_all()
results.save("output/307210830/")
```

## Detection only (no MCMC)

```python
pipeline = Pipeline("TIC 307210830", inference=False)
results = pipeline.run()

print(results.period)
results.plot("periodogram")
results.plot("phase")
```

## Override period from literature

```python
pipeline = Pipeline("TIC 307210830", period=3.3601, inference=True)
results = pipeline.run()
```

## Custom MCMC settings

```python
pipeline = Pipeline(
    "TIC 307210830",
    inference=True,
    chains=4,
    draws=4000,
)
results = pipeline.run()
results.diagnostics   # check R-hat, ESS
```

## CLI examples

```bash
# Full pipeline
tess-pipeline TIC307210830

# Detection only, BLS cross-check
tess-pipeline TIC307210830 --no-inference --search both --output ./results/

# Override period, save PDF report
tess-pipeline TIC307210830 --period 3.3601 --save-report --output ./results/

# Force fresh download
tess-pipeline TIC307210830 --force-download
```

## Accessing results

```python
# Stellar parameters
print(results.stellar["r_star"])      # R★ (solar radii)
print(results.stellar["rho_star"])    # ρ★ (g/cm³)

# Planet parameters with uncertainties
p = results.planet
print(f"Rp = {p['rp_earth']:.2f} ± {p['rp_earth_err']:.2f} R_Earth")
print(f"a  = {p['a_au']:.4f} ± {p['a_au_err']:.4f} au")
print(f"Teq = {p['t_eq']:.0f} K")

# ArviZ posterior diagnostics
import arviz as az
az.summary(results.posterior)
az.plot_trace(results.posterior)
az.plot_corner(results.posterior)
```
