# TESS Exoplanet Pipeline

Documentation for the `tess_pipeline` package, including the analysis workflow, public API, and notebook-backed examples.

## What this project does

The pipeline resolves a TESS target, loads or downloads light curves, preprocesses them, searches for periodic transit signatures, estimates stellar properties, fits a transit model, and exports results and diagnostics.

## Start here

```{toctree}
---
maxdepth: 2
---
installation
pipeline
methodology
api
examples
notebooks
```

## Public entry points

The package currently exposes three primary classes:

* `tess_pipeline.TESSAnalysis`
* `tess_pipeline.Pipeline`
* `tess_pipeline.PipelineResults`

## Next steps

The first documentation pass will turn the existing Markdown guides into a Sphinx site and add API reference pages generated from the package itself.
