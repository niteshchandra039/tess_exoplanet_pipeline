"""Example: run the TESS analysis step-by-step (notebook-friendly)."""

from __future__ import annotations

from tess_pipeline import TESSAnalysis


def main() -> None:
    analysis = TESSAnalysis(
        "TIC 307210830",
        inference=True,
        search_method="tls",
        chains=4,
        draws=2000,
        plots=True,
        output_dir="output",
    )

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
    analysis.results.plot_all()
    analysis.save()


if __name__ == "__main__":
    main()
