"""
visualization/posterior.py — Posterior distribution visualization.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def plot_corner(
    posterior: Any,
    var_names: list[str] | None = None,
) -> plt.Figure:
    """
    Corner plot of the posterior distribution using corner.py.

    Parameters
    ----------
    posterior : arviz.InferenceData
    var_names : list[str] | None
        Parameters to include. Defaults to key transit parameters.
    """
    try:
        import corner
        import numpy as np
    except ImportError:
        try:
            import arviz as az
            if var_names is None:
                var_names = ["rp_r_star", "b", "t14", "t0", "u1", "u2"]
            available = list(posterior.posterior.data_vars)
            var_names = [v for v in var_names if v in available]
            axes = az.plot_pair(
                posterior,
                var_names=var_names,
                marginal=True,
                visuals={"point_estimate": True},
            )
            fig = plt.gcf()
            fig.suptitle("Posterior Corner Plot", y=1.01, fontsize=11)
            return fig
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "arviz or corner required for corner plot", ha="center", va="center")
            return fig

    if var_names is None:
        # Default key parameters in our posterior
        var_names = ["period", "rp_r_star", "b", "t14", "u1", "u2", "rho_star"]

    post_group = posterior.posterior
    if hasattr(post_group, "to_dataset"):
        ds = post_group.to_dataset()
    else:
        ds = post_group

    available = list(ds.data_vars)
    var_names = [v for v in var_names if v in available]

    labels_dict = {
        "period": "Period [d]",
        "rp_r_star": "Rₚ/R★",
        "b": "Impact Parameter b",
        "t14": "Duration T₁₄ [d]",
        "u1": "u₁",
        "u2": "u₂",
        "rho_star": "Density ρ★ [g/cm³]",
    }
    flat_samps = ds.stack(sample=("chain", "draw"))

    samples_list = []
    labels = []
    for v in var_names:
        vals = flat_samps[v].values
        # vals shape could be (n_planets, n_samples) or (n_samples,)
        if vals.ndim == 2:
            n_pl = vals.shape[0]
            for i in range(n_pl):
                samples_list.append(vals[i, :])
                labels.append(labels_dict.get(v, v) + f" p{i+1}")
        else:
            samples_list.append(vals)
            labels.append(labels_dict.get(v, v))

    samples = np.column_stack(samples_list)

    if samples.shape[0] < samples.shape[1]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(
            0.5, 0.5,
            f"Too few MCMC samples ({samples.shape[0]}) for {samples.shape[1]} dimensions.\n"
            "Increase draws and tune to generate the corner plot.",
            ha="center", va="center", color="#dc2626", fontsize=10, fontweight="medium"
        )
        return fig

    fig = corner.corner(
        samples,
        labels=labels,
        color="#2563eb",
        hist_kwargs={"color": "#1d4ed8", "fill": True, "alpha": 0.25},
        show_titles=True,
        title_fmt=".5f",
        title_kwargs={"fontsize": 9, "fontweight": "medium"},
        label_kwargs={"fontsize": 10},
    )
    return fig


def plot_trace(
    posterior: Any,
    var_names: list[str] | None = None,
) -> plt.Figure:
    """
    MCMC trace plots.

    Parameters
    ----------
    posterior : arviz.InferenceData
    var_names : list[str] | None
        Parameters to plot. Defaults to key transit parameters.
    """
    try:
        import arviz as az
    except ImportError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "arviz required for trace plot", ha="center", va="center")
        return fig

    if var_names is None:
        var_names = ["rp_r_star", "b", "t14", "t0", "q1", "q2", "sigma_gp", "rho_gp"]

    available = list(posterior.posterior.data_vars)
    var_names = [v for v in var_names if v in available]

    az.plot_trace(posterior, var_names=var_names)
    fig = plt.gcf()
    fig.suptitle("MCMC Trace Plots", y=1.01, fontsize=11)
    return fig


def plot_posterior_predictive(
    lc: Any,
    posterior: Any,
    model_outputs: dict[str, Any] | None,
    n_samples: int = 100,
) -> plt.Figure:
    """
    Posterior predictive check: plot a sample of posterior draws
    overlaid on the data.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5))

    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    ax.scatter(time, flux, s=0.5, color="steelblue", alpha=0.4, rasterized=True, label="Data")

    # If posterior predictive samples exist, draw them
    if model_outputs and model_outputs.get("flux_model") is not None:
        ax.plot(
            time, model_outputs["flux_model"],
            color="red", linewidth=1.5, zorder=5, label="Posterior median",
        )

    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title("Posterior Predictive Check")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
