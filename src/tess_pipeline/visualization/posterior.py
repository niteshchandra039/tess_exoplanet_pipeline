"""
visualization/posterior.py — Posterior distribution visualization.
"""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt

LABELS_DICT = {
    "period": "Period $P$ [d]",
    "rp_r_star": "$R_p/R_*$",
    "b": "Impact parameter $b$",
    "t14": "Duration $T_{14}$ [d]",
    "t0": "Epoch $t_0$ [BTJD]",
    "u1": "$u_1$",
    "u2": "$u_2$",
    "q1": "$q_1$",
    "q2": "$q_2$",
    "rho_star": "Density $\\rho_*$ [g/cm$^3$]",
    "sigma_gp": "GP $\\sigma$",
    "rho_gp": "GP $\\rho$",
}


def plot_corner(
    posterior: Any,
    var_names: list[str] | None = None,
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    Corner plot of the posterior distribution using corner.py.
    """
    import numpy as np

    try:
        import corner
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
            fig.suptitle(f"TIC {tic_id} | Sectors: {sectors_str} | MCMC Posterior Corner Plot", y=1.01, fontsize=11, fontweight="bold")
            return fig
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "arviz or corner required for corner plot", ha="center", va="center")
            return fig

    if var_names is None:
        var_names = ["period", "rp_r_star", "b", "t14", "u1", "u2", "rho_star"]

    post_group = posterior.posterior
    if hasattr(post_group, "to_dataset"):
        ds = post_group.to_dataset()
    else:
        ds = post_group

    available = list(ds.data_vars)
    var_names = [v for v in var_names if v in available]

    labels_dict = LABELS_DICT
    flat_samps = ds.stack(sample=("chain", "draw"))

    samples_list = []
    labels = []
    for v in var_names:
        vals = flat_samps[v].values
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
            ha="center", va="center", color="#dc2626", fontsize=10, fontweight="bold"
        )
        return fig

    fig = corner.corner(
        samples,
        labels=labels,
        color="#0f766e",
        hist_kwargs={"color": "#0d9488", "fill": True, "alpha": 0.3},
        show_titles=True,
        title_fmt=".5f",
        title_kwargs={"fontsize": 9, "fontweight": "normal"},
        label_kwargs={"fontsize": 10},
    )

    # Add explanatory text in the top-right empty space
    desc = (
        "MCMC Parameter Guide:\n"
        "--------------------\n"
        "P          : Orbital period [days]\n"
        "Rₚ/R★      : Planet/stellar radius ratio\n"
        "b          : Impact parameter\n"
        "T₁₄        : Transit duration [days]\n"
        "u₁, u₂     : Limb darkening coeffs\n"
        "ρ★         : Stellar density [g/cm³]"
    )
    fig.text(0.62, 0.75, desc, fontsize=9.5, family="monospace", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", alpha=0.9, ec="#cbd5e1"))

    fig.suptitle(f"TIC {tic_id} | Sectors: {sectors_str} | MCMC Posterior Corner Plot", y=1.02, fontsize=11, fontweight="bold")
    return fig


def plot_trace(
    posterior: Any,
    var_names: list[str] | None = None,
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    MCMC trace plots.
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

    fig_size = (12, 1.6 * len(var_names))
    axes = None
    try:
        axes = az.plot_trace(posterior, var_names=var_names, backend_kwargs={"figsize": fig_size})
    except (TypeError, ValueError):
        try:
            axes = az.plot_trace(posterior, var_names=var_names, figsize=fig_size)
        except (TypeError, ValueError):
            axes = az.plot_trace(posterior, var_names=var_names)
    
    fig = plt.gcf()
    # Clean up labels and spacing to prevent clutter
    if axes is not None:
        import numpy as np
        axes_2d = np.atleast_2d(axes)
        for i, v in enumerate(var_names):
            if i < len(axes_2d):
                label = LABELS_DICT.get(v, v)
                axes_2d[i, 0].set_xlabel(label, fontsize=8.5)
                axes_2d[i, 1].set_ylabel(label, fontsize=8.5)
                axes_2d[i, 1].set_xlabel("Draw", fontsize=8.5)

    for ax in fig.axes:
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(8.5)
        ax.yaxis.label.set_size(8.5)
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=9, fontweight="normal")

    fig.suptitle(f"TIC {tic_id} | Sectors: {sectors_str} | MCMC Trace Plots", y=0.99, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.4)
    return fig


def plot_posterior_predictive(
    lc: Any,
    posterior: Any,
    model_outputs: dict[str, Any] | None,
    n_samples: int = 100,
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    Posterior predictive check: plot a sample of posterior draws
    overlaid on the data.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5))

    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    ax.scatter(time, flux, s=0.5, color="gray", alpha=0.3, rasterized=True, label="Data")

    # If posterior predictive samples exist, draw them
    if model_outputs and model_outputs.get("flux_model") is not None:
        ax.plot(
            time, model_outputs["flux_model"],
            color="#dc2626", linewidth=1.5, zorder=5, label="Posterior Median Model",
        )

    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | MCMC Posterior Predictive Check", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig
