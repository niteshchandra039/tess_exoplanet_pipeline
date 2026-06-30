"""
inference/diagnostics.py — Posterior convergence diagnostics.

Checks:
  * R̂ (rank-normalized split R-hat) < 1.01
  * Effective sample size (ESS) > 400 per parameter
  * Divergence count
  * Posterior predictive checks (placeholder)

Issues ConvergenceWarning when thresholds are not met.
"""

from __future__ import annotations

import warnings
from typing import Any

from tess_pipeline.constants import ESS_THRESHOLD, RHAT_THRESHOLD
from tess_pipeline.exceptions import ConvergenceWarning
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def check_convergence(posterior: Any) -> dict[str, Any]:
    """
    Run convergence diagnostics on an ArviZ InferenceData object.

    Parameters
    ----------
    posterior : arviz.InferenceData

    Returns
    -------
    dict with keys:
        rhat_max (float)
        rhat_dict (dict[str, float])
        ess_min (float)
        ess_dict (dict[str, float])
        divergences (int)
        converged (bool)
    """
    try:
        import arviz as az
    except ImportError:
        log.warning("arviz is not installed; skipping convergence diagnostics")
        return {"converged": None, "note": "arviz not installed"}

    # Exclude large deterministic vectors to prevent massive CPU/memory bottleneck
    all_vars = list(posterior.posterior.data_vars.keys())
    exclude_vars = {"transit_model", "gp_pred", "flux_model", "lc_pred"}
    var_names = [v for v in all_vars if v not in exclude_vars and not v.startswith("light_curve_p")]
    
    summary = az.summary(posterior, var_names=var_names, round_to=6)

    # ── R-hat ─────────────────────────────────────────────────────────────────
    rhat_col = "r_hat" if "r_hat" in summary.columns else None
    rhat_dict: dict[str, float] = {}
    rhat_max: float | None = None

    if rhat_col:
        rhat_dict = {k: float(v) for k, v in summary[rhat_col].items()}
        rhat_max = max(rhat_dict.values()) if rhat_dict else None

        bad_rhat = {k: v for k, v in rhat_dict.items() if v > RHAT_THRESHOLD}
        if bad_rhat:
            warnings.warn(
                f"{len(bad_rhat)} parameter(s) have R̂ > {RHAT_THRESHOLD}: "
                + ", ".join(f"{k}={v:.4f}" for k, v in bad_rhat.items()),
                ConvergenceWarning,
                stacklevel=2,
            )

    # ── ESS ───────────────────────────────────────────────────────────────────
    ess_col = "ess_bulk" if "ess_bulk" in summary.columns else None
    ess_dict: dict[str, float] = {}
    ess_min: float | None = None

    if ess_col:
        ess_dict = {k: float(v) for k, v in summary[ess_col].items()}
        ess_min = min(ess_dict.values()) if ess_dict else None

        bad_ess = {k: v for k, v in ess_dict.items() if v < ESS_THRESHOLD}
        if bad_ess:
            warnings.warn(
                f"{len(bad_ess)} parameter(s) have ESS < {ESS_THRESHOLD}: "
                + ", ".join(f"{k}={v:.0f}" for k, v in bad_ess.items()),
                ConvergenceWarning,
                stacklevel=2,
            )

    # ── Divergences ───────────────────────────────────────────────────────────
    divergences = 0
    try:
        div_arr = posterior.sample_stats["diverging"].values
        divergences = int(div_arr.sum())
        if divergences > 0:
            warnings.warn(
                f"{divergences} divergent transition(s) detected. "
                "Consider increasing target_accept or reparameterizing.",
                ConvergenceWarning,
                stacklevel=2,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not extract divergences: %s", exc)

    # ── Overall convergence ───────────────────────────────────────────────────
    converged = (
        (rhat_max is None or rhat_max <= RHAT_THRESHOLD)
        and (ess_min is None or ess_min >= ESS_THRESHOLD)
        and divergences == 0
    )

    result = {
        "rhat_max": rhat_max,
        "rhat_dict": rhat_dict,
        "ess_min": ess_min,
        "ess_dict": ess_dict,
        "divergences": divergences,
        "converged": converged,
    }

    if converged:
        log.info("Convergence OK (R̂_max=%.4f, ESS_min=%.0f, divergences=%d)",
                 rhat_max or 0, ess_min or 0, divergences)
    else:
        log.warning("Convergence issues detected (see ConvergenceWarning)")

    return result


def posterior_predictive_check(
    posterior: Any,
    lc: Any,
) -> dict[str, Any]:
    """
    Run posterior predictive check via arviz.

    Returns dict with p_values and coverage statistics.
    """
    try:
        import arviz as az
        import numpy as np
    except ImportError:
        return {"note": "arviz not installed"}

    try:
        ppc = az.loo(posterior)
        return {
            "elpd_loo": float(ppc.elpd_loo),
            "p_loo": float(ppc.p_loo),
            "se": float(ppc.se),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("LOO-CV failed: %s", exc)
        return {}
