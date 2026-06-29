"""Inference dependency checks."""

from __future__ import annotations

from tess_pipeline.exceptions import InferenceNotInstalledError

_INFERENCE_PACKAGES: tuple[tuple[str, str], ...] = (
    ("exoplanet", "exoplanet"),
    ("exoplanet_core", "exoplanet-core"),
    ("pymc", "pymc"),
    ("celerite2", "celerite2"),
    ("arviz", "arviz"),
)


def check_inference_installed() -> None:
    """
    Verify all packages required for Bayesian inference are importable.

    Raises
    ------
    InferenceNotInstalledError
        If one or more inference packages are missing.
    """
    missing: list[str] = []
    for import_name, pip_name in _INFERENCE_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        packages = ", ".join(missing)
        raise InferenceNotInstalledError(
            "Bayesian inference packages are missing or incomplete: "
            f"{packages}. Install the full stack with:\n"
            "  pip install -e \".[inference]\"\n"
            "from the tess_exoplanet_pipeline project root."
        )
