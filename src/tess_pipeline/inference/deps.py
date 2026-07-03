"""Inference dependency checks."""

from __future__ import annotations

import importlib
from tess_pipeline.exceptions import InferenceNotInstalledError

_INFERENCE_PACKAGES: tuple[tuple[str, str], ...] = (
    ("exoplanet", "exoplanet"),
    ("exoplanet_core", "exoplanet-core"),
    ("pymc", "pymc"),
    ("pytensor", "pytensor"),
    ("celerite2", "celerite2"),
    ("celerite2.pymc", "celerite2.pymc"),
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
            importlib.import_module(import_name)
        except ImportError as exc:
            missing.append(f"{pip_name} (error: {exc})")

    if missing:
        packages = ", ".join(missing)
        raise InferenceNotInstalledError(
            "Bayesian inference packages are missing, incomplete, or failing to import:\n"
            f"  {packages}\n\n"
            "Please reinstall the package or resolve the import issues with:\n"
            "  pip install -e .\n"
            "from the tess_exoplanet_pipeline project root."
        )
