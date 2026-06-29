from __future__ import annotations

import numpy as np


def test_tls_module_importable() -> None:
    import tess_pipeline.transit.tls as tls

    assert hasattr(tls, "run_tls")
