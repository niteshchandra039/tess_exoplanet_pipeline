import numpy as np

from tess_pipeline.transit.batman_model import make_batman_model


def test_make_batman_model_callable() -> None:
    # Do not execute batman import here; only verify function is defined.
    assert callable(make_batman_model)
