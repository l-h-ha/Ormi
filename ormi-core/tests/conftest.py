from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

###
###
###

@pytest.fixture
def matrix() -> np.ndarray:
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

@pytest.fixture(params=["cpu", "cuda:0"])
def nn_device(request):
    if request.param == "cuda:0":
        if importlib.util.find_spec("ormi_cuda") is None:
            pytest.skip("ormi-cuda is not installed")

        import ormi_core as oc

        try:
            oc.Tensor([0.0], device=request.param).numpy()
        except oc.DeviceError:
            if os.environ.get("ORMI_WHEEL_EXPECT_CUDA") == "1":
                raise

            pytest.skip("No usable CUDA device")

    return request.param
