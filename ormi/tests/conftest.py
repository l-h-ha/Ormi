import importlib.util
import os

import pytest

import ormi as om

###
###
###

@pytest.fixture(params=["cpu", "cuda:0"])
def api_device(request):
    if request.param == "cpu":
        return request.param

    if importlib.util.find_spec("ormi_cuda") is None:
        if os.environ.get("ORMI_WHEEL_EXPECT_CUDA") == "1":
            pytest.fail("ormi-cuda is required for this test run")

        pytest.skip("ormi-cuda is not installed")

    try:
        om.Tensor([0.0], device=request.param).numpy()
    except om.DeviceError:
        if os.environ.get("ORMI_WHEEL_EXPECT_CUDA") == "1":
            raise

        pytest.skip("No usable CUDA device")

    return request.param
