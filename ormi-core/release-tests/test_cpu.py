import math
import os

import numpy as np
import pytest

import ormi_core as oc

###
###
###

def test_installed_release_version():
    assert oc.__version__ == os.environ["ORMI_RELEASE_VERSION"]

def test_cpu_works_without_a_jit_compiler():
    oc.jit.clear()

    lhs_data = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    rhs_data = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
    result = oc.Tensor(lhs_data) @ oc.Tensor(rhs_data)

    np.testing.assert_allclose(
        result.numpy(),
        lhs_data @ rhs_data,
        rtol=1e-4,
        atol=1e-3,
    )
    assert oc.jit.info().entries == 0

def test_mean_is_available_from_the_installed_wheel():
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = oc.Tensor(values)

    np.testing.assert_allclose(
        tensor.mean((0, 2), keepdims=True).numpy(),
        values.mean(axis=(0, 2), keepdims=True),
    )
    np.testing.assert_allclose(
        oc.ops.mean(tensor).item(),
        values.mean(),
    )

def test_cpu_plan_replays_parameter_updates():
    parameter = oc.Tensor(2.0)
    gradient = oc.Tensor(1.0)
    next_parameter = oc.sgd(parameter, gradient, 0.1)
    plan = oc.Plan([], [(parameter, next_parameter)])

    plan.run()
    assert parameter.item() == np.float32(1.9)

    plan.run()
    assert parameter.item() == np.float32(1.8)

def test_cpu_caputo_works_without_a_jit_compiler():
    oc.jit.clear()

    spacing = 0.25
    times = np.arange(5, dtype=np.float64) * spacing
    values = 1.0 + 2.0 * times
    result = oc.Tensor(values, dtype=oc.float64).caputo(0.5, spacing)

    assert not result.realized
    expected = np.zeros_like(times)
    expected[1:] = 2.0 * np.sqrt(times[1:]) / math.gamma(1.5)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert oc.jit.info().entries == 0

@pytest.mark.parametrize("dtype,numpy_dtype,tolerance", [
    (oc.float16, np.float16, 5e-3), (oc.float32, np.float32, 1e-6), (oc.float64, np.float64, 1e-12),
])
def test_cpu_views_broadcasting_and_reductions_without_a_compiler(dtype, numpy_dtype, tolerance):
    values = (np.arange(24).reshape(2, 3, 4) / 20).astype(numpy_dtype)
    x = oc.Tensor(values, dtype=dtype)
    view = x.transpose((2, 0, 1))
    output = (view * view + 1.0).mean((0, 2), keepdims=True)
    assert not view.realized and not output.realized
    plan = oc.Plan([output])
    assert not view.realized and not output.realized

    for data in (values, values + numpy_dtype(0.25)):
        x.copy_from(data)
        plan.run()
        expected = (data.transpose(2, 0, 1) ** 2 + 1).mean(axis=(0, 2), keepdims=True)
        np.testing.assert_allclose(output.numpy(), expected, rtol=tolerance, atol=tolerance)

def test_cpu_lazy_higher_derivatives_and_shared_outputs_without_a_compiler():
    values = np.arange(8, dtype=np.float64).reshape(2, 1, 4) / 10
    x = oc.Tensor(values, dtype=oc.float64)
    output = (x * x * x * x).sum()
    first = oc.grad(output, x)
    second = oc.grad(first.sum(), x)
    third = oc.grad(second.sum(), x)
    outputs = [output, first, second, third]
    assert all(not value.realized for value in outputs)
    plan = oc.Plan(outputs)
    assert all(not value.realized for value in outputs)

    for data in (values, values + 0.25):
        x.copy_from(data)
        plan.run()
        np.testing.assert_allclose(output.item(), (data**4).sum(), rtol=1e-12)
        np.testing.assert_allclose(first.numpy(), 4 * data**3, rtol=1e-12)
        np.testing.assert_allclose(second.numpy(), 12 * data**2, rtol=1e-12)
        np.testing.assert_allclose(third.numpy(), 24 * data, rtol=1e-12)
