import math
import os

import numpy as np

import ormi_core as oc


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
