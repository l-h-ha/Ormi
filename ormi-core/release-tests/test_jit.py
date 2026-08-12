import numpy as np

import ormi_core as oc


def test_uncached_cpu_fusion_compiles_and_runs():
    oc.jit.clear()

    values = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    x = oc.Tensor(values)
    square = x * x
    summed = square + x
    exponential = summed.exp()
    result = exponential.sin()

    np.testing.assert_allclose(
        result.numpy(),
        np.sin(np.exp(values * values + values)),
        rtol=1e-6,
    )
    assert not square.realized
    assert not summed.realized
    assert not exponential.realized
    assert oc.jit.info().entries == 1


def test_lazy_autograd_from_the_installed_wheel():
    x = oc.Tensor(2.0, dtype=oc.float64)
    y = x * x * x
    derivative = oc.diff(y, x)

    assert not y.realized
    assert derivative.item() == 12.0


def test_float32_vector_math_and_scalar_tail():
    oc.jit.clear()

    values = np.linspace(-0.9, 0.9, 32 * 1024 + 3, dtype=np.float32)
    result = oc.Tensor(values).exp().log().sin().cos()

    expected = np.cos(np.sin(np.log(np.exp(values))))
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)
