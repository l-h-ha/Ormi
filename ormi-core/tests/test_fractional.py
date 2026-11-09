from __future__ import annotations

import math

import numpy as np

import ormi_core as oc

###
###
###

def test_caputo_method_and_function_are_lazy_and_analytic():
    spacing = 0.125
    times = np.arange(9, dtype=np.float64) * spacing
    values = 2.0 + 3.0 * times
    tensor = oc.Tensor(values, dtype=oc.float64)

    method = tensor.caputo(0.5, spacing)
    function = oc.ops.caputo(tensor, 0.5, spacing)

    assert not method.realized
    assert not function.realized

    expected = np.zeros_like(times)
    expected[1:] = 3.0 * np.sqrt(times[1:]) / math.gamma(1.5)
    np.testing.assert_allclose(method.numpy(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(function.numpy(), method.numpy())

def test_caputo_gradient_remains_lazy_and_satisfies_adjoint_identity():
    spacing = 0.04
    positions = np.arange(31, dtype=np.float64)
    values = np.sin(0.17 * positions) + 0.01 * positions**2
    seeds = np.cos(0.11 * positions) - 0.03 * positions
    tensor = oc.Tensor(values, dtype=oc.float64)
    seed = oc.Tensor(seeds, dtype=oc.float64)
    output = tensor.caputo(0.63, spacing)
    transpose = output.grad([tensor], seed)[0]

    assert not output.realized
    assert not transpose.realized

    lhs = np.dot(output.numpy(), seeds)
    rhs = np.dot(values, transpose.numpy())
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)

def test_higher_order_caputo_uses_and_differentiates_initial_derivative():
    alpha = 1.5
    spacing = 0.125
    times = np.arange(9, dtype=np.float64) * spacing
    values = 2.0 + 3.0 * times + 4.0 * times**2
    seeds = np.cos(0.2 * np.arange(9, dtype=np.float64))
    tensor = oc.Tensor(values, dtype=oc.float64)
    initial_derivative = oc.Tensor(np.array(3.0), dtype=oc.float64)
    seed = oc.Tensor(seeds, dtype=oc.float64)

    output = tensor.caputo(
        alpha,
        spacing,
        initial_derivative=initial_derivative,
    )
    gradients = output.grad([tensor, initial_derivative], seed)

    assert not output.realized
    assert not gradients[0].realized
    assert not gradients[1].realized

    expected = np.zeros_like(times)
    expected[1:] = (
        8.0
        * np.power(times[1:], 2.0 - alpha)
        / math.gamma(3.0 - alpha)
    )
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-12, atol=1e-12)

    lhs = np.dot(expected, seeds)
    rhs = (
        np.dot(values, gradients[0].numpy())
        + 3.0 * gradients[1].item()
    )
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
