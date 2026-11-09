from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

###
###
###

def reference_step(input, parameters, learning_rate):
    weight, bias, output_weight, output_bias = parameters
    phase = input @ weight + bias
    sine = np.sin(phase)
    basis = sine * (1.0 - weight * weight)
    residual = basis @ output_weight + output_bias
    loss = np.mean(residual * residual)
    upstream = 2.0 * residual / input.shape[0]
    slope = np.cos(phase) * (1.0 - weight * weight) * output_weight.T
    weight_gradient = np.sum(
        upstream * (slope * input - 2.0 * sine * weight * output_weight.T),
        axis=0,
        keepdims=True,
    )
    bias_gradient = np.sum(upstream * slope, axis=0)
    output_weight_gradient = basis.T @ upstream
    output_bias_gradient = np.sum(upstream, axis=0)
    gradients = [weight_gradient, bias_gradient, output_weight_gradient, output_bias_gradient]
    updated = [
        parameter - learning_rate * gradient
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    return loss, updated

###
###
###

@pytest.mark.parametrize("tier", ["direct", "optimized"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_framework_pinn_matches_analytic_sgd_steps(nn_device, tier, batch, dtype, monkeypatch):
    # Benchmark sources are not part of the installed wheel.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from benchmarks.frameworks.adapters import ormi
    from benchmarks.frameworks.model import Spec
    from benchmarks.frameworks.workloads import LEARNING_RATE, make_data

    spec = Spec("pinn", "test", batch, 1, 3, dtype=dtype)
    data = make_data(spec)
    original = {name: value.copy() for name, value in data.values.items()}
    parameters = [
        data.values[name].copy()
        for name in ("first_weight", "first_bias", "second_weight", "second_bias")
    ]
    case = ormi.prepare(spec, data, tier, nn_device)
    rtol, atol = (1e-5, 1e-7) if dtype == "float32" else (1e-11, 2e-12)

    for _ in range(3):
        loss, parameters = reference_step(data.values["input"], parameters, LEARNING_RATE)
        case.run()
        actual = case.values()
        assert len(actual) == 5

        for result, expected in zip(actual, [loss, *parameters], strict=True):
            np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

    for name, value in original.items():
        np.testing.assert_array_equal(data.values[name], value)
