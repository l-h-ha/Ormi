from __future__ import annotations

import math

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

def sigmoid(value):
    return np.exp(-np.logaddexp(0, -value))

def gelu(value):
    return value * (1 + np.vectorize(math.erf)(value / np.sqrt(2))) / 2

ACTIVATIONS = [
    ("relu", nn.ReLU, lambda x: np.maximum(x, 0)),
    ("leaky_relu", nn.LeakyReLU, lambda x: np.where(x > 0, x, 0.01 * x)),
    ("gelu", nn.GELU, gelu),
    ("silu", nn.SiLU, lambda x: x * sigmoid(x)),
    ("sigmoid", nn.Sigmoid, sigmoid),
    ("tanh", nn.Tanh, np.tanh),
    ("softplus", nn.Softplus, lambda x: np.logaddexp(0, x)),
    ("softsign", nn.Softsign, lambda x: x / (1 + np.abs(x))),
    ("mish", nn.Mish, lambda x: x * np.tanh(np.logaddexp(0, x))),
]

###
###
###

@pytest.mark.parametrize("name,module,reference", ACTIVATIONS)
@pytest.mark.parametrize("dtype", [oc.float16, oc.float32, oc.float64])
def test_activation_modules_match_functions_and_numpy(nn_device, name, module, reference, dtype):
    values = np.array([-1000, -4, -0.2, 0, 0.3, 5, 1000], dtype=np.float64).reshape(1, 1, 7)
    numpy_dtype = {oc.float16: np.float16, oc.float32: np.float32, oc.float64: np.float64}[dtype]
    values = values.astype(numpy_dtype)
    input = oc.Tensor(values, dtype, nn_device)
    functional = getattr(nn, name)(input)
    output = module()(input)
    assert output.shape == values.shape
    assert output.dtype == dtype
    assert not output.realized and not functional.realized
    oc.Plan([output, functional]).run()
    tolerance = {oc.float16: 2e-3, oc.float32: 2e-6, oc.float64: 1e-12}[dtype]
    np.testing.assert_allclose(output.numpy(), reference(values.astype(np.float64)), rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(output.numpy(), functional.numpy())

@pytest.mark.parametrize("name,module,reference", ACTIVATIONS)
def test_activations_accept_scalars_empty_and_strided_tensors(nn_device, name, module, reference):
    input = oc.Tensor(np.arange(12, dtype=np.float64).reshape(3, 4) / 4 - 1, oc.float64, nn_device)
    output = module()(input.transpose((1, 0)))
    np.testing.assert_allclose(output.numpy(), reference(input.numpy().T), atol=1e-12)
    assert module()(oc.Tensor(0.25, oc.float64, nn_device)).shape == ()
    assert module()(oc.Tensor(np.empty((0, 3)), oc.float64, nn_device)).numpy().shape == (0, 3)

@pytest.mark.parametrize("name", ["tanh", "sigmoid", "softplus", "gelu"])
def test_smooth_activations_have_lazy_third_order_derivatives(nn_device, name):
    values = np.array([-1.5, -0.25, 0, 0.7, 2], dtype=np.float64)
    input = oc.Tensor(values, oc.float64, nn_device)
    roots = [getattr(nn, name)(input)]

    for _ in range(3):
        roots.append(oc.grad(roots[-1].sum(), [input])[0])

    assert all(not root.realized for root in roots)
    oc.Plan(roots).run()
    s = sigmoid(values)

    if name == "tanh":
        t = np.tanh(values)
        expected = [1 - t*t, -2*t*(1-t*t), (-2+6*t*t)*(1-t*t)]
    elif name == "sigmoid":
        expected = [s*(1-s), s*(1-s)*(1-2*s), s*(1-s)*(1-6*s+6*s*s)]
    elif name == "softplus":
        expected = [s, s*(1-s), s*(1-s)*(1-2*s)]
    else:
        density = np.exp(-values*values/2) / np.sqrt(2*np.pi)
        cdf = (1 + np.vectorize(math.erf)(values/np.sqrt(2))) / 2
        expected = [cdf+values*density, (2-values*values)*density, (values**3-4*values)*density]

    for actual, reference in zip(roots[1:], expected, strict=True):
        np.testing.assert_allclose(actual.numpy(), reference, rtol=2e-11, atol=2e-12)

def test_piecewise_gradient_conventions(nn_device):
    input = oc.Tensor([-2.0, 0.0, 3.0], oc.float64, nn_device)
    relu = nn.relu(input)
    leaky = nn.leaky_relu(input, 0.2)
    np.testing.assert_allclose(oc.grad(relu.sum(), [input])[0].numpy(), [0, 0, 1])
    np.testing.assert_allclose(oc.grad(leaky.sum(), [input])[0].numpy(), [0.2, 0.2, 1])
    first = oc.grad(leaky.sum(), [input])[0]
    np.testing.assert_array_equal(oc.grad(first.sum(), [input])[0].numpy(), [0, 0, 0])

@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_and_log_softmax_are_stable_for_general_class_axes(nn_device, axis):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4) * 1000 - 7000
    input = oc.Tensor(values, oc.float64, nn_device)
    shifted = values - values.max(axis=axis, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    probability = nn.Softmax(axis)(input)
    logarithm = nn.LogSoftmax(axis)(input)
    assert not probability.realized and not logarithm.realized
    oc.Plan([probability, logarithm]).run()
    np.testing.assert_allclose(probability.numpy(), np.exp(log_probability), atol=1e-12)
    np.testing.assert_allclose(logarithm.numpy(), log_probability, atol=1e-12)

def test_activation_errors_and_nan_propagation(nn_device):
    for name, module, _ in ACTIVATIONS:
        with pytest.raises(oc.OperationError):
            module()(None)

        with pytest.raises(oc.DTypeError):
            getattr(nn, name)(oc.Tensor([1], oc.int32, nn_device))

        assert np.isnan(module()(oc.Tensor([np.nan], oc.float64, nn_device)).numpy()).all()

    for slope in [-0.1, np.inf, np.nan]:
        with pytest.raises(oc.OperationError):
            nn.LeakyReLU(slope)

    with pytest.raises(oc.ShapeError):
        nn.softmax(oc.Tensor(1.0, device=nn_device))

    with pytest.raises(oc.ShapeError):
        nn.softmax(oc.Tensor(np.empty((2, 0)), device=nn_device))
