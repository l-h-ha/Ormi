from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

def normalization(values, normalized_shape, eps):
    axes = tuple(range(values.ndim - len(normalized_shape), values.ndim))
    centered = values - values.mean(axis=axes, keepdims=True)
    inverse_stddev = 1 / np.sqrt((centered * centered).mean(axis=axes, keepdims=True) + eps)
    return centered, inverse_stddev, centered * inverse_stddev, axes

###
###
###

@pytest.mark.parametrize("shape,normalized_shape", [
    ((4,), (4,)),
    ((2, 3), (2, 3)),
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 4)),
    ((2, 1, 3, 2, 4), (2, 4)),
])
@pytest.mark.parametrize("dtype,numpy_dtype,tolerance", [
    (oc.float32, np.float32, 3e-5),
    (oc.float64, np.float64, 2e-12),
])
def test_layer_norm_generalized_shapes_and_affine_gradients(
    nn_device, shape, normalized_shape, dtype, numpy_dtype, tolerance
):
    values = np.linspace(-2, 3, np.prod(shape)).reshape(shape).astype(numpy_dtype)
    weights = np.linspace(0.5, 1.5, np.prod(normalized_shape)).reshape(normalized_shape).astype(numpy_dtype)
    biases = np.linspace(-0.3, 0.2, weights.size).reshape(normalized_shape).astype(numpy_dtype)
    direction = np.cos(np.arange(values.size)).reshape(shape).astype(numpy_dtype)
    input = oc.Tensor(values, dtype, nn_device)
    weight = oc.Tensor(weights, dtype, nn_device)
    bias = oc.Tensor(biases, dtype, nn_device)
    output = nn.layer_norm(input, normalized_shape, weight, bias, eps=0.03)
    objective = (output * oc.Tensor(direction, dtype, nn_device)).sum()
    gradients = oc.grad(objective, [input, weight, bias])
    assert not output.realized and all(not gradient.realized for gradient in gradients)
    assert output.dtype == dtype and output.shape == shape
    oc.Plan([output, *gradients]).run()

    centered, inverse_stddev, normalized, axes = normalization(values.astype(np.float64), normalized_shape, 0.03)
    upstream = direction * weights
    average_gradient = upstream.mean(axis=axes, keepdims=True)
    average_product = (upstream * centered).mean(axis=axes, keepdims=True)
    expected_input = inverse_stddev * (
        upstream - average_gradient - centered * inverse_stddev**2 * average_product
    )
    leading_axes = tuple(range(values.ndim - len(normalized_shape)))
    expected_weight = direction * normalized
    expected_bias = direction

    if leading_axes:
        expected_weight = expected_weight.sum(axis=leading_axes)
        expected_bias = expected_bias.sum(axis=leading_axes)

    for actual, expected in zip(
        [output, *gradients],
        [normalized * weights + biases, expected_input, expected_weight, expected_bias],
        strict=True,
    ):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=tolerance, atol=tolerance)

@pytest.mark.parametrize("dtype", [oc.float16, oc.float32, oc.float64])
def test_layer_norm_module_owns_initialized_leaf_parameters(nn_device, dtype):
    layer = nn.LayerNorm((2, 3), eps=0.02, dtype=dtype, device=nn_device)
    assert layer.normalized_shape == (2, 3) and layer.eps == 0.02
    assert layer.parameters() == [layer.weight, layer.bias]

    for parameter, expected in [(layer.weight, 1), (layer.bias, 0)]:
        assert parameter.realized and parameter.op == "leaf"
        assert parameter.dtype == dtype and parameter.shape == (2, 3)
        np.testing.assert_array_equal(parameter.numpy(), np.full((2, 3), expected))

    input = oc.Tensor(np.arange(24).reshape(4, 2, 3), dtype, nn_device)
    output = layer(input)
    functional = nn.layer_norm(input, (2, 3), layer.weight, layer.bias, layer.eps)
    assert not output.realized
    oc.Plan([output, functional]).run()
    np.testing.assert_array_equal(output.numpy(), functional.numpy())
    assert nn.LayerNorm(3, elementwise_affine=False).parameters() == []
    no_bias = nn.LayerNorm(3, bias=False)
    assert no_bias.bias is None and no_bias.parameters() == [no_bias.weight]

def test_layer_norm_accepts_strided_lazy_inputs_and_affine_tensors(nn_device):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 5
    source = oc.Tensor(values, oc.float64, nn_device)
    input = (source + 0.25).transpose((1, 0, 2))
    weight_values = np.arange(8, dtype=float).reshape(4, 2).T / 4 + 0.5
    weight = oc.Tensor(weight_values.T.copy(), oc.float64, nn_device).transpose((1, 0))
    bias = weight * 0.1
    output = nn.layer_norm(input, (2, 4), weight, bias)
    assert not input.realized and not weight.is_contiguous() and not bias.realized
    _, _, normalized, _ = normalization((values + 0.25).transpose((1, 0, 2)), (2, 4), 1e-5)
    np.testing.assert_allclose(output.numpy(), normalized * weight_values + weight_values * 0.1, atol=2e-12)

def test_layer_norm_third_order_derivatives_match_closed_form(nn_device):
    values = np.array([-0.6, 1.2], dtype=np.float64)
    input = oc.Tensor(values, oc.float64, nn_device)
    direction = oc.Tensor([1.0, -1.0], oc.float64, nn_device)
    eps = 0.3
    output = nn.layer_norm(input, 2, eps=eps)
    first = oc.grad((output * direction).sum(), [input])[0]
    second = oc.grad((first * direction).sum(), [input])[0]
    third = oc.grad((second * direction).sum(), [input])[0]
    assert all(not tensor.realized for tensor in [output, first, second, third])
    oc.Plan([output, first, second, third]).run()
    difference = (values[0] - values[1]) / 2
    variance = difference**2 + eps
    coefficients = [
        eps * variance**-1.5,
        -3 * eps * difference * variance**-2.5,
        3 * eps * (4 * difference**2 - eps) * variance**-3.5,
    ]

    for actual, coefficient in zip([first, second, third], coefficients, strict=True):
        np.testing.assert_allclose(actual.numpy(), [coefficient, -coefficient], rtol=2e-11, atol=2e-12)

def test_layer_norm_float16_uses_float32_statistics_and_affine_parameters(nn_device):
    values = (1000 + np.arange(24).reshape(2, 3, 4) * 2).astype(np.float16)
    layer = nn.LayerNorm((3, 4), eps=1e-8, device=nn_device)
    input = oc.Tensor(values, oc.float16, nn_device)
    output = layer(input)
    _, _, normalized, _ = normalization(values.astype(np.float32), (3, 4), 1e-8)
    assert output.dtype == oc.float16
    np.testing.assert_allclose(output.numpy(), normalized.astype(np.float16), atol=1e-3, rtol=1e-3)
    gradients = oc.grad((output * output).sum(), [input, layer.weight, layer.bias])
    assert [gradient.dtype for gradient in gradients] == [oc.float16, oc.float32, oc.float32]
    assert all(np.isfinite(gradient.numpy()).all() for gradient in gradients)

def test_layer_norm_constant_rows_empty_batches_and_optional_affine(nn_device):
    layer = nn.LayerNorm(4, device=nn_device)
    input = oc.Tensor(np.full((2, 4), 1000), device=nn_device)
    output = layer(input)
    np.testing.assert_array_equal(output.numpy(), np.zeros((2, 4)))
    np.testing.assert_array_equal(oc.grad(output.sum(), [input])[0].numpy(), np.zeros((2, 4)))
    empty = oc.Tensor(np.empty((0, 2, 4)), device=nn_device)
    result = layer(empty)
    assert result.numpy().shape == (0, 2, 4)
    gradients = oc.grad(result.sum(), layer.parameters())
    assert all(np.count_nonzero(gradient.numpy()) == 0 for gradient in gradients)
    bias = oc.Tensor([1.0, 2, 3, 4], device=nn_device)
    np.testing.assert_array_equal(nn.layer_norm(input, 4, bias=bias).numpy(), np.tile(bias.numpy(), (2, 1)))

def test_layer_norm_can_train_in_a_reused_plan(nn_device):
    layer = nn.LayerNorm(3, dtype=oc.float64, device=nn_device)
    input = oc.Tensor([[1.0, 2.0, -1.0], [-2.0, 0.0, 3.0]], oc.float64, nn_device)
    output = layer(input)
    loss = (output * output).mean()
    parameters = layer.parameters()
    gradients = oc.grad(loss, parameters)
    updates = [(parameter, oc.sgd(parameter, gradient, 0.05))
               for parameter, gradient in zip(parameters, gradients, strict=True)]
    plan = oc.Plan([loss], updates)
    losses = []

    for _ in range(6):
        plan.run()
        losses.append(loss.item())

    assert np.isfinite(losses).all() and losses[-1] < losses[0]
    assert layer.parameters() == parameters

    if nn_device == "cuda:0":
        replay = plan.record()
        del plan, layer, parameters, gradients, updates
        replay.run()
        assert np.isfinite(loss.item())

def test_layer_norm_rejects_invalid_metadata_without_realizing_input(nn_device):
    source = oc.Tensor(np.arange(6).reshape(2, 3), device=nn_device)
    input = source + 0.25

    for shape in [(), (0,), (-1,), (2,), (1, 2, 3)]:
        with pytest.raises(oc.ShapeError):
            nn.layer_norm(input, shape)

    for eps in [0, -1, np.inf, np.nan, 1e-300, 1e100]:
        with pytest.raises(oc.OperationError):
            nn.layer_norm(input, 3, eps=eps)

    for eps in [0, -1, np.inf, np.nan]:
        with pytest.raises(oc.OperationError):
            nn.LayerNorm(3, eps=eps)

    for shape in [(), (0,), (-1,)]:
        with pytest.raises(oc.ShapeError):
            nn.LayerNorm(shape)

    with pytest.raises(oc.OperationError):
        nn.layer_norm(None, 3)

    with pytest.raises(oc.DTypeError):
        nn.layer_norm(oc.Tensor([1, 2, 3], oc.int32, nn_device), 3)

    with pytest.raises(oc.DTypeError):
        nn.LayerNorm(3, dtype=oc.int32)

    with pytest.raises(oc.ShapeError):
        nn.layer_norm(input, 3, weight=source)

    with pytest.raises(oc.DTypeError):
        nn.layer_norm(input, 3, weight=oc.Tensor([1.0, 1, 1], oc.float64, nn_device))

    if nn_device == "cuda:0":
        with pytest.raises(oc.DeviceError):
            nn.layer_norm(input, 3, bias=oc.Tensor([0.0, 0, 0]))

    assert not input.realized
