from __future__ import annotations

import gc

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

@pytest.mark.parametrize("shape,axis", [((4, 3), -1), ((2, 3, 4), 1), ((2, 1, 3, 2, 4), -2)])
@pytest.mark.parametrize("dtype,np_dtype,tolerance", [
    (oc.float16, np.float16, 4e-3), (oc.float32, np.float32, 3e-5), (oc.float64, np.float64, 3e-12),
])
def test_batch_norm_generalized_axes_gradients_and_statistics(nn_device, shape, axis, dtype, np_dtype, tolerance):
    axis %= len(shape)
    axes = tuple(i for i in range(len(shape)) if i != axis)
    features = shape[axis]
    broadcast = [1] * len(shape)
    broadcast[axis] = features
    values = np.sin(np.arange(np.prod(shape))).reshape(shape).astype(np_dtype)
    direction = np.cos(np.arange(values.size)).reshape(shape).astype(np_dtype)
    layer = nn.BatchNorm(features, eps=0.03, momentum=0.25, axis=axis, dtype=dtype, device=nn_device)
    input = oc.Tensor(values, dtype, nn_device)
    output = layer(input)
    variables = [input, layer.weight, layer.bias]
    gradients = oc.grad((output * oc.Tensor(direction, dtype, nn_device)).sum(), variables)
    assert not output.realized and all(not gradient.realized for gradient in gradients)
    np.testing.assert_array_equal(layer.running_mean.numpy(), np.zeros(features))
    assert layer.parameters() == variables[1:]
    assert layer.buffers() == [layer.running_mean, layer.running_var]
    assert layer.running_mean.dtype == (oc.float32 if dtype == oc.float16 else dtype)
    oc.Plan([output, *gradients]).run()

    precise = values.astype(np.float64)
    average = precise.mean(axis=axes, keepdims=True)
    centered = precise - average
    variance = (centered**2).mean(axis=axes, keepdims=True)
    inverse = 1 / np.sqrt(variance + 0.03)
    normalized = centered * inverse
    upstream = direction.astype(np.float64)
    expected_input = inverse * (
        upstream - upstream.mean(axis=axes, keepdims=True)
        - centered * inverse**2 * (upstream * centered).mean(axis=axes, keepdims=True)
    )
    expected_weight = (normalized * upstream).sum(axis=axes)
    expected_bias = upstream.sum(axis=axes)

    for actual, expected in zip([output, *gradients], [normalized, expected_input, expected_weight, expected_bias]):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=tolerance, atol=tolerance)

    count = values.size // features
    np.testing.assert_allclose(layer.running_mean.numpy(), average.reshape(-1) * 0.25, atol=tolerance)
    np.testing.assert_allclose(
        layer.running_var.numpy(), 0.75 + variance.reshape(-1) * count / (count - 1) * 0.25,
        rtol=tolerance, atol=tolerance,
    )

def test_batch_norm_reused_plan_and_recording_update_state(nn_device):
    layer = nn.BatchNorm(3, momentum=0.2, dtype=oc.float64, device=nn_device)
    values = np.array([[1, 3, 7], [3, 9, -1]], dtype=np.float64)
    input = oc.Tensor(values, oc.float64, nn_device)
    output = layer(input)
    plan = oc.Plan([output])
    executions = 0

    def check():
        retained = 0.8**executions
        np.testing.assert_allclose(layer.running_mean.numpy(), values.mean(0) * (1 - retained), atol=1e-12)
        np.testing.assert_allclose(
            layer.running_var.numpy(), retained + values.var(0, ddof=1) * (1 - retained), atol=1e-12,
        )

    for _ in range(3):
        plan.run()
        executions += 1
        check()

    if nn_device == "cuda:0":
        replay = plan.record()
        # Recording captures the next execution without executing its device work.
        check()
        del plan
        gc.collect()

        for _ in range(3):
            replay.run()
            executions += 1
            check()

    layer.eval()
    inference = layer(input)
    expected = (values - layer.running_mean.numpy()) / np.sqrt(layer.running_var.numpy() + layer.eps)
    np.testing.assert_allclose(inference.numpy(), expected, atol=1e-12)
    check()

def test_batch_norm_shared_layer_calls_follow_construction_order(nn_device):
    layer = nn.BatchNorm(2, momentum=0.5, dtype=oc.float64, device=nn_device)
    a = np.array([[1, 2], [3, 6]], dtype=np.float64)
    b = a + 10
    first = layer(oc.Tensor(a, oc.float64, nn_device))
    second = layer(oc.Tensor(b, oc.float64, nn_device))
    plan = oc.Plan([second, first])  # Root order must not reverse state updates.
    mean = np.zeros(2)
    variance = np.ones(2)

    for _ in range(3):
        plan.run()

        for values in [a, b]:
            mean = mean * 0.5 + values.mean(0) * 0.5
            variance = variance * 0.5 + values.var(0, ddof=1) * 0.5

        np.testing.assert_allclose(layer.running_mean.numpy(), mean, atol=1e-12)
        np.testing.assert_allclose(layer.running_var.numpy(), variance, atol=1e-12)

def test_batch_norm_gradient_only_plan_keeps_updates_and_third_derivatives(nn_device):
    layer = nn.BatchNorm(1, eps=0.3, momentum=0.25, affine=False, dtype=oc.float64, device=nn_device)
    values = np.array([[-0.6], [1.2]])
    input = oc.Tensor(values, oc.float64, nn_device)
    direction = oc.Tensor([[1.0], [-1.0]], oc.float64, nn_device)
    output = layer(input)
    first = oc.grad((output * direction).sum(), [input])[0]
    second = oc.grad((first * direction).sum(), [input])[0]
    third = oc.grad((second * direction).sum(), [input])[0]
    assert not output.realized and not third.realized
    plan = oc.Plan([third])
    difference = (values[0, 0] - values[1, 0]) / 2
    coefficient = 0.9 * (4 * difference**2 - 0.3) * (difference**2 + 0.3)**-3.5

    for execution in range(1, 4):
        plan.run()
        np.testing.assert_allclose(third.numpy(), [[coefficient], [-coefficient]], atol=1e-11)
        np.testing.assert_allclose(layer.running_mean.numpy(), values.mean(0) * (1 - 0.75**execution))

def test_batch_norm_shared_layer_gradients_keep_each_update_once(nn_device):
    layer = nn.BatchNorm(2, momentum=0.5, dtype=oc.float64, device=nn_device)
    values = np.array([[1.0, 2], [3, 7], [-2, -3]])
    input = oc.Tensor(values, oc.float64, nn_device)
    a = layer(input)
    b = layer(input + 10)
    direction = oc.Tensor([[1.0, -1], [2, 0.5], [-1, 3]], oc.float64, nn_device)
    gradient = oc.grad(((a + b) * direction).sum(), [input])[0]
    plan = oc.Plan([gradient])
    mean = np.zeros(2)
    variance = np.ones(2)

    for _ in range(3):
        plan.run()

        for batch in [values, values + 10]:
            mean = mean * 0.5 + batch.mean(0) * 0.5
            variance = variance * 0.5 + batch.var(0, ddof=1) * 0.5

        np.testing.assert_allclose(layer.running_mean.numpy(), mean, atol=1e-12)
        np.testing.assert_allclose(layer.running_var.numpy(), variance, atol=1e-12)
        assert np.isfinite(gradient.numpy()).all()

def test_batch_norm_mixed_half_state_and_momentum_endpoints(nn_device):
    values = (1000 + np.arange(24).reshape(2, 3, 4) * 2).astype(np.float16)
    input = oc.Tensor(values, oc.float16, nn_device)

    for momentum in [0.0, 1.0]:
        layer = nn.BatchNorm(4, momentum=momentum, device=nn_device)
        output = layer(input)
        assert output.dtype == oc.float16
        centered = values.astype(np.float32) - values.astype(np.float32).mean(axis=(0, 1), keepdims=True)
        expected = centered / np.sqrt((centered**2).mean(axis=(0, 1), keepdims=True) + 1e-5)
        np.testing.assert_allclose(output.numpy(), expected.astype(np.float16), rtol=2e-3, atol=2e-3)
        expected_mean = values.astype(np.float32).mean(axis=(0, 1)) * momentum
        np.testing.assert_allclose(layer.running_mean.numpy(), expected_mean, atol=1e-5)

def test_batch_norm_untracked_eval_empty_batches_and_strided_input(nn_device):
    layer = nn.BatchNorm(3, affine=False, track_running_stats=False, axis=0, device=nn_device)
    layer.eval()
    assert layer.parameters() == [] and layer.buffers() == []
    assert layer.running_mean is None and layer.running_var is None
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    input = oc.Tensor(values, device=nn_device).transpose((1, 0, 2))
    precise = values.transpose((1, 0, 2))
    centered = precise - precise.mean(axis=(1, 2), keepdims=True)
    expected = centered / np.sqrt((centered**2).mean(axis=(1, 2), keepdims=True) + 1e-5)
    np.testing.assert_allclose(layer(input).numpy(), expected, rtol=3e-6, atol=3e-6)
    tracked = nn.BatchNorm(3, device=nn_device)
    empty = tracked(oc.Tensor(np.empty((0, 3)), device=nn_device))
    assert empty.numpy().shape == (0, 3)
    np.testing.assert_array_equal(tracked.running_var.numpy(), np.ones(3))
    tracked.eval()
    assert tracked(oc.Tensor([[1, 2, 3]], device=nn_device)).numpy().shape == (1, 3)

def test_batch_norm_invalid_configuration_is_lazy(nn_device):
    for kwargs in [{"num_features": 0}, {"num_features": -1}, {"num_features": 2, "eps": 0},
                   {"num_features": 2, "momentum": np.nan}, {"num_features": 2, "momentum": -0.1},
                   {"num_features": 2, "momentum": 1.1}]:
        with pytest.raises((oc.ShapeError, oc.OperationError)):
            nn.BatchNorm(**kwargs)

    source = oc.Tensor(np.ones((2, 3)), device=nn_device)
    input = source + 0.5

    for layer in [nn.BatchNorm(2, device=nn_device), nn.BatchNorm(3, axis=3, device=nn_device)]:
        with pytest.raises(oc.ShapeError):
            layer(input)

    assert not input.realized

    with pytest.raises(oc.ShapeError):
        nn.BatchNorm(3, device=nn_device)(oc.Tensor([[1, 2, 3]], device=nn_device))
