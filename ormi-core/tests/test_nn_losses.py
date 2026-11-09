from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

@pytest.mark.parametrize("name", ["mse_loss", "l1_loss", "huber_loss", "smooth_l1_loss"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_regression_loss_values_and_gradients(nn_device, name, reduction):
    values = np.array([[[-3., -0.2, 0., 0.2, 3.]]])
    input = oc.Tensor(values, oc.float64, nn_device)
    target = oc.zeros_like(input)
    kwargs = {"delta": 0.5} if name == "huber_loss" else {"beta": 0.5} if name == "smooth_l1_loss" else {}
    loss = getattr(nn, name)(input, target, reduction=reduction, **kwargs)
    first = oc.grad(loss.sum(), [input])[0]
    second = oc.grad(first.sum(), [input])[0]
    assert not loss.realized and not first.realized and not second.realized
    oc.Plan([loss, first, second]).run()
    magnitude = np.abs(values)

    if name == "mse_loss":
        expected, gradient, curvature = values**2, 2*values, np.full_like(values, 2)
    elif name == "l1_loss":
        expected, gradient, curvature = magnitude, np.sign(values), np.zeros_like(values)
    else:
        expected = np.where(magnitude <= 0.5, values**2/2, 0.5*(magnitude-0.25))
        gradient = np.clip(values, -0.5, 0.5)
        curvature = (magnitude < 0.5).astype(float)

        if name == "smooth_l1_loss":
            expected, gradient, curvature = expected/0.5, gradient/0.5, curvature/0.5

    if reduction == "sum":
        expected = expected.sum()
    elif reduction == "mean":
        expected = expected.mean()
        gradient, curvature = gradient/values.size, curvature/values.size

    np.testing.assert_allclose(loss.numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(first.numpy(), gradient, atol=1e-12)
    np.testing.assert_allclose(second.numpy(), curvature, atol=1e-12)

def test_regression_losses_promote_half_and_preserve_shape_validation(nn_device):
    input = oc.Tensor([1000., -1000.], oc.float16, nn_device)
    target = oc.zeros_like(input)
    assert nn.mse_loss(input, target).dtype == oc.float32
    assert nn.mse_loss(input, target).item() == 1000000
    np.testing.assert_array_equal(nn.smooth_l1_loss(input, target, beta=0, reduction="none").numpy(), [1000, 1000])

    for loss in [nn.mse_loss, nn.l1_loss, nn.huber_loss, nn.smooth_l1_loss]:
        with pytest.raises(oc.ShapeError):
            loss(input, oc.Tensor([1.], device=nn_device))

        with pytest.raises(oc.OperationError):
            loss(input, target, reduction="median")

    with pytest.raises(oc.OperationError):
        nn.huber_loss(input, target, delta=0)

    with pytest.raises(oc.OperationError):
        nn.smooth_l1_loss(input, target, beta=-1)

@pytest.mark.parametrize("dtype", [oc.float16, oc.float32, oc.float64])
def test_binary_logits_loss_is_stable_and_twice_differentiable(nn_device, dtype):
    values = np.array([-1000., -20., -1., 0., 1., 20., 1000.])
    labels = np.array([0., 1., 0.3, 1., 0.7, 0., 1.])
    input = oc.Tensor(values, dtype, nn_device)
    target = oc.Tensor(labels, dtype, nn_device)
    weight = oc.Tensor([2.], dtype, nn_device)
    loss = nn.binary_cross_entropy_with_logits(input, target, reduction="none", pos_weight=weight)
    first = oc.grad(loss.sum(), [input])[0]
    second = oc.grad(first.sum(), [input])[0]
    assert all(not value.realized for value in [loss, first, second])
    oc.Plan([loss, first, second]).run()
    labels = target.numpy().astype(float)
    s = np.exp(-np.logaddexp(0, -values))
    expected = 2*labels*np.logaddexp(0, -values)+(1-labels)*np.logaddexp(0, values)
    gradient = (1-labels)*s-2*labels*(1-s)
    curvature = ((1-labels)+2*labels)*s*(1-s)
    tolerance = 1e-3 if dtype == oc.float16 else 2e-6 if dtype == oc.float32 else 1e-12
    np.testing.assert_allclose(loss.numpy(), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(first.numpy(), gradient, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(second.numpy(), curvature, rtol=tolerance, atol=tolerance)

def test_probability_bce_clipping_and_kl_zero_targets(nn_device):
    input = oc.Tensor([0., 0.25, 0.75, 1.], oc.float64, nn_device)
    target = oc.Tensor([0., 1., 0., 1.], oc.float64, nn_device)
    probability = np.clip(input.numpy(), 1e-7, 1-1e-7)
    expected = -target.numpy()*np.log(probability)-(1-target.numpy())*np.log1p(-probability)
    np.testing.assert_allclose(nn.binary_cross_entropy(input, target, reduction="none").numpy(), expected, atol=1e-12)
    assert np.isfinite(nn.binary_cross_entropy(input, target, eps=1e-300).item())

    for eps in [0, 0.5, np.nan]:
        with pytest.raises(oc.OperationError):
            nn.binary_cross_entropy(input, target, eps=eps)

    log_probabilities = oc.Tensor(np.log([[0.4, 0.6], [0.5, 0.5]]), oc.float64, nn_device)
    distribution = oc.Tensor([[0., 1.], [0.5, 0.5]], oc.float64, nn_device)
    loss = nn.kl_div(log_probabilities, distribution)
    first = oc.grad(loss, [log_probabilities])[0]
    np.testing.assert_allclose(loss.numpy(), -np.log(0.6)/2, atol=1e-12)
    np.testing.assert_array_equal(first.numpy(), -distribution.numpy()/2)
    log_target = oc.Tensor([[-np.inf, 0.], [np.log(0.5), np.log(0.5)]], oc.float64, nn_device)
    np.testing.assert_allclose(nn.kl_div(log_probabilities, log_target, log_target=True).numpy(), loss.numpy(), atol=1e-12)
