from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

@pytest.mark.parametrize("shape,axis", [((4,), -1), ((2, 3, 4), -1), ((2, 4, 3), 1), ((4, 2, 1, 3), 0)])
@pytest.mark.parametrize("smoothing", [0., 0.2])
def test_indexed_cross_entropy_and_hessian_for_arbitrary_batch_axes(nn_device, shape, axis, smoothing):
    axis %= len(shape)
    values = np.linspace(-3, 2, np.prod(shape)).reshape(shape)
    target_shape = shape[:axis]+shape[axis+1:]
    labels = (np.arange(np.prod(target_shape, dtype=int)).reshape(target_shape) % shape[axis]).astype(np.int32)
    input = oc.Tensor(values, oc.float64, nn_device)
    target = oc.Tensor(labels, oc.int32, nn_device)
    loss = nn.cross_entropy(input, target, axis=axis, label_smoothing=smoothing)
    first = oc.grad(loss, [input])[0]
    direction = np.linspace(-1, 2, values.size).reshape(shape)
    second = oc.grad((first*oc.Tensor(direction, oc.float64, nn_device)).sum(), [input])[0]
    assert not loss.realized and not first.realized and not second.realized
    oc.Plan([loss, first, second]).run()
    shifted = values-values.max(axis=axis, keepdims=True)
    p = np.exp(shifted)/np.exp(shifted).sum(axis=axis, keepdims=True)
    log_p = shifted-np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    expected_distribution = np.zeros(shape)
    np.put_along_axis(expected_distribution, np.expand_dims(labels, axis), 1, axis=axis)
    expected_distribution = (1-smoothing)*expected_distribution+smoothing/shape[axis]
    expected_loss = -(expected_distribution*log_p).sum(axis=axis).mean()
    expected_first = (p-expected_distribution)/labels.size
    expected_second = p*(direction-(p*direction).sum(axis=axis, keepdims=True))/labels.size
    np.testing.assert_allclose(loss.numpy(), expected_loss, atol=2e-12)
    np.testing.assert_allclose(first.numpy(), expected_first, atol=2e-12)
    np.testing.assert_allclose(second.numpy(), expected_second, atol=2e-11)

def test_soft_targets_reductions_and_ignore_index(nn_device):
    values = np.array([[1000., 999., -1000.], [-1000., -999., -998.]])
    input = oc.Tensor(values, oc.float64, nn_device)
    labels = oc.Tensor([1, -100], oc.int32, nn_device)
    none = nn.cross_entropy(input, labels, reduction="none")
    average = nn.cross_entropy(input, labels)
    total = nn.cross_entropy(input, labels, reduction="sum")
    assert none.shape == (2,)
    np.testing.assert_allclose(none.numpy(), [1+np.log1p(np.exp(-1)), 0], atol=1e-12)
    assert average.item() == total.item()
    first = oc.grad(average, [input])[0]
    np.testing.assert_array_equal(first.numpy()[1], [0, 0, 0])
    logp = nn.log_softmax(input)
    np.testing.assert_allclose(nn.nll_loss(logp, labels).numpy(), average.numpy(), atol=1e-12)
    dense = oc.Tensor([[0.2, 0.8, 0.], [0., 0.1, 0.9]], oc.float64, nn_device)
    shifted = values-values.max(axis=-1, keepdims=True)
    reference = shifted-np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    for reduction in ["none", "sum", "mean"]:
        expected = -(dense.numpy()*reference).sum(axis=-1)

        if reduction == "sum":
            expected = expected.sum()
        elif reduction == "mean":
            expected = expected.mean()

        np.testing.assert_allclose(nn.cross_entropy(input, dense, reduction=reduction).numpy(), expected, atol=1e-12)

def test_classification_target_errors_empty_batches_and_invalid_labels(nn_device):
    input = oc.Tensor([[1., 2., 3.]], oc.float64, nn_device)

    for label in [-1, 3, 2147483647]:
        target = oc.Tensor([label], oc.int32, nn_device)
        # Invalid indices report NaN without an out-of-bounds device read.
        assert np.isnan(nn.cross_entropy(input, target).item())

    for target in [oc.Tensor([1.], oc.float64, nn_device), oc.Tensor([[1]], oc.int32, nn_device)]:
        with pytest.raises(oc.ShapeError):
            nn.cross_entropy(input, target)

    with pytest.raises(oc.OperationError):
        nn.cross_entropy(input, oc.Tensor([1], oc.int32, nn_device), label_smoothing=1.1)

    empty = oc.Tensor(np.empty((0, 3)), oc.float64, nn_device)
    labels = oc.Tensor(np.empty((0,), np.int32), oc.int32, nn_device)
    assert nn.cross_entropy(empty, labels, reduction="none").numpy().shape == (0,)
    assert nn.cross_entropy(empty, labels, reduction="sum").item() == 0
    assert np.isnan(nn.cross_entropy(empty, labels).item())

def test_cross_entropy_replay_preserves_network_and_parameter_storage(nn_device):
    layer = nn.Linear(2, 3, dtype=oc.float64, device=nn_device, seed=9)
    input = oc.Tensor([[1., 0.], [0., 1.]], oc.float64, nn_device)
    target = oc.Tensor([0, 2], oc.int32, nn_device)
    loss = nn.cross_entropy(layer(input), target)
    parameters = layer.parameters()
    gradients = oc.grad(loss, parameters)
    updates = [(parameter, oc.sgd(parameter, gradient, 0.1)) for parameter, gradient in zip(parameters, gradients, strict=True)]
    plan = oc.Plan([loss], updates)
    losses = []

    for _ in range(8):
        plan.run()
        losses.append(loss.item())

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert layer.parameters() == parameters

def test_classification_accepts_strided_logits_and_labels(nn_device):
    values = np.arange(12, dtype=float).reshape(3, 2, 2) / 3
    input = oc.Tensor(values, oc.float64, nn_device)
    label_storage = oc.Tensor([[2, 0], [1, 2]], oc.int32, nn_device)
    target = label_storage.transpose((1, 0))
    logits = input.transpose((1, 2, 0))
    loss = nn.cross_entropy(logits, target)
    gradient = oc.grad(loss, [input])[0]
    logical = values.transpose((1, 2, 0))
    shifted = logical - logical.max(axis=-1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    expected = probability.copy()

    for row in range(2):
        for column in range(2):
            expected[row, column, target.numpy()[row, column]] -= 1

    np.testing.assert_allclose(gradient.numpy(), expected.transpose((2, 0, 1)) / 4, atol=1e-12)

def test_cuda_nn_recording_outlives_the_plan_and_model(nn_device):
    if nn_device == "cpu":
        pytest.skip("CUDA recording test")

    model = nn.Sequential([
        nn.Linear(2, 4, device=nn_device, seed=13),
        nn.GELU(),
        nn.Linear(4, 3, device=nn_device, seed=14),
    ])
    input = oc.Tensor([[1., 0.], [0., 1.]], device=nn_device)
    target = oc.Tensor([0, 2], oc.int32, nn_device)
    loss = nn.cross_entropy(model(input), target)
    parameters = model.parameters()
    gradients = oc.grad(loss, parameters)
    updates = [(parameter, oc.sgd(parameter, gradient, 0.1)) for parameter, gradient in zip(parameters, gradients, strict=True)]
    plan = oc.Plan([loss], updates)
    plan.run()
    initial = loss.item()
    replay = plan.record()
    del plan, model, updates, parameters, gradients

    for _ in range(5):
        replay.run()

    assert np.isfinite(loss.item())
    assert loss.item() < initial
