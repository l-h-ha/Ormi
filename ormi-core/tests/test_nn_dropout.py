from __future__ import annotations

import gc

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

@pytest.mark.parametrize("dtype", [oc.float16, oc.float32, oc.float64])
def test_dropout_mask_distribution_scale_and_repeatability(nn_device, dtype):
    input = oc.Tensor(np.ones((4, 5, 1000)), dtype, nn_device)
    first = nn.Dropout(p=0.25, seed=2**64 - 1)
    second = nn.Dropout(p=0.25, seed=2**64 - 1)
    assert first.parameters() == [] and first.buffers() == []
    assert first.p == 0.25 and first.seed == 2**64 - 1

    for _ in range(2):
        a = first(input)
        b = second(input)
        assert not a.realized and a.shape == input.shape and a.dtype == dtype
        np.testing.assert_array_equal(a.numpy(), b.numpy())
        values = a.numpy()
        assert abs(np.count_nonzero(values) / values.size - 0.75) < 0.02
        np.testing.assert_allclose(values[values != 0], 4 / 3, rtol=5e-4)

def test_dropout_reused_plan_refreshes_masks_and_reuses_them_for_gradients(nn_device):
    values = np.linspace(0.5, 1.5, 256).reshape(2, 4, 32)
    input = oc.Tensor(values, oc.float64, nn_device)
    layer = nn.Dropout(0.5, seed=73)
    output = layer(input * input)
    first = oc.grad(output.sum(), [input])[0]
    second = oc.grad(first.sum(), [input])[0]
    third = oc.grad(second.sum(), [input])[0]
    plan = oc.Plan([output, first, second, third])
    previous = None

    def check():
        nonlocal previous
        result = output.numpy()
        mask = np.where(result == 0, 0.0, 2.0)

        if previous is not None:
            assert not np.array_equal(result, previous)

        previous = result.copy()
        np.testing.assert_allclose(result, values**2 * mask, atol=1e-12)
        np.testing.assert_allclose(first.numpy(), 2 * values * mask, atol=1e-12)
        np.testing.assert_array_equal(second.numpy(), 2 * mask)
        np.testing.assert_array_equal(third.numpy(), np.zeros_like(values))

    for _ in range(3):
        plan.run()
        check()

    layer.eval()  # This prepared graph keeps its training behavior after the mode change.
    assert layer(input) is input

    if nn_device == "cuda:0":
        replay = plan.record()
        del plan, layer
        gc.collect()

        for _ in range(3):
            replay.run()
            check()

def test_dropout_gradient_only_execution_does_not_generate_a_second_mask(nn_device):
    input = oc.Tensor(np.ones(1024), device=nn_device)
    output = nn.dropout(input, seed=15)
    gradient = oc.grad(output.sum(), [input])[0]
    plan = oc.Plan([gradient])
    previous = None

    for _ in range(3):
        plan.run()
        current = gradient.numpy()
        assert set(np.unique(current)) == {0, 2}

        if previous is not None:
            assert not np.array_equal(current, previous)

        previous = current.copy()

    np.testing.assert_array_equal(output.numpy(), gradient.numpy())

def test_dropout_discards_unexecuted_graphs_without_consuming_randomness(nn_device):
    input = oc.Tensor(np.ones(1024), device=nn_device)
    layer = nn.Dropout(seed=14)
    unused = layer(input)
    del unused
    actual = layer(input)
    reference = nn.dropout(input, seed=14)
    np.testing.assert_array_equal(actual.numpy(), reference.numpy())
    strided = oc.Tensor(np.ones((32, 32)), device=nn_device).transpose((1, 0))
    output = nn.dropout(strided + 0.5, p=0.5, seed=42)
    assert output.shape == (32, 32)
    assert set(np.unique(output.numpy())) == {0, 3}

def test_dropout_probability_edges_and_validation(nn_device):
    input = oc.Tensor([1.0, 2, 3], device=nn_device)
    assert nn.dropout(input, p=0) is input
    assert nn.dropout(input, training=False) is input
    all_dropped = nn.dropout(input, p=1)
    np.testing.assert_array_equal(all_dropped.numpy(), np.zeros(3))
    np.testing.assert_array_equal(oc.grad(all_dropped.sum(), [input])[0].numpy(), np.zeros(3))

    for p in [-0.01, 1.01, np.inf, np.nan]:
        with pytest.raises(oc.OperationError):
            nn.Dropout(p)

        with pytest.raises(oc.OperationError):
            nn.dropout(input, p, training=False)

    with pytest.raises(oc.DTypeError):
        nn.dropout(oc.Tensor([1, 2], oc.int32, nn_device))

    empty = nn.dropout(oc.Tensor(np.empty((0, 4)), device=nn_device), seed=1)
    assert empty.numpy().shape == (0, 4)
