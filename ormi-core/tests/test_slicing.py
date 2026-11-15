import gc
import weakref

import numpy as np
import pytest

import ormi_core as oc

###
###
###

@pytest.mark.parametrize("index", [
    slice(None), slice(None, None, -1), (), Ellipsis,
    (slice(1, None), slice(None, None, 2), slice(None)),
    (1, slice(None), slice(None, None, -1)),
    (Ellipsis, -1), (slice(None), 1),
    (slice(1, 1), Ellipsis), (slice(None), slice(None, None, -2), 2),
    (np.int64(-1), Ellipsis), (slice(100, 200), Ellipsis),
    (slice(None, None, 2**63 - 1), Ellipsis),
])
def test_slices_match_numpy(index, nn_device):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    tensor = oc.Tensor(values, dtype=oc.float64, device=nn_device)
    result = tensor[index]
    np.testing.assert_array_equal(result.numpy(), values[index])

@pytest.mark.parametrize("dtype, numpy_dtype", [
    (oc.float16, np.float16), (oc.float32, np.float32),
    (oc.float64, np.float64), (oc.int32, np.int32), (oc.bool_, np.bool_),
])
def test_strided_slice_and_concat_preserve_dtype(dtype, numpy_dtype, nn_device):
    values = np.arange(12).reshape(3, 4).astype(numpy_dtype)
    tensor = oc.Tensor(values, dtype=dtype, device=nn_device).transpose((1, 0))
    parts = [tensor[::-1, ::2], tensor[:, :0], tensor]
    result = oc.concat(parts, axis=1)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result.numpy(), np.concatenate([values.T[::-1, ::2], values.T[:, :0], values.T], axis=1))

def test_slicing_preserves_lazy_third_derivatives_and_plan_reuse(nn_device):
    values = np.arange(6, dtype=np.float64).reshape(2, 3)/4
    tensor = oc.Tensor(values, dtype=oc.float64, device=nn_device)
    selected = tensor[::-1, ::2]
    loss = (selected*selected*selected).sum()
    first = oc.grad(loss, tensor)
    second = oc.grad(first.sum(), tensor)
    third = oc.grad(second.sum(), tensor)
    assert not loss.realized and not first.realized and not second.realized and not third.realized
    plan = oc.Plan([loss, first, second, third])
    mask = np.zeros_like(values)
    mask[::-1, ::2] = 1
    for current in [values, -values - 0.25]:
        tensor.copy_from(current)
        plan.run()
        np.testing.assert_allclose(first.numpy(), mask*3*current*current, atol=1e-12)
        np.testing.assert_allclose(second.numpy(), mask*6*current, atol=1e-12)
        np.testing.assert_allclose(third.numpy(), mask*6, atol=1e-12)

def test_concat_accumulates_repeated_parents_and_third_derivatives(nn_device):
    values = np.arange(6, dtype=np.float64).reshape(2, 3)/4
    tensor = oc.Tensor(values, dtype=oc.float64, device=nn_device)
    joined = oc.ops.concat([tensor[:, ::-1], tensor, tensor[:, :0]], axis=-1)
    loss = (joined*joined*joined).sum()
    first = oc.grad(loss, tensor)
    second = oc.grad(first.sum(), tensor)
    third = oc.grad(second.sum(), tensor)
    oc.Plan([loss, first, second, third]).run()
    np.testing.assert_allclose(first.numpy(), 6*values*values, atol=1e-12)
    np.testing.assert_allclose(second.numpy(), 12*values, atol=1e-12)
    np.testing.assert_allclose(third.numpy(), np.full_like(values, 12), atol=1e-12)

def test_integer_selection_and_noncontiguous_seed(nn_device):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    tensor = oc.Tensor(values, dtype=oc.float64, device=nn_device)
    selected = tensor[1, :, ::-1]
    seed_values = np.arange(12, dtype=np.float64).reshape(4, 3)
    seed = oc.Tensor(seed_values, dtype=oc.float64, device=nn_device).transpose((1, 0))
    gradient = oc.grad(selected, tensor, seed)
    expected = np.zeros_like(values)
    expected[1, :, ::-1] = seed_values.T
    np.testing.assert_array_equal(gradient.numpy(), expected)

def test_empty_and_scalar_indexing(nn_device):
    empty = oc.Tensor(np.empty((0, 3)), dtype=oc.float64, device=nn_device)
    assert empty[::-1].shape == (0, 3)
    assert oc.concat([empty, empty]).shape == (0, 3)
    scalar = oc.Tensor(-0.0, dtype=oc.float64, device=nn_device)
    assert scalar[()] is scalar
    assert scalar[...] is scalar
    bits = oc.concat([scalar.reshape((1,)), scalar.reshape((1,))]).numpy().view(np.uint64)
    np.testing.assert_array_equal(bits, np.array([-0.0, -0.0]).view(np.uint64))

def test_index_and_concat_errors():
    tensor = oc.Tensor(np.zeros((2, 3)), dtype=oc.float64)
    for index in [2, -3, (0, 0, 0), (Ellipsis, Ellipsis)]:
        with pytest.raises(IndexError):
            tensor[index]
    for index in [None, True, [0], "x"]:
        with pytest.raises(TypeError):
            tensor[index]
    with pytest.raises(ValueError):
        tensor[::0]
    with pytest.raises(oc.OperationError):
        oc.concat([])
    with pytest.raises(oc.ShapeError):
        oc.concat([tensor, oc.Tensor([1.0], dtype=oc.float64)])
    with pytest.raises(oc.DTypeError):
        oc.concat([tensor, oc.Tensor(np.zeros((2, 3)), dtype=oc.float32)])
    with pytest.raises(oc.ShapeError):
        oc.concat([tensor], axis=2)
    assert oc.concat is oc.ops.concat

def test_grad_overloads_keep_tensor_and_sequence_inputs_distinct():
    tensor = oc.Tensor([1.0, 2.0], dtype=oc.float64)
    loss = (tensor*tensor).sum()
    assert isinstance(oc.grad(loss, tensor), oc.Tensor)
    assert isinstance(oc.grad(loss, input=tensor), oc.Tensor)
    assert isinstance(loss.grad(tensor), oc.Tensor)
    for gradients in [oc.grad(loss, [tensor]), oc.grad(loss, inputs=(tensor,)), loss.grad([tensor])]:
        assert isinstance(gradients, list)
        np.testing.assert_array_equal(gradients[0].numpy(), [2.0, 4.0])

def test_slice_concat_mixed_third_derivatives_and_recording(nn_device):
    values = np.arange(6, dtype=np.float64).reshape(2, 3)/8 - 0.2
    tensor = oc.Tensor(values, dtype=oc.float64, device=nn_device)
    joined = oc.concat([tensor[:, ::-1], tensor[:, ::2], tensor[:, :0]], axis=1)
    total = joined.sum()
    loss = total*total*total*total
    first = oc.grad(loss, tensor)
    mixed = oc.grad(first[0, 0], tensor)
    third = oc.grad(mixed[1, 1], tensor)
    roots = [joined, loss, first, mixed, third]
    assert all(not root.realized for root in roots)
    plan = oc.Plan(roots)
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)
    weights = np.broadcast_to([2.0, 1.0, 2.0], values.shape)
    batches = [values, -values - 0.15, values + 0.3]

    def check(current):
        expected = np.concatenate([current[:, ::-1], current[:, ::2]], axis=1)
        summed = expected.sum()
        np.testing.assert_array_equal(joined.numpy(), expected)
        np.testing.assert_allclose(loss.item(), summed**4, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(first.numpy(), 4*summed**3*weights, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(mixed.numpy(), 24*summed**2*weights, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(third.numpy(), 48*summed*weights, rtol=1e-11, atol=1e-12)

    for current in batches:
        tensor.copy_from(current)
        plan.run()
        check(current)
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

    if nn_device == "cpu":
        return

    before = [root.numpy().copy() for root in roots]
    replay = plan.record()
    for root, expected in zip(roots, before, strict=True):
        np.testing.assert_array_equal(root.numpy(), expected)
    for current in reversed(batches):
        tensor.copy_from(current)
        replay.run()
        check(current)
    reference = weakref.ref(plan)
    del plan
    gc.collect()
    assert reference() is None
    for current in batches:
        tensor.copy_from(current)
        replay.run()
        check(current)
