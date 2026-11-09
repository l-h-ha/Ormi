import numpy as np
import pytest

import ormi_core as oc

###
###
###

@pytest.mark.parametrize("dtype,np_dtype", [
    (oc.float16, np.float16), (oc.float32, np.float32), (oc.float64, np.float64),
    (oc.int32, np.int32), (oc.bool_, np.bool_),
])
@pytest.mark.parametrize("shape", [(), (2, 1, 3), (2, 0, 3)])
def test_copy_from_numpy_and_tensor_preserves_leaf_metadata(nn_device, dtype, np_dtype, shape):
    values = np.asarray(np.arange(np.prod(shape)).reshape(shape) + 1, dtype=np_dtype)
    destination = oc.Tensor(np.zeros(shape, dtype=np_dtype), dtype, nn_device)
    assert destination.copy_from(values) is None
    np.testing.assert_array_equal(destination.numpy(), values)
    source = oc.Tensor(values, dtype, nn_device)
    destination.copy_from(source)
    destination.copy_from(destination)
    np.testing.assert_array_equal(destination.numpy(), values)
    assert destination.op == "leaf" and destination.parents == []
    assert destination.shape == shape and destination.dtype == dtype

def test_copy_from_handles_strided_and_readonly_numpy_arrays(nn_device):
    values = np.arange(24, dtype=np.float64).reshape(4, 6).T[::-1, ::2]
    values.setflags(write=False)
    destination = oc.Tensor(np.zeros(values.shape), oc.float64, nn_device)
    destination.copy_from(values)
    np.testing.assert_array_equal(destination.numpy(), values)

def test_copy_from_supports_cpu_cuda_transfers_and_temporary_sources(nn_device):
    values = np.arange(12, dtype=np.float32).reshape(2, 1, 3, 2)
    destination = oc.Tensor(np.zeros_like(values), oc.float32, nn_device)
    destination.copy_from(oc.Tensor(values, oc.float32))
    np.testing.assert_array_equal(destination.numpy(), values)
    host = oc.Tensor(np.zeros_like(values), oc.float32)
    host.copy_from(destination)
    np.testing.assert_array_equal(host.numpy(), values)
    destination.copy_from(values + np.float32(2))
    np.testing.assert_array_equal(destination.numpy(), values + 2)

def test_copy_from_rejects_invalid_batches_without_realizing_graphs(nn_device):
    values = np.ones((2, 3), dtype=np.float32)
    destination = oc.Tensor(values, device=nn_device)
    output = destination.sin()

    for source in [np.ones((3, 2), np.float32), oc.Tensor(np.ones((3, 2), np.float32), device=nn_device)]:
        with pytest.raises(oc.ShapeError, match="shapes"):
            destination.copy_from(source)

    for source in [
        np.ones((2, 3), np.float64),
        values.astype(">f4"),
        oc.Tensor(values, oc.float64, nn_device),
    ]:
        with pytest.raises(oc.DTypeError, match="dtypes"):
            destination.copy_from(source)

    with pytest.raises(oc.OperationError, match="source must be realized"):
        destination.copy_from(output)

    with pytest.raises(oc.OperationError, match="realized leaf"):
        output.copy_from(values)

    assert not output.realized
    np.testing.assert_array_equal(destination.numpy(), values)
    output.realize()

    with pytest.raises(oc.OperationError, match="realized leaf"):
        output.copy_from(values)

    transposed = oc.Tensor(np.ones((3, 2), np.float32), device=nn_device).transpose((1, 0)).realize()

    with pytest.raises(oc.OperationError, match="source must be contiguous"):
        destination.copy_from(transposed)

    for source in [None, [[1, 2, 3], [4, 5, 6]], "batch"]:
        with pytest.raises(TypeError, match="Tensor or NumPy array"):
            destination.copy_from(source)

def test_copy_from_is_a_data_write_not_an_autograd_edge(nn_device):
    source = oc.Tensor([[2.0], [3.0]], oc.float64, nn_device)
    destination = oc.Tensor([[0.0], [0.0]], oc.float64, nn_device)
    output = (destination * destination).sum()
    destination.copy_from(source)
    first = oc.grad(output, destination)
    second = oc.grad(first.sum(), destination)
    unrelated = oc.grad(output, source)
    assert not output.realized and not first.realized and not second.realized
    oc.Plan([first, second, unrelated]).run()
    np.testing.assert_array_equal(first.numpy(), [[4.0], [6.0]])
    np.testing.assert_array_equal(second.numpy(), [[2.0], [2.0]])
    np.testing.assert_array_equal(unrelated.numpy(), np.zeros((2, 1)))
