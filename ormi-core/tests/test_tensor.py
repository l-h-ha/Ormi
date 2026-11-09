from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc

###
###
###

@pytest.mark.parametrize(
    ("dtype", "numpy_dtype", "values"),
    [
        (oc.float64, np.float64, [1.25, -2.5]),
        (oc.float32, np.float32, [1.25, -2.5]),
        (oc.float16, np.float16, [1.25, -2.5]),
        (oc.int32, np.int32, [1, -2]),
        (oc.bool_, np.bool_, [True, False]),
    ],
)
def test_round_trips_every_public_dtype(dtype, numpy_dtype, values):
    tensor = oc.Tensor(values, dtype=dtype)
    result = tensor.numpy()

    assert tensor.dtype == dtype
    assert result.dtype == numpy_dtype
    np.testing.assert_array_equal(result, np.asarray(values, dtype=numpy_dtype))

def test_tensor_metadata_and_device_string(matrix):
    tensor = oc.Tensor(matrix, device="cpu")

    assert tensor.device == oc.cpu()
    assert tensor.shape == (2, 2)
    assert tensor.strides == (2, 1)
    assert tensor.ndim == 2
    assert tensor.numel == 4
    assert tensor.realized
    assert tensor.is_contiguous()
    assert repr(tensor).startswith("Tensor(shape=[2, 2]")

def test_lazy_creation_realizes_on_numpy():
    tensor = oc.ones((2, 3))

    assert not tensor.realized
    np.testing.assert_array_equal(tensor.numpy(), np.ones((2, 3), dtype=np.float32))
    assert tensor.realized

def test_numpy_export_honors_view_strides(matrix):
    tensor = oc.Tensor(matrix)
    transposed = tensor.transpose()
    expanded = oc.Tensor([[1.0], [2.0]]).expand((2, 3))

    assert transposed.shape == (2, 2)
    assert not transposed.is_contiguous()
    np.testing.assert_array_equal(transposed.numpy(), matrix.T)
    np.testing.assert_array_equal(
        expanded.numpy(),
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
    )

def test_item_requires_one_element():
    scalar = oc.Tensor(3.5)
    assert scalar.shape == ()
    assert scalar.item() == pytest.approx(3.5)

    with pytest.raises(oc.OperationError):
        oc.Tensor([1.0, 2.0]).item()
