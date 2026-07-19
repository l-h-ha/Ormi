from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc


def test_scalar_arithmetic_is_lazy_and_broadcasts(matrix):
    tensor = oc.Tensor(matrix)
    result = 10.0 - (tensor * 2.0 + 1.0) / 2.0

    assert not result.realized
    np.testing.assert_allclose(result.numpy(), 10.0 - (matrix * 2.0 + 1.0) / 2.0)


def test_generalized_batched_matmul():
    lhs_data = np.arange(2 * 1 * 2 * 3, dtype=np.float32).reshape(2, 1, 2, 3)
    rhs_data = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)
    result = oc.Tensor(lhs_data) @ oc.Tensor(rhs_data)

    assert result.shape == (2, 4, 2, 2)
    np.testing.assert_allclose(result.numpy(), lhs_data @ rhs_data)


def test_multi_axis_reduction_and_keepdims():
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = oc.Tensor(values).sum((0, -1), keepdims=True)

    assert result.shape == (1, 3, 1)
    np.testing.assert_allclose(result.numpy(), values.sum(axis=(0, -1), keepdims=True))


def test_cast_and_elementary_functions():
    values = np.array([0.25, 0.5], dtype=np.float32)
    tensor = oc.Tensor(values)

    assert tensor.astype(oc.float64).dtype == oc.float64
    np.testing.assert_allclose(tensor.exp().log().numpy(), values, rtol=1e-6)
    np.testing.assert_allclose(tensor.sin().numpy(), np.sin(values), rtol=1e-6)
    np.testing.assert_allclose(tensor.cos().numpy(), np.cos(values), rtol=1e-6)


def test_boolean_numerical_operations_fail_at_construction():
    tensor = oc.Tensor([True, False], dtype=oc.bool_)
    with pytest.raises(oc.DTypeError):
        _ = tensor + tensor
    with pytest.raises(oc.DTypeError):
        _ = tensor.sum()
