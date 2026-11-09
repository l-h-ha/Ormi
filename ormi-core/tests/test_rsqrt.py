from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc

###
###
###

@pytest.mark.parametrize("dtype,numpy_dtype,tolerance", [
    (oc.float16, np.float16, 1e-3),
    (oc.float32, np.float32, 3e-7),
    (oc.float64, np.float64, 3e-15),
])
def test_rsqrt_values_strides_and_special_values(nn_device, dtype, numpy_dtype, tolerance):
    values = np.array([0.25, 1, 2, 9, 0, -0.0, -1, np.inf, np.nan], dtype=numpy_dtype).reshape(3, 3)
    input = oc.Tensor(values, dtype, nn_device).transpose((1, 0))
    output = input.rsqrt()
    functional = oc.ops.rsqrt(input)
    assert not output.realized and not functional.realized
    assert output.dtype == dtype and output.shape == (3, 3)
    oc.Plan([output, functional]).run()

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (1 / np.sqrt(values.T.astype(np.float64))).astype(numpy_dtype)

    np.testing.assert_allclose(output.numpy(), expected, rtol=tolerance, equal_nan=True)
    np.testing.assert_array_equal(output.numpy(), functional.numpy())
    assert np.signbit(output.numpy()[2, 1])  # rsqrt(-0) is negative infinity.
    assert oc.Tensor(np.empty((0, 3)), dtype, nn_device).rsqrt().numpy().shape == (0, 3)
    assert oc.Tensor(4.0, dtype, nn_device).rsqrt().item() == 0.5

def test_rsqrt_has_lazy_third_order_derivatives(nn_device):
    values = np.array([0.25, 0.8, 1, 2, 9], dtype=np.float64)
    input = oc.Tensor(values, oc.float64, nn_device)
    roots = [input.rsqrt()]

    for _ in range(3):
        roots.append(oc.grad(roots[-1].sum(), [input])[0])

    assert all(not root.realized for root in roots)
    oc.Plan(roots).run()
    expected = [values**-0.5, -0.5 * values**-1.5, 0.75 * values**-2.5, -1.875 * values**-3.5]

    for root, reference in zip(roots, expected, strict=True):
        np.testing.assert_allclose(root.numpy(), reference, rtol=2e-13, atol=1e-14)

def test_rsqrt_preserves_float64_range_and_rejects_nonfloating_inputs(nn_device):
    values = np.array([1e-300, 1e-100, 1e100, 1e300])
    np.testing.assert_allclose(
        oc.ops.rsqrt(oc.Tensor(values, oc.float64, nn_device)).numpy(),
        1 / np.sqrt(values),
        rtol=3e-15,
        atol=0,
    )

    with pytest.raises(oc.OperationError):
        oc.ops.rsqrt(None)

    for dtype in (oc.int32, oc.bool_):
        with pytest.raises(oc.DTypeError):
            oc.Tensor([1], dtype, nn_device).rsqrt()
