import numpy as np

import ormi_core as oc


def test_cuda_pointwise_fusion_and_copy():
    values = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    x = oc.Tensor(values, device="cuda")
    square = x * x
    summed = square + x
    exponential = summed.exp()
    result = exponential.sin()

    np.testing.assert_allclose(
        result.numpy(),
        np.sin(np.exp(values * values + values)),
        rtol=1e-6,
    )
    assert not square.realized
    assert not summed.realized
    assert not exponential.realized


def test_cuda_generalized_matmul_and_reduction():
    lhs_data = np.arange(2 * 1 * 2 * 3, dtype=np.float32).reshape(2, 1, 2, 3)
    rhs_data = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)
    lhs = oc.Tensor(lhs_data, device="cuda")
    rhs = oc.Tensor(rhs_data, device="cuda")
    result = (lhs @ rhs).sum((0, 2))

    expected = (lhs_data @ rhs_data).sum(axis=(0, 2))
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


def test_cuda_lazy_autograd():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = oc.Tensor(values, device="cuda")
    loss = (x * x).sum()
    gradient = loss.grad([x])[0]

    assert not loss.realized
    assert not gradient.realized
    np.testing.assert_allclose(gradient.numpy(), values * 2.0)
