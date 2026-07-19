from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc


def test_functional_sgd_preserves_parameter():
    parameter = oc.Tensor([3.0, 5.0])
    gradient = oc.Tensor([6.0, 10.0])
    updated = oc.sgd(parameter, gradient, 0.1)

    assert not updated.realized
    np.testing.assert_allclose(updated.numpy(), np.array([2.4, 4.0], dtype=np.float32))
    np.testing.assert_allclose(parameter.numpy(), np.array([3.0, 5.0], dtype=np.float32))


def test_grad_scaler_fuses_unscale_and_updates_policy():
    policy = oc.ScalePolicy()
    policy.interval = 1
    scaler = oc.GradScaler(8.0, policy)
    parameter = oc.Tensor(3.0)
    scaled_gradient = oc.Tensor(48.0)

    updated = scaler.step(parameter, scaled_gradient, 0.1)

    assert updated is not None
    assert scaler.value == pytest.approx(16.0)
    assert updated.item() == pytest.approx(2.4)


def test_grad_scaler_skips_non_finite_update():
    scaler = oc.GradScaler(8.0)
    result = scaler.step(oc.Tensor(3.0), oc.Tensor(np.inf), 0.1)

    assert result is None
    assert scaler.value == pytest.approx(4.0)


def test_autocast_scope_restores_policy():
    lhs = oc.Tensor([[1.0, 2.0]])
    rhs = oc.Tensor([[3.0], [4.0]])

    with oc.autocast():
        lowered = lhs @ rhs
        assert lowered.dtype == oc.float16

    restored = lhs @ rhs
    assert restored.dtype == oc.float32
