from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc

###
###
###

def test_gradient_graph_is_lazy():
    x = oc.Tensor([[3.0, 3.0], [3.0, 3.0]])
    loss = (x * x).sum()
    gradient = loss.grad([x])[0]

    assert not loss.realized
    assert not gradient.realized
    assert loss.item() == pytest.approx(36.0)
    np.testing.assert_allclose(gradient.numpy(), np.full((2, 2), 6.0, dtype=np.float32))

def test_non_scalar_gradient_uses_explicit_seed():
    x = oc.Tensor([1.0, 2.0, 3.0])
    y = x * x
    seed = oc.Tensor([1.0, 2.0, 3.0])
    gradient = oc.grad(y, [x], seed)[0]

    np.testing.assert_allclose(gradient.numpy(), np.array([2.0, 8.0, 18.0], dtype=np.float32))

def test_repeated_differentiation():
    x = oc.Tensor(2.0, dtype=oc.float64)
    y = x * x * x * x * x

    assert oc.diff(y, x, 5).item() == pytest.approx(120.0)
    assert oc.diff(y, x, 6).item() == pytest.approx(0.0)
