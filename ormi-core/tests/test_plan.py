from __future__ import annotations

import numpy as np
import pytest

import ormi_core as oc


def test_plan_replays_outputs():
    values = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    input = oc.Tensor(values)
    output = (input * input + input).sin()
    plan = oc.Plan([output])

    assert plan.steps == 1
    assert plan.storage_buffers == 1
    assert plan.storage_bytes == values.nbytes

    plan.run()
    np.testing.assert_allclose(
        output.numpy(),
        np.sin(values * values + values),
    )


def test_plan_applies_updates_after_each_run():
    parameter = oc.Tensor(2.0)
    gradient = oc.Tensor(1.0)
    next_parameter = oc.sgd(parameter, gradient, 0.1)
    plan = oc.Plan([], [(parameter, next_parameter)])

    plan.run()
    assert parameter.item() == pytest.approx(1.9)

    plan.run()
    assert parameter.item() == pytest.approx(1.8)


def test_cpu_plan_rejects_recording():
    input = oc.Tensor(2.0)
    output = input * input
    plan = oc.Plan([output])
    plan.run()

    with pytest.raises(oc.OperationError, match="does not support recording"):
        plan.record()
