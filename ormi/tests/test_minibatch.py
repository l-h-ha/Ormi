import numpy as np
import pytest

import ormi as om
from ormi import nn

###
###
###

def test_minibatches_reuse_a_classifier_plan_and_match_numpy_sgd(api_device):
    shape = (2, 1, 3, 2)
    weight_values = np.array([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]])
    bias_values = np.array([0.1, -0.2, 0.05])
    weight = om.Tensor(weight_values, om.float64, api_device)
    bias = om.Tensor(bias_values, om.float64, api_device)
    model = nn.Linear(weight, bias)
    inputs = om.Tensor(np.zeros(shape), om.float64, api_device)
    targets = om.Tensor(np.zeros(shape[:-1], np.int32), om.int32, api_device)
    loss = nn.cross_entropy(model(inputs), targets)
    parameters = model.parameters()
    gradients = om.grad(loss, parameters)
    plan = om.Plan([loss, *gradients], [
        (parameter, om.sgd(parameter, gradient, 0.1))
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ])
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)

    for batch in range(4):
        values = np.sin(np.arange(12) + batch).reshape(shape)
        labels = ((np.arange(6) + batch) % 3).reshape(shape[:-1]).astype(np.int32)
        logits = values @ weight_values + bias_values
        log_probability = logits - logits.max(-1, keepdims=True)
        log_probability -= np.log(np.exp(log_probability).sum(-1, keepdims=True))
        expected_loss = -log_probability.reshape(-1, 3)[np.arange(6), labels.reshape(-1)].mean()
        upstream = np.exp(log_probability).reshape(-1, 3)
        upstream[np.arange(6), labels.reshape(-1)] -= 1
        upstream /= 6
        expected_weight_gradient = values.reshape(-1, 2).T @ upstream
        expected_bias_gradient = upstream.sum(0)

        inputs.copy_from(values)
        targets.copy_from(labels)

        if batch == 0:
            assert not loss.realized and all(not gradient.realized for gradient in gradients)

        plan.run()
        assert loss.item() == pytest.approx(expected_loss)
        np.testing.assert_allclose(gradients[0].numpy(), expected_weight_gradient, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gradients[1].numpy(), expected_bias_gradient, rtol=1e-12, atol=1e-12)
        weight_values -= 0.1 * expected_weight_gradient
        bias_values -= 0.1 * expected_bias_gradient
        np.testing.assert_allclose(weight.numpy(), weight_values, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bias.numpy(), bias_values, rtol=1e-12, atol=1e-12)
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage
        assert model.parameters()[0] is weight and model.parameters()[1] is bias

def test_feeding_preserves_coupled_higher_order_gradients_and_replay(api_device):
    shape = (2, 1, 3, 1)
    inputs = om.Tensor(np.zeros(shape), om.float64, api_device)
    targets = om.Tensor(np.zeros(shape), om.float64, api_device)
    weight = om.Tensor(1.5, om.float64, api_device)
    centered = inputs - inputs.mean(axes=(0, 1, 2), keepdims=True)
    output = weight * weight * centered * centered * centered
    second = om.partial(output, inputs, order=2)
    loss = nn.mse_loss(second, targets)
    gradient = om.grad(loss, weight)
    curvature = om.grad(gradient, weight)
    plan = om.Plan([loss, second, gradient, curvature], [(weight, om.sgd(weight, gradient, 1e-4))])
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)
    expected_weight = 1.5

    def check_batch(run, batch):
        nonlocal expected_weight
        values = np.sin(np.arange(6) + batch).reshape(shape)
        labels = np.cos(np.arange(6) + batch).reshape(shape)
        coefficient = 6 * (values - values.mean()) * (5 / 6)**2
        predicted = coefficient * expected_weight**2
        expected_gradient = (4 * expected_weight * coefficient * (predicted - labels)).mean()
        expected_curvature = (12 * expected_weight**2 * coefficient**2 - 4 * coefficient * labels).mean()
        inputs.copy_from(values)
        targets.copy_from(labels)
        run()
        np.testing.assert_allclose(second.numpy(), predicted, rtol=1e-12, atol=1e-12)
        assert loss.item() == pytest.approx(((predicted - labels)**2).mean())
        assert gradient.item() == pytest.approx(expected_gradient)
        assert curvature.item() == pytest.approx(expected_curvature)
        expected_weight -= 1e-4 * expected_gradient
        assert weight.item() == pytest.approx(expected_weight)

    assert all(not tensor.realized for tensor in [output, second, loss, gradient, curvature])
    check_batch(plan.run, 0)
    check_batch(plan.run, 1)
    assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

    if api_device != "cpu":
        replay = plan.record()
        assert weight.item() == pytest.approx(expected_weight)
        del plan
        check_batch(replay.run, 2)
        check_batch(replay.run, 3)
