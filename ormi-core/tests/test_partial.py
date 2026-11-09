import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

@pytest.mark.parametrize("dtype", [oc.float16, oc.float32, oc.float64])
def test_single_input_grad_preserves_the_sequence_api_and_explicit_seed(nn_device, dtype):
    x = oc.Tensor([[2.0], [3.0]], dtype, nn_device)
    y = x * x
    loss = y.sum()
    seed = oc.Tensor([[2.0], [4.0]], dtype, nn_device)
    roots = [
        oc.grad(loss, x), loss.grad(x), oc.grad(loss, input=x),
        oc.grad(loss, [x])[0], loss.grad([x])[0], oc.grad(loss, inputs=[x])[0],
        oc.grad(y, x, seed=seed), y.grad(x, seed),
    ]
    assert all(isinstance(value, oc.Tensor) and not value.realized for value in roots)
    assert isinstance(oc.grad(loss, [x]), list)

    with pytest.raises(oc.ShapeError, match="explicit seed"):
        oc.grad(y, x)

    oc.Plan(roots).run()

    for value in roots[:6]:
        np.testing.assert_array_equal(value.numpy(), [[4.0], [6.0]])

    for value in roots[6:]:
        np.testing.assert_array_equal(value.numpy(), [[8.0], [24.0]])

@pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 1, 2, 1), (0, 1), (2, 0, 3, 1)])
def test_partial_preserves_arbitrary_point_shapes_and_empty_axes(nn_device, shape):
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 10
    x = oc.Tensor(values, oc.float64, nn_device)
    y = x * x * x
    first = oc.partial(y, x)
    second = oc.partial(y, x, order=2)
    third = oc.partial(y, x, order=3)
    fourth = oc.partial(y, x, order=4)
    roots = [first, second, third, fourth]
    assert all(value.shape == shape and not value.realized for value in roots)
    assert not y.realized
    oc.Plan(roots).run()

    for actual, expected in zip(roots, [3 * values**2, 6 * values, np.full(shape, 6), np.zeros(shape)]):
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)

@pytest.mark.parametrize("dtype,tolerance", [(oc.float16, 1e-3), (oc.float32, 1e-6), (oc.float64, 1e-12)])
def test_partial_does_not_sum_interacting_points(nn_device, dtype, tolerance):
    x = oc.Tensor([[2.0], [4.0], [6.0], [8.0]], dtype, nn_device)
    y = x - x.mean(axes=(0,), keepdims=True)
    first = oc.partial(y, x)
    second = oc.partial(y, x, order=2)
    summed = oc.grad(y.sum(), x)
    oc.Plan([first, second, summed]).run()
    np.testing.assert_allclose(first.numpy(), np.full((4, 1), 0.75), atol=tolerance)
    np.testing.assert_allclose(second.numpy(), np.zeros((4, 1)), atol=tolerance)
    np.testing.assert_allclose(summed.numpy(), np.zeros((4, 1)), atol=tolerance)

def test_partial_handles_dense_cross_point_dependencies_at_higher_orders(nn_device):
    values = np.array([[0.2], [0.4], [0.8]])
    mixing = np.array([[1.0, 2.0, -1.0], [0.5, 3.0, 2.0], [2.0, -0.5, 4.0]])
    x = oc.Tensor(values, oc.float64, nn_device)
    z = oc.Tensor(mixing, oc.float64, nn_device) @ x
    y = z * z * z
    roots = [oc.partial(y, x, order=order) for order in range(1, 5)]
    assert all(not value.realized for value in [y, *roots])
    oc.Plan(roots).run()
    z_values = mixing @ values
    diagonal = np.diag(mixing).reshape(-1, 1)
    expected = [
        3 * z_values**2 * diagonal,
        6 * z_values * diagonal**2,
        6 * diagonal**3,
        np.zeros_like(values),
    ]

    for actual, reference in zip(roots, expected):
        np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-12, atol=1e-12)

def test_partial_distinguishes_pure_and_mixed_coordinate_derivatives(nn_device):
    values = np.arange(12, dtype=np.float64).reshape(2, 1, 3, 2) / 10
    points = oc.Tensor(values, oc.float64, nn_device)
    x = points @ oc.Tensor([[1.0], [0.0]], oc.float64, nn_device)
    t = points @ oc.Tensor([[0.0], [1.0]], oc.float64, nn_device)
    u = x * t
    u_x = oc.partial(u, points, coordinate=0)
    u_xx = oc.partial(u, points, coordinate=0, order=2)
    u_tt = oc.partial(u, points, coordinate=1, order=2)
    u_xt = oc.partial(u_x, points, coordinate=1)
    roots = [u_x, u_xx, u_tt, u_xt]
    assert all(value.shape == (2, 1, 3, 1) and not value.realized for value in roots)
    oc.Plan(roots).run()
    expected = [
        values[..., 1:2],
        np.zeros_like(values[..., :1]),
        np.zeros_like(values[..., :1]),
        np.ones_like(values[..., :1]),
    ]

    for actual, reference in zip(roots, expected):
        np.testing.assert_allclose(actual.numpy(), reference, atol=1e-12)

def test_partial_selects_channels_on_strided_inputs_and_outputs(nn_device):
    values = np.arange(12, dtype=np.float64).reshape(2, 3, 2) / 10
    points = oc.Tensor(values, oc.float64, nn_device).transpose((1, 0, 2))
    output = points.transpose((1, 0, 2)).sin().transpose((1, 0, 2))
    first = oc.partial(output, points, coordinate=-1, component=-1)
    second = oc.partial(output, points, coordinate=1, component=1, order=2)
    cross = oc.partial(output, points, coordinate=0, component=1)
    assert not points.is_contiguous() and not output.is_contiguous()
    oc.Plan([first, second, cross]).run()
    selected = values.transpose(1, 0, 2)[..., 1:2]
    np.testing.assert_allclose(first.numpy(), np.cos(selected), atol=1e-12)
    np.testing.assert_allclose(second.numpy(), -np.sin(selected), atol=1e-12)
    np.testing.assert_array_equal(cross.numpy(), np.zeros_like(selected))

def test_partial_uses_the_input_dtype_after_a_differentiable_cast(nn_device):
    x = oc.Tensor([[0.25], [0.5]], oc.float64, nn_device)
    cast = x.astype(oc.float32)
    result = oc.partial(cast * cast, x)
    assert result.dtype == oc.float64 and not result.realized
    np.testing.assert_array_equal(result.numpy(), [[0.5], [1.0]])

def test_partial_handles_zero_stride_points_without_assuming_independence(nn_device):
    scalar = oc.Tensor([[0.5]], oc.float64, nn_device)
    points = scalar.expand((2, 3, 1))
    centered = points - points.mean(axes=(0, 1), keepdims=True)
    output = centered * centered + points * points * points
    first = oc.partial(output, points)
    second = oc.partial(output, points, order=2)
    scalar_gradient = oc.grad(first.sum(), scalar)
    assert not points.realized and not first.realized and not second.realized
    oc.Plan([first, second, scalar_gradient]).run()
    np.testing.assert_allclose(first.numpy(), np.full((2, 3, 1), 0.75), atol=1e-12)
    np.testing.assert_allclose(second.numpy(), np.full((2, 3, 1), 3 + 2 * (5 / 6)**2), atol=1e-12)
    assert scalar_gradient.item() == pytest.approx(18.0)

@pytest.mark.parametrize("track_running_stats", [False, True])
def test_partial_handles_batch_normalization_coupling(nn_device, track_running_stats):
    values = np.array([[0.2], [0.7], [-0.3]])
    points = oc.Tensor(values, oc.float64, nn_device)
    layer = nn.BatchNorm(
        1, eps=0.25, momentum=0.25, affine=False,
        track_running_stats=track_running_stats, dtype=oc.float64, device=nn_device,
    )
    output = layer(points)
    result = oc.partial(output, points)
    assert not output.realized and not result.realized
    centered = values - values.mean(axis=0)
    variance = np.mean(centered**2, axis=0) + layer.eps
    count = values.shape[0]
    expected = (1 - 1 / count - centered**2 / (count * variance)) / np.sqrt(variance)
    plan = oc.Plan([result])

    for step in range(1, 3):
        plan.run()
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12, atol=1e-12)

        if track_running_stats:
            retained = 0.75**step
            np.testing.assert_allclose(layer.running_mean.numpy(), values.mean(0) * (1 - retained), atol=1e-12)
            np.testing.assert_allclose(
                layer.running_var.numpy(), retained + values.var(0, ddof=1) * (1 - retained), atol=1e-12,
            )

def test_partial_parameter_gradients_survive_training_and_replay(nn_device):
    values = np.array([[0.2], [0.4]])
    mixing = np.array([[1.0, 0.5], [0.25, 2.0]])
    points = oc.Tensor(values, oc.float64, nn_device)
    weight = oc.Tensor(1.5, oc.float64, nn_device)
    z = oc.Tensor(mixing, oc.float64, nn_device) @ points
    output = weight * weight * z * z * z
    second = oc.partial(output, points, order=2)
    loss = second.sum()
    gradient = oc.grad(loss, weight)
    curvature = oc.grad(gradient, weight)
    assert all(not tensor.realized for tensor in [output, second, loss, gradient, curvature])
    plan = oc.Plan([loss, gradient, curvature], [(weight, oc.sgd(weight, gradient, 0.001))])
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)
    diagonal = np.diag(mixing).reshape(-1, 1)
    coefficient = float((6 * (mixing @ values) * diagonal**2).sum())
    expected_weight = 1.5

    def check_step(run):
        nonlocal expected_weight
        run()
        assert loss.item() == pytest.approx(expected_weight**2 * coefficient)
        assert gradient.item() == pytest.approx(2 * expected_weight * coefficient)
        assert curvature.item() == pytest.approx(2 * coefficient)
        expected_weight -= 0.001 * 2 * expected_weight * coefficient
        assert weight.item() == pytest.approx(expected_weight)
        assert weight.realized and weight.op == "leaf"

    for _ in range(2):
        check_step(plan.run)
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

    if nn_device != "cpu":
        replay = plan.record()
        assert weight.item() == pytest.approx(expected_weight)
        del plan

        for _ in range(2):
            check_step(replay.run)

def test_partial_validates_shapes_selectors_and_orders_before_execution():
    x = oc.Tensor(np.ones((2, 3)), oc.float64)
    y = oc.Tensor(np.ones((2, 2)), oc.float64)

    with pytest.raises(oc.ShapeError, match="coordinate"):
        oc.partial(y, x)

    with pytest.raises(oc.ShapeError, match="component"):
        oc.partial(y, x, coordinate=0)

    for coordinate, component in [(3, 0), (-4, 0), (0, 2), (0, -3)]:
        with pytest.raises(oc.ShapeError, match="out of range"):
            oc.partial(y, x, coordinate=coordinate, component=component)

    with pytest.raises(oc.OperationError, match="positive"):
        oc.partial(y, x, coordinate=0, component=0, order=0)

    with pytest.raises(TypeError):
        oc.partial(y, x, coordinate=0, component=0, order=-1)

    with pytest.raises(oc.ShapeError, match="matching point dimensions"):
        oc.partial(oc.Tensor(np.ones((3, 1))), x, coordinate=0)

    with pytest.raises(oc.ShapeError, match="scalar"):
        oc.partial(oc.Tensor(1.0), oc.Tensor(2.0))

    with pytest.raises(oc.DTypeError, match="floating-point"):
        oc.partial(oc.Tensor([[1]], oc.int32), oc.Tensor([[1.0]]))

    for shape in [(2, 0), (0,)]:
        with pytest.raises(oc.ShapeError):
            oc.partial(oc.Tensor(np.empty(shape)), oc.Tensor(np.empty(shape)))

def test_partial_returns_lazy_zeros_for_disconnected_inputs(nn_device):
    x = oc.Tensor([[1.0], [2.0]], oc.float64, nn_device)
    y = oc.Tensor([[3.0], [4.0]], oc.float64, nn_device)
    result = oc.partial(y, x, order=3)
    assert not result.realized
    np.testing.assert_array_equal(result.numpy(), np.zeros((2, 1)))
