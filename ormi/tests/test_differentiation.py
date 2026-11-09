import numpy as np

import ormi as om
import ormi_core as core

###
###
###

def test_differentiation_reuses_the_core_api_and_stays_lazy(api_device):
    assert om.grad is core.grad
    assert om.partial is core.partial
    x = core.Tensor([[2.0], [4.0]], core.float64, api_device)
    y = x - x.mean(axes=(0,), keepdims=True)
    own_point = om.partial(y, x)
    summed = om.grad(y.sum(), x)
    second = om.partial(y, x, order=2)
    assert all(not value.realized for value in [y, own_point, summed, second])
    om.Plan([own_point, summed, second]).run()
    np.testing.assert_array_equal(own_point.numpy(), [[0.5], [0.5]])
    np.testing.assert_array_equal(summed.numpy(), [[0.0], [0.0]])
    np.testing.assert_array_equal(second.numpy(), [[0.0], [0.0]])
