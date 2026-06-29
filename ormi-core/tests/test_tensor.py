import numpy as np
import ormi_core as oc
from ormi_core import Tensor

def test_tensor_square_sum_grad_cpu():
    x = Tensor([[3, 3], [3, 3]], requires_grad=True)

    y = (x * x).sum()
    grads = y.grad([x])
    g = grads[0]

    assert g is not None

    np.testing.assert_allclose(y.numpy(), np.array(36.0, dtype=np.float32))
    np.testing.assert_allclose(
        g.numpy(),
        np.array([[6, 6], [6, 6]], dtype=np.float32),
    )

def test_public_training_step_cpu():
    x = oc.Tensor([[3, 3], [3, 3]], requires_grad=True)

    loss = (x * x).sum()
    grads = loss.grad([x])

    tape = oc.ComputeTape()
    tape.capture([loss, *grads])
    
    optim = oc.SGD([x], grads, lr=0.1)

    tape.execute()
    optim.step()

    np.testing.assert_allclose(
        x.numpy(),
        np.array([[2.4, 2.4], [2.4, 2.4]], dtype=np.float32),
        rtol=1e-6,
    )