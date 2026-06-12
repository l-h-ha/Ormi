import pytest
from numpy.testing import assert_allclose
from ormi_core.tensor import Tensor

def test_initialization(compute_device):
    t = Tensor([1.0, 2.0, 3.0], device=compute_device, requires_grad=True)
    
    assert t.shape == (3,)
    assert t.requires_grad is True
    assert t.device.type.name.lower() == compute_device
    assert_allclose(t.numpy(), [1.0, 2.0, 3.0])

def test_binary_operations(compute_device):
    a = Tensor([4.0, 6.0], device=compute_device)
    b = Tensor([2.0, 3.0], device=compute_device)

    assert_allclose((a + b).numpy(), [6.0, 9.0])
    assert_allclose((a - b).numpy(), [2.0, 3.0])
    assert_allclose((a * b).numpy(), [8.0, 18.0])
    assert_allclose((a / b).numpy(), [2.0, 2.0])
    assert_allclose((-a).numpy(), [-4.0, -6.0])
    assert_allclose((a ** 2.0).numpy(), [16.0, 36.0])

def test_reverse_operations(compute_device):
    a = Tensor([2.0, 4.0], device=compute_device)

    # Python triggers __r*__ methods when the left operand is a native scalar
    assert_allclose((10.0 + a).numpy(), [12.0, 14.0])
    assert_allclose((10.0 - a).numpy(), [8.0, 6.0])
    assert_allclose((10.0 * a).numpy(), [20.0, 40.0])
    assert_allclose((10.0 / a).numpy(), [5.0, 2.5])

def test_matmul(compute_device):
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], device=compute_device)
    b = Tensor([[2.0, 0.0], [0.0, 2.0]], device=compute_device)
    
    c = a @ b
    
    assert c.shape == (2, 2)
    assert_allclose(c.numpy(), [[2.0, 4.0], [6.0, 8.0]])

def test_matmul_type_enforcement(compute_device):
    a = Tensor([[1.0, 2.0]], device=compute_device)
    with pytest.raises(TypeError):
        _ = a @ 2.0

def test_view_operations(compute_device):
    # (2, 3)
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=compute_device)
    
    b = a.reshape((3, 2))
    assert b.shape == (3, 2)
    assert_allclose(b.numpy(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    c = a.transpose()
    assert c.shape == (3, 2)
    assert_allclose(c.numpy(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

def test_autograd_graph(compute_device):
    a = Tensor([2.0, 3.0], device=compute_device, requires_grad=True)
    b = Tensor([4.0, 5.0], device=compute_device, requires_grad=True)
    
    # Z = (A * B) + A
    # dZ/dA = B + 1  -> [5.0, 6.0]
    # dZ/dB = A      -> [2.0, 3.0]
    z = (a * b) + a
    z.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert_allclose(a.grad.numpy(), [5.0, 6.0])
    assert_allclose(b.grad.numpy(), [2.0, 3.0])

def test_zero_grad(compute_device):
    a = Tensor([1.0], device=compute_device, requires_grad=True)
    b = a * 2.0
    b.backward()
    
    assert a.grad is not None
    
    a.zero_grad()
    assert a.grad is None