import numpy as np
from ormi_core.tensor import Tensor

X = Tensor(data=np.array([2.0, 3.0], dtype=np.float32), device="cpu", requires_grad=True)
Y = Tensor(data=np.array([4.0, 5.0], dtype=np.float32), device="cpu", requires_grad=True)

# M = X * Y = [8.0, 15.0]
# Z = M + X = [10.0, 18.0]
M = X * Y
Z = M + X

# .numpy() triggers realization
print(f"Forward Pass Result Z: {Z.numpy()}") 

Z.backward()

# dZ/dX: Y + 1 -> [5.0, 6.0]
if X.grad is not None:
    print(f"dX: {X.grad.numpy()}")
else:
    print("dX: None")

# dZ/dY: X -> [2.0, 3.0]
if Y.grad is not None:
    print(f"dY: {Y.grad.numpy()}")
else:
    print("dY: None")