import numpy as np
from ormi_core.tensor import Tensor

print("Allocating leaf tensors...")
X = Tensor(data=np.random.rand(2029, 1202), device="cuda")
W = Tensor(data=np.random.rand(2029, 1202), device="cuda")
b = Tensor(data=np.random.rand(2029, 1202), device="cuda")

print("Building graph...")
Z = (W * (X ** 2.0)) + b

print("Realizing graph...")
Z.realize()

print(f"Result: {Z.numpy()}")
# Expected [7.0, 13.0, 23.0, 37.0]