import numpy as np
from ormi_core.tensor import Tensor

print("matmul graph break test")

# (2, 3)
W = Tensor(data=np.array([[1.0, 2.0, 3.0], 
                          [4.0, 5.0, 6.0]]), device="cuda")

# (3, 1)
X = Tensor(data=np.array([[1.0], 
                          [2.0], 
                          [3.0]]), device="cuda")

Z = W @ X

print("Shape:", Z.shape)
print("Result:\n", Z.numpy())
"""
Expected:
Shape: (2, 1)
Result:
 [[14.]
 [32.]]
"""