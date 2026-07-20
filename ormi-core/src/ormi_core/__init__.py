from ._ext._C import (
    Autocast,
    BufferError,
    DType,
    DTypeError,
    Device,
    DeviceError,
    MemoryError,
    OperationError,
    OrmiError,
    ShapeError,
    Tensor,
    bool_,
    cpu,
    cuda,
    cudaError,
    diff,
    dtype_name,
    dtype_size,
    float16,
    float32,
    float64,
    grad,
    int32,
    jit,
    jitError,
    ops,
    optim,
    synchronize,
    tensor,
)
from .autocast import autocast

full = ops.full
zeros = ops.zeros
ones = ops.ones
zeros_like = ops.zeros_like
ones_like = ops.ones_like
sgd = optim.sgd
GradScaler = optim.GradScaler
ScalePolicy = optim.ScalePolicy

__version__ = "0.1.0a5"

__all__ = [
    "Autocast",
    "BufferError",
    "DType",
    "DTypeError",
    "Device",
    "DeviceError",
    "GradScaler",
    "MemoryError",
    "OperationError",
    "OrmiError",
    "ScalePolicy",
    "ShapeError",
    "Tensor",
    "autocast",
    "bool_",
    "cpu",
    "cuda",
    "cudaError",
    "diff",
    "dtype_name",
    "dtype_size",
    "float16",
    "float32",
    "float64",
    "full",
    "grad",
    "int32",
    "jit",
    "jitError",
    "ones",
    "ones_like",
    "ops",
    "optim",
    "sgd",
    "synchronize",
    "tensor",
    "zeros",
    "zeros_like",
]
