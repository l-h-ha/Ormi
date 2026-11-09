from ._ext._C import (
    Autocast,
    BackendError,
    BufferError,
    DType,
    DTypeError,
    Device,
    DeviceError,
    MemoryError,
    OperationError,
    OrmiError,
    Plan,
    Replay,
    ShapeError,
    Tensor,
    bool_,
    cpu,
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
    partial,
    synchronize,
    tensor,
)
from .autocast import autocast
from . import nn

###
###
###

full = ops.full
zeros = ops.zeros
ones = ops.ones
zeros_like = ops.zeros_like
ones_like = ops.ones_like
sgd = optim.sgd
GradScaler = optim.GradScaler
ScalePolicy = optim.ScalePolicy

__version__ = "0.1.0a7"

__all__ = [
    "Autocast",
    "BackendError",
    "BufferError",
    "DType",
    "DTypeError",
    "Device",
    "DeviceError",
    "GradScaler",
    "MemoryError",
    "OperationError",
    "OrmiError",
    "Plan",
    "Replay",
    "ScalePolicy",
    "ShapeError",
    "Tensor",
    "autocast",
    "bool_",
    "cpu",
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
    "nn",
    "ones",
    "ones_like",
    "ops",
    "optim",
    "partial",
    "sgd",
    "synchronize",
    "tensor",
    "zeros",
    "zeros_like",
]
