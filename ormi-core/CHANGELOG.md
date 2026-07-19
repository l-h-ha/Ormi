# Changelog

## 0.1.0a3 - 2026-07-19

First alpha of the current ORMI Core runtime.

### Added

- Lazy tensors with Float64, Float32, Float16, Int32, and Bool storage.
- CPU execution on Windows and Linux.
- An optional CUDA 12.4 backend loaded on first CUDA use.
- Pointwise arithmetic, elementary functions, casts, views, reductions, and
  broadcasted all-rank matmul.
- Reverse-mode automatic differentiation and repeated integer-order
  differentiation.
- Functional SGD, Float16 autocast, finite-gradient checks, and dynamic gradient
  scaling.
- CPU pointwise fusion through an automatically detected C++ compiler and a
  persistent kernel cache.
- CUDA pointwise fusion through NVRTC.
- Optional OpenBLAS 0.3.33 dispatch for compatible CPU matmul operations.
- Generated pybind11 bindings and type stubs from YAML API definitions.
- Python wheels for CPython 3.10-3.13 on 64-bit Windows and Linux.
- C++ and Python tests, Google Benchmark suites, and a PINN training example.

### Known Limitations

- Fractional differentiation is not implemented.
- CPU fusion currently supports contiguous, equal-shape Float32 and Float64
  pointwise graphs.
- CUDA fusion currently supports Float32 and Float64 pointwise graphs.
- SGD does not yet provide momentum or weight decay.
- The backend ABI is versioned but not stable for third-party plugins.
