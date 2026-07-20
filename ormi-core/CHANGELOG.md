# Changelog

## 0.1.0a5 - 2026-07-20

### Changed

- CPU wheels now contain only `ormi-core` and bundled OpenBLAS.
- The CUDA 12.4 backend and its runtime libraries are distributed separately as
  the version-matched `ormi-cuda` package.
- `pip install "ormi-core[cuda]"` installs both packages while preserving lazy
  CUDA loading.

### Added

- Published-wheel checks verify CPU-only package contents and the matching CUDA
  add-on on Windows and Linux.

## 0.1.0a4 - 2026-07-20

### Added

- Post-release wheel checks for clean CPU execution and uncached CPU JIT across
  CPython 3.10-3.13 on Windows and Linux.
- Real-GPU verification covering CUDA fusion, generalized matmul, reductions,
  copies, and lazy autograd.

### Fixed

- Windows backend loading now resolves dependencies from registered DLL
  directories and reports the Windows loader error code on failure.
- Windows wheels keep their private CUDA library directory available for lazy
  backend loading and NVRTC builtins discovery.

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
