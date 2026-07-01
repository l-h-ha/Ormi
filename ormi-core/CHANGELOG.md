# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] - 2026-06-29

### Added

- Added a typed ORMI error hierarchy for C++ and Python boundary errors, including `OrmiError`, `DeviceError`, `DTypeError`, `ShapeError`, `RegistryError`, `OperationError`, `BufferError`, `NodeError`, `MemoryError`, `ValueError`, `AutogradError`, and `RuntimeError`.
- Added failure and validation macros in `error.h`: `ORMI_WARN`, `ORMI_FAIL`, `ORMI_THROW`, `ORMI_CHECK`, `ORMI_CHECK_EX`, and `ORMI_CHECK_EQ`.
- Added the `Buffer` abstraction as the core data owner, including allocation ownership, dtype metadata, device metadata, scalar item reads, formatted tensor printing, and `empty`, `full`, `scalar`, `zeros`, `ones`, `zeros_like`, and `ones_like` helpers.
- Added explicit dtype support for `Float32`, `Float16`, `Int32`, and `Bool`, with `dtype_size`, `to_string`, and stream formatting support.
- Added a thread-local `std::pmr::unsynchronized_pool_resource` node allocator for lower-overhead graph node allocation.
- Added `GradContext` for reverse-mode gradient accumulation keyed by source node.
- Added seeded `grad(root, inputs, seed)` and higher-order `diff(y, x, order)` APIs.
- Added broadcast-aware gradient helpers, including `unbroadcast`, so pointwise, reduction, matmul, expand, reshape, and transpose adjoints can reduce gradients back to the original operand shapes.
- Added view metadata support through `TensorView`, contiguous stride calculation, stride inference, and view-op registration.
- Added lazy view operations for `reshape`, `transpose`, and `expand`, with shape inference, stride inference, and backward rules.
- Added reduction attributes for `sum`, including axis-specific reduction and `keepdims`.
- Added batched and broadcasted matmul shape inference and adjoints.
- Added `Tape`, topology sorting, `ComputeTape`, `ZeroGradTape`, and CUDA graph capture/replay through `CudaGraphTape`.
- Added `optim::SGD` with CPU and CUDA backend implementations.
- Added CPU optimizer kernels and CUDA optimizer kernels for SGD, including Float32 and Float16 CUDA paths.
- Added a CPU allocator with 64-byte alignment, cached freed blocks, live allocation tracking, dtype-aware fills, host/device copy methods, and explicit cache clearing.
- Added a CUDA allocator with CUDA primary context management, stream-ordered memory pool support, configurable release threshold via `ORMI_CUDA_MEMPOOL_RELEASE_THRESHOLD`, live allocation tracking, dtype-aware fills, async copies, OOM retry handling, and cache trimming.
- Added global CUDA launch configuration through `CudaConfig`, including optimizer, elementwise, and reduction thread counts.
- Added shared CUDA stream and cuBLAS handle helpers, TF32 capability detection, NVRTC PTX compilation helpers, and contiguous-copy kernels for non-contiguous CUDA views.
- Added a backend graph-break registry so operations such as `MATMUL` and `SUM` can dispatch to specialized kernels outside the fused pointwise JIT.
- Added CPU graph-break kernels for matmul and sum, including OpenBLAS matmul, batched/view-aware offsets, axis reductions, and Float16/Int32 reduction paths where supported.
- Added CUDA graph-break kernels for matmul and sum, including cuBLAS execution through `cublasGemmEx` and `cublasGemmStridedBatchedEx`, Float32/Float16 support, TF32 compute selection, contiguous staging for incompatible views, cached NVRTC reduction kernels, and configurable launch sizes.
- Added shape/stride-aware CPU JIT compilation for fused pointwise graphs, with scalar handling, broadcasting support, non-fusible node boundaries, stable generated shared library caching, `ORMI_CPU_JIT_OPT` validation, and `ORMI_CPU_JIT_CACHE_DIR` cache placement.
- Added shape/stride-aware CUDA JIT compilation for fused pointwise graphs, with scalar handling, broadcasting support, non-fusible node boundaries, generated kernel caching, and configurable launch sizes.
- Added YAML-driven generated Python bindings from `csrc/bindings/defs/*.yaml` using `generate_bindings.py`.
- Added generated Python type stubs for `_C` and `_C.ops`, plus the `py.typed` marker for typed package consumers.
- Added generated Python bindings for devices, dtypes, buffers, nodes, tensor views, autograd, compute tapes, optimizers, pointwise ops, BLAS ops, reductions, and view ops.
- Added `numpy_buffer.cpp`/`numpy_buffer.h` helpers for NumPy-to-buffer and buffer-to-NumPy conversion.
- Added public Python wrappers for `Tensor`, `ComputeTape`, and `SGD`.
- Added public Python `Tensor` APIs for device normalization, lazy realization, NumPy conversion, arithmetic, matmul, reshape, transpose, `sum`, axis reductions, and explicit gradient queries.
- Added MIT license files for the workspace and Python packages.
- Added C++ examples and benchmark programs for CPU training, CUDA training, SGD, compute tapes, and CUDA graph replay.
- Added new CMake wiring for `ormi_engine`, generated pybind sources, conditional CUDA source inclusion, CUDA runtime DLL copying on Windows, OpenBLAS linking, `magic_enum`, GTest, and examples.
- Added CMake generation steps for native binding source, `_C` stubs, `_C.ops` stubs, and `py.typed`.
- Added backend and core tests for lowerings, optimizers, tapes, CUDA graph tape, matmul kernels, sum kernels, runtime graph breakers, view ops, shape logic, allocator behavior, autograd, and pointwise operations.

### Changed

- Replaced the old `memory` layer with `Buffer` and allocator-backed ownership.
- Updated `Node` construction so leaf nodes inherit device, dtype, shape, and strides from their buffers.
- Updated internal `Node` construction to require at least one parent and to validate matching parent devices and dtypes.
- Updated internal nodes to inherit `requires_grad` when any parent requires gradients.
- Updated operation constructors such as pointwise ops, matmul, reductions, and views to accept `const std::shared_ptr<Node>&` and use pooled allocation.
- Updated shape inference to use a registry-based system with clearer errors for broadcast and registry failures.
- Updated pointwise adjoints to accumulate gradients through `GradContext` and reduce broadcasted gradients back to input shapes.
- Updated `sum` to support global reductions, axis reductions, and `keepdims`.
- Updated `matmul` to support batched broadcasting in shape inference and gradient construction.
- Updated `reshape`, `transpose`, and `expand` to behave as metadata-only view operations during runtime execution.
- Updated runtime execution to realize parent graphs recursively, run view ops without allocation, dispatch graph-break handlers, and synchronize CUDA streams.
- Updated CPU and CUDA JIT compilers to treat realized non-fusible nodes as leaves while still recomputing lowerable graph nodes for tape execution.
- Updated CPU JIT lowering registration for pointwise operations through a centralized lowering registry.
- Updated CUDA JIT lowering registration and kernel launch behavior to use the shared CUDA configuration.
- Updated CPU and CUDA matmul kernels to accept tensor views and optional output buffers.
- Updated CPU and CUDA sum kernels to reuse output buffers by clearing them before accumulation.
- Updated CUDA matmul to stage incompatible non-contiguous or irregular batched views into contiguous temporary buffers before cuBLAS execution.
- Updated CUDA reduction and JIT kernels to cache compiled kernels by generated source.
- Updated allocator interfaces to include `memset`, `memcpy`, dtype-aware `fill_n`, host/device copy operations, host visibility checks, and cache clearing.
- Updated pybind module initialization to delegate the native surface to generated bindings and expose the extension as `_C`.
- Updated Python package exports to expose `Tensor`, `ComputeTape`, and `SGD` from `ormi_core`.
- Updated Python package metadata to version `0.1.0a2`.
- Updated the build to skip tests and examples under `SKBUILD`.
- Updated `README.md` with beginner-friendly installation checks, source build instructions, native test/example build steps, platform compatibility, CUDA and CPU build behavior, and advanced configuration variables.
- Updated CUDA availability checks in tests to probe ORMI's native CUDA path instead of checking for third-party package presence.
- Updated `uv.lock` metadata.

### Fixed

- Fixed gradient accumulation for graphs where multiple paths contribute to the same input gradient.
- Fixed broadcasted backward behavior for pointwise operations, reductions, expand, and batched matmul.
- Fixed shape diagnostics for incompatible broadcasting, invalid reshape volumes, and invalid matmul dimensions.
- Fixed allocator safety checks for double frees, unknown pointers, duplicate allocation bookkeeping, and allocation size overflow.
- Fixed CUDA allocator cleanup to synchronize before freeing live allocations and to warn about leaked live buffers.
- Fixed CUDA out-of-memory handling by trimming or retrying supported memory pools before raising an ORMI memory error.
- Fixed buffer item reads to validate bounds and support Float16 conversion.
- Fixed CUDA and CPU output buffer reuse paths to clear reduction outputs before accumulation.
- Fixed core runtime error paths to use ORMI-specific exception types instead of raw `std::runtime_error` paths.
- Fixed CPU-only optimizer builds so CPU optimizer code no longer includes CUDA headers unless `ORMI_USE_CUDA` is enabled.
- Fixed clean CI wheel builds by replacing the external `half.hpp` requirement with internal Float16 conversion helpers.
- Fixed CUDA 12.4 wheel builds by using the CUDA Driver API default memory pool call available in the release workflow toolkit.

### Removed

- Removed `csrc/core/memory.cpp` and `csrc/core/memory.h` in favor of `Buffer`.
- Removed old handwritten Python binding files for BLAS, device, graph, memory, operation enum, pointwise, reduce, and view bindings in favor of generated bindings.
- Removed old backend kernel header names such as `cpu_matmul.h`, `cpu_sum.h`, `cuda_matmul.cuh`, and `cuda_sum.cuh`; the new headers use operation-local names such as `matmul.h`, `sum.h`, `matmul.cuh`, and `sum.cuh`.
- Removed obsolete Python example scripts `example/test_autograd.py`, `example/test_graph.py`, and `example/test_matmul.py`.
- Removed the generated Windows linker artifact `ormi_backend.cp313-win_amd64.ilk` from the Python extension output tree.
- Removed the old monolithic `ormi_backend.pyi` extension stub in favor of generated `_C/__init__.pyi` and `_C/ops.pyi` stubs.
- Removed unused `cuda12` and `cuda13` package extras that installed third-party CUDA Python packages but were not used by ORMI.

### Tests

- Expanded the C++ test suite across allocator, buffer, shape, node, autograd, pointwise, view, reduction, matmul, backend runtime, graph breakers, CPU/CUDA JIT, CPU/CUDA lowerings, CPU/CUDA optimizers, CPU/CUDA kernels, tapes, and CUDA graph capture.
- Added optimizer and tape-specific tests for SGD, topology sorting, compute tape execution, zero-grad behavior, and CUDA graph capture/replay.
- Updated existing operation tests to cover broadcasting, gradients, views, dtype/device checks, and backend execution behavior.
- Updated Python tensor tests to cover public gradient computation, compute tape capture/execution, and SGD parameter updates.

### Notes

- CPU matmul currently requires `Float32` because OpenBLAS does not provide native FP16 execution.
- CUDA matmul currently supports `Float32` and `Float16`; CUDA sum currently supports `Float32` and `Float16`.
- CPU JIT compilation requires `g++` to be available on `PATH`; cache location can be overridden with `ORMI_CPU_JIT_CACHE_DIR`.
- CUDA acceleration is enabled when the CUDA toolkit is found at configure time.
