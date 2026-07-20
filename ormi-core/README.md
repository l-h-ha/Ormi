# ormi-core

`ormi-core` is ORMI's lazy tensor, automatic-differentiation, and C++ execution runtime for Python and C++.

Version `0.1.0a5` is an alpha release for 64-bit Windows and Linux. CPU is always available. CUDA is an optional backend plugin: importing ORMI and running CPU work do not load CUDA.

## Install

```bash
pip install ormi-core --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

This installs the CPU package, including OpenBLAS `0.3.33`. It contains no CUDA libraries.

Install the matching CUDA `12.4` backend and its redistributable runtime libraries with:

```bash
pip install "ormi-core[cuda]" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

The CUDA plugin is loaded only when a CUDA device is requested.

## Python API

```python
import ormi_core as oc

x = oc.Tensor([1.0, 2.0, 3.0])
y = (x * x + 1.0).sum()
dx = oc.grad(y, [x])[0]

print(y.numpy())
print(dx.numpy())
```

The alpha surface includes:

- `Tensor`, `Device`, and `DType` with Float64, Float32, Float16, Int32, and Bool storage.
- Lazy creation, pointwise arithmetic, elementary unary operations, casts, views, multi-axis reduction, and all-rank broadcasted matmul.
- Lazy reverse-mode `grad`, repeated `diff`, and higher-order differentiation.
- Autocast, functional SGD, and dynamic `GradScaler` support.
- CPU JIT cache inspection and clearing through `ormi_core.jit`.
- Generated type stubs for the root module and `ops`, `optim`, and `jit` submodules.

Fractional differentiation is planned but is not part of `0.1.0a5`.

## User Requirements

| Capability | Requirement |
| --- | --- |
| Python | CPython `3.10`, `3.11`, `3.12`, or `3.13`; NumPy `>=1.24`. |
| Windows wheel | 64-bit Windows and the Microsoft Visual C++ runtime supplied by supported Windows installations. |
| Linux wheel | x86-64 Linux with glibc `>=2.28` (`manylinux_2_28`). |
| CPU execution | No CUDA Toolkit, OpenBLAS installation, or compiler is required for ordinary kernels. OpenBLAS `0.3.33` is bundled in `ormi-core` wheels. |
| CPU JIT fusion | A supported C++20 compiler is required only when realizing an eligible fused graph that is not already in the persistent cache. Absence is reported at that realization, not at import. |
| CUDA execution | Install `ormi-core[cuda]`. An NVIDIA CUDA-capable GPU and NVIDIA driver `>=550.54.14` on Linux or `>=551.61` on Windows are required. The CUDA Toolkit is not required. |

The CUDA driver floors above are the CUDA `12.4` GA toolkit-driver versions, not the lower CUDA 12.x minor-compatibility floors. This release uses the conservative full-compatibility requirement.

The release build emits CUDA code for compute capabilities `5.0`, `5.2`, `6.0`, `6.1`, `7.0`, `7.5`, `8.0`, `8.6`, `8.9`, and `9.0`, with forward-compatible PTX for `9.0`. CUDA failures are lazy: a machine without a usable NVIDIA driver can import ORMI and run CPU tensors, but requesting `cuda` raises `DeviceError` or `cudaError`.

### CPU JIT Compiler Resolution

ORMI resolves a compiler once, on the first uncached eligible CPU fusion:

1. If `ORMI_CPU_JIT_CXX` is nonempty, ORMI resolves that executable or path and uses it.
2. Otherwise Windows probes `cl.exe`, `clang-cl.exe`, `g++.exe`, then `clang++.exe`, followed by Visual Studio discovery outside a developer shell.
3. Otherwise Linux probes `g++`, `clang++`, then `c++`.
4. If no usable compiler is found, ORMI raises `jitError` at fused realization. Non-JIT CPU operations remain usable.

`ORMI_CPU_JIT_CXX_ID` optionally declares how ORMI should invoke an override or filters autodetection. Values are case-insensitive: `gcc`, `clang`, `clang-cl`, and `msvc` (`cl` is an alias). This supports compiler launchers and wrappers whose executable names do not identify their command-line syntax.

## Builder Requirements

| Build | Requirement |
| --- | --- |
| C++ core | CMake `>=3.18` and a C++20 compiler. |
| Verified Linux toolchains | GCC `13` and Clang `18` on Ubuntu `24.04`. |
| Verified Windows toolchains | MSVC from Visual Studio 2022, `clang-cl`, and GCC in MSYS2 UCRT64. |
| Python extension | CPython `>=3.10`, scikit-build-core `>=0.8`, pybind11 `>=2.11`, and PyYAML `>=6.0`. Isolated `pip` builds install these build dependencies. |
| Tests | GoogleTest is fetched by CMake when `ORMI_BUILD_TESTS=ON`. |
| Benchmarks | Google Benchmark is fetched by CMake when `ORMI_BUILD_BENCHMARKS=ON`. |
| OpenBLAS | Optional. `ORMI_ENABLE_OPENBLAS=ON` requires the exact OpenBLAS `0.3.33` CMake package. Windows release builds pin it through `vcpkg.json`. |
| CUDA | Optional. `ORMI_ENABLE_CUDA=ON` requires the CUDA Toolkit, NVCC with C++20 support, cuBLAS, CUDA Runtime, CUDA Driver import library, and NVRTC. Published wheels use Toolkit `12.4`. |
| Network | Required on the first dependency-fetching configure, and for isolated Python build dependencies. |

Windows and Linux are the only implemented platform adapters. Core tensors and operations contain no operating-system branches; a new platform requires one process/filesystem/shared-library adapter and one CMake platform selection branch.

## Build From Source

The default C++ build is CPU-only, uses ORMI's direct CPU matmul, and does not build Python:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

Build the Python package through its isolated PEP 517 configuration:

```bash
python -m pip install .
```

Enable optional backends explicitly:

```bash
cmake -S . -B build -DORMI_ENABLE_OPENBLAS=ON -DORMI_ENABLE_CUDA=ON
cmake --build build --parallel
```

Normal C++ consumers link `ormi::core`. Code that directly consumes the optional CUDA plugin can additionally link `ormi::cuda` when `ORMI_ENABLE_CUDA=ON`.

## Binding Generation

The Python API is declared in `csrc/bindings/defs/*.yaml`. CMake runs `csrc/bindings/generate.py` to produce the pybind11 registration unit and `.pyi` files from the same declarations. `glue.cpp` contains only Python/NumPy conversion and ergonomic argument handling that cannot be expressed as direct pybind11 bindings.

## Runtime Configuration

| Name | Default | Meaning |
| --- | --- | --- |
| `ORMI_CPU_JIT_CXX` | unset | Compiler executable name or path override. |
| `ORMI_CPU_JIT_CXX_ID` | auto | Override syntax: `gcc`, `clang`, `clang-cl`, or `msvc`. |
| `ORMI_CPU_JIT_TIMEOUT_MS` | `120000` | Compiler deadline in milliseconds; `0` disables the deadline. |
| `ORMI_CPU_JIT_CACHE_DIR` | platform user cache | Persistent compiled-kernel cache directory. |
| `ORMI_CPU_JIT_CACHE_MAX_BYTES` | `268435456` | Persistent cache budget; `0` disables automatic eviction. |

## License

MIT. Bundled third-party runtime libraries retain their own licenses.
