# ormi-core

A C++ accelerated tensor and autograd engine for Python.

`ormi-core` gives you a small Python `Tensor` API backed by native CPU kernels and optional NVIDIA CUDA kernels.

## Installation

Most users should install the published wheel:

```bash
pip install ormi-core --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

## What Works

| Platform | Status |
| --- | --- |
| Python | Python `>=3.10`; wheels are configured for CPython `3.10`, `3.11`, `3.12`, and `3.13`. |
| Linux | Published wheels target `x86_64` manylinux `2_28`. |
| Windows | Published wheels target 64-bit Windows. |
| macOS / MacBook | No published wheel or CI job yet. CPU source builds may work, but macOS is currently unverified. NVIDIA CUDA is not available on modern macOS. |
| CPU | Works without CUDA. CPU matmul currently uses OpenBLAS and supports `Float32`. |
| CUDA | Optional. Release wheels are built with CUDA Toolkit `12.4`; source builds use the CUDA Toolkit found on your machine. CUDA matmul and sum currently support `Float32` and `Float16`. |

Use `device="cpu"`, `device="cuda"`, or `device="cuda:<id>"` in Python. If you leave the device as `None`, ORMI uses CPU.

## Build From Source

Build from source if you are developing ORMI, building for a platform without a wheel, or changing the native C++/CUDA code.

### Prerequisites

- Python `>=3.10`.
- CMake `>=3.18`.
- A C++23 compiler.
- Python build helpers: `scikit-build-core`, `pybind11`, and `build`. `pip` installs these automatically for normal package builds, but direct CMake builds still need `pybind11` importable from Python.
- OpenBLAS and `cblas.h`.
- `half.hpp`.
- Internet access the first time CMake fetches `magic_enum`, and the first time native tests fetch GoogleTest.
- `g++` on `PATH` if you want to use the CPU JIT at runtime.
- Optional CUDA support: NVIDIA CUDA Toolkit, NVRTC, cuBLAS, and a compatible NVIDIA driver.

On Windows, the current CI setup uses MSVC, Ninja, and OpenBLAS from vcpkg:

```powershell
vcpkg install openblas:x64-windows
```

### Editable Python Build

From the repository root:

```bash
cd ormi-core
python -m pip install -U pip
python -m pip install -e .
```

On Windows with vcpkg OpenBLAS, set the vcpkg toolchain before installing:

```powershell
cd ormi-core
$env:CMAKE_GENERATOR = "Ninja"
$env:SKBUILD_CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
python -m pip install -U pip ninja
python -m pip install -e .
```

### Build A Wheel

```bash
cd ormi-core
python -m pip install -U pip build
python -m build --wheel
python -m pip install dist/*.whl
```

On Windows, use the same `CMAKE_GENERATOR` and `SKBUILD_CMAKE_ARGS` settings shown above before running `python -m build --wheel`.

### Build C++ Tests And Examples

Use this path when you want to run the native test suite:

```bash
cd ormi-core
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

On Windows with vcpkg:

```powershell
cd ormi-core
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
ctest --test-dir build --output-on-failure
```

## CUDA Builds

If CUDA is not installed, ORMI builds the CPU backend only.

If CUDA is installed and CMake can find it, ORMI automatically enables the CUDA backend. The release workflow builds with CUDA Toolkit `12.4`.

CUDA support is implemented in the native C++/CUDA extension.

To point the build at a specific CUDA Toolkit, set `CUDACXX` before building:

```bash
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
python -m pip install -e .
```

On Windows:

```powershell
$env:CUDACXX = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"
python -m pip install -e .
```

`CUDA_PATH` is also used by the runtime CUDA JIT to find CUDA headers. If it is not set, ORMI falls back to `/usr/local/cuda/include`.

## Advanced Configuration

Most users do not need these variables.

| Name | Used for | Default / behavior |
| --- | --- | --- |
| `ORMI_CPU_JIT_OPT` | CPU JIT optimization | `-O3`; valid values are `-O0`, `-O1`, `-O2`, `-O3`, and `-Ofast`. |
| `ORMI_CPU_JIT_CACHE_DIR` | CPU JIT cache location | System temp directory under `ormi/cpu_jit`. |
| `ORMI_CUDA_MEMPOOL_RELEASE_THRESHOLD` | CUDA memory-pool release threshold | `8589934592` bytes, or 8 GiB. Use `0` to release memory aggressively. |
| `CUDA_PATH` | CUDA runtime JIT header lookup | Falls back to `/usr/local/cuda/include`. |
| `CUDACXX` | CUDA compiler path | Lets you choose the `nvcc` used by CMake. |
| `SKBUILD_CMAKE_ARGS` | scikit-build CMake arguments | Useful for passing a vcpkg toolchain file on Windows. |
| `CMAKE_GENERATOR` | CMake generator selection | Windows wheel builds use `Ninja`. |

## Release Workflow

Release builds run when a tag matching `ormi-core-v*` is pushed. The workflow builds wheels on `ubuntu-latest` and `windows-2022`, builds a source distribution on Ubuntu, creates a GitHub Release, and publishes a simple package index at:

```text
https://l-h-ha.github.io/Ormi/whl/
```

## License

MIT
