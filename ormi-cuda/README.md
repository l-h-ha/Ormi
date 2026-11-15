# ormi-cuda

`ormi-cuda` is the optional CUDA plugin for `ormi-core`.

## Users

### Install

For Python training, follow the [ORMI CUDA installation](../README.md#install).
To use `ormi_core` directly, install the matching extra:

```bash
python -m pip install "ormi-core[cuda]==0.1.0a8" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

### Requirements

Use CPython 3.10-3.13 on x86-64 Windows or Linux. See the
[core requirements](../ormi-core/README.md#requirements) for platform details.
Version 0.1.0a8 wheels bundle CUDA 12.8 Runtime, cuBLAS, NVRTC, and matching
builtins. CUDA execution needs an NVIDIA GPU and
[NVIDIA driver](https://www.nvidia.com/Download/index.aspx/)
version `570.124.06` or newer on Linux or `572.61` or newer on Windows. Users do
not need a C++ compiler, CMake, Ninja, or the CUDA Toolkit.

The plugin loads on first CUDA use. CPU tensors and `ormi_core` imports do not load it.

### Usage

```python
import ormi_core as oc

x = oc.tensor([1.0, 2.0], device="cuda:0")
print((x * x).sum().item())
```

`cuda:0` selects the first GPU. Reading the result executes the graph.

### Kernel Cache

NVRTC compiles fused kernels to PTX for the GPU driver. ORMI caches it on disk.
`ORMI_CUDA_JIT_CACHE_DIR` overrides the cache directory, and
`ORMI_CUDA_JIT_CACHE_MAX_BYTES` sets its byte budget. The default budget is
`268435456`. Set it to `0` to disable automatic removal of old cache entries.

## Builders

### Requirements

Start with a [source checkout](../README.md#from-source).
Building requires [CMake 3.18+](https://cmake.org/download/), a supported
[C++20 compiler](../ormi-core/README.md#compilers), and the
[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
with NVCC, CUDA Runtime, CUDA Driver import library, cuBLAS, NVRTC, and cuRAND
headers. Dropout uses only cuRAND's Philox headers. Wheel builds use Toolkit
12.8, with 12.8.1 pinned on Windows. CMake enforces no exact Toolkit version.
On Windows, use an x64 Visual Studio 2022 developer shell with MSVC as
NVCC's host compiler. On Linux, use a host compiler supported by the Toolkit.

CUDA 12.8 wheels require driver `570.124.06` or newer on Linux or `572.61` or
newer on Windows for CUDA JIT, per the
[CUDA 12.8 Update 1 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html).
They add compute capability 10.0 to the existing GPU targets, with generic
PTX for newer devices. NVRTC and its builtins must come from the same Toolkit.

Version 0.1.0a8 fusion selects the highest NVRTC-supported target no
greater than the GPU's compute capability. The driver compiles the PTX.
Runtime compilation needs no additional compiler installation.

### Runtime Tests

Install CMake 3.21+ and [Ninja](https://ninja-build.org/) for these presets.
Running the CUDA tests requires a supported GPU and NVIDIA driver.
From the repository root:

```bash
cd ormi-core
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
cd ..
```

### Wheel

From the repository root, build the plugin wheel into `dist/`:

```bash
python -m pip wheel --no-deps --wheel-dir dist ./ormi-cuda
```

This requires CPython 3.10-3.13. Build isolation installs Python build dependencies,
CMake, and Ninja as needed. `--no-deps` builds only the plugin wheel.
Using it requires the matching `ormi-core` package.
Local wheels depend on the installed Toolkit libraries. The release workflow
bundles those libraries for distribution.

See the [core guide](../ormi-core/README.md) for supported GPU
architectures, source-build options, and runtime compiler requirements.
