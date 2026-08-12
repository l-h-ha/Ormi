# ormi-cuda

`ormi-cuda` is the optional CUDA 12.4 plugin for `ormi-core` 0.1.0a6.

## Install

Install it through the matching `ormi-core` extra:

```bash
python -m pip install "ormi-core[cuda]==0.1.0a6" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

The wheel includes CUDA Runtime, cuBLAS, and NVRTC libraries. Users need an
NVIDIA GPU and an [NVIDIA driver](https://www.nvidia.com/Download/index.aspx/)
version `550.54.14` or newer on Linux or `551.61` or newer on Windows. Users do
not need the CUDA Toolkit.

The plugin is loaded only when a CUDA device is requested. Installing it does
not change CPU startup or require CUDA during `import ormi_core`.

Fused CUDA kernels are compiled on first use and stored as PTX in the user's
cache. `ORMI_CUDA_JIT_CACHE_DIR` overrides the cache directory, and
`ORMI_CUDA_JIT_CACHE_MAX_BYTES` sets its byte budget. The default budget is
`268435456`; `0` disables automatic eviction.

## Build

Building requires the ORMI monorepo checkout, CMake 3.18 or newer, a C++20
compiler, and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
with NVCC, CUDA Runtime, CUDA Driver import library, cuBLAS, and NVRTC. Release
wheels are built with CUDA Toolkit 12.4; CMake does not currently enforce an
exact Toolkit version.

Build and test the CUDA runtime from `ormi-core`:

```bash
cd ormi-core
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
```

On Windows, run these commands from a Visual Studio 2022 developer shell.

Build the separately installable wheel from the repository root:

```bash
python -m pip wheel ./ormi-cuda
```

See [../ormi-core/README.md](../ormi-core/README.md) for supported GPU
architectures, source-build options, and runtime compiler requirements.
