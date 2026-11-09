# ormi-cuda

`ormi-cuda` is the optional CUDA 12.4 plugin for `ormi-core` 0.1.0a7.

## Install

Install it through the matching `ormi-core` extra:

```bash
python -m pip install "ormi-core[cuda]==0.1.0a7" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

The wheel includes CUDA Runtime, cuBLAS, and NVRTC libraries. Users need an
NVIDIA GPU and an [NVIDIA driver](https://www.nvidia.com/Download/index.aspx/)
version `550.54.14` or newer on Linux or `551.61` or newer on Windows. Users do
not need the CUDA Toolkit.

The plugin is loaded only when a CUDA device is requested. Importing
`ormi_core` and using CPU tensors do not load the CUDA backend.

NVRTC compiles fused CUDA kernels to GPU code (PTX). ORMI saves it on disk
for reuse in later runs.
`ORMI_CUDA_JIT_CACHE_DIR` overrides the cache directory, and
`ORMI_CUDA_JIT_CACHE_MAX_BYTES` sets its byte budget. The default budget is
`268435456`. Set it to `0` to disable automatic removal of old cache entries.

## Build

Building requires a checkout of the ORMI repository, CMake 3.18 or newer, a C++20
compiler, and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
with NVCC, CUDA Runtime, CUDA Driver import library, cuBLAS, NVRTC, and cuRAND
headers. Dropout needs only cuRAND's Philox headers. Release
wheels are built with CUDA Toolkit 12.4. CMake does not currently enforce an
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
