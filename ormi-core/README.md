# ormi-core

`ormi-core` is ORMI's lazy tensor, automatic-differentiation, and execution
library. Version `0.1.0a6` supports x86-64 Windows and Linux.

## Users

CPU:

```bash
python -m pip install "ormi-core==0.1.0a6" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

CPU and CUDA:

```bash
python -m pip install "ormi-core[cuda]==0.1.0a6" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

```python
import ormi_core as oc

x = oc.Tensor([1.0, 2.0, 3.0])
y = (x * x + 1.0).sum()
dx = oc.grad(y, [x])[0]
```

### Requirements

| Use | Requirement |
| --- | --- |
| Python | [CPython 3.10-3.13](https://www.python.org/downloads/) and NumPy 1.24 or newer. |
| Windows | 64-bit Windows. |
| Linux | x86-64 Linux with glibc 2.28 or newer. |
| CPU | No OpenBLAS or SLEEF installation. The wheel includes OpenBLAS 0.3.33 and SLEEF 3.9.0. |
| CPU fusion | A supported C++20 compiler when an uncached fused kernel is first realized. |
| CUDA | The CUDA extra, a supported NVIDIA GPU, and a compatible [NVIDIA driver](https://www.nvidia.com/Download/index.aspx/). No CUDA Toolkit. |

Published CUDA wheels use CUDA 12.4. The minimum supported driver is
`550.54.14` on Linux and `551.61` on Windows. The wheels target compute
capabilities 5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, and 9.0, with PTX
for newer 9.x devices. Float16 fusion requires compute capability 5.3 or newer.

CUDA loads only when a CUDA device is requested. Importing `ormi_core` and
using CPU tensors does not require CUDA.

### CPU Fusion Compiler

Compiler lookup happens only when an uncached fused CPU kernel is needed:

1. `ORMI_CPU_JIT_CXX`, when set, supplies the executable name or path.
2. Windows tries `cl.exe`, `clang-cl.exe`, `g++.exe`, and `clang++.exe`, then
   searches for Visual Studio.
3. Linux tries `g++`, `clang++`, and `c++`.
4. Failure raises `jitError` at fused realization. Import and non-fused CPU
   operations still work.

Automatic candidates must pass a C++20 syntax and standard-library header
probe. Setting `ORMI_CPU_JIT_CXX` skips that probe; if the specified compiler
fails, ORMI does not fall back to another compiler.

With `ORMI_CPU_JIT_CXX` set, `ORMI_CPU_JIT_CXX_ID` specifies its command-line
syntax. Without `ORMI_CPU_JIT_CXX`, it restricts automatic selection to that
compiler family. Values are case-insensitive: `gcc`, `clang`, `clang-cl`, and
`msvc`; `cl` is an alias for `msvc`.

`ORMI_CPU_JIT_TARGET` selects the generated CPU instruction set. Its default,
`auto`, uses AVX2 when the CPU and operating system support it and otherwise
uses the baseline instruction set. `baseline` and `avx2` are explicit
case-insensitive overrides; requesting unsupported AVX2 raises `jitError` only
when a fused CPU kernel is first compiled.

On AVX2 CPUs with F16C, ORMI also uses packed Float16 conversion and arithmetic
for eligible fused kernels. F16C is detected automatically and is not required;
CPUs without F16C use the scalar Float16 path.

Published wheels use SLEEF 3.9.0 for eight-lane Float32 `exp`, `log`, `sin`,
and `cos` inside eligible AVX2 fused kernels. ORMI passes these functions to
generated kernels through its own table, so the user's JIT compiler does not
need SLEEF headers or libraries. Non-AVX2 systems use scalar standard-library
math.

Compiler installers and instructions:

- Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  (Visual Studio 2022 is tested),
  [LLVM/Clang](https://releases.llvm.org/), or
  [MSYS2 UCRT64 GCC](https://www.msys2.org/).
- Linux: [GCC](https://gcc.gnu.org/install/) or
  [LLVM/Clang](https://releases.llvm.org/). Ubuntu users can follow the
  [Ubuntu GCC setup guide](https://ubuntu.com/developers/docs/howto/gcc-setup/).

## Builders

| Feature | Requirement |
| --- | --- |
| Core | [CMake 3.18+](https://cmake.org/download/) and a C++20 compiler. [Ninja](https://ninja-build.org/) is required by the Ninja presets. |
| Linux toolchains tested in CI | GCC 13 and Clang 18 on Ubuntu 24.04. |
| Windows toolchains tested in CI | Visual Studio 2022 MSVC, clang-cl, and MSYS2 UCRT64 GCC. |
| Python build | CPython 3.10-3.13. Build isolation installs scikit-build-core 1.0+, pybind11 2.11+, and PyYAML 6.0+. |
| Tests and benchmarks | Network access on the first configure; CMake fetches GoogleTest and Google Benchmark. Tests also need a matching C compiler. |
| Sanitizers | Windows x64: Visual Studio 2022 MSVC with [C++ AddressSanitizer](https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170). Linux: Clang with [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) and [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html). |
| OpenBLAS | Optional; exactly [OpenBLAS 0.3.33](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.33) with a discoverable CMake package. |
| SLEEF | Optional; exactly [SLEEF 3.9.0](https://github.com/shibatch/sleef/releases/tag/3.9.0) with a discoverable CMake package. |
| CUDA | Optional; [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) with C++20-capable NVCC, CUDA Runtime, CUDA Driver import library, cuBLAS, and NVRTC. Release wheels use Toolkit 12.4. |

Only Windows and Linux have platform implementations.

### CPU

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

Compiler-specific presets are `cpu-gcc-debug`, `cpu-clang-debug`,
`cpu-msvc-debug`, and `cpu-clang-cl-debug`. Release benchmarks use
`cpu-bench`.

Run the instrumented CPU suite on Windows x64 from a Visual Studio 2022
developer shell:

```powershell
cmake --preset cpu-msvc-sanitize
cmake --build --preset cpu-msvc-sanitize
ctest --preset cpu-msvc-sanitize
```

On Linux, use Clang AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
cmake --preset cpu-clang-sanitize
cmake --build --preset cpu-clang-sanitize
ctest --preset cpu-clang-sanitize
```

### OpenBLAS

```bash
cmake -S . -B build-openblas -G Ninja -DORMI_ENABLE_OPENBLAS=ON -DCMAKE_PREFIX_PATH=/path/to/openblas
cmake --build build-openblas --parallel
ctest --test-dir build-openblas --output-on-failure
```

On Windows, replace `CMAKE_PREFIX_PATH` with:

```text
-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

The repository's `vcpkg.json` pins the Windows dependency used by release
builds. Follow Microsoft's [vcpkg setup guide](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started)
to install vcpkg.

Enable vectorized Float32 transcendental functions with:

```bash
cmake -S . -B build-sleef -G Ninja -DORMI_ENABLE_SLEEF=ON -DCMAKE_PREFIX_PATH=/path/to/sleef
cmake --build build-sleef --parallel
ctest --test-dir build-sleef --output-on-failure
```

On Windows, the vcpkg toolchain in the preceding example supplies SLEEF as
well as OpenBLAS.

### CUDA

```bash
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
```

On Windows, run this from a Visual Studio 2022 developer shell. Release
benchmarks use `cuda-bench`.

### Python And C++

Build the CPU Python package from source:

```bash
python -m pip install .
```

That source build has OpenBLAS, SLEEF, and CUDA disabled unless explicitly
configured.
Build the separately installable CUDA wheel from the repository root with:

```bash
python -m pip wheel ./ormi-cuda
```

For C++, use CMake `add_subdirectory` and link `ormi::core`. When CUDA is
enabled, C++ applications that call the CUDA backend directly may also link
`ormi::cuda`. A `find_package(ormi)` package is not installed yet.

## Runtime Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `ORMI_CPU_THREADS` | Detected physical cores | Threads used by large contiguous CPU JIT maps; `1` forces serial execution. |
| `ORMI_CPU_JIT_CXX` | Unset | Compiler executable name or path. |
| `ORMI_CPU_JIT_CXX_ID` | Auto | Compiler syntax: `gcc`, `clang`, `clang-cl`, or `msvc`. |
| `ORMI_CPU_JIT_TARGET` | `auto` | CPU instruction set: `auto`, `baseline`, or `avx2`. |
| `ORMI_CPU_JIT_TIMEOUT_MS` | `120000` | Compile timeout in milliseconds; `0` disables it. |
| `ORMI_CPU_JIT_CACHE_DIR` | User cache | Persistent fused-kernel cache directory. |
| `ORMI_CPU_JIT_CACHE_MAX_BYTES` | `268435456` | Cache budget in bytes; `0` disables eviction. |
| `ORMI_CUDA_JIT_CACHE_DIR` | User cache | Persistent CUDA source and PTX cache directory. |
| `ORMI_CUDA_JIT_CACHE_MAX_BYTES` | `268435456` | CUDA cache budget in bytes; `0` disables eviction. |

Bindings and `.pyi` files are generated from `csrc/bindings/defs/*.yaml`.

## License

MIT. Bundled dependencies retain their own licenses.
