# Changelog

## 0.1.0a8

### Added

- Added tensor indexing with integers, slices, and ellipsis, plus concatenation
  on any axis. Both support lazy execution and higher-order derivatives.
- Added Windows and Linux CI for Python-only pull requests, with cached CPU
  core wheels reused when core sources and build settings match.
- Added CPU/CUDA numerical derivative checks, including changed-input Plan
  runs and replay after the original Plan is released.

### Changed

- Updated CUDA wheels to Toolkit 12.8 and added compute capability 10.0,
  retaining the existing GPU targets and generic PTX.
- Bumped the backend ABI from 18 to 20. Install matching core and CUDA plugin
  versions.
- Bounded CPU fusion kernels and compiler batches to limit generated source
  size. Reused shared scalar branches and removed loops from scalar kernels.
- Limited gradient traversal to the depths needed by the requested inputs,
  retaining graph edges for higher derivatives.
- Separated wheel installation from source-build instructions and revised
  the benchmark guide.

### Fixed

- Prevented signed overflow in stride calculations and contiguity checks.
  Corrected empty-shape counts when large dimensions precede a zero.
- Selected an NVRTC-supported PTX target for GPUs newer than the bundled
  compiler. Disk-cache keys include the selected target.
- Kept single-Tensor and Tensor-sequence `grad` overloads distinct after
  adding tensor indexing.

## 0.1.0a7

### Added

- Added `ormi` builds and publishing alongside `ormi-core` and `ormi-cuda`,
  with version-pin checks and clean-install tests on Windows and Linux with
  CPython 3.10-3.13.
- Added C++ neural networks with YAML-generated Python bindings: Linear,
  Sequential, LayerNorm, BatchNorm, Dropout, common activations and losses,
  `rsqrt`, and indexed selection.
- Added custom Python layers with named parameters, buffers, and children.
  Python builds require pybind11 3.0 or newer.
- Added `batches` and `fit` with seeded shuffling, progress reporting, one
  Plan per batch shape, and short final batches kept by default. Updated the
  PINN and minibatch examples to use `fit`.
- Added `Tensor.copy_from` for updating input storage used by Plans and CUDA
  replay. Source and destination shapes and dtypes must match.
- Added model-only checkpoints through `ormi.save` and `ormi.load`, covering
  parameters, buffers, and shared tensors.
- Added single-input `grad` and lazy higher-order `partial` derivatives for
  interacting points.

### Changed

- Added CPU execution without a compatible C++ compiler, using cached fused
  kernels or precompiled operations. Invalid settings and compiler failures
  raise errors.
- Added C++20 syntax and header checks to automatic CPU compiler discovery.
- Added span traversal for strided CPU unary and broadcast inputs. Skip fusion
  source generation for compiler-free Plans with an empty JIT cache.
- Parallelized large AVX2 matmuls across CPU workers with right-hand panel reuse.
- Bumped the backend ABI to 18 for indexed operations, dropout masks, and
  scalar scatter. Core and CUDA plugin ABIs must match.
- Separated Plan construction (`plan/build.cpp`), execution (`plan/run.cpp`),
  and recording and replay (`plan/record.cpp`) under `core/runtime/`.
- Separated CPU JIT compiler discovery, invocation, and cache handling, and
  CUDA compilation, kernel loading, and execution.

### Fixed

- Fixed SLEEF DLL staging for Windows C++ examples.
- Deferred layers appended during a Sequential forward pass until its next call.
- Fixed CPU JIT reduction indices for singleton axes with equal strides.
- Released CPU JIT cache locks before repairing invalid libraries, reported
  cache inspection errors, and discarded stale compiler probe results.
- Initialized CUDA graph handles to null for cleanup after failed capture.
- Pinned Linux CUDA wheel repair to `patchelf 0.17.2.4` to avoid cuBLAS load failures.
- Fixed process tests that assumed the child reached `main()` before a timeout.
  Linked the MinGW helper statically for tests that replace `PATH`, and fixed
  GCC builds of CUDA grouped-launch tests.

## 0.1.0a6 - 2026-08-12

### Changed

- Corrected the cold CPU JIT benchmark to use a distinct artifact per iteration
  and release cache leases before cleanup. Measured compiler discovery separately
  from compilation. Removed redundant version probes for canonical compiler
  names and explicit `ORMI_CPU_JIT_CXX_ID` values; retained inspection for
  unusually named wrappers without an ID. On the Windows development system,
  automatic clang-cl resolution fell from a 58.0 ms median to 2.14 ms. A corrected
  five-build run of the same 11-operation kernel measured 552 ms total, including
  522.8 ms in compile-and-link. A clang time trace for a larger generated kernel
  measured 567 ms in the frontend and 25 ms in backend code generation.
- Included `<bit>` and `<cstdint>` only for Float16, `<cmath>` only for
  transcendental operations, and `<immintrin.h>` only for x86 intrinsics in
  generated CPU JIT source. Replaced the unconditional `<memory>` dependency
  with a portable alignment helper. In an eight-sample exact-source ABBA on
  Windows, median compile-and-link time fell from 654.910 to 182.658 ms with
  clang-cl, 673.805 to 235.072 ms with MSVC, 2002.463 to 425.312 ms with MinGW
  GCC, and 641.377 to 189.168 ms with Clang's GNU-style driver. Old and new
  sources produced identical exported-kernel instruction bytes with all four
  compilers.
- Expanded the cold CPU JIT benchmark to five 11-operation dependency classes:
  header-free Float32 arithmetic, scalar transcendental math, vector math,
  vector reduction, and Float16 arithmetic. Added generated-header categories
  alongside loop-planning, emission, compilation, cache, loading, source-size,
  and cache-status metrics. Across two seven-repetition clang-cl passes, with
  the second in reverse case order, uncached compile-and-link medians were
  521.728-592.616 ms for arithmetic, 691.430-792.502 ms for scalar math,
  776.261-919.000 ms for reduction, 841.516-1041.340 ms for vector math, and
  906.378-1032.970 ms for Float16.
- Evaluated generated-header costs with exact-source clang-cl controls. Against
  a 188.430 ms header-free median, forced `<cmath>` added 62.010 ms, forced
  `<immintrin.h>` added 193.252 ms, and the Float16 set of `<bit>`, `<cstdint>`,
  and `<immintrin.h>` added 249.650 ms. A separate interleaved run measured
  `<cmath>` plus `<immintrin.h>` at 290.346 ms above its header-free control.

  A reusable union precompiled-header experiment reduced the corresponding
  compile from 457.546 to 186.540 ms with the same 226 emitted kernel
  instructions. Building its 7,368,968-byte artifact cost 533.038 ms, increasing
  estimated first-compile time from 457.546 to 719.578 ms and reaching
  break-even around the second header-heavy kernel. Compiler-specific PCH
  caching was not adopted.
- Added a compiler-startup benchmark independent of source generation and
  linking. Through ORMI's process runner, startup measured 65.908 ms for clang-cl
  and 50.552 ms for MSVC on the Windows development system. A balanced
  exact-source clang-cl split measured 89.934 ms for object generation,
  99.740 ms for Microsoft linking, and 240.124 ms for the existing one-shot
  compile-and-link path. In separate linker experiments, `lld-link` was 4.07%
  slower than Microsoft linking in a balanced comparison; `/NOENTRY` appeared
  11.42% faster for arithmetic but failed valid scalar-math kernels with
  unresolved `expf` and `sinf`. Neither linker change was adopted.
- Added per-workload CPU JIT entry counts, generated-source bytes, and library
  bytes to framework benchmark JSON. Isolated current-package runs produced
  one kernel for pointwise, one for reduction, two for dense gradients, and
  seven for the second-order PINN, with cold Plan preparation of 0.717, 0.764,
  1.022, and 3.852 seconds respectively.
- Added optional batch preparation of fused backend steps after Plan scheduling.
  Immediate realization and backends without batch preparation use the
  single-step path. CPU preparation emits uncached expressions into one
  translation unit, loads distinct symbols from one shared library, and
  publishes each expression in the memory cache. On the development system,
  five fresh-process medium PINN runs had a cold-preparation median of
  0.825 seconds, versus the preceding 3.852-second isolated run, and produced
  one artifact instead of seven. Five medium dense runs had a 0.834-second
  median versus the preceding 1.022-second run and produced one artifact
  instead of two. A second process reused the PINN artifact, reducing
  preparation from 742.4 to 5.3 milliseconds without compilation. A cached
  seven-expression batch lookup measured 1.274 microseconds total during
  Plan construction.
- Added explicit planning of contiguous innermost reduced axes to CPU JIT
  reductions. Float32 arithmetic uses packed AVX2 with a portable generated-loop
  fallback; Float64 and non-AVX2 execution use the existing path. On the
  development system, the seeded `[256, 64, 64] -> [64]` two-axis mean fell
  from 0.987 to 0.242 milliseconds in direct mode and from 1.071 to 0.220
  milliseconds with a prepared Plan. Two repeated prepared runs measured a
  0.340-millisecond geometric median for PyTorch and 1.716 milliseconds for
  TensorFlow on the same inputs. The full one-thread Float32 corpus passed
  numerical validation; ORMI had the lowest warmed median in all 12 direct
  and all 12 prepared cases on this machine.
- Added a tracked framework benchmark corpus with seeded pointwise, reduction,
  dense-gradient, and second-order PINN workloads for ORMI, PyTorch, and
  TensorFlow. Separated preparation, first execution, and warmed timing; fixed
  framework thread counts; validated complete numerical results before timing;
  and reported unavailable modes. Added build, platform, processor, Python,
  timestamp, case-selection, and explicitly configured CPU JIT compiler
  metadata to labeled JSON results.

  A local one-thread Float32 checkpoint with 20 rounds, 10 calls per round, and five
  warmups produced 72 validated result rows. ORMI had the lowest warmed median
  in all 12 direct and all 12 optimized cases. Geometric-mean speedups over
  PyTorch were 4.88x for direct and 8.71x for optimized execution; over
  TensorFlow, they were 20.16x and 25.05x respectively. All three frameworks
  slowed by similar factors against the preceding saved checkpoint. A second
  ORMI process reused all 12 prepared artifacts without creating entries and
  reduced geometric-mean Plan preparation from 402.1 to 3.27 milliseconds,
  a 123.2x improvement.
- Added OpenBLAS architecture-kernel detection to Float32 GEMM dispatch.
  Compatible large products use ORMI's runtime-guarded AVX2/FMA kernels when
  OpenBLAS is absent or reports its generic core. Tuned OpenBLAS builds retain
  the existing crossover policy; Float64 retains the existing dispatch.
  Across contiguous, transposed-left, skinny, square, and 512-cubed products
  on the development system, ORMI's kernels sustained 69-105 GFLOP/s versus
  3.6-3.8 GFLOP/s for the bundled generic OpenBLAS. The cross-framework large
  dense-gradient median fell from 2.718 to 0.194 milliseconds in direct mode
  and from 2.475 to 0.162 milliseconds with a prepared Plan.
- Separated CPU matmul into tensor and batch traversal, one-product kernel
  selection, and portable or architecture-specific execution. Kept typed
  policy and portable loops compiler-visible for small-product and GEMV performance.
- Moved CPU JIT setup outside timed Plan-preparation benchmarks and added
  alias-chain depths from 128 through 4,096. The corrected Release benchmark
  measured one-element replay at 35.7 ns and a 31-step fifth-order derivative
  replay at 1.22 microseconds.
- Separated runtime lowering from step execution. Immediate realization follows
  `Tensor::realize -> schedule::run -> lower -> Step::run`: the scheduler lowers
  the root, traverses dependencies outside that step, and executes in postorder.
  Limited `Step` to allocation, aliasing, execution, and failure cleanup. Used
  realized buffers for immediate completion tracking and retained the pending
  set for Plan construction. Started immediate group discovery at unresolved
  root-step inputs and stopped at realized tensors, avoiding a second full
  graph walk for fully fused roots and grouping pointwise siblings beneath
  direct operations.

  In nine-repetition local Release measurements, warm fused-root medians at
  depths 1, 4, 16, and 64 changed from 1.69, 2.56, 7.59, and 28.77 microseconds
  to 1.37, 2.03, 5.15, and 22.57 microseconds. A longer depth-64 run measured
  16.54 microseconds.

  Added direct execution of a ready root Step when lowering leaves no
  unresolved input, bypassing an empty grouping table and a 16-frame traversal
  stack. In a bracketed nine-repetition Release A/B/A at depths 1, 4, 16, and
  64, control medians were 1.482, 2.283, 6.090, and 19.275 microseconds. The two
  direct-root runs measured 1.215-1.263, 2.024-2.147, 5.790-5.794, and
  15.627-16.448 microseconds respectively. Changed scheduler benchmarks to
  adaptive duration instead of a short fixed iteration count.
- Split fusion expression construction and compiled-kernel cache identity into
  `expr` and `key` units. Kept graph eligibility, traversal, broadcasting,
  reductions, and auxiliary outputs in expression code; moved structural
  copying, matching, and hashing into `Key`. Updated CPU and CUDA compilers
  to include the key directly and separated key tests and benchmarks from
  expression tests.

  In isolated 11-repetition Release measurements at 256 nodes, key construction
  measured 2.36 microseconds versus 2.46 before the split; expression
  construction measured 25.36 versus 24.97 microseconds. Moved build-only
  benchmarks into `bench_build.cpp`, covering unary chains, shared and distinct
  inputs, balanced trees, and fast rejection. A nine-repetition Release run
  measured 256-operation unary, shared-input, distinct-input, and balanced
  graphs at 21.4, 22.9, 35.6, and 34.1 microseconds respectively. Deep and
  balanced graphs had similar costs at matching node and input counts.

  An adjacent A/B measured the 256-operation shared chain at 27.2 microseconds
  with the local PMR arena versus 41.1 microseconds with plain containers.
  Larger-arena experiments were not adopted because gains varied across graph
  shapes; retained the 1 KiB arena.
- Split the Windows platform adapter into process, toolchain, dynamic-library,
  filesystem, CPU-query, and UTF conversion units. Kept `platform.h` as the
  runtime interface and declared complete source lists in each platform sidecar.
- Removed the empty top-level `ormi` placeholder project from the workspace.
- Narrowed the CPU JIT compilation coordinator. Moved target and vector-math
  selection to target policy, compiler command construction and invocation to
  compiler discovery, and work estimation to loop planning. Assigned
  memory-cache lookup, loop/code preparation, and publication to `compile.cpp`;
  assigned persistent-cache lookup, compilation, library loading, and repair
  to `artifact.cpp`.
- Split CPU JIT benchmarks into cold compilation, memory and disk lookup,
  library loading, execution preparation, source emission, prepared execution,
  and cached execution. Fresh-cache compilation measured 542.6 milliseconds;
  disk lookup plus library loading measured 0.579 milliseconds. Added a bounded
  four-entry per-thread front cache before the immutable shared table. In
  seven-repetition Release measurements, eight-thread memory lookup fell from
  249.3 to 35.3 nanoseconds and cached execution from 242.1 to 60.2 nanoseconds;
  single-thread cached execution changed from 88.9 to 92.2 nanoseconds.
  Alternating two kernels through generic lookup measured 60.0 nanoseconds on
  one thread and 22.4 nanoseconds at eight threads.
- Replaced backend-specific device identity with catalog-driven resolution.
  `Device::cpu()` resolves the single declared host backend; optional devices
  use `Device::from(name, index)` or strings such as `"cuda:0"`. Replaced
  per-backend predicates with `name`, `indexed`, and `is_host`, and removed the
  CUDA-specific constructor and Python helper. Backend implementations report
  operational failures through `BackendError`. Retained cached backend
  pointers for steady-state dispatch.
- Grouped Backend ABI 14 into mandatory `memory` and optional `ops`
  capabilities. Installation validates device-memory callbacks and fills
  omitted ordinary and fractional operations with throwing callbacks.
  Retained explicit fusion and recording capability queries and cached CPU
  and CUDA pointer dispatch. A seven-repetition Release check measured CPU
  allocator lookup at 2.78 ns and synchronization dispatch at 2.82 ns.
- Defined memory-transfer ownership: same-backend and host-to-backend copies
  use the destination backend; backend-to-host copies use the source.
  Transfers between accelerators offer the destination an optional direct-copy
  hook before staging through the single host backend. CUDA implements
  `within`, `from_host`, and `to_host` routes through this interface. In
  seven-repetition Release measurements, complete CPU buffer copies took
  12.8 ns for 64 bytes and 30.5 ns for 4 KiB, versus 5.61 ns and 23.7 ns for
  direct `memcpy`. Both paths measured within the 36-38 microsecond run-to-run
  range for 1 MiB.
- Added generic planning for compatible single-parent pointwise siblings.
  The CPU JIT recognizes sine/cosine pairs and retains paired vector-math
  lowering; backends without multi-output fusion use independent steps.
  In a five-repetition Release check of the maintained 23-step training
  sequence, the grouped batch-1,024 mean measured 118.4 microseconds versus
  141.4 microseconds ungrouped; smaller batches remained within run-to-run
  noise. Kept each parent's first output inline and allocated result vectors
  only for compatible siblings, removing a second pass and one-element
  vectors along unary chains. In a nine-repetition Release A/B at 256
  operations, an ungrouped unary chain fell from 90.2 to 51.1 microseconds,
  64 groups of four fell from 88.5 to 74.8 microseconds, and a 256-output
  fanout retained its 39.8-microsecond CPU median. Added grouping benchmarks
  for binary and unary chains, fanout, and independent groups.
- Added generic auxiliary outputs to CUDA pointwise fusion, with one typed
  output argument per tensor and one stable pointer array per prepared call.
  Covered Float16, Float32, Float64, empty tensors, and recorded CUDA graphs;
  reductions retained single-output execution. A shared-parent `neg`/`exp`
  benchmark used one plan step instead of two. Seven-repetition median
  completion time fell from 16.3 to 9.91 microseconds at one element, 16.2 to
  9.44 microseconds at 1,024 elements, and 136.5 to 113.5 microseconds at
  1,048,576 elements. The independent-launch control remained within 1.5%
  of its prior measurements.
- Split CUDA fusion benchmarks into direct execution, prepared execution,
  fresh-output allocation, reusable Plan execution, CUDA graph replay, and
  ordinary realization. Used manual launch-to-completion timing after a
  duration-based warmup for synchronized cases, excluding graph construction
  and output cleanup. In an 11-repetition Windows Release check, reusable Plan
  medians measured 10.57 microseconds at one element and 10.44 microseconds at
  1,024 elements, versus 15.17 and 18.85 microseconds for ordinary one-shot
  realization. Prepared and Plan execution both measured 10.44 microseconds
  at 1,024 elements. Added idle-stream host-submission controls for expression
  construction, backend preparation, output allocation, and launch submission.
  Removed a redundant `cudaSetDevice` from transient fusion execution,
  reducing direct one-shot completion from 12.08 to 9.68 microseconds at one
  element and from 11.86 to 9.55 microseconds at 1,024 elements; prepared
  execution remained within run-to-run noise.
- Standardized CUDA training benchmarks on a 50-millisecond completion-based
  warmup for ordinary Plan execution and recorded graph replay. For the
  maintained 23-step training graph, Windows Release median completion
  measured 198.2 versus 45.9 microseconds at batch 1 and 189.4 versus
  72.2 microseconds at batch 1,024. An isolated 15-repetition batch-32
  confirmation measured 211.4 versus 51.2 microseconds. Kept recording
  explicit because capture has a one-time cost and single-kernel Plans
  did not show this benefit.
- Hoisted tensor element counts out of CPU pointwise, cast, SGD, finite-check,
  and reduction loop conditions, removing repeated library calls and enabling
  MSVC vectorization of contiguous Float32, Float64, and Int32 arithmetic.
  The affected saved-gradient multiply fell from 167.75 to 5.50 microseconds
  at 32,768 elements.
- Retained expanded inputs when CPU JIT reductions select direct stride
  planning. This enabled large global reductions over broadcast views to
  split their largest reduced dimension while preserving the flat pointwise
  broadcast path.
- Derived fixed CPU matmul dimensions directly from tensors, removing
  per-call shape-metadata allocations. Added contiguous row-wise and
  transposed-left matrix-vector paths and runtime-selected x86-64 Float32
  AVX2 kernels, retaining portable execution for other processors. AVX2
  kernels keep up to 32 output columns in registers across the shared
  dimension, use four independent accumulators for contiguous matrix-vector
  rows, and use a four-row by 16-column tile for transposed-left products.
  In a bracketed MSVC benchmark, the 16 by 1,024 by 32 training-gradient median
  fell from 21.56 to 16.47 microseconds, with its CPU-time median improving by
  25.9%. In the packaged MSVC, SLEEF, and OpenBLAS configuration, the prepared
  23-operation training plan fell from 237.65 to 123.04 microseconds at batch
  1,024; batches 1 and 32 remained near 4.12 and 9.40 microseconds.
- Enabled fused multiply-add in Float32 AVX2 matmul kernels on processors
  reporting both AVX2 and FMA support, retaining the portable fallback.
  In a controlled MSVC object-level comparison, the 1,024 by 16 by 32
  forward-product median fell from 30.34 to 20.44 microseconds and the
  16 by 1,024 by 32 transposed-left median from 27.03 to 22.42 microseconds.
- Added a runtime-guarded AVX2 row-sum kernel for larger contiguous Float32
  reductions across leading or middle blocks of axes. Kept output columns
  in registers across reduced rows and retained portable execution for
  small inputs and other processors. The 1,024 by 32 first-axis sum fell
  from 10.07 to 1.12 microseconds, and the repeated 23-operation training-plan
  mean fell from 130.46 to 121.02 microseconds in the same packaged build.
- Selected SLEEF's approximately 3.5-ULP AVX2 variants for Float32 sine and
  cosine, retaining 1-ULP exponential and logarithm variants. Wide-range
  pointwise accuracy tests, including special values, remained bounded
  within 2e-6. In a paired build comparison, the batch-1,024 training-plan
  mean fell from 151.21 to 111.87 microseconds, the sine step from 32.89 to
  22.56 microseconds, and the cosine chain from 37.35 to 25.70 microseconds.
- Called SLEEF AVX2 functions directly from CPU JIT kernels when the core
  and runtime compiler use compatible vector ABIs. Retained buffered
  C-compatible callbacks for mixed Windows toolchains and included the
  selected ABI in JIT cache identity. At batch 1,024, the fused sine step
  fell from 21.48 to 17.93 microseconds, the fused cosine chain from 22.53
  to 18.04 microseconds, and the full 23-step training-plan median from
  126.68 to 92.43 microseconds.
- Applied `__vectorcall` to paired AVX2 sine/cosine callbacks with MSVC and
  clang-cl, replacing aligned stack temporaries with vector-register
  arguments and results. In a bracketed paired-output benchmark, medians
  fell from 76.1 to 64.5 nanoseconds at 32 elements, 1.334 to 1.000
  microseconds at 1,024 elements, and 38.21 to 34.00 microseconds at
  32,768 elements.
- Counted paired SLEEF sine/cosine nodes as one vector-math call in CPU
  parallel-work estimates, retaining separate scalar-fallback and nested
  trigonometric costs. A six-to-one participant sweep selected three
  participants for the 32,768-element pair. Compared with six participants,
  the prepared-kernel median fell from 48.15 to 30.16 microseconds and the
  matching training-plan step from 59.66 to 34.10 microseconds.
- Balanced CPU worker ranges in whole units and added a dedicated
  notification for each selected slot. Kept unused workers asleep and
  active-slot worker ownership stable across runs. The 32,768-element
  paired sine/cosine median fell from 30.16 to 18.77 microseconds and its
  matching training-plan step from 34.10 to 17.05 microseconds.
- Partitioned training-sized Float32 AVX2 matrix products across two
  persistent-worker row ranges, retaining serial execution for small
  products. In a bracketed comparison, the 1,024 by 16 by 32 forward product
  fell from 20.37 to 9.35 microseconds and the 16 by 1,024 by 32 transposed-left
  product from 20.35 to 9.38 microseconds. Retained internal AVX2 execution
  for these shapes based on direct OpenBLAS crossover benchmarks.
- Fixed CPU JIT target-name lifetime by referencing the static selected-target
  name instead of a temporary string.
- Added lazy left-sided Caputo differentiation on uniform grids for orders
  between zero and two, excluding one. Orders below one use the L1 scheme;
  orders above one use the L2 scheme and require the first derivative at the
  history origin.
- Added CPU and CUDA Caputo execution for Float32 and Float64 with arbitrary
  strided axes. Gradients cover the sampled history and initial derivative;
  discrete transposes remain lazy for higher-order autograd. Tests cover
  analytic histories, convergence, exact adjoints, CPU/CUDA parity, and Python.
- Reused CUDA L2 second differences for histories of 64 samples or more to reduce
  global-memory reads. Shorter histories use the direct single-kernel path.
- Shared immutable Caputo coefficients with adjoints. CUDA plans upload
  coefficients once and retain the L2 difference workspace, eliminating
  fractional setup allocations and coefficient transfers during repeated runs
  and recorded graphs.
- Exposed `tensor.caputo(...)` and `ormi_core.ops.caputo(...)` in the generated
  Python API. Forward and adjoint benchmarks cover both schemes through
  8,192-sample histories, including CUDA plan preparation and recording.
- Extended CPU and CUDA pointwise fusion to contiguous broadcast inputs,
  including scalar, rank-expanded, and multi-axis broadcasting, and to
  transposed and expanded views.
- Added Float16 pointwise fusion on CPU and CUDA with operation-by-operation
  half-precision rounding.
- Fused floating-point casts on CPU and CUDA, including Float16 autocast
  chains with Float32 operations and accumulation.
- Added CPU JIT vectorization hints for 64-byte alignment and non-aliasing
  outputs. Eligible contiguous Float32 and Float64 arithmetic chains support
  linear, scalar, and repeated contiguous broadcast spans, with baseline or
  AVX2 code selected for the host.
- Added the CPU target to JIT cache entries. Compiler discovery is deferred
  until an uncached kernel requires compilation.
- Added persistent CUDA fusion source and PTX caching. Keys include generated
  source, GPU compute capability, NVRTC version, and the CUDA JIT ABI.
  Per-entry locks serialize concurrent misses; publication is staged, and
  least-recently-used eviction limits the default cache to 256 MiB.
- Deferred process-lifetime CUDA module cleanup to context teardown instead of
  calling `cuModuleUnload` during static destruction. This fixes a driver
  access violation after instrumented reduction workloads.
- Removed the compilation mutex from CPU JIT execution through an immutable
  published kernel registry. Calls with eight or fewer inputs keep input
  pointers on the stack.
- Separated CPU JIT compilation, caching, and library loading from prepared
  invocation and reduction execution.
- Separated CUDA NVRTC compilation, PTX loading, and kernel publication from
  prepared launch state and reduction execution. Lifecycle tests cover unrun
  prepared reductions, direct/prepared parity, and concurrent kernel lookup.
- Unified CPU and CUDA compiled-kernel expression identity. CUDA cache matching
  includes the output shape used for tiling and code generation.
- Replaced the state-free `DeviceAccess` wrapper with a direct internal backend
  accessor.
- Added a persistent CPU worker pool for large flat and contiguous-broadcast
  pointwise JIT kernels. Broadcast ranges align to complete contiguous spans;
  work estimates include measured transcendental costs, and active participants
  are limited to available work. The maximum worker count follows detected
  physical cores on Windows and Linux, with an `ORMI_CPU_THREADS` override.
  Smaller kernels run on the caller thread. A 65,536-element arithmetic control
  measured 6.01 microseconds while the batch-1,024 training plan gained parallel
  transcendental execution.
- Added parallel CPU JIT `sum` and `mean` reductions with independent outputs.
  Trailing-axis reductions partition output values, first-axis reductions
  partition complete 32-column tiles, and general direct-stride reductions
  partition outer-dimension slabs. Input element count determines whether to
  parallelize. Contiguous global reductions partition the flat input;
  transposed and expanded global reductions partition the largest reduced
  dimension independently of traversal order. Direct loops place the lowest
  worst-case physical input stride innermost. Eligible Float32 and Float64
  loops with innermost strides of zero or one use independent lane accumulators
  and a scalar tail. Conflicting two-dimensional layouts use 16-by-16 cache
  tiles with worker ranges aligned to complete tiles. Higher-rank conflicting
  layouts tile the worker split dimension with one complementary physical-stride
  dimension and traverse remaining dimensions directly. Each worker produces
  one typed partial; Float16 uses Float32 partials, and `mean` division follows
  the combine. Smaller reductions run on the caller thread.
- Replaced per-element division and modulo in CPU JIT pointwise traversal with
  aligned arbitrary-rank stride tables for transposed and expanded inputs.
  Large two-dimensional transposes use 32-by-32 tiles.
- Enabled ranged CPU JIT transcendental execution independently of SIMD
  eligibility. Published builds use SLEEF 3.9.0 for eight-lane AVX2 Float32
  `exp`, `log`, `sin`, and `cos` through a compiler-neutral function table;
  scalar tails and non-AVX2 systems use standard-library math. On the development
  system, the prepared 32,768-element `cos(x) * gradient` kernel measured
  47.5 us between scalar runs of 85.1 us and 82.1 us. The complete batch-1024
  training plan measured 358 us between scalar runs of 450 us and 426 us.
- Added logical-dimension traversal with physical input strides for CPU JIT
  reductions fused from transposed or expanded maps. Two-dimensional
  single-axis reductions tile 32 adjacent outputs when every input is
  contiguous or broadcast along the output axis, covering ordinary first-axis
  and transposed last-axis reductions. Higher-rank reductions use 16-value
  tiles along the first surviving output dimension and ordinary loops for
  remaining output and reduced dimensions.
- Improved sequential CPU JIT Float16 and Float32 reduction accuracy by
  combining 32-value Float32 blocks in a Float64 running total.
- Improved Float32 and F16C SIMD reduction accuracy by processing 32 vector
  steps per Float32 lane block before combining lanes in Float64. Vectorized
  strided reductions use Float32 lanes with a Float64 scalar combine.
  Million-element contiguous, transposed, and packed fused reductions have a
  0.5-parts-per-million error limit; benchmarks report relative error and
  throughput.
- Added a portable CPU row kernel for matrix products with contiguous
  right-hand rows and outputs, up to one million scalar products. Larger
  products initially used OpenBLAS. Benchmarks cover contiguous and transposed
  GEMM shapes used by training adjoints. On the development system, the
  optimized batch-1024 training plan fell from 1.46 ms to 0.97 ms without
  regressing batches 1 or 32.
- Flattened ordinary CPU `sum` and `mean` reductions over contiguous consecutive
  axes into outer, reduced, and inner spans. On the development system, a
  `[1024, 32]` first-axis sum fell from 541 us to 10.3 us, and the prepared
  batch-1024 training step fell from 0.97 ms to 0.45 ms.
- Improved small CPU GEMMs with physically contiguous transposed left operands
  by traversing the shared dimension first and reusing contiguous right-hand
  rows while accumulating output tiles. In a bracketed A/B measurement on the
  development system, the `16 x 1024` by `1024 x 32` weight-gradient GEMM fell
  from 73.1 us to 60.2 us and its full training plan fell from 349.9 us to
  326.5 us. The prepared Python batch-1024 training step fell from 0.448 ms to
  0.398 ms without regressing smaller batches.
- Separated CPU JIT code generation into kernel setup, typed value emission,
  pointwise traversal, and reduction traversal.
- Added vectorization of eligible Float32 and Float64 arithmetic maps fused
  into global or trailing-axis `sum` and `mean`, including scalar tails.
- Added auto-detected F16C execution for eligible Float16 arithmetic maps with
  scalar, repeated contiguous-span, and fixed-value broadcasts, including maps
  fused into global or trailing-axis `sum` and `mean`. Mixed Float16/Float32
  fusion includes Float32 master-tensor casts and autocast Float32 reductions.
  Kernels preserve operation-by-operation Float16 rounding and accumulate
  reductions in Float32. Scalar execution is used when F16C is unavailable or
  explicitly disabled.
- Added CPU and CUDA fusion of eligible pointwise chains into global and
  partial-axis sums, including multi-axis reductions and `keepdims`, without
  materializing the mapped tensor.
- Optimized CUDA fused partial reductions for inputs contiguous or broadcast
  along a surviving output dimension. Each warp handles 32 output coordinates;
  each lane accumulates one output across remaining dimensions, eliminating
  strided warp reads, partial-buffer allocation, and the finish kernel.
  Other layouts use the general block-reduction path.
- Added physical-storage traversal for CUDA fused global reductions when all
  inputs share the same dense layout. This avoids logical-coordinate division
  and modulo for transposed inputs; mixed and broadcast layouts use logical
  traversal.
- Added lazy floating-point mean reductions to the C++ and generated Python
  APIs on CPU and CUDA, including multi-axis and `keepdims` forms, higher-order
  gradients, Float32 accumulation for Float16, empty-input NaN behavior, and
  fused pointwise-to-mean execution.
- Extended backend ABI 11 with fractional execution, its optional initial
  derivative, move-only operation setup, and optional fusion preparation.
  Runtime steps own prepared state and include it in Plan storage metrics.
  The backend interface supports accepting or declining typed fusion
  expressions, asynchronous same-device copies, and optional prepared
  execution recording.
- Unified backend startup and lazy plugin loading through one entry table and
  a shared initialization, validation, and installation path. CPU is built in;
  CUDA loads on first use.
- Centralized backend IDs, names, plugin libraries, index policy, host-memory
  role, and initialization in a private catalog. Backend implementations
  provide capabilities.
- Moved backend build declarations beside their implementations. CMake
  generates the private runtime catalog and uses registered targets for
  installation and staging.
- Moved platform declarations beside the Windows and Linux adapters. CMake
  selects one adapter and applies its core source, loader library, export,
  plugin filename, runtime-path, and staging traits.
- Switched the CPU memory pool to standard C++ aligned allocation on all
  platforms with 64-byte alignment. Benchmarks cover pooled reuse and uncached
  allocation.
- Separated public C++ headers into `<ormi/core/...>` for users and
  `<ormi/backend/...>` for backend implementations. Registry, dispatch,
  realization, and graph-building headers remain private. The backend interface
  exposes the fusion instruction schema; a public-only compile test checks the
  boundary.
- Replaced the public `Device` function-table accessor with a backend name.
  Runtime dispatch uses a private inline accessor to the cached backend pointer.
- Added inline temporary traversal storage for common small fusion graphs,
  with allocator fallback for larger graphs.
- Added CPU and CUDA benchmarks for construction, forward execution, backward
  execution, optimizer execution, and complete execution of the same fixed-shape
  two-layer training step. Optimizer results are applied to parameter buffers.
  CUDA benchmarks also measure one-time graph capture and steady-state replay.
- Added `ormi::core::Plan` for explicitly owned prepared execution. Plans keep
  graphs alive, resolve traversal, fusion, operation functions, and parameter
  updates once, and reuse output storage through a linear step list. CPU and
  CUDA tests cover moves, fusion, aliases, shared dependencies, lazy gradients,
  and optimizer updates. CUDA parameter updates are stream ordered, and warmed
  plans support CUDA Graph capture. Realized tensors are fixed input boundaries,
  Plan outputs must be lazy, and all update values are evaluated before any
  destination is modified.
- Unified ordinary realization and prepared Plan execution steps for fusion
  selection, input boundaries, aliasing, allocation, execution, and failure
  cleanup.
- Added a shared iterative postorder scheduler for ordinary realization and
  Plan preparation, removing graph-depth-dependent C++ call-stack growth.
  Immediate realization executes completed steps directly; Plans retain the
  ordered steps for replay.
- Resolved CPU and CUDA fused kernels during Plan preparation. Replays refresh
  tensor addresses and invoke retained callables directly, avoiding expression
  hashing and structural cache lookup. CUDA Plans retain launch geometry,
  argument slots, and reduction partial storage. CPU Plans retain one typed
  partial per worker for parallel fused reductions, eliminating heap allocations
  from replay.
- Replaced CPU worker-pool ownership locking with an atomic flag. Nested or
  competing submissions fall back to serial execution; worker handoff uses
  condition variables. A dedicated benchmark measures submission latency
  independently of generated kernel work.
- Added move-only replay creation from warmed Plans with `plan.record()`.
  Recording is an optional backend capability with record, replay, and release
  hooks. CUDA uses CUDA Graphs; CPU reports recording as unsupported. Replays
  own their prepared Plans, keeping tensors and buffers valid after the
  original `Plan` is destroyed.
- Exposed `Plan` and `Replay` in the generated Python API, including outputs,
  optional `(destination, value)` updates, prepared step and storage counts,
  execution, and recording.
- Added `Plan::storage_buffers()` and `Plan::storage_bytes()` for storage
  implied by the prepared step list. Benchmarks track both metrics for
  fixed-shape training and first-, third-, and fifth-order neural-network
  derivatives.

### Fixed

- Added fail-fast CPU sanitizer presets running the complete CPU suite in CI:
  MSVC AddressSanitizer across Windows x64 DLL boundaries, and Clang
  AddressSanitizer, LeakSanitizer, and UndefinedBehaviorSanitizer with frame
  pointers on Linux.
- Added isolated Float32 and Float64 CPU JIT checks for each CMake
  compiler-matrix row: compile uncached kernels, then load both artifacts in a
  second process without resolving an installed compiler.
- Fixed explicit GCC, Clang, and clang-cl presets to select matching C and C++
  compiler drivers for C dependencies.
- Added Windows backend dependency lookup through `PATH` when libraries are
  absent from registered package directories.

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
