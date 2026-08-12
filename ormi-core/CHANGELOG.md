# Changelog

## Unreleased

## 0.1.0a6 - 2026-08-12

### Changed

- The cold CPU JIT benchmark now gives every iteration a distinct artifact and
  releases its cache lease before cleanup, so repetitions cannot silently turn
  into cache hits. Compiler discovery is measured separately because production
  caches the selected toolchain once. Canonical compiler names and explicit
  `ORMI_CPU_JIT_CXX_ID` values no longer launch a redundant version probe;
  unusually named wrappers without an ID are still inspected. On the Windows
  development system, automatic clang-cl resolution fell from a 58.0 ms median
  to 2.14 ms. A corrected five-build run measured 552 ms total, including
  522.8 ms in compile-and-link, for the same 11-operation kernel. One clang
  time trace for a larger generated kernel attributed 567 ms to the compiler
  frontend and 25 ms to backend code generation, identifying generated-header
  parsing as the next cold-path target.
- Generated CPU JIT source now includes `<bit>` and `<cstdint>` only for
  Float16, `<cmath>` only for transcendental operations, and `<immintrin.h>`
  only when it emits x86 intrinsics. A small portable alignment helper replaces
  the unconditional `<memory>` dependency. In an eight-sample exact-source
  ABBA on Windows, median compile-and-link time fell from 654.910 to 182.658 ms
  with clang-cl, 673.805 to 235.072 ms with MSVC, 2002.463 to 425.312 ms with
  MinGW GCC, and 641.377 to 189.168 ms with Clang's GNU-style driver. The old
  and new sources produced identical exported-kernel instruction bytes with
  all four compilers, so warmed execution code is unchanged.
- The cold CPU JIT benchmark now covers five 11-operation dependency classes:
  header-free Float32 arithmetic, scalar transcendental math, vector math,
  vector reduction, and Float16 arithmetic. It reports the generated header
  categories alongside loop planning, emission, compilation, cache, loading,
  source size, and cache status. Across two seven-repetition clang-cl passes,
  with the second launched in reverse case order, compile-and-link medians were
  521.728-592.616 ms for arithmetic, 691.430-792.502 ms for scalar math,
  776.261-919.000 ms for reduction, 841.516-1041.340 ms for vector math, and
  906.378-1032.970 ms for Float16. Every measurement was an uncached build.
  The stable ordering identifies intrinsic and Float16 support as the largest
  remaining generated-source costs without attributing the full difference to
  header parsing alone.
- Exact-source clang-cl controls isolated generated-header parsing from kernel
  structure. Against a 188.430 ms header-free median, forced `<cmath>` added
  62.010 ms, forced `<immintrin.h>` added 193.252 ms, and the Float16 set of
  `<bit>`, `<cstdint>`, and `<immintrin.h>` added 249.650 ms. A separate
  interleaved run measured `<cmath>` plus `<immintrin.h>` at 290.346 ms above
  its header-free control. A reusable union precompiled header reduced the
  corresponding 457.546 ms compile to 186.540 ms without changing the 226
  emitted kernel instructions, but building its 7,368,968-byte artifact cost
  533.038 ms. The first compile would therefore regress from 457.546 to
  719.578 ms and reuse only breaks even around the second header-heavy kernel.
  Because Plan preparation already batches its kernels into one translation
  unit and persistent artifacts avoid recompilation, compiler-specific PCH
  caching was not added without broader workload evidence.
- A compiler-startup benchmark now measures the process floor independently of
  source generation and linking. Through ORMI's process runner, startup measured
  65.908 ms for clang-cl and 50.552 ms for MSVC on the Windows development
  system. A balanced exact-source clang-cl split measured 89.934 ms for object
  generation, 99.740 ms for Microsoft linking, and 240.124 ms for the existing
  one-shot compile-and-link path. `lld-link` was 4.07% slower than Microsoft
  linking in a separate balanced comparison. `/NOENTRY` appeared 11.42% faster
  for arithmetic but failed valid scalar-math kernels with unresolved `expf`
  and `sinf`, so neither linker change was adopted.
- Framework benchmark JSON now records CPU JIT entries, generated-source bytes,
  and library bytes added by each workload. Isolated current-package runs found
  one kernel for pointwise, one for reduction, two for dense gradients, and
  seven for the second-order PINN; cold plan preparation measured 0.717, 0.764,
  1.022, and 3.852 seconds respectively. This identifies repeated compiler
  invocations, rather than graph planning or code emission, as the next cold
  preparation problem.
- Plan construction now finishes scheduling before it prepares backend work.
  Backends may prepare fused steps together through one optional function, while
  immediate realization and backends without that function retain the existing
  single-step path. CPU preparation emits all uncached expressions into one
  translation unit, loads their distinct symbols from one shared library, and
  publishes each expression in the existing memory cache. On the development
  system, five fresh-process medium PINN runs had a cold-preparation median of
  0.825 seconds, down from the preceding 3.852-second isolated run, and produced
  one artifact instead of seven. Five medium dense runs had a 0.834-second
  median versus the preceding 1.022-second run and produced one artifact instead
  of two. Reusing the PINN artifact from a second process reduced preparation
  from 742.4 to 5.3 milliseconds without invoking the compiler. A cached
  seven-expression batch lookup measured 1.274 microseconds total, paid only
  while constructing the Plan.
- CPU JIT reductions now plan contiguous innermost reduced axes explicitly.
  Float32 arithmetic uses a packed AVX2 branch with a portable generated-loop
  fallback; Float64 and non-AVX2 execution retain the existing path. On the
  development system, the seeded `[256, 64, 64] -> [64]` two-axis mean fell
  from 0.987 to 0.242 milliseconds in direct mode and from 1.071 to 0.220
  milliseconds with a prepared plan. Two repeated prepared runs measured
  PyTorch at a 0.340-millisecond geometric median and TensorFlow at 1.716
  milliseconds on the same inputs. The full one-thread Float32 corpus passed
  numerical validation and ORMI had the lowest warmed median in all 12 direct
  and all 12 prepared cases on this machine.
- A tracked framework benchmark corpus now runs the same seeded pointwise,
  reduction, dense-gradient, and second-order PINN workloads through ORMI,
  PyTorch, and TensorFlow. It separates preparation, first execution, and
  warmed timing; fixes framework thread counts; compares complete numerical
  results before timing; reports unavailable modes explicitly; and writes
  labeled machine-readable JSON with build, platform, processor, Python,
  timestamp, case-selection, and explicitly configured CPU JIT compiler
  metadata for multi-machine evidence. A current one-thread Float32 checkpoint
  using 20 rounds, 10 calls per round, and five warmups produced 72 validated
  result rows. ORMI had the lowest warmed median in all 12 direct and all 12
  optimized cases; its geometric-mean speedups were 4.88x and 8.71x over
  PyTorch, and 20.16x and 25.05x over TensorFlow, for direct and optimized
  execution respectively. Against the preceding saved checkpoint, all three
  frameworks slowed by similar factors, separating machine-wide run variance
  from an ORMI-specific regression. A second ORMI process reused every one of
  the 12 prepared artifacts, created no new entries, and reduced geometric-mean
  Plan preparation from 402.1 to 3.27 milliseconds, a 123.2x improvement.
- Float32 GEMM dispatch now checks whether the linked OpenBLAS has
  architecture-specific kernels. Compatible large products use ORMI's
  runtime-guarded AVX2/FMA kernels when OpenBLAS is absent or reports its
  generic core; tuned OpenBLAS builds retain the existing crossover policy,
  and Float64 behavior is unchanged. Across contiguous, transposed-left,
  skinny, square, and 512-cubed products on the development system, ORMI's
  kernels sustained 69-105 GFLOP/s versus 3.6-3.8 GFLOP/s from the bundled
  generic OpenBLAS. The cross-framework large dense-gradient median fell from
  2.718 to 0.194 milliseconds in direct mode and from 2.475 to 0.162
  milliseconds with a prepared plan.
- CPU matmul now reads sequentially as tensor and batch traversal, one-product
  kernel selection, then portable or architecture-specific execution. The
  typed policy and portable loops remain compiler-visible internal headers;
  benchmark experiments showed that hiding either behind runtime `void*`
  dispatch or aliased descriptors regressed small products and GEMV.
- Plan preparation benchmarks now warm CPU JIT setup outside the timed region
  and cover alias-chain depths from 128 through 4,096. The corrected Release
  benchmark measured a one-element replay at 35.7 ns and a 31-step fifth-order
  derivative replay at 1.22 microseconds. Reserving planner containers and its
  traversal stack did not improve large graphs consistently, so the sequential
  Plan implementation and scheduler API remain unchanged.
- Runtime lowering is now separate from step execution. The immediate path reads
  `Tensor::realize -> schedule::run -> lower -> Step::run`: the scheduler lowers
  the requested root once, walks the dependencies left outside that step, and
  executes steps in postorder. `Step` owns only allocation, aliasing, execution,
  and failure cleanup. Immediate realization uses a tensor's realized buffer as
  its completion state instead of also maintaining an `unordered_set`; Plan
  construction keeps its set because its steps have not run yet. Group discovery
  for immediate realization now starts at unresolved root-step inputs and stops
  at realized tensors. Fully fused roots therefore avoid a second full graph
  walk, while pointwise siblings below direct operations remain groupable. In
  nine-repetition local Release measurements, warm fused-root medians at depths
  1, 4, 16, and 64 changed from 1.69, 2.56, 7.59, and 28.77 microseconds to
  1.37, 2.03, 5.15, and 22.57 microseconds; a longer depth-64 run measured
  16.54 microseconds. When lowering leaves no unresolved input, immediate
  realization now runs the ready root Step directly instead of constructing an
  empty grouping table and a 16-frame traversal stack. In a bracketed
  nine-repetition Release A/B/A, the control measured 1.482, 2.283, 6.090, and
  19.275 microseconds at depths 1, 4, 16, and 64. The two direct-root runs
  measured 1.215-1.263, 2.024-2.147, 5.790-5.794, and 15.627-16.448
  microseconds. Scheduler benchmarks now use adaptive duration rather than a
  short fixed iteration count.
- Fusion expression construction and compiled-kernel cache identity now have
  separate `expr` and `key` units. Expression code owns graph eligibility,
  traversal, broadcasting, reductions, and auxiliary outputs; `Key` owns only
  structural copying, matching, and hashing. CPU and CUDA compilers include the
  key directly, and key tests and benchmarks no longer share the expression
  test file. In isolated 11-repetition Release measurements at 256 nodes, key
  construction measured 2.36 microseconds versus 2.46 before the split, while
  expression construction measured 25.36 versus 24.97 microseconds. Build-only
  benchmarks now live in `bench_build.cpp` and cover unary chains, shared and
  distinct inputs, balanced trees, and fast rejection. A nine-repetition
  Release run measured 256-operation unary, shared-input, distinct-input, and
  balanced graphs at 21.4, 22.9, 35.6, and 34.1 microseconds. Similar deep and
  balanced graphs cost about the same when their node and input counts match,
  so traversal depth is not the scaling problem. An adjacent A/B measured the
  256-operation shared chain at 27.2 microseconds with the local PMR arena and
  41.1 microseconds with plain containers. Larger local arenas did not improve
  all graph shapes consistently, so the existing 1 KiB capacity remains.
- The Windows platform adapter is now separated by responsibility into process,
  toolchain, dynamic-library, filesystem, CPU-query, and UTF conversion units.
  `platform.h` remains the single runtime interface, and each platform sidecar
  now declares its complete source list. No platform call or runtime behavior
  changed.
- The empty top-level `ormi` placeholder project has been removed from the
  workspace. The repository now contains only the implemented `ormi-core`
  package and its optional `ormi-cuda` plugin.
- CPU JIT compilation now has a narrower sequential coordinator. Target and
  vector-math selection live with target policy, compiler command construction
  and invocation live with compiler discovery, and work estimation lives with
  loop planning. `compile.cpp` owns memory-cache lookup, loop/code preparation,
  and publication; `artifact.cpp` owns persistent-cache lookup, compilation,
  library loading, and repair. Generated kernels, cache identities, and
  steady-state execution are unchanged.
- CPU JIT benchmarks now separate cold compilation, memory and disk lookup,
  library loading, execution preparation, source emission, prepared execution,
  and cached execution. A fresh-cache compile measured 542.6 milliseconds,
  while a disk lookup plus library load measured 0.579 milliseconds, confirming
  that compiler startup dominates first use. Cached compilation now checks a
  bounded four-entry per-thread front cache before the immutable shared table.
  In seven-repetition Release measurements, eight-thread memory lookup fell
  from 249.3 to 35.3 nanoseconds and cached execution from 242.1 to 60.2
  nanoseconds; single-thread cached execution changed from 88.9 to 92.2
  nanoseconds. Alternating two kernels takes 60.0 nanoseconds on one thread and
  22.4 nanoseconds at eight threads without specializing for either graph.
- Device identity is now catalog-driven. `Device::cpu()` resolves the single
  declared host backend, while optional devices use `Device::from(name, index)`
  or a device string such as `"cuda:0"`. The public `name`, `indexed`, and
  `is_host` properties replace per-backend predicates, and the CUDA-specific
  constructor and Python helper have been removed. Backend implementations now
  report operational failures through `BackendError` instead of requiring a
  core exception type for every backend. Device creation still caches the
  resolved backend pointer, so steady-state dispatch is unchanged.
- Backend ABI 14 groups the raw function table into mandatory `memory` and
  optional `ops` capabilities. Backend installation validates only the device
  memory mechanics, then fills omitted ordinary operations and fractional
  execution with direct throwing function pointers once. Fusion and recording
  remain explicitly queryable capabilities for planning. CPU and CUDA keep
  cached pointer dispatch; a seven-repetition Release check measured CPU
  allocator lookup at 2.78 ns and synchronization dispatch at 2.82 ns.
- Memory transfers now have explicit ownership. Same-backend copies use the
  destination backend, host-to-backend copies use the destination, and
  backend-to-host copies use the source. Transfers between different
  accelerators first offer the destination an optional direct-copy hook, then
  stage through the single host backend when it declines. CUDA implements
  `within`, `from_host`, and `to_host` routes without CPU/CUDA identity checks.
  In seven-repetition Release measurements, the complete CPU buffer-copy path
  took 12.8 ns for 64 bytes and 30.5 ns for 4 KiB, versus 5.61 ns and 23.7 ns
  for direct `memcpy`; both paths were within the 36-38 microsecond run-to-run
  range for 1 MiB.
- Planning now groups compatible single-parent pointwise siblings without
  naming particular operations in the core scheduler. The CPU JIT privately
  recognizes sine/cosine pairs and retains its paired vector-math lowering;
  backends without multi-output fusion fall back to independent steps. In a
  five-repetition Release check of the maintained 23-step training sequence,
  the grouped batch-1,024 mean was 118.4 microseconds versus 141.4
  microseconds ungrouped. Smaller batches remained within run-to-run noise.
  Group discovery now keeps the first output for each parent inline and emits
  result vectors only when compatible siblings actually exist, eliminating a
  second pass and one-element vector allocations along unary chains. In a
  nine-repetition Release A/B at 256 operations, an ungrouped unary chain fell
  from 90.2 to 51.1 microseconds; 64 groups of four fell from 88.5 to 74.8
  microseconds; and a 256-output fanout retained the same 39.8-microsecond CPU
  median. Dedicated grouping benchmarks cover binary and unary chains, fanout,
  and many independent groups.
- CUDA pointwise fusion now emits and launches generic auxiliary outputs using
  one typed output argument per tensor. Prepared calls retain one stable output
  pointer array, including Float16, Float32, Float64, empty-tensor, and recorded
  CUDA graph paths; reductions remain deliberately single-output. A generic
  shared-parent `neg`/`exp` benchmark now uses one plan step instead of two.
  Seven-repetition median completion time fell from 16.3 to 9.91 microseconds
  at one element, 16.2 to 9.44 microseconds at 1,024 elements, and 136.5 to
  113.5 microseconds at 1,048,576 elements. The independent-launch control
  remained within 1.5% of its prior measurements.
- CUDA fusion benchmarks now separate direct execution, prepared execution,
  fresh-output allocation, reusable Plan execution, CUDA graph replay, and
  ordinary realization. Synchronized cases use manual launch-to-completion
  timing after a duration-based warmup, so graph construction and output
  cleanup cannot distort the measured interval. In an 11-repetition Windows
  Release check, reusable Plan medians were 10.57 microseconds at one element
  and 10.44 microseconds at 1,024 elements, versus 15.17 and 18.85 microseconds
  for ordinary one-shot realization. Prepared execution and Plan execution
  converged at 10.44 microseconds for 1,024 elements, showing no measurable
  reusable core-dispatch penalty. New idle-stream host-submission controls
  isolate expression construction, backend preparation, output allocation,
  and launch submission. They exposed a redundant `cudaSetDevice` in transient
  fusion execution; removing it reduced direct one-shot completion from 12.08
  to 9.68 microseconds at one element and from 11.86 to 9.55 microseconds at
  1,024 elements, while prepared execution remained within run-to-run noise.
- CUDA training benchmarks now give ordinary Plan execution and recorded graph
  replay the same 50-millisecond completion-based warmup before timing. For the
  maintained 23-step training graph, Windows Release median completion time
  fell from 198.2 to 45.9 microseconds at batch 1 and from 189.4 to 72.2
  microseconds at batch 1,024. An isolated 15-repetition batch-32 confirmation
  measured 211.4 versus 51.2 microseconds. Recording remains explicit because
  capture has a one-time cost and single-kernel Plans do not show this benefit.
- CPU pointwise, cast, SGD, finite-check, and reduction loops now read tensor
  element counts once before iteration. This removes a library call from every
  loop condition and allows MSVC to vectorize contiguous Float32, Float64, and
  Int32 arithmetic. The affected saved-gradient multiply fell from 167.75 to
  5.50 microseconds at 32,768 elements.
- CPU JIT reductions now retain expanded inputs when deciding whether direct
  stride planning is required. Large global reductions over broadcast views can
  therefore split their largest reduced dimension without changing the flat
  pointwise broadcast path.
- CPU matmul execution now derives fixed dimensions directly from its tensors
  instead of rebuilding allocating shape metadata on every call. Dedicated
  row-wise and transposed-left matrix-vector paths also expose contiguous loops
  to the compiler. On x86-64, larger Float32 products select AVX2 kernels at
  runtime while retaining the portable implementation for other processors.
  The AVX2 kernels keep up to 32 output columns in registers across the shared
  dimension and reduce contiguous matrix-vector rows with four independent
  accumulators. Transposed-left products use a four-row by 16-column tile so
  each right-hand vector feeds four output rows. In a bracketed MSVC benchmark,
  the 16 by 1,024 by 32 training-gradient median fell from 21.56 to 16.47
  microseconds, with its CPU-time median improving by 25.9%.
  In the packaged MSVC, SLEEF, and OpenBLAS configuration, the prepared
  23-operation training plan fell from 237.65 to 123.04 microseconds at batch
  1,024 while batches 1 and 32 remained near 4.12 and 9.40 microseconds.
- Float32 AVX2 matmul kernels now use fused multiply-add instructions when the
  processor reports both AVX2 and FMA support, with the portable path retained
  otherwise. In a controlled MSVC object-level comparison, the 1,024 by 16 by
  32 forward-product median fell from 30.34 to 20.44 microseconds and its
  16 by 1,024 by 32 transposed-left counterpart fell from 27.03 to 22.42.
- Contiguous Float32 reductions across a leading or middle block of axes now
  use a runtime-guarded AVX2 row-sum kernel for larger inputs. It keeps output
  columns in registers across the reduced rows and retains the portable path
  for small inputs and other processors. The 1,024 by 32 first-axis sum fell
  from 10.07 to 1.12 microseconds, and the repeated 23-operation training-plan
  mean fell from 130.46 to 121.02 microseconds in the same packaged build.
- SLEEF-enabled Float32 sine and cosine now use its approximately 3.5-ULP AVX2
  variants; exponential and logarithm retain their 1-ULP variants. Pointwise
  accuracy remains bounded within 2e-6 over ORMI's wide-range tests, including
  special values. In a paired build comparison, the batch-1,024 training-plan
  mean fell from 151.21 to 111.87 microseconds, with the sine step falling from
  32.89 to 22.56 microseconds and the cosine chain from 37.35 to 25.70.
- CPU JIT kernels now call SLEEF's AVX2 functions directly when the core and
  runtime compiler use compatible vector ABIs. Mixed Windows toolchains retain
  the buffered C-compatible callback path, and the selected ABI is part of the
  JIT cache identity. At batch 1,024, the fused sine step fell from 21.48 to
  17.93 microseconds, the fused cosine chain from 22.53 to 18.04, and the full
  23-step training-plan median from 126.68 to 92.43 microseconds.
- Paired AVX2 sine and cosine callbacks now use `__vectorcall` with MSVC and
  clang-cl. Inputs and both results stay in vector registers instead of passing
  through aligned stack temporaries. In a bracketed paired-output benchmark,
  the 32-, 1,024-, and 32,768-element medians improved from 76.1 to 64.5
  nanoseconds, 1.334 to 1.000 microseconds, and 38.21 to 34.00 microseconds.
- CPU parallel work estimates now count paired SLEEF sine and cosine nodes as
  one vector-math call while retaining separate costs for scalar fallback and
  nested trigonometric operations. A six-to-one participant sweep selected
  three participants for the 32,768-element pair; compared with six, its
  prepared-kernel median fell from 48.15 to 30.16 microseconds and the matching
  training-plan step fell from 59.66 to 34.10 microseconds.
- CPU workers now receive balanced whole-unit ranges when work does not divide
  evenly, and each selected slot has a dedicated notification. Unused workers
  remain asleep while active slots retain stable worker ownership across runs.
  The 32,768-element paired sine/cosine median fell from 30.16 to 18.77
  microseconds, and its matching training-plan step fell from 34.10 to 17.05
  microseconds.
- Training-sized Float32 AVX2 matrix products now use two persistent-worker row
  partitions while small products remain serial. In a bracketed comparison,
  the 1,024 by 16 by 32 forward product fell from 20.37 to 9.35 microseconds
  and the 16 by 1,024 by 32 transposed-left product fell from 20.35 to 9.38
  microseconds. Direct OpenBLAS crossover benchmarks retain the internal AVX2
  path for these shapes.
- CPU JIT target context now refers to the static selected-target name instead
  of returning a string view into a temporary value.
- Added lazy left-sided Caputo differentiation on uniform grids for orders
  between zero and two, excluding one. Orders below one use the L1 scheme;
  orders above one use the L2 scheme and require the first derivative at the
  history origin.
- Caputo runs on CPU and CUDA for Float32 and Float64, supports arbitrary
  strided axes, differentiates both the sampled history and initial derivative,
  and keeps every discrete transpose lazy for higher-order autograd. Tests cover
  analytic histories, convergence, exact adjoints, CPU/CUDA parity, and Python.
- CUDA L2 forward evaluation computes reusable second differences once for
  histories of 64 samples or more, reducing repeated global-memory reads while
  preserving the direct single-kernel path for shorter histories.
- Caputo nodes share one immutable coefficient set with their adjoints. CUDA
  plans upload those coefficients once and retain the L2 difference workspace,
  so repeated plan runs and recorded graphs perform no fractional setup
  allocation or coefficient transfer.
- Exposed Caputo as `tensor.caputo(...)` and `ormi_core.ops.caputo(...)` in the
  generated Python API. Forward and adjoint benchmarks cover both schemes
  through 8,192-sample histories, including CUDA plan preparation and
  recording.
- CPU and CUDA pointwise fusion now support contiguous broadcast inputs,
  including scalar, rank-expanded, and multi-axis broadcasting, as well as
  transposed and expanded views.
- CPU and CUDA pointwise fusion now supports Float16 while preserving
  operation-by-operation half-precision rounding.
- CPU and CUDA fusion now keeps floating-point casts inside generated kernels,
  including Float16 autocast chains with Float32 operations and accumulation.
- CPU JIT kernels now expose 64-byte alignment and output non-aliasing to the
  compiler, request loop vectorization for eligible contiguous Float32 and
  Float64 arithmetic chains with linear, scalar, or repeated contiguous
  broadcast spans, and select baseline or AVX2 code for the host.
- CPU JIT cache entries now include the CPU target, while compiler discovery is
  still deferred until an uncached kernel must be built.
- CUDA fusion now caches generated source and PTX across processes. Cache keys
  include the generated source, GPU compute capability, NVRTC version, and a
  CUDA JIT ABI; per-entry locks serialize concurrent misses, publication is
  staged, and least-recently-used entries keep the default cache within 256 MiB.
- CUDA fusion now leaves process-lifetime compiled modules to CUDA context
  teardown instead of calling `cuModuleUnload` from static destruction. This
  prevents a driver access violation after instrumented reduction workloads.
- CPU JIT execution now reads compiled kernels from an immutable published
  registry without taking the compilation mutex, and keeps input pointers on
  the stack for the common case of eight or fewer inputs.
- CPU JIT compilation, caching, and library loading are now separate from
  prepared invocation and reduction execution. The kernel entry point shows
  the complete compile-then-run sequence without mixing those mechanisms.
- CUDA fusion now follows the same sequence: NVRTC compilation, PTX loading,
  and kernel publication are separate from prepared launch state and reduction
  execution. Its kernel entry point only selects the device, resolves the
  compiled kernel, and prepares or runs the call. Lifecycle tests cover unrun
  prepared reductions, direct/prepared parity, and concurrent kernel lookup.
- CPU and CUDA compiled-kernel registries now share one fusion expression
  identity. CUDA cache matching includes the output shape used by its tiling
  and code generation, so shape-dependent kernels cannot be reused incorrectly.
- Runtime dispatch now accesses a device's backend through a direct internal
  function instead of the state-free `DeviceAccess` wrapper.
- Large flat and contiguous-broadcast pointwise JIT kernels now execute
  disjoint output ranges through a persistent CPU worker pool. Broadcast work
  is partitioned only at complete contiguous-span boundaries. Transcendental
  expressions contribute their measured cost to the parallel decision, and
  each submission limits active participants to the amount of available work.
  The maximum worker count follows the physical cores detected on Windows and
  Linux, `ORMI_CPU_THREADS` provides an explicit override, and smaller kernels
  remain on the caller thread. A 65,536-element arithmetic control remained at
  6.01 microseconds while the batch-1,024 training plan gained the parallel
  transcendental path.
- Large CPU JIT `sum` and `mean` reductions with independent outputs now use
  the same worker pool. Trailing-axis reductions partition output values,
  first-axis reductions partition complete 32-column tiles, and general
  direct-stride reductions partition outer-dimension slabs. The input element
  count controls whether parallel execution is worthwhile. Contiguous global
  reductions partition the flat input, while transposed and expanded global
  reductions choose the largest reduced dimension for worker partitioning
  independently from traversal order. Direct loops place the lowest worst-case
  physical input stride innermost without reconstructing flat coordinates.
  Eligible Float32 and Float64 direct loops whose innermost input strides are
  zero or one use independent lane accumulators with a scalar tail. Conflicting
  two-dimensional layouts use 16-by-16 cache tiles with worker ranges aligned
  to complete tiles. Higher-rank conflicting layouts tile the worker split
  dimension with one complementary physical-stride dimension while traversing
  all remaining dimensions directly. All paths combine one typed partial per
  worker; Float16 uses Float32 partials, `mean` division is applied once after
  the combine, and smaller reductions remain on the caller thread.
- CPU JIT pointwise fusion now traverses transposed and expanded inputs with
  aligned arbitrary-rank stride tables instead of per-element division and
  modulo. Large two-dimensional transposes use 32-by-32 tiles to retain input
  cache lines across adjacent output rows.
- CPU JIT transcendental maps can now use ranged execution independently from
  SIMD eligibility. Published builds also use SLEEF 3.9.0 for eight-lane AVX2
  Float32 `exp`, `log`, `sin`, and `cos`, passed to generated kernels through a
  compiler-neutral function table. Scalar tails and non-AVX2 systems retain
  standard-library math. On the development system, the prepared 32,768-element
  `cos(x) * gradient` kernel measured 47.5 us between scalar runs of 85.1 us
  and 82.1 us; the complete batch-1024 training plan measured 358 us between
  scalar runs of 450 us and 426 us.
- CPU JIT reductions fused from transposed or expanded maps now traverse
  logical dimensions directly and apply each input's physical strides without
  reconstructing flat coordinates. Two-dimensional single-axis reductions tile
  32 adjacent outputs whenever every input is contiguous or broadcast along
  that output axis. This covers both ordinary first-axis reductions and
  last-axis reductions over transposed inputs while keeping generated inner
  loops contiguous. Higher-rank reductions use 16-value tiles along the first
  surviving output dimension and retain ordinary loops for every other output
  and reduced dimension.
- Sequential CPU JIT Float16 and Float32 reductions now combine 32-value
  Float32 blocks in a Float64 running total. This bounds rounding growth
  without promoting every mapped value.
- Float32 and F16C SIMD reductions now process 32 vector steps per Float32
  lane block before combining lanes in Float64. Vectorized strided reductions
  retain their Float32 lanes and use a Float64 scalar combine. Million-element
  contiguous, transposed, and packed fused reductions are covered by a
  0.5-parts-per-million error limit, and their benchmarks report measured
  relative error alongside throughput.
- Small CPU matrix multiplications with contiguous right-hand rows and outputs
  now use a portable row kernel for up to one million scalar products. Larger
  products initially retained OpenBLAS. Benchmarks cover the
  contiguous and transposed GEMM shapes used by the training adjoints; the
  optimized batch-1024 training plan fell from 1.46 ms to 0.97 ms on the
  development system without regressing batches 1 or 32.
- Ordinary CPU `sum` and `mean` now flatten contiguous consecutive-axis
  reductions into outer, reduced, and inner spans instead of reconstructing
  coordinates with integer division for every element. On the development
  system, a `[1024, 32]` first-axis sum fell from 541 us to 10.3 us, and the
  prepared batch-1024 training step fell from 0.97 ms to 0.45 ms.
- Small CPU GEMMs with a physically contiguous transposed left operand now
  traverse the shared dimension first, reusing each contiguous right-hand row
  while accumulating the small output tile. In a bracketed A/B measurement on
  the development system, the `16 x 1024` by `1024 x 32` weight-gradient GEMM
  fell from 73.1 us to 60.2 us and its full training plan fell from 349.9 us to
  326.5 us. The prepared Python batch-1024 training step fell from 0.448 ms to
  0.398 ms without regressing the smaller batches.
- CPU JIT code generation is now separated into kernel setup, typed value
  emission, pointwise traversal, and reduction traversal.
- CPU JIT kernels now vectorize eligible Float32 and Float64 arithmetic maps
  fused into global or trailing-axis `sum` and `mean` reductions, including
  reduction lengths that require a scalar tail.
- CPU JIT kernels now use auto-detected F16C instructions for eligible Float16
  arithmetic maps, including scalar, repeated contiguous-span, and fixed-value
  broadcasts, and maps fused into global or trailing-axis `sum` and `mean`
  reductions. Mixed Float16/Float32 arithmetic fusion uses the same packed path,
  including Float32 master-tensor casts and autocast Float32 reductions. Kernels
  preserve operation-by-operation Float16 rounding, accumulate reductions in
  Float32, and use the scalar implementation when F16C is unavailable or
  explicitly disabled.
- CPU and CUDA now fuse eligible pointwise chains directly into global and
  partial-axis sums, including multi-axis reductions and `keepdims`, without
  materializing the mapped tensor.
- CUDA fused partial reductions now assign each warp to 32 surviving output
  coordinates when every input is contiguous or broadcast along that output
  dimension. Each lane accumulates one output across arbitrary remaining
  dimensions, avoiding strided warp reads, partial-buffer allocation, and the
  finish kernel. Other layouts retain the general block-reduction path.
- CUDA fused global reductions now traverse physical storage directly when
  every input has the same dense layout. Global `sum` and `mean` therefore
  avoid logical-coordinate division and modulo for transposed inputs, while
  mixed and broadcast layouts retain logical traversal.
- Added lazy floating-point mean reductions to the C++ and generated Python
  APIs on CPU and CUDA, including multi-axis and `keepdims` forms, higher-order
  gradients, Float32 accumulation for Float16, empty-input NaN behavior, and
  fused pointwise-to-mean execution.
- Backend ABI 11 adds fractional execution, its optional initial derivative,
  move-only operation setup, and optional fusion preparation to the backend
  function table. Prepared state is owned by the runtime step and contributes
  to plan storage metrics.
  Backends can still accept or decline typed fusion expressions, enqueue
  same-device copies without synchronizing, and optionally provide prepared
  execution recording.
- Backend startup and lazy plugin loading now use one entry table and the same
  initialization, validation, and installation path. CPU remains built in,
  while CUDA remains unresolved until first use.
- Backend IDs, names, plugin libraries, index policy, host-memory role, and
  initialization now come from one private catalog row. Backend implementations
  provide capabilities only, so plugin metadata cannot disagree with the
  registry's identity.
- Backend build declarations now live beside their implementations and generate
  the private runtime catalog during CMake configuration. The root build
  discovers declarations and iterates registered targets for installation and
  staging, so adding a backend requires no registry or central build branch.
- Platform declarations now live beside the Windows and Linux adapters. CMake
  selects exactly one adapter and applies its core source, loader library,
  export, plugin filename, runtime-path, and staging traits without platform
  branches in the root build or optional backend targets.
- The CPU memory pool now uses standard C++ aligned allocation on every
  platform while retaining 64-byte alignment. Benchmarks cover both pooled
  reuse and uncached allocation paths.
- C++ headers now separate the user API under `<ormi/core/...>` from the
  backend interface under `<ormi/backend/...>`. Registry, dispatch,
  realization, and graph-building headers remain private, while the backend
  interface exposes the complete fusion instruction schema needed to implement
  fused execution. A public-only compile test enforces this boundary.
- `Device` now exposes its backend name instead of its function table. Runtime
  dispatch reads the same cached pointer through a private inline accessor, so
  the public boundary adds no registry lookup to operation execution.
- Fusion planning now keeps its temporary traversal state in inline storage for
  common small graphs, while larger graphs retain the normal allocator
  fallback.
- CPU and CUDA benchmarks now measure construction, forward execution,
  backward execution, optimizer execution, and complete execution for the same
  fixed-shape two-layer training step, including applying each optimizer result
  to its parameter buffer. The CUDA benchmark also measures one-time graph
  capture and steady-state replay without exposing graph machinery through the
  core API.
- `ormi::core::Plan` now gives C++ callers explicit ownership of prepared
  execution. A plan keeps its graph alive, resolves traversal, fusion,
  operation functions, and parameter updates once, then reuses output storage
  while executing a linear step list. CPU and CUDA tests cover moves, fusion,
  aliases, shared dependencies, lazy gradients, and optimizer updates. CUDA
  parameter updates remain stream ordered, and warmed plans can be captured
  directly as CUDA Graphs. Realized tensors are fixed input boundaries, plan
  outputs must be lazy, and all update values are evaluated before any
  destination is modified.
- Ordinary realization and prepared plans now use the same internal execution
  step for fusion selection, input boundaries, aliasing, allocation, execution,
  and failure cleanup.
- Ordinary realization and plan preparation now traverse dependencies through
  one iterative postorder scheduler. Dependency traversal no longer grows the
  C++ call stack with graph depth, while immediate realization still runs each
  completed step directly and plans retain the same ordered steps for replay.
- CPU and CUDA plans now resolve fused kernels once during preparation. Replays
  refresh tensor addresses and invoke the retained callable directly, avoiding
  expression hashing and structural cache lookup. CUDA plans also retain launch
  geometry, argument slots, and reduction partial storage. CPU plans retain one
  typed partial per worker for parallel fused reductions, removing heap
  allocation from replay. Ordinary one-shot realization keeps the direct
  execution path.
- The CPU executor now claims its shared worker pool with an atomic flag.
  Nested or competing submissions take the existing serial fallback without
  recursively locking the pool, while the worker handoff remains the portable
  condition-variable implementation. A focused benchmark tracks submission
  latency independently from generated kernel work.
- A warmed `Plan` can now create a move-only replay with `plan.record()`.
  Recording is an optional backend capability with three hooks: record, replay,
  and release. CUDA implements those hooks with CUDA Graphs, while CPU reports
  that recording is unsupported. The replay owns the prepared plan, so its
  tensors and buffers remain valid after the original `Plan` is destroyed.
- The generated Python API now exposes `Plan` and `Replay`. Python callers
  provide plan outputs and optional `(destination, value)` update pairs, can
  inspect prepared step and storage counts, and can execute or record the same
  prepared runtime path used by C++ callers.
- `Plan::storage_buffers()` and `Plan::storage_bytes()` report the stable
  storage implied by the prepared step list. Benchmarks now track these values
  for fixed-shape training and first-, third-, and fifth-order neural-network
  derivatives.

### Fixed

- Added fail-fast CPU sanitizer presets. Windows x64 uses MSVC
  AddressSanitizer across ORMI's DLL boundary; Linux uses Clang
  AddressSanitizer, LeakSanitizer, and UndefinedBehaviorSanitizer with frame
  pointers. Both configurations run the complete CPU suite in CI.
- Each CMake compiler-matrix row now verifies its exact compiler in an isolated
  process by compiling uncached Float32 and Float64 CPU JIT kernels, then loading
  both artifacts in a second process without resolving an installed compiler.
- Explicit GCC, Clang, and clang-cl CMake presets now select matching C and C++
  compiler drivers, preventing C dependencies from inheriting incompatible
  command-line syntax.
- Lazy Windows backend loading can resolve toolkit dependencies from `PATH`
  when they are not present in ORMI's registered package directories.

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
