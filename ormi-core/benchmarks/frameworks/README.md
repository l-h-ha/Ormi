# Framework benchmarks

ORMI, PyTorch, and TensorFlow receive copies of identical NumPy inputs.
The runner times setup, first use, and calls after warmup separately.

## Workloads

| Workload | Timed work |
| --- | --- |
| `pointwise` | `(input * scale + bias) * (input - bias) + input * 0.125`, with broadcasting |
| `reduction` | Square `input * row + column`, then take its mean over axes 0 and 2 |
| `dense` | Mean squared error of `input @ weight + bias` against `target`, and gradients for weight and bias |
| `pinn` | Sine network, first and second input derivatives, parameter gradients, and an SGD step |

`quick` uses one medium shape per workload. `full` adds small and large shapes.
The default seed is `20260810`.

The ORMI PINN uses `nn.Sequential`, `nn.Linear`, and `nn.Sin` with the shared
parameter arrays. It draws no additional initial weights.
`model.parameters()` lists updates in weight, bias, weight, bias order.

## Setup

The benchmark scripts require a [repository checkout](../../../README.md#from-source).
Install `ormi-core` using the [core installation instructions](../../README.md#install).
For a source build, use the [core Python build instructions](../../README.md#python).
Install frameworks in the same Python environment. PyTorch and TensorFlow are
optional. Follow their platform instructions:

- [Install PyTorch](https://pytorch.org/get-started/locally/)
- [Install TensorFlow](https://www.tensorflow.org/install)

From the repository root, enter `ormi-core`:

```bash
cd ormi-core
python -m benchmarks.frameworks.run --suite quick --json .benchmarks/framework-quick.json
```

Run the commands below from that directory. They use CPU by default.
The default run compares all three frameworks in both modes.
When using the workspace's `.venv`, prefix each Python command with `uv run`.
To measure only ORMI, use `--frameworks ormi`.

Select cases with `--workloads` and `--scales`. `--list` shows them without
importing frameworks. Skipped cases appear under `Unavailable cases` without
timings. Check that list before reporting framework comparisons.

`--label` identifies the ORMI build and machine. Include the
platform, compiler, and enabled performance libraries, such as
`windows-msvc-openblas-sleef` or `linux-gcc-builtin-math`. The JSON also records
the machine architecture, processor string, Python executable, framework
versions, UTC timestamp, selected cases, thread count, JIT cache path, and
configured ORMI JIT compiler.

## Modes

In `direct` mode, ORMI rebuilds and executes its lazy graph on every call.
PyTorch and TensorFlow use eager execution. In `optimized` mode, the runner
reuses an ORMI `Plan`, a `torch.compile(fullgraph=True)` callable, or a
TensorFlow function with XLA JIT compilation.

Without a C++ compiler, both ORMI CPU modes can use cached fused kernels and
fall back to precompiled kernels. To measure only precompiled kernels, use an
empty JIT cache and set `ORMI_CPU_JIT_CXX` to `ormi-compiler-does-not-exist`.
Record compiler availability and whether the cache started empty.

## Timing

The runner reports:

- `prepare`: tensor and callable or Plan creation.
- `first`: first execution after preparation.
- `median`, `p10`, and `p90`: median, 10th percentile, and 90th percentile of
  warmed per-call execution times. Each sample is one round's elapsed time
  divided by `--calls-per-round`.
- ORMI CPU JIT cache entries and bytes created by each workload.

Each run gets a new ORMI CPU JIT cache unless `--jit-cache PATH` is supplied.
The first direct run may compile CPU kernels.
Plans may compile during `prepare`. PyTorch and TensorFlow usually compile
during `first` in optimized mode.
The runner does not reset the CUDA PTX disk cache.

To measure first-use costs separately, run each mode in its own process:

```bash
python -m benchmarks.frameworks.run --suite full --tiers direct
python -m benchmarks.frameworks.run --suite full --tiers optimized
```

When run together, both tiers share process initialization and may share caches.
Their warmed timings remain useful. Their `prepare` and `first` timings are
not fully cold.

To count the CPU JIT cache entries added by one ORMI Plan, select one workload
and one scale:

```bash
python -m benchmarks.frameworks.run --suite quick --workloads pinn --frameworks ormi --tiers optimized
```

`--threads` sets ORMI, OpenMP, MKL, OpenBLAS, PyTorch, and TensorFlow CPU thread
counts. Timers wait for accelerator work to finish. Outputs, gradients, and
updated parameters are compared after first execution, before warmup.
The reference is PyTorch direct when available, then the first direct case,
then the first remaining case.
A single case provides no cross-framework comparison. JSON results include
output means and ranges, gradient means and L2 norms, or loss and parameter
means, depending on the workload.

Results apply to the recorded versions, processor or GPU, compiler, threads,
and shapes. They establish no performance claims for other machines.
Keep JSON files when comparing architectures.

## Tests

Workload, input, comparison, and option tests run without PyTorch or TensorFlow:

```bash
python -m unittest discover -s benchmarks/frameworks/tests -v
```
