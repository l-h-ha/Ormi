# Training

Install `ormi` using the [package guide](../README.md#install).
Run `uv` commands from the repository root after [source setup](../../README.md#from-source).

```python
import numpy as np
import ormi as om
from ormi import nn

values = np.random.default_rng(7).normal(size=(100, 3)).astype(np.float32)
labels = np.argmax(values, axis=-1).astype(np.int32)
model = nn.Sequential([nn.Linear(3, 8, seed=7), nn.Tanh(), nn.Linear(8, 3, seed=8)])

def loss(inputs, targets):
    return nn.cross_entropy(model(inputs), targets)

history = om.fit(
    model, values, labels, loss=loss,
    update=lambda parameter, gradient: om.sgd(parameter, gradient, 0.01),
    epochs=30, batch_size=32, seed=7,
)
```

`fit` copies batches, runs Plans, and reports progress and loss. It returns
`{epoch: loss}` for reported epochs, starting at one. The example uses every
sample, including the short final batch.

`loss` receives one Tensor per NumPy array and builds the loss, including model
calls. It must return a Tensor with one floating-point value. Inputs, targets,
and other per-sample arrays can have any trailing dimensions. Physics losses
can use `partial(model(points), points, order=2)`. Derivatives include interactions
between points within each batch. Smaller batches cannot reproduce interactions
across the whole dataset.

`update(parameter, gradient)` builds the parameter's next value as a lazy Tensor.
The example uses SGD. Both callbacks run only when building a graph for a new
batch shape. Their Python counters, tensor reads, and value-dependent branches
do not rerun for later batches of that shape.

Batches with matching shapes reuse input tensors and a Plan. A short final
batch gets its Plan after the previous batch runs. Both share model state.
Set `drop_last=True` to skip the short batch.
`fit` calls `model.train()` and leaves it in training mode. All model parameters
and buffers must use one device. Inputs are copied there without changing dtype.
Supported NumPy dtypes are float16, float32, float64, int32, and bool in the
machine's byte order.

`fit` shuffles by default. A `seed` reproduces reshuffles across epochs.
Pass a NumPy `Generator` as `rng` to continue its sequence across calls. Do not
pass both `rng` and `seed`. Neither option is accepted with `shuffle=False`.

Every epoch is reported by default. `report_every=100` reads loss on epochs 1,
100, 200, and so on, plus the final epoch. Other epochs run without loss reads.
Reported loss is `sum(batch_size * batch_loss) / sum(batch_size)`, using each
batch's loss before its update. This weighting also applies to losses that sum
values or ignore targets.

`verbose=1` shows a terminal/notebook progress bar, or plain epoch logs when
output is redirected. `verbose=2` always prints epoch logs. `verbose=0` is
silent. These modes do not change loss reads, updates, or random draws. The bar
covers all batches and shows the current epoch without extra tensor reads.

Do not train the same model concurrently or add or replace its parameters, buffers,
or layers during training. Keep its mode and input shapes and dtypes fixed.
Errors stop training and close the progress bar without undoing completed updates.
Call `model.eval()` before evaluation. Use Plans and batching directly for custom
scheduling, mixed-precision loss scaling, CUDA recording, or other execution control.

## Minibatches

Build the training graph and Plan once, then copy each batch into its input tensors:

```python
import numpy as np
import ormi as om
from ormi import nn

rng = np.random.default_rng(7)
values = rng.normal(size=(100, 3)).astype(np.float32)
labels = rng.integers(0, 4, size=100, dtype=np.int32)
inputs = om.tensor(np.zeros((32, 3), dtype=np.float32))
targets = om.tensor(np.zeros(32, dtype=np.int32), dtype=om.int32)
model = nn.Linear(3, 4, seed=7)
loss = nn.cross_entropy(model(inputs), targets)
parameters = model.parameters()
gradients = om.grad(loss, parameters)
updates = [
    (parameter, om.sgd(parameter, gradient, 0.01))
    for parameter, gradient in zip(parameters, gradients, strict=True)
]
plan = om.Plan([loss], updates)

for batch_inputs, batch_targets in om.batches(
    values, labels, batch_size=32, shuffle=True, rng=rng, drop_last=True,
):
    inputs.copy_from(batch_inputs)
    targets.copy_from(batch_targets)
    plan.run()
```

`batches(*arrays, batch_size=..., shuffle=False, seed=None, rng=None, drop_last=False)`
returns a single-use iterator. Each batch is a tuple, even for one input.
Inputs must be NumPy arrays with matching lengths along axis 0. Other dimensions
and dtypes stay unchanged. Strided, read-only, and memory-mapped arrays work.
Convert tensors to NumPy before calling.

Unshuffled batches are views, so writes can change the source. Shuffling applies
one permutation to every array and returns copies, leaving sources unchanged.
A fixed `seed` repeats the order on each call. For reproducible reshuffling,
reuse one NumPy `Generator` as `rng` across epochs. Choose either `seed` or `rng`
and use it only with `shuffle=True`. Reproducibility requires the same NumPy
version and generator algorithm. NumPy's global random state is unchanged.

`batches()` checks arguments when you call it and shuffles when iteration starts.
An empty dataset, or one smaller than `batch_size` with `drop_last=True`,
yields no batches and leaves `rng` unchanged. Call `batches()` again for each
epoch. Do not resize the source arrays while iterating.

The final batch is kept without padding, even when smaller than `batch_size`.
`drop_last=True` discards short final batches. The loop above keeps only full
batches to reuse one Plan.

`copy_from` returns `None` and writes into a realized, contiguous leaf. Its shape,
dtype, allocation, and graph identity stay unchanged. Source shape and dtype
must match exactly. Any number of leading dimensions is supported. NumPy sources
can be strided or read-only. Tensor sources must be realized and contiguous.
Copies support CPU and CUDA tensors, including transfers between them, and
finish before returning. Temporary sources are safe.
Do not write to a source or destination concurrently with a copy or execution.

Copying adds no gradient dependencies and runs no graph. Rerun the Plan to
update results. All derivative orders read the new inputs on each run.
For CUDA replay, copy inputs before `replay.run()` after `replay = plan.record()`.

A short final batch needs another Plan. Build it after the previous batch runs
and reuse it in later epochs. Plans can share model parameters and buffers.
The complete example uses at most two Plans, selected by batch length. Other
shapes and dtypes stay fixed. Rebuild graphs and Plans after changing model mode.

Run the complete shuffled classification example using `fit`:

```bash
uv run python ormi/examples/minibatch.py
uv run python ormi/examples/minibatch.py --samples 257 --batch-size 32
uv run python ormi/examples/minibatch.py --samples 257 --batch-size 32 --drop-last
```

Examples show progress bars in terminals and supported notebooks. Redirecting
output or setting `--no-progress` selects plain-text loss logs. The PINN's
`--report-every` controls loss sampling. The bar performs no extra tensor reads.

## Model Checkpoints

```python
om.save(model, "model.npz")
om.load(model, "model.npz")
plan.run()
```

Create the model before loading. Files store named parameters and buffers,
including BatchNorm running statistics. Names, shapes, dtypes, and shared tensor
relationships must match. CPU and CUDA models can load the same file.
Loading writes into existing storage. Outputs update when the Plan or CUDA
replay runs again. Saving and loading execute no forward or derivative graphs.

`model.state()` returns a new dictionary of live tensor references. Shared tensors
appear under every registered path but are stored once. Built-in names include
`weight`, `bias`, `running_mean`, and `running_var`. Sequential prefixes names
with child indices, such as `0.weight`.
For custom modules, pass a local name when registering state or a child:

```python
self.register_parameter(self.weight, name="weight")
self.register_buffer(self.count, name="count")
self.register_module(self.layers, name="layers")
```

Names must be unique within a module and cannot contain `.`, `/`, `\\`, or null
bytes. Unnamed registrations use `parameter_N`, `buffer_N`, and numeric child
names. Use explicit names if registration order may change.

Files use versioned `.npz` arrays and JSON metadata without pickle. Loading
validates every entry before changing tensors. Copy failures can leave partial
updates. Saving writes and flushes a temporary file, then replaces the destination.
State must be contiguous. Do not train or change state during saving or loading.
Allow host RAM for unique tensor values and file buffers, including for CUDA models.

Checkpoints omit model structure, layer settings, mode, optimizer state,
random-generator and Dropout counters, epoch, shuffle progress, graphs, Plans,
and compiled kernels. Resuming training exactly requires saving extra state
separately.

```bash
uv run python ormi/examples/minibatch.py --save model.npz
uv run python ormi/examples/minibatch.py --load model.npz
```

The second command starts a new training loop with the saved model values.
