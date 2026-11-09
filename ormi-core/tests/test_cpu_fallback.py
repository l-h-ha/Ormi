import os
import subprocess
import sys

import pytest

###
###
###

@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("existing_cache", [False, True])
def test_missing_compiler_uses_precompiled_kernels(tmp_path, automatic, existing_cache):
    environment = os.environ.copy()
    environment["ORMI_CPU_JIT_CXX"] = "" if automatic else "ormi-compiler-does-not-exist"
    environment["ORMI_CPU_JIT_CXX_ID"] = "gcc"
    environment["ORMI_CPU_JIT_CACHE_DIR"] = str(tmp_path / "jit")
    if existing_cache:
        (tmp_path / "jit").mkdir()

    if automatic:
        environment["PATH"] = ""

    script = """
import os
from pathlib import Path

import numpy as np
import ormi_core as oc
x = oc.Tensor([1., 2., 3.])
y = (x * x + 1.).sum()
gradient = oc.grad(y, x)
assert not y.realized and not gradient.realized
plan = oc.Plan([y, gradient])
assert not y.realized and not gradient.realized
plan.run()
assert y.item() == 17.
np.testing.assert_array_equal(gradient.numpy(), [2., 4., 6.])
cache = Path(os.environ['ORMI_CPU_JIT_CACHE_DIR'])
assert not cache.exists() or not any(cache.iterdir())
assert oc.jit.info().entries == 0
"""
    subprocess.run([sys.executable, "-I", "-c", script], env=environment,
                   check=True, capture_output=True, text=True, timeout=30)

@pytest.mark.parametrize("invalid_id", [False, True])
def test_compiler_errors_are_not_silently_ignored(tmp_path, invalid_id):
    environment = os.environ.copy()
    environment["ORMI_CPU_JIT_CXX"] = sys.executable
    environment["ORMI_CPU_JIT_CXX_ID"] = "unknown" if invalid_id else "gcc"
    environment["ORMI_CPU_JIT_CACHE_DIR"] = str(tmp_path / "jit")
    script = """
import ormi_core as oc
x = oc.Tensor([1., 2.])
y = x * x + 1.
try:
    y.numpy()
except oc.jitError as error:
    expected = 'ORMI_CPU_JIT_CXX_ID must be' if INVALID_ID else 'CPU JIT compiler failed'
    assert expected in str(error), error
    assert not y.realized
else:
    raise AssertionError('Expected a compiler configuration or invocation error')
""".replace("INVALID_ID", repr(invalid_id))
    subprocess.run([sys.executable, "-I", "-c", script], env=environment,
                   check=True, capture_output=True, text=True, timeout=30)

@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_broadcast_gradients_and_plan_reuse_without_a_compiler(tmp_path, dtype):
    environment = os.environ.copy()
    environment["ORMI_CPU_JIT_CXX"] = "ormi-compiler-does-not-exist"
    environment["ORMI_CPU_JIT_CXX_ID"] = "gcc"
    environment["ORMI_CPU_JIT_CACHE_DIR"] = str(tmp_path / "jit")
    script = """
import numpy as np
import ormi_core as oc

dtype = np.dtype('DTYPE')
values = np.linspace(-0.5, 0.75, 24, dtype=dtype).reshape(2, 3, 4)
scales = np.array([[0.25], [0.75], [1.5]], dtype=dtype)
biases = np.array([-0.125, 0.0, 0.125, 0.25], dtype=dtype)
tensor_dtype = getattr(oc, 'DTYPE')
x = oc.Tensor(values, dtype=tensor_dtype)
scale = oc.Tensor(scales, dtype=tensor_dtype)
bias = oc.Tensor(biases, dtype=tensor_dtype)
z = x * scale + bias
coupled = z + z.sum((0,), keepdims=True)
square = coupled * coupled
loss = (square * square).sum()
first = oc.grad(loss, x)
second = oc.grad(first.sum(), x)
third = oc.grad(second.sum(), x)
scale_gradient = oc.grad(loss, scale)
bias_gradient = oc.grad(loss, bias)
outputs = [loss, first, second, third, scale_gradient, bias_gradient]
assert all(not value.realized for value in outputs)
plan = oc.Plan(outputs)
assert all(not value.realized for value in outputs)
metadata = (plan.steps, plan.storage_buffers, plan.storage_bytes)
tolerance = 2e-5 if dtype == np.dtype('float32') else 1e-12
for batch in (values, values * 0.5 - 0.125):
    x.copy_from(batch)
    plan.run()
    z_ref = batch * scales + biases
    c = z_ref + z_ref.sum(axis=0, keepdims=True)
    dz = 4 * (c ** 3 + (c ** 3).sum(axis=0, keepdims=True))
    expected = [
        (c ** 4).sum(),
        dz * scales,
        12 * 3 * scales ** 2 * (c ** 2 + (c ** 2).sum(axis=0, keepdims=True)),
        24 * 9 * scales ** 3 * (c + c.sum(axis=0, keepdims=True)),
        (dz * batch).sum(axis=(0, 2)).reshape(3, 1),
        dz.sum(axis=(0, 1)),
    ]
    for actual, reference in zip(outputs, expected):
        np.testing.assert_allclose(actual.numpy(), reference, rtol=tolerance, atol=tolerance)
    assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == metadata
assert oc.jit.info().entries == 0
""".replace("DTYPE", dtype)
    subprocess.run([sys.executable, "-I", "-c", script], env=environment,
                   check=True, capture_output=True, text=True, timeout=60)

@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["transposed", "expanded"])
def test_strided_unary_higher_derivatives_and_plan_reuse(tmp_path, dtype, layout):
    environment = os.environ.copy()
    environment["ORMI_CPU_JIT_CXX"] = "ormi-compiler-does-not-exist"
    environment["ORMI_CPU_JIT_CXX_ID"] = "gcc"
    environment["ORMI_CPU_JIT_CACHE_DIR"] = str(tmp_path / "jit")
    script = """
import numpy as np
import ormi_core as oc

dtype = np.dtype('DTYPE')
values = np.linspace(-0.5, 0.75, 24, dtype=dtype).reshape(2, 3, 4)
x = oc.Tensor(values, dtype=getattr(oc, 'DTYPE'))
repeats = 7 if LAYOUT == 'expanded' else 1
if repeats > 1:
    view = x.reshape((2, 3, 4, 1)).expand((2, 3, 4, repeats))
else:
    view = x.transpose((2, 0, 1))
sines = view.sin()
total = sines.sum()
loss = total * total
first = oc.grad(loss, x)
second = oc.grad(first.sum(), x)
third = oc.grad(second.sum(), x)
outputs = [loss, first, second, third]
assert not view.realized and not sines.realized
assert all(not value.realized for value in outputs)
plan = oc.Plan(outputs)
assert all(not value.realized for value in outputs)
metadata = (plan.steps, plan.storage_buffers, plan.storage_bytes)
tolerance = 3e-5 if dtype == np.dtype('float32') else 1e-12
for batch in (values, values * 0.5 - 0.125):
    x.copy_from(batch)
    plan.run()
    sine, cosine = np.sin(batch), np.cos(batch)
    s, c = sine.sum(), cosine.sum()
    expected = [s * s, 2 * s * cosine,
                2 * (c * cosine - s * sine), -4 * (c * sine + s * cosine)]
    for actual, reference in zip(outputs, expected):
        np.testing.assert_allclose(actual.numpy(), repeats ** 2 * reference, rtol=tolerance, atol=tolerance)
    assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == metadata
assert oc.jit.info().entries == 0
""".replace("DTYPE", dtype).replace("LAYOUT", repr(layout))
    subprocess.run([sys.executable, "-I", "-c", script], env=environment,
                   check=True, capture_output=True, text=True, timeout=30)
