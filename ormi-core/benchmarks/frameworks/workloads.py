from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model import DTypeName, Data, Spec, WorkloadName

LEARNING_RATE = 0.001

###
###
###

def quick_specs() -> list[Spec]:
    return [
        Spec("pointwise", "medium", 32, 16, 64),
        Spec("reduction", "medium", 32, 16, 64),
        Spec("dense", "medium", 32, 32, 64),
        Spec("pinn", "medium", 32, 1, 32),
    ]

def full_specs() -> list[Spec]:
    return [
        Spec("pointwise", "small", 1, 16, 32),
        Spec("pointwise", "medium", 32, 16, 64),
        Spec("pointwise", "large", 1024, 16, 128),
        Spec("reduction", "small", 1, 8, 32),
        Spec("reduction", "medium", 32, 16, 64),
        Spec("reduction", "large", 256, 64, 64),
        Spec("dense", "small", 1, 16, 16),
        Spec("dense", "medium", 32, 32, 64),
        Spec("dense", "large", 1024, 64, 32),
        Spec("pinn", "small", 1, 1, 16),
        Spec("pinn", "medium", 32, 1, 32),
        Spec("pinn", "large", 256, 1, 64),
    ]

def corpus(
    suite: str,
    workloads: set[WorkloadName] | None = None,
    scales: set[str] | None = None,
    dtypes: list[DTypeName] | None = None,
) -> list[Spec]:
    if suite == "quick":
        selected = quick_specs()
    elif suite == "full":
        selected = full_specs()
    else:
        raise ValueError(f"Unknown benchmark suite: {suite}")

    if workloads:
        selected = [spec for spec in selected if spec.name in workloads]

    if scales:
        selected = [spec for spec in selected if spec.scale in scales]

    requested_dtypes = dtypes or ["float32"]
    return [
        replace(spec, dtype=dtype)
        for spec in selected
        for dtype in requested_dtypes
    ]

def make_data(spec: Spec) -> Data:
    rng = np.random.default_rng(spec.seed)
    dtype = np.dtype(spec.dtype)

    def normal(shape: tuple[int, ...], scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, shape).astype(dtype)

    if spec.name == "pointwise":
        return Data({
            "input": normal((spec.batch, spec.width), 0.25),
            "scale": normal((spec.width,), 0.2),
            "bias": normal((spec.width,), 0.05),
        })

    if spec.name == "reduction":
        return Data({
            "input": normal(
                (spec.batch, spec.features, spec.width),
                0.25,
            ),
            "row": normal((spec.width,), 0.2),
            "column": normal((spec.features, 1), 0.05),
        })

    if spec.name == "dense":
        return Data({
            "input": normal((spec.batch, spec.features), 0.25),
            "target": normal((spec.batch, spec.width), 0.1),
            "weight": normal((spec.features, spec.width), 0.2),
            "bias": normal((spec.width,), 0.05),
        })

    if spec.name == "pinn":
        return Data({
            "input": rng.uniform(-1.0, 1.0, (spec.batch, 1)).astype(dtype),
            "first_weight": normal((1, spec.width), 0.5),
            "first_bias": normal((spec.width,), 0.1),
            "second_weight": normal((spec.width, 1), 0.2),
            "second_bias": normal((1,), 0.05),
        })

    raise ValueError(f"Unknown workload: {spec.name}")
