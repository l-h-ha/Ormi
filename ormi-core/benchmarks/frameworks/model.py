from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


Framework = Literal["ormi", "pytorch", "tensorflow"]
Tier = Literal["direct", "optimized"]
DTypeName = Literal["float16", "float32", "float64"]
WorkloadName = Literal["pointwise", "reduction", "dense", "pinn"]


@dataclass(frozen=True)
class Spec:
    name: WorkloadName
    scale: str
    batch: int
    features: int
    width: int
    dtype: DTypeName = "float32"
    seed: int = 20260810

    def __post_init__(self) -> None:
        if self.batch <= 0 or self.features <= 0 or self.width <= 0:
            raise ValueError("Workload dimensions must be positive.")

    @property
    def key(self) -> str:
        return (
            f"{self.name}/{self.scale}/{self.dtype}/"
            f"b{self.batch}-f{self.features}-w{self.width}"
        )


@dataclass(frozen=True)
class Data:
    values: dict[str, np.ndarray]


@dataclass(frozen=True)
class Observation:
    names: tuple[str, ...]
    values: tuple[float, ...]


@dataclass
class Case:
    framework: str
    tier: Tier
    mode: str
    prepare_ms: float
    run: Callable[[], None]
    values: Callable[[], tuple[np.ndarray, ...]]
    observe: Callable[[], Observation]
    synchronize: Callable[[], None]


@dataclass(frozen=True)
class Result:
    workload: str
    framework: str
    tier: str
    mode: str
    prepare_ms: float
    first_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    iterations_per_second: float
    observation: Observation

    def json(self) -> dict[str, object]:
        result = asdict(self)
        result["observation"] = dict(
            zip(self.observation.names, self.observation.values, strict=True)
        )
        return result


@dataclass(frozen=True)
class Unavailable:
    workload: str
    framework: str
    tier: str
    reason: str

    def json(self) -> dict[str, str]:
        return asdict(self)


class Unsupported(RuntimeError):
    pass


def output_observation(value: np.ndarray) -> Observation:
    array = np.asarray(value, dtype=np.float64)
    return Observation(
        ("mean", "minimum", "maximum"),
        (
            float(np.mean(array, dtype=np.float64)),
            float(np.min(array)),
            float(np.max(array)),
        ),
    )


def gradient_observation(
    loss: np.ndarray,
    gradients: Sequence[np.ndarray],
) -> Observation:
    names = ["loss"]
    values = [float(np.asarray(loss, dtype=np.float64).reshape(()))]
    for index, gradient in enumerate(gradients):
        array = np.asarray(gradient, dtype=np.float64)
        names.extend((f"gradient_{index}_mean", f"gradient_{index}_l2"))
        values.extend((
            float(np.mean(array, dtype=np.float64)),
            float(np.sqrt(np.sum(array * array, dtype=np.float64))),
        ))
    return Observation(tuple(names), tuple(values))


def step_observation(
    loss: np.ndarray,
    parameters: Sequence[np.ndarray],
) -> Observation:
    names = ["loss"]
    values = [float(np.asarray(loss, dtype=np.float64).reshape(()))]
    for index, parameter in enumerate(parameters):
        array = np.asarray(parameter, dtype=np.float64)
        names.append(f"parameter_{index}_mean")
        values.append(float(np.mean(array, dtype=np.float64)))
    return Observation(tuple(names), tuple(values))
