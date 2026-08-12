from __future__ import annotations

import random
import statistics
import time
from collections.abc import Sequence

import numpy as np

from .model import Case, Observation, Result, Spec, Unavailable


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def time_calls(case: Case, calls: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(calls):
        case.run()
    case.synchronize()
    return (time.perf_counter_ns() - start) / 1e6 / calls


def tolerances(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return 5e-2, 5e-3
    if dtype == "float32":
        return 5e-4, 5e-5
    return 1e-8, 1e-10


def verify(
    spec: Spec,
    cases: Sequence[Case],
    values: dict[tuple[str, str], tuple[np.ndarray, ...]],
) -> None:
    reference = next(
        (
            case for case in cases
            if case.framework == "PyTorch" and case.tier == "direct"
        ),
        next((case for case in cases if case.tier == "direct"), cases[0]),
    )
    expected = values[(reference.framework, reference.mode)]
    rtol, atol = tolerances(spec.dtype)

    for case in cases:
        actual = values[(case.framework, case.mode)]
        if len(actual) != len(expected):
            raise AssertionError(
                f"{case.framework} {case.mode} returned {len(actual)} "
                f"values instead of {len(expected)} for {spec.key}."
            )
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"value {index} from {case.framework} {case.mode} "
                    f"differs from {reference.framework} "
                    f"{reference.mode} for {spec.key}"
                ),
            )


def measure(
    spec: Spec,
    cases: list[Case],
    rounds: int,
    calls_per_round: int,
    warmup: int,
) -> tuple[list[Result], list[Unavailable]]:
    if not cases:
        return [], []

    first: dict[tuple[str, str], float] = {}
    values: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    observations: dict[tuple[str, str], Observation] = {}
    unavailable = []
    active = []
    for case in cases:
        key = (case.framework, case.mode)
        try:
            first[key] = time_calls(case, 1)
            values[key] = case.values()
            observations[key] = case.observe()
            active.append(case)
        except Exception as error:
            unavailable.append(Unavailable(
                workload=spec.key,
                framework=case.framework,
                tier=case.tier,
                reason=f"First execution failed: {error}",
            ))
    if not active:
        return [], unavailable
    verify(spec, active, values)

    warmed = []
    for case in active:
        try:
            time_calls(case, warmup)
            warmed.append(case)
        except Exception as error:
            unavailable.append(Unavailable(
                workload=spec.key,
                framework=case.framework,
                tier=case.tier,
                reason=f"Warmup failed: {error}",
            ))
    active = warmed
    if not active:
        return [], unavailable

    samples = {
        (case.framework, case.mode): []
        for case in active
    }
    order = random.Random(spec.seed)
    for _ in range(rounds):
        order.shuffle(active)
        for case in active:
            samples[(case.framework, case.mode)].append(
                time_calls(case, calls_per_round)
            )

    results = []
    for case in active:
        key = (case.framework, case.mode)
        durations = samples[key]
        median = statistics.median(durations)
        results.append(Result(
            workload=spec.key,
            framework=case.framework,
            tier=case.tier,
            mode=case.mode,
            prepare_ms=case.prepare_ms,
            first_ms=first[key],
            median_ms=median,
            p10_ms=percentile(durations, 0.1),
            p90_ms=percentile(durations, 0.9),
            iterations_per_second=1000.0 / median,
            observation=case.observe(),
        ))
    return (
        sorted(results, key=lambda result: result.median_ms),
        unavailable,
    )
