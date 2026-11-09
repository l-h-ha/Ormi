from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

from .measure import measure
from .model import (
    DTypeName,
    Framework,
    Result,
    Spec,
    Tier,
    Unavailable,
    Unsupported,
    WorkloadName,
)
from .workloads import corpus, make_data

ADAPTERS = {
    "ormi": "benchmarks.frameworks.adapters.ormi",
    "pytorch": "benchmarks.frameworks.adapters.pytorch",
    "tensorflow": "benchmarks.frameworks.adapters.tensorflow",
}

###
###
###

def jit_artifacts(path: Path) -> dict[str, int]:
    sources = {
        source.stem: source
        for source in path.glob("*.cpp")
        if source.is_file()
    }
    libraries = {
        library.stem: library
        for library in path.iterdir()
        if library.is_file() and library.suffix.lower() in {".dll", ".so", ".dylib"}
    }
    return {
        "entries": len(sources.keys() & libraries.keys()),
        "source_bytes": sum(source.stat().st_size for source in sources.values()),
        "library_bytes": sum(
            library.stat().st_size for library in libraries.values()
        ),
    }

def artifact_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {
        name: after[name] - before[name]
        for name in before
    }

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Compare ORMI, PyTorch, and TensorFlow using the same seeded inputs."
        )
    )
    result.add_argument("--suite", choices=("quick", "full"), default="quick")
    result.add_argument(
        "--workloads",
        choices=("pointwise", "reduction", "dense", "pinn"),
        nargs="+",
    )
    result.add_argument(
        "--scales",
        choices=("small", "medium", "large"),
        nargs="+",
    )
    result.add_argument(
        "--frameworks",
        choices=tuple(ADAPTERS),
        nargs="+",
        default=list(ADAPTERS),
    )
    result.add_argument(
        "--tiers",
        choices=("direct", "optimized"),
        nargs="+",
        default=["direct", "optimized"],
    )
    result.add_argument(
        "--dtypes",
        choices=("float16", "float32", "float64"),
        nargs="+",
        default=["float32"],
    )
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--label",
        default="",
        help=(
            "Build and machine label saved in JSON results, "
            "for example windows-msvc-openblas-sleef."
        ),
    )
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--rounds", type=int, default=10)
    result.add_argument("--calls-per-round", type=int, default=3)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument(
        "--jit-cache",
        type=Path,
        help=(
            "ORMI CPU JIT cache directory. By default, each run gets a "
            "new directory below .benchmarks/framework-jit."
        ),
    )
    result.add_argument("--json", type=Path)
    result.add_argument("--list", action="store_true")
    return result

def positive(arguments: argparse.Namespace) -> None:
    for name in ("threads", "rounds", "calls_per_round", "warmup"):
        if getattr(arguments, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")

def configure_environment(threads: int, configured_cache: Path | None) -> Path:
    jit_cache = configured_cache or (
        Path(".benchmarks/framework-jit") /
        f"{os.getpid()}-{time.time_ns()}"
    )
    jit_cache = jit_cache.resolve()
    jit_cache.mkdir(parents=True, exist_ok=True)
    value = str(threads)
    os.environ["ORMI_CPU_THREADS"] = value
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["ORMI_CPU_JIT_CACHE_DIR"] = str(jit_cache)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return jit_cache

def load_adapters(
    frameworks: list[Framework],
    threads: int,
) -> tuple[dict[Framework, ModuleType], dict[Framework, str]]:
    loaded = {}
    failed = {}

    for framework in frameworks:
        try:
            adapter = importlib.import_module(ADAPTERS[framework])
            adapter.configure(threads)
            loaded[framework] = adapter
        except Exception as error:
            failed[framework] = str(error)

    return loaded, failed

def print_specs(specs: list[Spec]) -> None:
    for spec in specs:
        print(spec.key)

def print_results(results: list[Result]) -> None:
    print(
        f"{'framework':<12} {'mode':<9} {'prepare':>10} {'first':>10} "
        f"{'median':>10} {'p10':>10} {'p90':>10} {'iter/s':>12}"
    )

    for result in results:
        print(
            f"{result.framework:<12} {result.mode:<9} "
            f"{result.prepare_ms:>9.3f}ms {result.first_ms:>9.3f}ms "
            f"{result.median_ms:>9.3f}ms {result.p10_ms:>9.3f}ms "
            f"{result.p90_ms:>9.3f}ms "
            f"{result.iterations_per_second:>12.1f}"
        )

###
###
###

def main() -> None:
    arguments = parser().parse_args()
    positive(arguments)
    specs = corpus(
        arguments.suite,
        set(arguments.workloads) if arguments.workloads else None,
        set(arguments.scales) if arguments.scales else None,
        list(arguments.dtypes),
    )

    if arguments.list:
        print_specs(specs)
        return

    jit_cache = configure_environment(
        arguments.threads,
        arguments.jit_cache,
    )
    adapters, import_failures = load_adapters(
        list(arguments.frameworks),
        arguments.threads,
    )

    versions = {
        framework: adapter.version()
        for framework, adapter in adapters.items()
    }
    versions.update({
        framework: f"unavailable: {reason}"
        for framework, reason in import_failures.items()
    })

    print(f"Python {platform.python_version()} on {platform.platform()}")

    if arguments.label:
        print(f"run label {arguments.label}")
    else:
        print("Run label unset. Add --label before citing these results.")

    print(", ".join(
        f"{framework} {versions[framework]}"
        for framework in arguments.frameworks
    ))
    print(
        f"device {arguments.device}, threads {arguments.threads}, "
        f"suite {arguments.suite}, rounds {arguments.rounds}, "
        f"calls/round {arguments.calls_per_round}"
    )
    print(f"ORMI CPU JIT cache {jit_cache}")
    print(
        "Preparation, first execution, and warmed execution are reported "
        "separately. Complete numerical outputs are checked before warmup."
    )

    all_results = []
    unavailable = []
    jit_workloads = []
    requested_tiers: list[Tier] = list(arguments.tiers)

    for spec in specs:
        jit_before = jit_artifacts(jit_cache)
        data = make_data(spec)
        cases = []

        for framework in arguments.frameworks:
            adapter = adapters.get(framework)

            if adapter is None:
                for tier in requested_tiers:
                    unavailable.append(Unavailable(
                        workload=spec.key,
                        framework=framework,
                        tier=tier,
                        reason=import_failures[framework],
                    ))

                continue

            for tier in requested_tiers:
                try:
                    cases.append(adapter.prepare(
                        spec,
                        data,
                        tier,
                        arguments.device,
                    ))
                except Unsupported as error:
                    unavailable.append(Unavailable(
                        workload=spec.key,
                        framework=framework,
                        tier=tier,
                        reason=str(error),
                    ))
                except Exception as error:
                    unavailable.append(Unavailable(
                        workload=spec.key,
                        framework=framework,
                        tier=tier,
                        reason=f"Preparation failed: {error}",
                    ))

        results, runtime_failures = measure(
            spec,
            cases,
            arguments.rounds,
            arguments.calls_per_round,
            arguments.warmup,
        )
        unavailable.extend(runtime_failures)
        print()
        print(spec.key)
        print_results(results)
        all_results.extend(results)
        jit_after = jit_artifacts(jit_cache)
        created = artifact_delta(jit_before, jit_after)
        jit_workloads.append({
            "workload": spec.key,
            **created,
        })

        if created["entries"]:
            print(
                f"ORMI CPU JIT created {created['entries']} cache "
                f"entr{'y' if created['entries'] == 1 else 'ies'} "
                f"({created['source_bytes']} source bytes, "
                f"{created['library_bytes']} library bytes)."
            )

    if unavailable:
        print()
        print("Unavailable cases")

        for item in unavailable:
            print(
                f"{item.workload}: {item.framework} {item.tier}: "
                f"{item.reason}"
            )

    if arguments.json:
        jit_summary = jit_artifacts(jit_cache)
        payload = {
            "environment": {
                "label": arguments.label or None,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "device": arguments.device,
                "threads": arguments.threads,
                "suite": arguments.suite,
                "rounds": arguments.rounds,
                "calls_per_round": arguments.calls_per_round,
                "warmup": arguments.warmup,
                "jit_cache": str(jit_cache),
                "ormi_cpu_jit_compiler": {
                    "executable": os.environ.get("ORMI_CPU_JIT_CXX"),
                    "id": os.environ.get("ORMI_CPU_JIT_CXX_ID"),
                    "target": os.environ.get("ORMI_CPU_JIT_TARGET", "auto"),
                },
                "workloads": arguments.workloads or [
                    "pointwise",
                    "reduction",
                    "dense",
                    "pinn",
                ],
                "scales": arguments.scales or (
                    ["medium"]
                    if arguments.suite == "quick"
                    else ["small", "medium", "large"]
                ),
                "frameworks": arguments.frameworks,
                "tiers": arguments.tiers,
                "dtypes": arguments.dtypes,
                "versions": versions,
            },
            "ormi_cpu_jit": {
                **jit_summary,
                "workloads": jit_workloads,
            },
            "results": [result.json() for result in all_results],
            "unavailable": [item.json() for item in unavailable],
        }
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

if __name__ == "__main__":
    main()
