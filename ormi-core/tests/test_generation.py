from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
GENERATOR = ROOT / "csrc" / "bindings" / "generate.py"
DEFINITIONS = ROOT / "csrc" / "bindings" / "defs"
SOURCE_STUBS = ROOT / "src" / "ormi_core" / "_ext" / "_C"


def load_generator() -> dict[str, object]:
    return runpy.run_path(str(GENERATOR))


def test_binding_definitions_generate_deterministically() -> None:
    generator = load_generator()
    paths = sorted(DEFINITIONS.glob("*.yaml"))

    first = generator["load_spec"](paths)
    second = generator["load_spec"](paths)

    assert generator["generate_cpp"](first) == generator["generate_cpp"](second)

    modules = [""] + generator["module_names"](first)
    for module in modules:
        first_stub = generator["generate_stub"](first, module)
        second_stub = generator["generate_stub"](second, module)
        assert first_stub == second_stub
        ast.parse(first_stub)

        filename = "__init__.pyi" if not module else f"{module}.pyi"
        assert (SOURCE_STUBS / filename).read_text(encoding="utf-8") == first_stub

        if module:
            marker = generator["generate_source_marker"](module)
            assert (SOURCE_STUBS / f"{module}.py").read_text(encoding="utf-8") == marker


def test_binding_generator_rejects_unknown_sections(tmp_path: Path) -> None:
    generator = load_generator()
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("surprise: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown section"):
        generator["load_spec"]([invalid])
