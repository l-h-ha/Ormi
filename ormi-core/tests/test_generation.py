from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
GENERATOR = ROOT / "csrc" / "bindings" / "generate.py"
DEFINITIONS = ROOT / "csrc" / "bindings" / "defs"
SOURCE_STUBS = ROOT / "src" / "ormi_core" / "_ext" / "_C"

###
###
###

def load_generator() -> dict[str, object]:
    return runpy.run_path(str(GENERATOR))

###
###
###

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

def test_grad_overloads_and_partial_are_generated_from_yaml() -> None:
    generator = load_generator()
    spec = generator["load_spec"](sorted(DEFINITIONS.glob("*.yaml")))
    stub = generator["generate_stub"](spec, "")
    assert "def grad(output: Tensor, input: Tensor, seed: Tensor | None = None) -> Tensor:" in stub
    assert "def grad(output: Tensor, inputs: Sequence[Tensor], seed: Tensor | None = None) -> list[Tensor]:" in stub
    assert "def grad(self, input: Tensor, seed: Tensor | None = None) -> Tensor:" in stub
    assert "def grad(self, inputs: Sequence[Tensor], seed: Tensor | None = None) -> list[Tensor]:" in stub
    assert "def partial(output: Tensor, input: Tensor, coordinate: int | None = None, component: int | None = None, order: int = 1) -> Tensor:" in stub
    assert "&ormi::core::partial" in generator["generate_cpp"](spec)

def test_tensor_copy_is_generated_from_yaml() -> None:
    generator = load_generator()
    spec = generator["load_spec"](sorted(DEFINITIONS.glob("*.yaml")))
    stub = generator["generate_stub"](spec, "")
    assert "def copy_from(self, source: Any) -> None:" in stub
    assert "&ormi::bindings::tensor_copy_from" in generator["generate_cpp"](spec)

def test_named_module_state_is_generated_from_yaml() -> None:
    generator = load_generator()
    spec = generator["load_spec"](sorted(DEFINITIONS.glob("*.yaml")))
    stub = generator["generate_stub"](spec, "nn")
    assert "def state(self) -> dict[str, Tensor]:" in stub
    assert "def register_parameter(self, parameter: Tensor, name: str = '') -> None:" in stub
    assert "def register_buffer(self, buffer: Tensor, name: str = '') -> None:" in stub
    assert "def register_module(self, module: Module, name: str = '') -> None:" in stub
    assert "&ormi::nn::Module::state" in generator["generate_cpp"](spec)

def test_nn_inheritance_and_exports_are_generated_from_yaml() -> None:
    generator = load_generator()
    spec = generator["load_spec"](sorted(DEFINITIONS.glob("*.yaml")))
    cpp = generator["generate_cpp"](spec)
    stub = generator["generate_stub"](spec, "nn")
    assert "py::class_<ormi::nn::Linear, ormi::nn::Module, py::smart_holder>" in cpp
    assert "py::class_<ormi::nn::Module, ormi::bindings::PyModule, py::smart_holder>" in cpp
    assert '"Linear", py::is_final()' in cpp
    assert '"Sin", py::is_final()' in cpp
    assert '"Sequential", py::is_final()' in cpp
    assert "class Linear(Module):" in stub
    assert "class Sin(Module):" in stub
    assert "class Sequential(Module):" in stub
    assert "py::class_<ormi::nn::LayerNorm, ormi::nn::Module, py::smart_holder>" in cpp
    assert "class LayerNorm(Module):" in stub
    assert "class BatchNorm(Module):" in stub
    assert "class Dropout(Module):" in stub
    assert "def register_buffer(" in stub
    assert "def train(" in stub
    assert "def dropout(" in stub
    assert "def layer_norm(" in stub
    assert "normalized_shape: int | Sequence[int]" in stub
    assert "def rsqrt(self) -> Tensor:" in generator["generate_stub"](spec, "")
    assert "def rsqrt(input: Tensor) -> Tensor:" in generator["generate_stub"](spec, "ops")
    assert "def mse_loss(" in stub
    assert "def register_parameter(self, parameter: Tensor, name: str = '') -> None:" in stub
    assert "def register_module(self, module: Module, name: str = '') -> None:" in stub

@pytest.mark.parametrize("final", ["true", 1, None])
def test_binding_generator_rejects_non_boolean_final(final) -> None:
    generator = load_generator()
    spec = generator["load_spec"](sorted(DEFINITIONS.glob("*.yaml")))
    module = next(cls for cls in spec["classes"] if cls["name"] == "Module")
    module["final"] = final

    with pytest.raises(TypeError, match="final must be a boolean"):
        generator["validate_spec"](spec)

def test_binding_generator_rejects_final_trampolines() -> None:
    generator = load_generator()
    spec = generator["load_spec"](sorted(DEFINITIONS.glob("*.yaml")))
    module = next(cls for cls in spec["classes"] if cls["name"] == "Module")
    module["final"] = True

    with pytest.raises(ValueError, match="final class cannot have a trampoline"):
        generator["validate_spec"](spec)
