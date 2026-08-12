from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


SECTIONS = {"includes", "exceptions", "enums", "classes", "functions"}


def load_spec(paths: list[Path]) -> dict[str, list[Any]]:
    spec: dict[str, list[Any]] = {key: [] for key in SECTIONS}

    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            raise TypeError(f"{path}: binding definition must be a mapping")

        unknown = set(document) - SECTIONS
        if unknown:
            raise ValueError(f"{path}: unknown section(s): {', '.join(sorted(unknown))}")

        for section, values in document.items():
            if not isinstance(values, list):
                raise TypeError(f"{path}: section '{section}' must be a list")
            spec[section].extend(values)

    validate_spec(spec)
    return spec


def require(item: dict[str, Any], fields: set[str], context: str) -> None:
    missing = fields - set(item)
    if missing:
        raise ValueError(f"{context}: missing field(s): {', '.join(sorted(missing))}")


def validate_annotation(value: str | None, context: str) -> None:
    if not value:
        return
    try:
        ast.parse(f"def check(value: {value}) -> None: ...")
    except SyntaxError as error:
        raise ValueError(f"{context}: invalid Python annotation '{value}'") from error


def validate_args(args: list[dict[str, Any]], context: str) -> None:
    names: set[str] = set()
    for arg in args:
        require(arg, {"name"}, context)
        if arg["name"] in names:
            raise ValueError(f"{context}: duplicate argument '{arg['name']}'")
        names.add(arg["name"])
        validate_annotation(arg.get("py_type"), f"{context}.{arg['name']}")


def validate_spec(spec: dict[str, list[Any]]) -> None:
    exports: dict[tuple[str, str], str] = {}

    for section in ("exceptions", "enums", "classes"):
        for item in spec[section]:
            if not isinstance(item, dict):
                raise TypeError(f"{section}: entries must be mappings")
            require(item, {"name", "cpp_type"}, section)
            module = item.get("module", "")
            key = (module, item["name"])
            if key in exports:
                raise ValueError(f"duplicate export '{item['name']}' in module '{module or 'root'}'")
            exports[key] = section

    function_signatures: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in spec["functions"]:
        require(item, {"python"}, "function")
        validate_args(item.get("args", []), f"function {item['python']}")
        validate_annotation(item.get("returns"), f"function {item['python']} return")
        key = (
            item.get("module", ""),
            item["python"],
            tuple(arg.get("py_type", "Any") for arg in item.get("args", [])),
        )
        if key in function_signatures:
            raise ValueError(f"duplicate function signature '{item['python']}'")
        function_signatures.add(key)

    for cls in spec["classes"]:
        for constructor in cls.get("constructors", []):
            validate_args(constructor.get("args", []), f"class {cls['name']} constructor")
        for prop in cls.get("properties_readonly", []) + cls.get("properties", []):
            require(prop, {"name"}, f"class {cls['name']} property")
            validate_annotation(prop.get("py_type"), f"class {cls['name']}.{prop['name']}")
        method_signatures: set[tuple[str, tuple[str, ...]]] = set()
        for method in cls.get("methods", []):
            require(method, {"name"}, f"class {cls['name']} method")
            validate_args(method.get("args", []), f"class {cls['name']}.{method['name']}")
            validate_annotation(method.get("returns"), f"class {cls['name']}.{method['name']} return")
            key = (
                method["name"],
                tuple(arg.get("py_type", "Any") for arg in method.get("args", [])),
            )
            if key in method_signatures:
                raise ValueError(f"class {cls['name']}: duplicate method signature '{method['name']}'")
            method_signatures.add(key)


def cpp_default(value: Any) -> str:
    if value is None:
        return "nullptr"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def py_default(arg: dict[str, Any]) -> str:
    value = arg.get("py_default", arg.get("default"))
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    return str(value)


def callable_expr(item: dict[str, Any], cpp_type: str | None = None) -> str:
    if "binding" in item:
        return f"&{item['binding']}"
    if "expr" in item:
        return item["expr"]
    if "cpp" not in item:
        raise ValueError(f"binding for '{item.get('name', item.get('python', '?'))}' has no callable")

    owner = cpp_type or item.get("namespace")
    if not owner:
        raise ValueError(f"binding for '{item.get('name', item.get('python', '?'))}' has no owner")
    raw = f"&{owner}::{item['cpp']}"
    if "overload" in item:
        return f"static_cast<{item['overload']}>({raw})"
    return raw


def emit_args(args: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for arg in args:
        expression = f'py::arg("{arg["name"]}")'
        if "default" in arg:
            expression += f" = {cpp_default(arg['default'])}"
        result.append(expression)
    return result


def emit_call(receiver: str, method: str, name: str, target: str, item: dict[str, Any]) -> list[str]:
    parts = [f'{receiver}.{method}("{name}", {target}']
    parts.extend(emit_args(item.get("args", [])))
    if item.get("is_operator"):
        parts.append("py::is_operator()")
    if policy := item.get("return_policy"):
        parts.append(f"py::return_value_policy::{policy}")
    if item.get("release_gil"):
        parts.append("py::call_guard<py::gil_scoped_release>()")
    return [",\n        ".join(parts) + ");"]


def module_names(spec: dict[str, list[Any]]) -> list[str]:
    names = {
        item.get("module", "")
        for section in ("exceptions", "enums", "classes", "functions")
        for item in spec[section]
    }
    names.discard("")
    return sorted(names)


def module_var(name: str) -> str:
    return f"module_{name.replace('.', '_')}"


def receiver(item: dict[str, Any]) -> str:
    module = item.get("module", "")
    return module_var(module) if module else "m"


def generate_cpp(spec: dict[str, list[Any]]) -> str:
    lines = [
        "// Generated by csrc/bindings/generate.py. Do not edit.",
        "",
        "#include <pybind11/pybind11.h>",
        "#include <pybind11/stl.h>",
    ]
    for include in sorted(set(spec["includes"])):
        if include.startswith("<"):
            lines.append(f"#include {include}")
        elif include.startswith(("ormi/", "pybind11/")):
            lines.append(f"#include <{include}>")
        else:
            lines.append(f'#include "{include}"')

    lines += ["", "namespace py = pybind11;", "", "namespace ormi::bindings {", "", "void bind_generated(py::module_& m) {"]

    for name in module_names(spec):
        lines.append(f'    auto {module_var(name)} = m.def_submodule("{name}");')
    if module_names(spec):
        lines.append("")

    exception_vars: dict[str, str] = {}
    for item in spec["exceptions"]:
        var = f"exception_{item['name']}"
        base = item.get("base", "RuntimeError")
        if base in exception_vars:
            base = f"{exception_vars[base]}.ptr()"
        elif base == "RuntimeError":
            base = "PyExc_RuntimeError"
        lines.append(
            f'    auto {var} = py::register_exception<{item["cpp_type"]}>('
            f'{receiver(item)}, "{item["name"]}", {base});'
        )
        exception_vars[item["name"]] = var
    if spec["exceptions"]:
        lines.append("")

    for item in spec["enums"]:
        var = f"enum_{item['name']}"
        lines.append(f'    auto {var} = py::enum_<{item["cpp_type"]}>({receiver(item)}, "{item["name"]}");')
        for value in item.get("values", []):
            lines.append(f'    {var}.value("{value["name"]}", {item["cpp_type"]}::{value["cpp"]});')
            for alias in value.get("aliases", []):
                lines.append(f'    {receiver(item)}.attr("{alias}") = {var}.attr("{value["name"]}");')
        lines.append("")

    for cls in spec["classes"]:
        template = cls["cpp_type"]
        if holder := cls.get("holder"):
            template += f", {holder}"
        var = f"class_{cls['name']}"
        lines.append(f'    auto {var} = py::class_<{template}>({receiver(cls)}, "{cls["name"]}");')

        for ctor in cls.get("constructors", []):
            if factory := ctor.get("factory"):
                target = f"py::init(&{factory})"
            else:
                target = f"py::init<{ctor.get('signature', '')}>()"
            parts = [f"    {var}.def({target}"] + emit_args(ctor.get("args", []))
            lines.append(",\n        ".join(parts) + ");")

        for field in cls.get("fields_readwrite", []):
            lines.append(f'    {var}.def_readwrite("{field["name"]}", &{cls["cpp_type"]}::{field["cpp"]});')
        for field in cls.get("fields_readonly", []):
            lines.append(f'    {var}.def_readonly("{field["name"]}", &{cls["cpp_type"]}::{field["cpp"]});')
        for prop in cls.get("properties_readonly", []):
            lines.append(
                f'    {var}.def_property_readonly("{prop["name"]}", '
                f'{callable_expr(prop, cls["cpp_type"])});'
            )
        for prop in cls.get("properties", []):
            lines.append(
                f'    {var}.def_property("{prop["name"]}", '
                f'{callable_expr(prop["get"], cls["cpp_type"])}, '
                f'{callable_expr(prop["set"], cls["cpp_type"])});'
            )
        for method in cls.get("methods", []):
            for emitted in emit_call(
                f"    {var}", "def", method["name"], callable_expr(method, cls["cpp_type"]), method
            ):
                lines.append(emitted)
        lines.append("")

    for item in spec["functions"]:
        for emitted in emit_call(
            f"    {receiver(item)}", "def", item["python"], callable_expr(item), item
        ):
            lines.append(emitted)
        lines.append("")

    lines += ["}", "", "} // namespace ormi::bindings", ""]
    return "\n".join(lines)


def stub_arg(arg: dict[str, Any]) -> str:
    value = f"{arg['name']}: {arg.get('py_type', 'Any')}"
    if "default" in arg:
        value += f" = {py_default(arg)}"
    return value


def stub_signature(item: dict[str, Any], *, method: bool = False) -> str:
    args = ["self"] if method else []
    args.extend(stub_arg(arg) for arg in item.get("args", []))
    return f"def {item.get('name', item.get('python'))}({', '.join(args)}) -> {item.get('returns', 'Any')}: ..."


def grouped(items: list[dict[str, Any]], key: str) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for item in items:
        name = item[key]
        if name not in groups:
            order.append(name)
        groups[name].append(item)
    return [groups[name] for name in order]


def emit_stub_class(lines: list[str], cls: dict[str, Any]) -> None:
    lines.append(f"class {cls['name']}:")
    body = False
    constructors = cls.get("constructors", [])
    for ctor in constructors:
        if len(constructors) > 1:
            lines.append("    @overload")
        args = ", ".join(["self"] + [stub_arg(arg) for arg in ctor.get("args", [])])
        lines.append(f"    def __init__({args}) -> None: ...")
        body = True
    for field in cls.get("fields_readwrite", []):
        lines.append(f"    {field['name']}: {field.get('py_type', 'Any')}")
        body = True
    for field in cls.get("fields_readonly", []):
        lines += ["    @property", f"    def {field['name']}(self) -> {field.get('py_type', 'Any')}: ..."]
        body = True
    for prop in cls.get("properties_readonly", []):
        lines += ["    @property", f"    def {prop['name']}(self) -> {prop.get('py_type', 'Any')}: ..."]
        body = True
    for prop in cls.get("properties", []):
        typ = prop.get("py_type", "Any")
        lines += [
            "    @property",
            f"    def {prop['name']}(self) -> {typ}: ...",
            f"    @{prop['name']}.setter",
            f"    def {prop['name']}(self, value: {typ}) -> None: ...",
        ]
        body = True
    methods = cls.get("methods", [])
    for group in grouped(methods, "name"):
        for method in group:
            if len(group) > 1:
                lines.append("    @overload")
            lines.append("    " + stub_signature(method, method=True))
            body = True
    if not body:
        lines.append("    pass")
    lines.append("")


def generate_stub(spec: dict[str, list[Any]], module: str) -> str:
    lines = [
        "# Generated by csrc/bindings/generate.py. Do not edit.",
        "from __future__ import annotations",
        "",
        "import enum",
        "from collections.abc import Sequence",
        "from typing import Any, ClassVar, overload",
        "",
    ]
    modules = module_names(spec)
    root_types = [item["name"] for section in ("enums", "classes") for item in spec[section] if not item.get("module")]
    if module and root_types:
        lines.append(f"from . import {', '.join(root_types)}")
        lines.append("")
    if not module:
        for name in modules:
            lines.append(f"from . import {name} as {name}")
        if modules:
            lines.append("")

    names: list[str] = []
    for section in ("exceptions", "enums", "classes"):
        names.extend(item["name"] for item in spec[section] if item.get("module", "") == module)
    names.extend(item["python"] for item in spec["functions"] if item.get("module", "") == module)
    if not module:
        names.extend(modules)
        for enum_item in spec["enums"]:
            if not enum_item.get("module"):
                for value in enum_item.get("values", []):
                    names.extend(value.get("aliases", []))
    names = list(dict.fromkeys(names))
    lines.append(f"__all__ = {names!r}")
    lines.append("")

    for item in spec["exceptions"]:
        if item.get("module", "") == module:
            lines.append(f"class {item['name']}({item.get('base', 'RuntimeError')}): ...")
    if any(item.get("module", "") == module for item in spec["exceptions"]):
        lines.append("")

    for item in spec["enums"]:
        if item.get("module", "") != module:
            continue
        lines.append(f"class {item['name']}(enum.Enum):")
        for value in item.get("values", []):
            lines.append(f"    {value['name']}: ClassVar[{item['name']}]")
        if not item.get("values"):
            lines.append("    pass")
        lines.append("")
        for value in item.get("values", []):
            for alias in value.get("aliases", []):
                lines.append(f"{alias}: {item['name']}")
        if any(value.get("aliases") for value in item.get("values", [])):
            lines.append("")

    for cls in spec["classes"]:
        if cls.get("module", "") == module:
            emit_stub_class(lines, cls)

    functions = [item for item in spec["functions"] if item.get("module", "") == module]
    for group in grouped(functions, "python"):
        for item in group:
            if len(group) > 1:
                lines.append("@overload")
            lines.append(stub_signature(item))
        lines.append("")

    source = "\n".join(lines)
    ast.parse(source)
    return source


def generate_source_marker(module: str) -> str:
    return (
        "# Generated by csrc/bindings/generate.py. Do not edit.\n"
        f'"""Source marker for the compiled ``_C.{module}`` submodule."""\n\n'
        "raise ImportError(\n"
        '    "This module is registered by the compiled ormi_core extension."\n'
        ")\n"
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content.encode("utf-8"):
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--stub-dir", required=True, type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--defs", required=True, nargs="+", type=Path)
    args = parser.parse_args()

    spec = load_spec(args.defs)
    if args.cpp:
        write(args.cpp, generate_cpp(spec))
    write(args.stub_dir / "__init__.pyi", generate_stub(spec, ""))
    for module in module_names(spec):
        write(args.stub_dir / f"{module}.pyi", generate_stub(spec, module))
        write(args.stub_dir / f"{module}.py", generate_source_marker(module))
    write(args.marker or args.stub_dir.parent / "py.typed", "")


if __name__ == "__main__":
    main()
