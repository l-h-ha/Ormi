from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SECTIONS = {
    "enums",
    "classes",
    "functions",
    "ops",
}


def cpp_default(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return value

    if value is None:
        return "nullptr"

    return str(value)


def py_default(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"

    if value is None:
        return "None"

    if isinstance(value, str):
        if value in {"True", "False", "None", "..."}:
            return value
        return repr(value)

    return repr(value)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    return data if data else {}


def collect_defs(defs: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "enums": [],
        "classes": [],
        "functions": [],
        "ops": [],
    }

    for path in defs:
        data = load_yaml(path)

        unknown = set(data.keys()) - SUPPORTED_SECTIONS
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise ValueError(f"{path}: unsupported binding section(s): {keys}")

        for key in merged:
            if key in data:
                section = data[key]
                if not isinstance(section, list):
                    raise TypeError(f"{path}: section '{key}' must be a list")
                merged[key].extend(section)

    return merged


def is_system_include(include: str) -> bool:
    return (
        include in {"memory", "vector", "string", "utility"}
        or include.startswith("pybind11/")
        or include.startswith("<")
    )


def collect_includes(spec: dict[str, Any]) -> list[str]:
    includes: set[str] = {
        "pybind11/pybind11.h",
        "pybind11/stl.h",
        "memory",
        "vector",
        "core/node.h",
    }

    for section in ("enums", "classes", "functions", "ops"):
        for item in spec.get(section, []):
            include = item.get("include")
            if include:
                includes.add(include)

            for extra in item.get("includes", []):
                includes.add(extra)

    return sorted(includes)


def emit_include(include: str) -> str:
    if include.startswith("<") and include.endswith(">"):
        return f"#include {include}"

    if is_system_include(include):
        return f"#include <{include}>"

    return f'#include "{include}"'


def qualified_function_expr(item: dict[str, Any]) -> str:
    namespace = item["namespace"]
    cpp = item["cpp"]
    overload = item.get("overload")

    raw = f"&{namespace}::{cpp}"

    if overload:
        return f"static_cast<{overload}>({raw})"

    return raw


def qualified_method_expr(cls: dict[str, Any], method: dict[str, Any]) -> str:
    cpp_type = cls["cpp_type"]
    cpp = method["cpp"]
    overload = method.get("overload")

    raw = f"&{cpp_type}::{cpp}"

    if overload:
        return f"static_cast<{overload}>({raw})"

    return raw


def emit_args(args: list[dict[str, Any]], indent: str) -> list[str]:
    lines: list[str] = []

    for i, arg in enumerate(args):
        comma = "," if i + 1 < len(args) else ""
        name = arg["name"]

        if "default" in arg:
            default = cpp_default(arg["default"])
            lines.append(f'{indent}py::arg("{name}") = {default}{comma}')
        else:
            lines.append(f'{indent}py::arg("{name}"){comma}')

    return lines


def emit_function_bind(
    lines: list[str],
    receiver: str,
    item: dict[str, Any],
) -> None:
    py_name = item["python"]
    args = item.get("args", [])
    fn = qualified_function_expr(item)

    lines.append(f'    {receiver}.def("{py_name}", {fn}')

    if args:
        lines[-1] += ","
        lines.extend(emit_args(args, indent="        "))
        lines.append("    );")
    else:
        lines[-1] += ");"

    lines.append("")


def emit_enum_bind(lines: list[str], enum: dict[str, Any]) -> None:
    py_name = enum["name"]
    cpp_type = enum["cpp_type"]
    values = enum.get("values", [])

    lines.append(f'    py::enum_<{cpp_type}>(m, "{py_name}")')

    if values:
        for i, value in enumerate(values):
            name = value["name"]
            cpp = value["cpp"]
            semi = ";" if i + 1 == len(values) else ""
            lines.append(f'        .value("{name}", {cpp_type}::{cpp}){semi}')
    else:
        lines[-1] += ";"

    lines.append("")

    for value in values:
        cpp = value["cpp"]

        for alias in value.get("aliases", []):
            lines.append(f'    m.attr("{alias}") = py::cast({cpp_type}::{cpp});')

    if values:
        lines.append("")


def emit_class_bind(lines: list[str], cls: dict[str, Any]) -> None:
    py_name = cls["name"]
    cpp_type = cls["cpp_type"]
    holder = cls.get("holder")

    if holder:
        lines.append(f'    py::class_<{cpp_type}, {holder}>(m, "{py_name}")')
    else:
        lines.append(f'    py::class_<{cpp_type}>(m, "{py_name}")')

    chain: list[str] = []

    for ctor in cls.get("constructors", []):
        signature = ctor.get("signature", "")
        args = ctor.get("args", [])

        if signature:
            chain.append(f"        .def(py::init<{signature}>()")
        else:
            chain.append("        .def(py::init<>()")

        if args:
            chain[-1] += ","
            chain.extend(emit_args(args, indent="            "))
            chain.append("        )")
        else:
            chain[-1] += ")"

    for prop in cls.get("properties_readonly", []):
        name = prop["name"]
        expr = prop["expr"]

        chain.append(
            f'        .def_property_readonly("{name}", '
            f'[](const {cpp_type}& self) {{ return {expr}; }})'
        )

    for prop in cls.get("properties", []):
        name = prop["name"]
        getter = prop["get"]
        setter = prop["set"]

        chain.append(
            f'        .def_property("{name}", '
            f'[](const {cpp_type}& self) {{ return {getter}; }}, '
            f'[]({cpp_type}& self, auto value) {{ {setter}; }})'
        )

    for method in cls.get("methods", []):
        name = method["name"]
        args = method.get("args", [])
        fn = qualified_method_expr(cls, method)

        if args:
            chain.append(f'        .def("{name}", {fn},')
            chain.extend(emit_args(args, indent="            "))
            chain.append("        )")
        else:
            chain.append(f'        .def("{name}", {fn})')

    if not chain:
        lines[-1] += ";"
        lines.append("")
        return

    for i, part in enumerate(chain):
        if i + 1 == len(chain):
            lines.append(part + ";")
        else:
            lines.append(part)

    lines.append("")


def generate_cpp(spec: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("// Generated file. Do not edit by hand.")
    lines.append("// Source: csrc/bindings/defs/*.yaml")
    lines.append("")

    for include in collect_includes(spec):
        lines.append(emit_include(include))

    lines.append("")
    lines.append("namespace py = pybind11;")
    lines.append("")
    lines.append("namespace ormi::bindings {")
    lines.append("")
    lines.append("void bind_generated(py::module_& m) {")
    lines.append("")

    for enum in spec.get("enums", []):
        emit_enum_bind(lines, enum)

    for cls in spec.get("classes", []):
        emit_class_bind(lines, cls)

    for fn in spec.get("functions", []):
        emit_function_bind(lines, "m", fn)

    if spec.get("ops", []):
        lines.append('    auto ops = m.def_submodule("ops");')
        lines.append("")

        for op in spec.get("ops", []):
            emit_function_bind(lines, "ops", op)

    lines.append("}")
    lines.append("")
    lines.append("} // namespace ormi::bindings")
    lines.append("")

    return "\n".join(lines)


def stub_type(value: str | None, *, context: str) -> str:
    if not value:
        return "Any"

    if "..." in value:
        raise ValueError(f"{context}: invalid stub type contains ellipsis: {value}")

    try:
        ast.parse(f"def __ormi_typecheck__(x: {value}) -> None: ...")
    except SyntaxError as exc:
        raise ValueError(f"{context}: invalid Python type annotation: {value}") from exc

    return value


def stub_return(value: str | None, *, context: str) -> str:
    if not value:
        return "Any"

    if "..." in value:
        raise ValueError(f"{context}: invalid return type contains ellipsis: {value}")

    try:
        ast.parse(f"def __ormi_typecheck__() -> {value}: ...")
    except SyntaxError as exc:
        raise ValueError(f"{context}: invalid Python return annotation: {value}") from exc

    return value


def stub_arg(arg: dict[str, Any], *, context: str) -> str:
    name = arg["name"]
    typ = stub_type(arg.get("py_type"), context=f"{context}.{name}")

    if "default" in arg:
        return f"{name}: {typ} = {py_default(arg['default'])}"

    return f"{name}: {typ}"


def stub_signature(
    py_name: str,
    args: list[dict[str, Any]],
    returns: str | None,
    *,
    context: str,
    self_arg: bool = False,
) -> str:
    parts: list[str] = []

    if self_arg:
        parts.append("self")

    for arg in args:
        parts.append(stub_arg(arg, context=context))

    ret = stub_return(returns, context=context)
    return f"def {py_name}({', '.join(parts)}) -> {ret}: ..."


def group_by_python_name(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []

    for item in items:
        py_name = item["python"]
        if py_name not in groups:
            groups[py_name] = []
            order.append(py_name)
        groups[py_name].append(item)

    return [(name, groups[name]) for name in order]


def emit_stub_function_group(lines: list[str], items: list[dict[str, Any]], *, indent: str = "") -> None:
    if not items:
        return

    overloaded = len(items) > 1

    for item in items:
        py_name = item["python"]
        args = item.get("args", [])
        returns = item.get("returns")

        if overloaded:
            lines.append(f"{indent}@overload")

        sig = stub_signature(
            py_name,
            args,
            returns,
            context=f"function {py_name}",
            self_arg=False,
        )
        lines.append(f"{indent}{sig}")

    lines.append("")


def emit_stub_enum(lines: list[str], enum: dict[str, Any]) -> None:
    name = enum["name"]
    values = enum.get("values", [])

    lines.append(f"class {name}(enum.Enum):")

    if values:
        for value in values:
            value_name = value["name"]
            lines.append(f"    {value_name}: ClassVar[{name}]")
    else:
        lines.append("    pass")

    lines.append("")

    for value in values:
        for alias in value.get("aliases", []):
            lines.append(f"{alias}: {name}")

    if values:
        lines.append("")


def emit_stub_constructor(lines: list[str], cls: dict[str, Any], ctor: dict[str, Any], *, overloaded: bool) -> None:
    class_name = cls["name"]
    args = ctor.get("args", [])

    if overloaded:
        lines.append("    @overload")

    sig = stub_signature(
        "__init__",
        args,
        "None",
        context=f"class {class_name}.__init__",
        self_arg=True,
    )
    lines.append(f"    {sig}")


def emit_stub_class(lines: list[str], cls: dict[str, Any]) -> None:
    name = cls["name"]
    constructors = cls.get("constructors", [])
    properties_readonly = cls.get("properties_readonly", [])
    properties = cls.get("properties", [])
    methods = cls.get("methods", [])

    lines.append(f"class {name}:")

    body_started = False

    if constructors:
        overloaded = len(constructors) > 1

        for ctor in constructors:
            emit_stub_constructor(lines, cls, ctor, overloaded=overloaded)
            body_started = True

        lines.append("")

    for prop in properties_readonly:
        prop_name = prop["name"]
        prop_type = stub_return(prop.get("py_type"), context=f"class {name}.{prop_name}")

        lines.append("    @property")
        lines.append(f"    def {prop_name}(self) -> {prop_type}: ...")
        body_started = True

    for prop in properties:
        prop_name = prop["name"]
        prop_type = stub_return(prop.get("py_type"), context=f"class {name}.{prop_name}")

        lines.append("    @property")
        lines.append(f"    def {prop_name}(self) -> {prop_type}: ...")
        lines.append(f"    @{prop_name}.setter")
        lines.append(f"    def {prop_name}(self, value: {prop_type}) -> None: ...")
        body_started = True

    if (properties_readonly or properties) and methods:
        lines.append("")

    method_groups: dict[str, list[dict[str, Any]]] = {}
    method_order: list[str] = []

    for method in methods:
        method_name = method["name"]
        if method_name not in method_groups:
            method_groups[method_name] = []
            method_order.append(method_name)
        method_groups[method_name].append(method)

    for method_name in method_order:
        group = method_groups[method_name]
        overloaded = len(group) > 1

        for method in group:
            if overloaded:
                lines.append("    @overload")

            sig = stub_signature(
                method_name,
                method.get("args", []),
                method.get("returns"),
                context=f"class {name}.{method_name}",
                self_arg=True,
            )
            lines.append(f"    {sig}")
            body_started = True

    if not body_started:
        lines.append("    pass")

    lines.append("")


def collect_top_level_all(spec: dict[str, Any]) -> list[str]:
    names: list[str] = []

    def add(name: str) -> None:
        if name not in names:
            names.append(name)

    for enum in spec.get("enums", []):
        add(enum["name"])
        for value in enum.get("values", []):
            for alias in value.get("aliases", []):
                add(alias)

    for cls in spec.get("classes", []):
        add(cls["name"])

    for fn in spec.get("functions", []):
        add(fn["python"])

    if spec.get("ops", []):
        add("ops")

    return names


def collect_ops_all(spec: dict[str, Any]) -> list[str]:
    names: list[str] = []

    for op in spec.get("ops", []):
        py_name = op["python"]
        if py_name not in names:
            names.append(py_name)

    return names


def emit_all(lines: list[str], names: list[str]) -> None:
    rendered = ", ".join(repr(name) for name in names)
    lines.append(f"__all__: list[str] = [{rendered}]")
    lines.append("")


def generate_module_stub(spec: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import enum")
    lines.append("from collections.abc import Sequence")
    lines.append("from typing import Any, ClassVar, overload")
    lines.append("")
    lines.append("from . import ops as ops")
    lines.append("")

    emit_all(lines, collect_top_level_all(spec))

    for enum in spec.get("enums", []):
        emit_stub_enum(lines, enum)

    for cls in spec.get("classes", []):
        emit_stub_class(lines, cls)

    for _, group in group_by_python_name(spec.get("functions", [])):
        emit_stub_function_group(lines, group)

    source = "\n".join(lines)
    validate_stub_source(source, "module stub")
    return source


def generate_ops_stub(spec: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from collections.abc import Sequence")
    lines.append("from typing import Any, overload")
    lines.append("")
    lines.append("from . import Node")
    lines.append("")

    emit_all(lines, collect_ops_all(spec))

    for _, group in group_by_python_name(spec.get("ops", [])):
        emit_stub_function_group(lines, group)

    source = "\n".join(lines)
    validate_stub_source(source, "ops stub")
    return source


def validate_stub_source(source: str, label: str) -> None:
    if "..." in source:
        # Ellipsis is valid in function bodies, but not inside annotations.
        # The annotation validators catch annotation misuse. This check only rejects
        # the exact bad pattern pybind11-stubgen produced.
        if "Sequence[...]" in source or "list[...]" in source or "tuple[...]" in source:
            raise ValueError(f"{label}: invalid ellipsis type annotation generated")

    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"{label}: generated invalid Python syntax") from exc


def write_text(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--stub-out")
    parser.add_argument("--ops-stub-out")
    parser.add_argument("--py-typed")
    parser.add_argument("--defs", nargs="+", required=True)
    args = parser.parse_args()

    defs = [Path(p) for p in args.defs]

    spec = collect_defs(defs)

    out = Path(args.out)
    write_text(out, generate_cpp(spec))
    print(f"Generated {out}")

    if args.stub_out:
        stub_out = Path(args.stub_out)
        write_text(stub_out, generate_module_stub(spec))
        print(f"Generated {stub_out}")

    if args.ops_stub_out:
        ops_stub_out = Path(args.ops_stub_out)
        write_text(ops_stub_out, generate_ops_stub(spec))
        print(f"Generated {ops_stub_out}")

    if args.py_typed:
        py_typed = Path(args.py_typed)
        py_typed.parent.mkdir(parents=True, exist_ok=True)
        py_typed.touch()
        print(f"Touched {py_typed}")


if __name__ == "__main__":
    main()