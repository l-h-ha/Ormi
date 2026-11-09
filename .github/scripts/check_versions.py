from __future__ import annotations

import os
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

###
###
###

def check_versions(root: Path, tag: str | None = None) -> str:
    projects = {}

    for name in ("ormi", "ormi-core", "ormi-cuda"):
        with (root / name / "pyproject.toml").open("rb") as stream:
            projects[name] = tomllib.load(stream)["project"]

        if projects[name]["name"] != name:
            raise ValueError(f"{name}/pyproject.toml must declare project name {name!r}")

    version = str(Version(projects["ormi-core"]["version"]))

    for name, project in projects.items():
        if project["version"] != version:
            raise ValueError(f"{name} version is {project['version']!r}, but expected {version!r}")

    pins = (
        ("ormi", None, "ormi-core"),
        ("ormi", "cuda", "ormi-cuda"),
        ("ormi-core", "cuda", "ormi-cuda"),
        ("ormi-cuda", None, "ormi-core"),
    )

    for name, extra, dependency in pins:
        project = projects[name]
        source = f"{name}[{extra}]" if extra else name
        group = project.get("dependencies", [])

        if extra is not None:
            group = project.get("optional-dependencies", {}).get(extra, [])

        requirements = [Requirement(value) for value in group]
        matches = [value for value in requirements if canonicalize_name(value.name) == dependency]
        expected = f"{dependency}=={version}"

        if len(matches) != 1:
            raise ValueError(f"{source} must declare exactly one {expected} dependency")

        requirement = matches[0]

        if (
            str(requirement.specifier) != f"=={version}"
            or requirement.url is not None
            or requirement.marker is not None
            or requirement.extras
        ):
            raise ValueError(f"{source} must require {expected} without extras or conditions")

    if tag is not None and tag != f"ormi-core-v{version}":
        raise ValueError(f"Release tag is {tag!r}, but expected 'ormi-core-v{version}'")

    return version

###
###
###

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    tag = os.environ.get("GITHUB_REF_NAME") if os.environ.get("GITHUB_REF_TYPE") == "tag" else None

    try:
        version = check_versions(root, tag)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from None

    print(f"ormi, ormi-core, and ormi-cuda: {version}")

    output = os.environ.get("GITHUB_OUTPUT")

    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(f"version={version}\n")
