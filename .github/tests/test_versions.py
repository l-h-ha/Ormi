import json
from pathlib import Path
import runpy

import pytest

check_versions = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "check_versions.py"))["check_versions"]

###
###
###

@pytest.fixture
def projects():
    return {
        "ormi": {"version": "0.1.0a6", "dependencies": ["ormi-core==0.1.0a6"], "cuda": ["ormi-cuda==0.1.0a6"]},
        "ormi-core": {"version": "0.1.0a6", "dependencies": ["numpy>=1.24"], "cuda": ["ormi-cuda==0.1.0a6"]},
        "ormi-cuda": {"version": "0.1.0a6", "dependencies": ["ormi-core==0.1.0a6"], "cuda": []},
    }

def write_projects(root, projects):
    for name, project in projects.items():
        directory = root / name
        directory.mkdir(exist_ok=True)
        content = (
            f"[project]\nname = {json.dumps(project.get('name', name))}\n"
            f"version = {json.dumps(project['version'])}\n"
            f"dependencies = {json.dumps(project['dependencies'])}\n"
            f"[project.optional-dependencies]\ncuda = {json.dumps(project['cuda'])}\n"
        )
        (directory / "pyproject.toml").write_text(content, encoding="utf-8")

###
###
###

def test_matching_versions_and_tag(tmp_path, projects):
    write_projects(tmp_path, projects)
    assert check_versions(tmp_path) == "0.1.0a6"
    assert check_versions(tmp_path, "ormi-core-v0.1.0a6") == "0.1.0a6"

@pytest.mark.parametrize("package", ["ormi", "ormi-core", "ormi-cuda"])
def test_rejects_mismatched_versions(tmp_path, projects, package):
    projects[package]["version"] = "0.1.0a7"
    write_projects(tmp_path, projects)

    with pytest.raises(ValueError, match="version"):
        check_versions(tmp_path)

@pytest.mark.parametrize("package,group", [
    ("ormi", "dependencies"), ("ormi", "cuda"), ("ormi-core", "cuda"), ("ormi-cuda", "dependencies"),
])
@pytest.mark.parametrize("change", ["missing", "duplicate", "old", "range", "marker", "extra", "url"])
def test_rejects_incorrect_pins(tmp_path, projects, package, group, change):
    dependency = "ormi-cuda" if group == "cuda" else "ormi-core"
    replacements = {
        "missing": [],
        "duplicate": [f"{dependency}==0.1.0a6"] * 2,
        "old": [f"{dependency}==0.1.0a5"],
        "range": [f"{dependency}>=0.1.0a6"],
        "marker": [f"{dependency}==0.1.0a6; python_version >= '3.11'"],
        "extra": [f"{dependency}[cuda]==0.1.0a6"],
        "url": [f"{dependency} @ https://example.invalid/package.whl"],
    }
    projects[package][group] = replacements[change]
    write_projects(tmp_path, projects)

    with pytest.raises(ValueError, match=dependency):
        check_versions(tmp_path)

def test_accepts_normalized_dependency_names(tmp_path, projects):
    projects["ormi"]["dependencies"] = ["ORMI_core == 0.1.0a6"]
    write_projects(tmp_path, projects)
    assert check_versions(tmp_path) == "0.1.0a6"

def test_rejects_wrong_project_name(tmp_path, projects):
    projects["ormi"]["name"] = "other"
    write_projects(tmp_path, projects)

    with pytest.raises(ValueError, match="project name"):
        check_versions(tmp_path)

def test_rejects_wrong_tag(tmp_path, projects):
    write_projects(tmp_path, projects)

    with pytest.raises(ValueError, match="Release tag"):
        check_versions(tmp_path, "ormi-core-v0.1.0a5")

def test_rejects_invalid_version(tmp_path, projects):
    projects["ormi-core"]["version"] = "invalid"
    write_projects(tmp_path, projects)

    with pytest.raises(ValueError):
        check_versions(tmp_path)
