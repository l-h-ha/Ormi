from pathlib import Path
import shlex

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

###
###
###

@pytest.fixture(scope="module")
def workflow():
    path = ROOT / ".github" / "workflows" / "python.yml"
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

@pytest.mark.parametrize("event", ["push", "pull_request"])
@pytest.mark.parametrize("pattern", [
    ".github/workflows/**", ".github/scripts/**", ".github/tests/**",
    "pyproject.toml", "pytest.ini", "conftest.py", "uv.lock", "ormi/**/*.py", "ormi/**/*.pyi", "ormi/**/py.typed",
    "ormi/pyproject.toml", "ormi/pyrightconfig.json", "ormi-core/pyproject.toml",
    "ormi-core/CMakeLists.txt", "ormi-core/CMakePresets.json", "ormi-core/csrc/**",
    "ormi-core/src/**", "ormi-core/tests/**", "ormi-core/benchmarks/frameworks/**/*.py",
    "ormi-core/release-tests/**",
    "ormi-cuda/pyproject.toml", "ormi-cuda/src/**",
])
def test_python_and_binding_changes_trigger_checks(workflow, event, pattern):
    assert pattern in workflow["on"][event]["paths"]

def test_push_and_pr_paths_match(workflow):
    assert workflow["on"]["push"]["paths"] == workflow["on"]["pull_request"]["paths"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in workflow["on"]
    assert "pull_request_target" not in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"

def test_python_matrix_stays_small(workflow):
    job = workflow["jobs"]["test"]
    assert job["strategy"]["matrix"] == {"os": ["ubuntu-24.04", "windows-2022"]}
    assert job["env"]["UV_PYTHON"] == "3.12"
    assert int(job["timeout-minutes"]) <= 15

def test_build_uses_exact_core_cache_and_cpu_only_options(workflow):
    steps = workflow["jobs"]["test"]["steps"]
    cache = next(step for step in steps if step.get("id") == "core-wheel")
    key = cache["with"]["key"]
    assert "runner.os" in key
    assert "runner.arch" in key
    assert "steps.python.outputs.python-version" in key

    for pattern in ["ormi-core/csrc/**", "ormi-core/src/**", "ormi-core/pyproject.toml",
                    "ormi-core/CMakeLists.txt", "ormi-core/CMakePresets.json", ".github/workflows/python.yml"]:
        assert pattern in key

    assert "restore-keys" not in cache["with"]

    build = next(step for step in steps if step.get("name") == "Build CPU core wheel")
    assert build["if"] == "steps.core-wheel.outputs.cache-hit != 'true'"
    command = shlex.split(build["run"])
    assert command[:6] == ["uv", "build", "--wheel", "--package", "ormi-core", "--out-dir"]

    for option in ["ORMI_ENABLE_CUDA", "ORMI_ENABLE_OPENBLAS", "ORMI_ENABLE_SLEEF",
                   "ORMI_BUILD_TESTS", "ORMI_BUILD_BENCHMARKS", "ORMI_BUILD_EXAMPLES"]:
        assert f"cmake.define.{option}=OFF" in command
        assert f'"cmake.define.{option}=OFF"' in build["run"]

    assert command.count("--config-setting") == 6

def test_tests_use_locked_dependencies_and_built_core(workflow):
    steps = workflow["jobs"]["test"]["steps"]
    commands = [step["run"] for step in steps if "run" in step]
    assert "uv sync --locked --dev --no-editable --no-install-package ormi-core --reinstall-package ormi" in commands
    install = next(command for command in commands if command.startswith("uv pip install"))
    assert "--no-index --no-deps --find-links build-python-wheels ormi-core" in install
    assert "uv pip check" in commands
    test = next(step["run"] for step in steps if step.get("name") == "Test Python core and API")
    assert "uv run --no-sync python -m pytest" in test
    assert "--import-mode" not in test
    assert "ormi-core/tests ormi/tests" in test
    assert "ormi-core/benchmarks/frameworks/tests" in test
    fallback, jit = steps[-2:]
    assert "ormi-core/release-tests/test_cpu.py" in fallback["run"]
    assert "test_packages.py::test_core_install_is_cpu_only" in fallback["run"]
    assert fallback["env"]["ORMI_CPU_JIT_CXX"] == "ormi-compiler-does-not-exist"
    assert "ormi-core/release-tests/test_jit.py" in jit["run"]
    assert "ORMI_CPU_JIT_CXX" not in jit["env"]
    assert fallback["env"]["ORMI_CPU_JIT_CACHE_DIR"] != jit["env"]["ORMI_CPU_JIT_CACHE_DIR"]

    for release in [fallback, jit]:
        assert "ormi/release-tests" in release["run"]
        assert "--import-mode" not in release["run"]
        assert release["env"]["ORMI_RELEASE_VERSION"] == "${{ steps.versions.outputs.version }}"

@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_binding_only_changes_do_not_run_cpp_matrix(workflow, event):
    path = ROOT / ".github" / "workflows" / "runtime.yml"
    runtime = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    paths = runtime["on"][event]["paths"]
    assert paths.index("!ormi-core/csrc/bindings/**") > paths.index("ormi-core/csrc/**")
    assert "ormi-core/csrc/**" in workflow["on"][event]["paths"]
