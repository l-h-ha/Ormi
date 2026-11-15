from pathlib import Path
import tomllib

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

###
###
###

@pytest.fixture
def cuda_build():
    project = tomllib.loads((ROOT / "ormi-cuda" / "pyproject.toml").read_text(encoding="utf-8"))
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8"))
    return project["tool"]["cibuildwheel"], workflow["jobs"]["build_cuda_wheels"]

def test_windows_toolkit_and_paths_match(cuda_build):
    build, job = cuda_build
    install = next(step for step in job["steps"] if step.get("name") == "Install CUDA (Windows)")
    assert install["uses"] == "Jimver/cuda-toolkit@v0.2.36"
    assert install["with"]["cuda"] == "12.8.1"
    directory = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
    assert build["windows"]["environment"]["CUDACXX"] == f"{directory}/nvcc.exe"
    assert f'{directory};{directory}/x64' in build["windows"]["repair-wheel-command"]

def test_linux_toolkit_and_paths_match(cuda_build):
    build, _ = cuda_build
    linux = build["linux"]
    install = next(command for command in linux["before-all"] if "cuda-compiler" in command)
    packages = install.split()[4:]
    assert set(packages) == {
        "cuda-compiler-12-8", "cuda-cudart-devel-12-8", "libcublas-devel-12-8",
        "cuda-nvrtc-devel-12-8", "cuda-driver-devel-12-8", "libcurand-devel-12-8",
    }
    environment = linux["environment"]
    assert environment["CUDACXX"] == "/usr/local/cuda-12.8/bin/nvcc"
    assert "/usr/local/cuda-12.8/bin:" in environment["PATH"]
    assert environment["LD_LIBRARY_PATH"].startswith("/usr/local/cuda-12.8/lib64:")

def test_wheels_keep_existing_targets_and_add_blackwell(cuda_build):
    build, _ = cuda_build
    architectures = build["config-settings"]["cmake.define.CMAKE_CUDA_ARCHITECTURES"].split(";")
    assert architectures == [
        "50-real", "52-real", "60-real", "61-real", "70-real", "75-real",
        "80-real", "86-real", "89-real", "90", "100-real",
    ]

def test_wheel_repair_excludes_system_driver_and_separate_core(cuda_build):
    build, _ = cuda_build
    assert "--exclude libcuda.so.1 --exclude libormi_core.so" in build["linux"]["repair-wheel-command"]
    assert "--exclude nvcuda.dll --exclude ormi_core.dll" in build["windows"]["repair-wheel-command"]
