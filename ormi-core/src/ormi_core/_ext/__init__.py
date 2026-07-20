import os
from pathlib import Path

_dll_directories = []

if os.name == "nt":
    extension_dir = Path(__file__).resolve().parent
    _dll_directories.append(os.add_dll_directory(str(extension_dir)))

    libraries_dir = extension_dir.parent.parent / "ormi_core.libs"
    if libraries_dir.is_dir():
        _dll_directories.append(os.add_dll_directory(str(libraries_dir)))
        # NVRTC locates its builtins DLL through PATH when compilation begins.
        os.environ["PATH"] = (
            str(libraries_dir) + os.pathsep + os.environ.get("PATH", "")
        )

try:
    from . import _C
except ImportError as error:
    raise ImportError(
        "ORMI Core failed to load its compiled extension.\n"
        f"Original error: {error}"
    ) from error

__all__ = ["_C"]
