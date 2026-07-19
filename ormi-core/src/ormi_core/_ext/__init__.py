import os
from pathlib import Path

_dll_directories = []

if os.name == "nt":
    _dll_directories.append(
        os.add_dll_directory(str(Path(__file__).resolve().parent))
    )

try:
    from . import _C
except ImportError as error:
    raise ImportError(
        "ORMI Core failed to load its compiled extension.\n"
        f"Original error: {error}"
    ) from error

__all__ = ["_C"]
