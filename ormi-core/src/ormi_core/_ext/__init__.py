try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "ORMI Core failed to load the native backend.\n"
        f"Original error: {e}"
    ) from e

__all__ = ["_C"]