import sys as _sys

from ormi_core import *
from ormi_core import __all__ as _core_exports, __version__ as __version__

from .checkpoint import load as load, save as save
from .data import batches as batches
from .training import fit as fit

###
###
###

__all__ = [*_core_exports, "save", "load", "batches", "fit"]

# Both import styles use the same modules.
_sys.modules[__name__ + ".nn"] = nn
_sys.modules[__name__ + ".ops"] = ops
_sys.modules[__name__ + ".optim"] = optim
_sys.modules[__name__ + ".jit"] = jit
