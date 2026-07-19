from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def matrix() -> np.ndarray:
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
