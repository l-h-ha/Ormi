from __future__ import annotations

import unittest

import numpy as np

from benchmarks.frameworks.measure import percentile, verify
from benchmarks.frameworks.model import Case, Observation, Spec


def case(framework: str) -> Case:
    return Case(
        framework=framework,
        tier="direct",
        mode="eager",
        prepare_ms=0.0,
        run=lambda: None,
        values=lambda: (np.empty(0),),
        observe=lambda: Observation((), ()),
        synchronize=lambda: None,
    )


class MeasurementTest(unittest.TestCase):
    def test_percentile_interpolates_ordered_samples(self) -> None:
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)

    def test_verification_compares_complete_values(self) -> None:
        spec = Spec("pointwise", "test", 1, 1, 3)
        reference = case("PyTorch")
        candidate = case("ORMI")
        values = {
            ("PyTorch", "eager"): (np.array([0.0, 1.0, 2.0]),),
            ("ORMI", "eager"): (np.array([0.0, 2.0, 1.0]),),
        }

        with self.assertRaises(AssertionError):
            verify(spec, [reference, candidate], values)


if __name__ == "__main__":
    unittest.main()
