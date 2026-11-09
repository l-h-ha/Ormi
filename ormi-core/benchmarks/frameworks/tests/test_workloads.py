from __future__ import annotations

import unittest

import numpy as np

from benchmarks.frameworks.model import Spec
from benchmarks.frameworks.workloads import corpus, make_data

###
###
###

class WorkloadTest(unittest.TestCase):
    def test_full_suite_has_three_scales_per_workload_and_dtype(self) -> None:
        specs = corpus("full", dtypes=["float32", "float64"])

        self.assertEqual(len(specs), 24)
        self.assertEqual(len({spec.key for spec in specs}), len(specs))
        self.assertEqual(
            {spec.scale for spec in specs},
            {"small", "medium", "large"},
        )

    def test_generation_is_deterministic(self) -> None:
        spec = Spec("dense", "test", 4, 3, 2)

        first = make_data(spec).values
        second = make_data(spec).values

        self.assertEqual(first.keys(), second.keys())

        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_requested_dtype_reaches_every_input(self) -> None:
        for name in ("pointwise", "reduction", "dense", "pinn"):
            with self.subTest(workload=name):
                spec = Spec(name, "test", 4, 3, 2, dtype="float64")
                data = make_data(spec)
                self.assertTrue(all(
                    value.dtype == np.float64
                    for value in data.values.values()
                ))

    def test_workload_and_scale_filters_compose(self) -> None:
        specs = corpus(
            "full",
            workloads={"dense", "pinn"},
            scales={"large"},
        )

        self.assertEqual(
            [spec.key for spec in specs],
            [
                "dense/large/float32/b1024-f64-w32",
                "pinn/large/float32/b256-f1-w64",
            ],
        )

    def test_non_positive_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Spec("pointwise", "test", 0, 1, 1)

###
###
###

if __name__ == "__main__":
    unittest.main()
