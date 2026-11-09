from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.frameworks.run import artifact_delta, jit_artifacts, parser

###
###
###

class RunnerTest(unittest.TestCase):
    def test_label_accepts_build_and_machine_description(self) -> None:
        arguments = parser().parse_args([
            "--label",
            "windows-msvc-openblas-sleef",
        ])

        self.assertEqual(
            arguments.label,
            "windows-msvc-openblas-sleef",
        )

    def test_jit_artifacts_counts_complete_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "first.cpp").write_bytes(b"source")
            (path / "first.dll").write_bytes(b"library")
            (path / "second.cpp").write_bytes(b"other")
            (path / "unrelated.lock").write_bytes(b"")

            self.assertEqual(
                jit_artifacts(path),
                {
                    "entries": 1,
                    "source_bytes": 11,
                    "library_bytes": 7,
                },
            )

    def test_artifact_delta_reports_new_workload_artifacts(self) -> None:
        self.assertEqual(
            artifact_delta(
                {"entries": 1, "source_bytes": 20, "library_bytes": 30},
                {"entries": 3, "source_bytes": 50, "library_bytes": 90},
            ),
            {"entries": 2, "source_bytes": 30, "library_bytes": 60},
        )

###
###
###

if __name__ == "__main__":
    unittest.main()
