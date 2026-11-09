import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

###
###
###

@pytest.fixture(scope="module")
def type_reports():
    if importlib.util.find_spec("pyright") is None:
        pytest.skip("Pyright is not installed. Run uv sync to install the development dependencies.")

    package = Path(__file__).parents[1]
    reports = {}

    for name, files in [("valid", []), ("invalid", ["tests/typing/invalid.py"])]:
        result = subprocess.run(
            [
                sys.executable, "-m", "pyright",
                "--project", str(package / "pyrightconfig.json"),
                "--pythonpath", sys.executable,
                "--outputjson", *files,
            ],
            cwd=package,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
        )
        assert result.returncode in (0, 1), result.stdout + result.stderr
        assert not result.stderr, result.stderr
        reports[name] = json.loads(result.stdout)

    return reports

###
###
###

def test_valid_usage_has_no_type_errors_or_import_warnings(type_reports):
    report = type_reports["valid"]
    assert report["summary"]["filesAnalyzed"] == 3
    assert report["generalDiagnostics"] == [], json.dumps(report, indent=2)

def test_invalid_usage_reports_the_expected_errors(type_reports):
    path = Path(__file__).parent / "typing" / "invalid.py"
    expected = set()

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        _, marker, rules = line.partition("# error: ")

        if not marker:
            continue

        for rule in rules.split(","):
            expected.add((line_number, rule.strip()))

    report = type_reports["invalid"]
    actual = set()

    for diagnostic in report["generalDiagnostics"]:
        assert diagnostic["severity"] == "error", diagnostic
        assert Path(diagnostic["file"]).resolve() == path.resolve(), diagnostic
        actual.add((diagnostic["range"]["start"]["line"], diagnostic["rule"]))

    assert actual == expected, json.dumps(report, indent=2)
