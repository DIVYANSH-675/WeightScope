import json
import subprocess
import sys


def test_artifact_commands_round_trip(tmp_path):
    data = tmp_path / "data.json"
    data.write_text('[{"value":1},{"value":3}]')

    def run(*args, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "weightscope.artifact_cli", *map(str, args)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == expected, result.stdout + result.stderr

    manifest = tmp_path / "manifest.json"
    run("manifest", "--inputs", "data.json", "--base", tmp_path, "--output", manifest)
    run(
        "verify",
        "--input",
        manifest,
        "--base",
        tmp_path,
        "--output",
        tmp_path / "verification.json",
    )
    run("summary", "--input", data, "--output", tmp_path / "summary.json")
    assert json.loads((tmp_path / "summary.json").read_text())["metrics"]["value"]["mean"] == 2
    run("csv", "--input", data, "--output", tmp_path / "report.csv")
    run("compare", "--before", data, "--after", data, "--output", tmp_path / "comparison.json")
    run(
        "bootstrap",
        "--input",
        data,
        "--key",
        "value",
        "--samples",
        "10",
        "--output",
        tmp_path / "interval.json",
    )
    data.write_text('[{"value":9}]')
    run(
        "verify",
        "--input",
        manifest,
        "--base",
        tmp_path,
        "--output",
        tmp_path / "changed.json",
        expected=1,
    )
