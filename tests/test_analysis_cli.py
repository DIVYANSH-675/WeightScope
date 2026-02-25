import json
import subprocess
import sys


def run_cli(*args, expected=0):
    result = subprocess.run(
        [sys.executable, "-m", "weightscope.analysis_cli", *map(str, args)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def test_distribution_command_records_statistics_and_input_digest(tmp_path):
    source = tmp_path / "metrics.json"
    source.write_text('[{"loss":1},{"loss":2},{"loss":3},{"loss":100}]')
    output = tmp_path / "distribution.json"
    run_cli("distribution", "--input", source, "--key", "loss", "--output", output)
    report = json.loads(output.read_text())
    assert report["distribution"]["median"] == 2.5
    assert len(report["source"]["sha256"]) == 64
    run_cli("distribution", "--input", source, "--key", "loss", "--output", output, expected=2)
