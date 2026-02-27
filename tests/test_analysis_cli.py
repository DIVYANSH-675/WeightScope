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


def test_paired_command_aligns_runs_and_rejects_missing_partners(tmp_path):
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text('[{"seed":1,"loss":4},{"seed":2,"loss":10}]')
    after.write_text('[{"seed":2,"loss":8},{"seed":1,"loss":2}]')
    output = tmp_path / "comparison.json"
    run_cli(
        "paired",
        "--before",
        before,
        "--after",
        after,
        "--key",
        "loss",
        "--samples",
        "20",
        "--output",
        output,
    )
    report = json.loads(output.read_text())
    assert report["pairs"]["after"] == [2, 8]
    assert report["bootstrap"]["lower"] == report["bootstrap"]["upper"] == -2
    after.write_text('[{"seed":1,"loss":2}]')
    invalid = tmp_path / "invalid.json"
    run_cli(
        "paired",
        "--before",
        before,
        "--after",
        after,
        "--key",
        "loss",
        "--output",
        invalid,
        expected=2,
    )
    assert not invalid.exists()


def test_grouped_command_keeps_experimental_conditions_separate(tmp_path):
    source = tmp_path / "metrics.jsonl"
    source.write_text(
        '{"condition":"clean","loss":1}\n{"condition":"noisy","loss":3}\n{"condition":"clean","loss":2}\n'
    )
    output = tmp_path / "groups.json"
    run_cli(
        "grouped",
        "--input",
        source,
        "--key",
        "loss",
        "--group-key",
        "condition",
        "--output",
        output,
    )
    groups = json.loads(output.read_text())["groups"]
    assert groups[0]["group"] == "clean" and groups[0]["mean"] == 1.5
    assert groups[1]["group"] == "noisy" and groups[1]["count"] == 1


def test_curve_command_orders_steps_and_exposes_causal_diagnostics(tmp_path):
    source = tmp_path / "curve.json"
    source.write_text('[{"step":2,"loss":1},{"step":0,"loss":3},{"step":1,"loss":2}]')
    output = tmp_path / "diagnostics.json"
    run_cli(
        "curve",
        "--input",
        source,
        "--key",
        "loss",
        "--step-key",
        "step",
        "--window",
        "2",
        "--threshold",
        "2",
        "--output",
        output,
    )
    curve = json.loads(output.read_text())["curve"]
    assert curve["steps"] == [0, 1, 2] and curve["rolling"] == [None, 2.5, 1.5]
    assert curve["threshold_step"] == 1 and curve["trend"]["slope"] == -1
    source.write_text('[{"step":0,"loss":1},{"step":0,"loss":2}]')
    run_cli(
        "curve",
        "--input",
        source,
        "--key",
        "loss",
        "--step-key",
        "step",
        "--output",
        tmp_path / "invalid.json",
        expected=2,
    )
