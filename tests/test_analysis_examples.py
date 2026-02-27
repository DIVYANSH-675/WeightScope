import json
import subprocess
import sys
from pathlib import Path


def test_published_analysis_examples_are_executable(tmp_path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    workflows = [
        ["distribution", "--input", examples / "analysis-before.json", "--key", "score"],
        [
            "paired",
            "--before",
            examples / "analysis-before.json",
            "--after",
            examples / "analysis-after.json",
            "--key",
            "score",
            "--samples",
            "20",
        ],
        [
            "grouped",
            "--input",
            examples / "analysis-before.json",
            "--key",
            "score",
            "--group-key",
            "condition",
        ],
        [
            "curve",
            "--input",
            examples / "analysis-curve.json",
            "--key",
            "loss",
            "--step-key",
            "step",
        ],
    ]
    for index, arguments in enumerate(workflows):
        output = tmp_path / f"report-{index}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "weightscope.analysis_cli",
                *map(str, arguments),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert isinstance(json.loads(output.read_text()), dict)
