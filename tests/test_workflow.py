import json
import subprocess
import sys


def test_command_line_round_trip(tmp_path):
    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "weightscope", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr

    run("bank", "--count", "5", "--steps", "2", "--output", tmp_path / "bank.npz")
    run("train", "--data", tmp_path / "bank.npz", "--epochs", "2", "--output", tmp_path / "train")
    run(
        "evaluate",
        "--data",
        tmp_path / "bank.npz",
        "--checkpoint",
        tmp_path / "train/best.pt",
        "--output",
        tmp_path / "report.json",
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert 0 <= report["mae"] <= 1 and report["count"] == 10
    split = json.loads((tmp_path / "train/split.json").read_text())
    assert not set(split["train"]["run_ids"]) & set(split["test"]["run_ids"])
