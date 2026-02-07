import json

import pytest

from weightscope.artifacts import write_json


def test_invalid_json_cannot_leave_a_partial_output(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        write_json(path, {"loss": float("nan")})
    assert not path.exists()
    write_json(path, {"label": "μ", "loss": 0.5})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json(path, {"loss": 1})
    assert path.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8"))["label"] == "μ"


def test_streamed_digest_has_known_value_and_detects_changes(tmp_path):
    from weightscope.artifacts import file_digest

    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    assert (
        file_digest(path, 2) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    path.write_bytes(b"abd")
    assert file_digest(path) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(ValueError):
        file_digest(path, 0)


def test_manifest_paths_are_portable_and_sorted(tmp_path):
    from weightscope.artifacts import build_manifest

    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    manifest = build_manifest(["b.txt", "a.txt", "a.txt"], tmp_path, {"seed": 7})
    assert [row["path"] for row in manifest["files"]] == ["a.txt", "b.txt"]
    assert manifest["metadata"] == {"seed": 7}
    assert manifest["files"][0]["bytes"] == 1
    with pytest.raises(ValueError, match="inside"):
        build_manifest(["../outside.txt"], tmp_path)


def test_manifest_verification_detects_changed_and_missing_inputs(tmp_path):
    from weightscope.artifacts import build_manifest, verify_manifest

    path = tmp_path / "data.txt"
    path.write_text("original")
    manifest = build_manifest(["data.txt"], tmp_path)
    assert verify_manifest(manifest, tmp_path)["valid"]
    path.write_text("edited")
    assert verify_manifest(manifest, tmp_path)["files"][0]["status"] == "changed"
    path.unlink()
    assert verify_manifest(manifest, tmp_path)["files"][0]["status"] == "missing"
    manifest["files"][0]["path"] = "../outside.txt"
    with pytest.raises(ValueError, match="inside"):
        verify_manifest(manifest, tmp_path)


def test_metric_summary_streams_jsonl_and_ignores_boolean_flags(tmp_path):
    from weightscope.artifacts import read_records, summarize_metrics

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"loss":1,"ok":true}\n\n{"loss":3}\n')
    summary = summarize_metrics(read_records(path))
    assert summary["rows"] == 2
    assert summary["metrics"] == {
        "loss": {"count": 2, "mean": 2.0, "min": 1.0, "max": 3.0, "std": 1.0}
    }
    with pytest.raises(ValueError, match="finite"):
        summarize_metrics([{"loss": float("nan")}])


def test_csv_preserves_quoted_values_and_missing_columns(tmp_path):
    import csv

    from weightscope.artifacts import export_csv

    path = tmp_path / "metrics.csv"
    export_csv([{"loss": 1.0, "label": "first,\nsecond"}, {"loss": 2.0, "step": 1}], path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["label"] == "first,\nsecond"
    assert rows[0]["step"] == "" and rows[1]["step"] == "1"
    with pytest.raises(FileExistsError):
        export_csv([{"loss": 0}], path)


def test_comparison_reports_signed_changes_and_undefined_zero_baselines():
    from weightscope.artifacts import compare_metrics

    report = compare_metrics([{"loss": 4, "cost": 0, "old": 1}], [{"loss": 2, "cost": 1, "new": 3}])
    assert report["metrics"]["loss"]["relative_change"] == -0.5
    assert report["metrics"]["cost"]["relative_change"] is None
    assert report["only_before"] == ["old"] and report["only_after"] == ["new"]


def test_bootstrap_is_seeded_and_handles_constant_observations():
    from weightscope.artifacts import bootstrap_mean

    first = bootstrap_mean([1, 2, 4, 8], samples=100, seed=7)
    assert first == bootstrap_mean([1, 2, 4, 8], samples=100, seed=7)
    assert 1 <= first["lower"] <= first["upper"] <= 8
    constant = bootstrap_mean([3, 3, 3], samples=10)
    assert constant["lower"] == constant["upper"] == 3
    with pytest.raises(ValueError):
        bootstrap_mean([1], confidence=1)
