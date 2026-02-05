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
