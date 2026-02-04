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
