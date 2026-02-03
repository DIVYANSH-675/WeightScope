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
