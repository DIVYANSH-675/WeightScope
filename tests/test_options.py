import sys

import pytest

from weightscope import cli


def test_invalid_learning_rate_fails_before_loading_a_bank(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["weightscope", "train", "--learning-rate", "nan"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "learning-rate must" in capsys.readouterr().err


def test_invalid_weight_decay_fails_before_loading_a_bank(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["weightscope", "train", "--weight-decay", "-1"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "weight-decay must" in capsys.readouterr().err
