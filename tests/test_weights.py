import copy

import numpy as np
import torch

from weightscope.model import (
    TinyTransformer,
    WeightRegressor,
    descriptors,
    group_split,
    load_bank,
    make_bank,
)


def transformed(model):
    other = copy.deepcopy(model)
    with torch.no_grad():
        for h in range(len(model.q)):
            # Well-conditioned, nonorthogonal invertible changes of head basis.
            a = (
                torch.eye(model.q.shape[-1], dtype=model.q.dtype)
                + torch.randn_like(model.q[h, : model.q.shape[-1]]) * 0.05
            )
            b = torch.eye(a.shape[0], dtype=a.dtype) + torch.randn_like(a) * 0.05
            other.q[h].copy_(model.q[h] @ a)
            other.k[h].copy_(model.k[h] @ torch.linalg.inv(a).T)
            other.v[h].copy_(model.v[h] @ b)
            other.o[h].copy_(torch.linalg.inv(b) @ model.o[h])
        for name in ("q", "k", "v", "o"):
            weight = getattr(other, name)
            weight.copy_(weight.flip(0))
    return other


def test_attention_and_regressor_are_basis_and_head_invariant():
    torch.manual_seed(1)
    model = TinyTransformer().double()
    other = transformed(model)
    tokens = torch.randint(12, (4, 6))
    torch.testing.assert_close(model(tokens), other(tokens))

    def features(m):
        return descriptors(*(getattr(m, n).unsqueeze(0) for n in ("q", "k", "v", "o")))

    regressor = WeightRegressor(288).double()
    torch.testing.assert_close(regressor(features(model)), regressor(features(other)))


def test_run_groups_never_leak():
    ids = np.repeat(np.arange(10), 3)
    partitions = group_split(ids, 1)
    groups = [set(ids[p]) for p in partitions]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    assert sorted(np.concatenate(partitions)) == list(range(30))


def test_bank_contains_real_checkpoints_and_regressor_learns(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "bank.npz"
    np.savez(path, **make_bank(5, 2, 3))
    x, y, ids = load_bank(path)
    assert len(x) == 10 and len(set(ids)) == 5
    assert not torch.equal(x[0], x[1])
    model = WeightRegressor(x.shape[-1])
    model.fit_normalizer(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = float((model(x) - y).square().mean().detach())
    for _ in range(20):
        loss = (model(x) - y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float((model(x) - y).square().mean().detach()) < initial


def test_invalid_run_ids_cannot_silently_drop_rows():
    import pytest

    for ids in ([0, 1, 2, 3, 4, float("nan")], [0, 1, 2, 3, 4, float("inf")], [[0, 1, 2, 3, 4]]):
        with pytest.raises(ValueError, match="run_ids"):
            group_split(ids)
    ids = np.array(["first", "second", "third", "fourth", "fifth", "first"])
    partitions = group_split(ids)
    assert sorted(np.concatenate(partitions)) == list(range(len(ids)))
    assert any(0 in part and 5 in part for part in partitions)


def test_attention_descriptors_reject_empty_and_nonfinite_weights():
    import pytest

    q = torch.zeros(2, 1, 3, 2)
    o = torch.zeros(2, 1, 2, 3)
    with pytest.raises(ValueError, match="nonempty"):
        descriptors(q[:0], q[:0], q[:0], o[:0])
    q[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        descriptors(q, q, q, o)


def test_invalid_normalization_data_cannot_corrupt_buffers():
    import pytest

    model = WeightRegressor(4)
    original = model.center.clone()
    for features in (
        torch.empty(0, 2, 4),
        torch.ones(2, 2, 3),
        torch.full((2, 2, 4), float("nan")),
    ):
        with pytest.raises(ValueError):
            model.fit_normalizer(features)
        torch.testing.assert_close(model.center, original)
    model.fit_normalizer(torch.ones(2, 2, 4))
    assert torch.isfinite(model(torch.ones(1, 2, 4))).all()


def test_prediction_rows_keep_snapshot_run_identifiers():
    from weightscope.model import prediction_rows

    rows = prediction_rows([0.7, 0.9], [0.8, 0.8], ["run-a", "run-a"])
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[1]["run_id"] == "run-a"
    np.testing.assert_allclose([row["error"] for row in rows], [-0.1, 0.1])


def test_regression_metrics_distinguish_bias_and_magnitude():
    import pytest

    from weightscope.model import regression_metrics

    report = regression_metrics([0, 2], [1, 1])
    assert report == {"mae": 1.0, "rmse": 1.0, "bias": 0.0}
    with pytest.raises(ValueError):
        regression_metrics([float("nan")], [1])
