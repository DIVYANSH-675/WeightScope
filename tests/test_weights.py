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
