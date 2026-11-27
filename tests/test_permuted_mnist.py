"""Sanity checks on the Permuted MNIST stream (synthetic data, no download)."""

from pathlib import Path

import pytest
import torch

from softresets.config import load_config
from softresets.data import PermutedMNISTStream

N, D = 100, 784


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    return torch.rand(N, D, generator=g), torch.randint(0, 10, (N,), generator=g)


def make(data, **kw):
    kw = {"num_tasks": 5, "batch_size": 16, "seed": 0} | kw
    return PermutedMNISTStream(*data, **kw)


def test_permutation_is_valid_and_applied(data):
    x, y = data
    task = make(data).task(2)
    assert sorted(task.permutation.tolist()) == list(range(D))
    assert torch.equal(task.x, x[:, task.permutation])
    assert torch.equal(task.y, y)  # labels never change
    inverse = torch.argsort(task.permutation)
    assert torch.equal(task.x[:, inverse], x)


def test_tasks_differ_and_are_reproducible(data):
    s1, s2, other = make(data), make(data), make(data, seed=1)
    perms = [s1.permutation(k) for k in range(5)]
    assert all(not torch.equal(perms[i], perms[j]) for i in range(5) for j in range(i + 1, 5))
    assert all(torch.equal(s1.permutation(k), s2.permutation(k)) for k in range(5))
    assert not torch.equal(s1.permutation(0), other.permutation(0))


def test_first_task_identity_option(data):
    assert torch.equal(make(data, permute_first_task=False).permutation(0), torch.arange(D))


@pytest.mark.parametrize("epochs", [1, 3])
def test_batches_cover_every_sample_once_per_epoch(data, epochs):
    task = make(data, epochs_per_task=epochs).task(0)
    batches = list(task.batches())
    assert len(batches) == task.num_steps == epochs * 7  # ceil(100 / 16) = 7
    assert all(len(b[0]) == 16 for b in batches[:6]) and len(batches[6][0]) == 100 - 6 * 16
    seen = torch.cat([b[0] for b in batches[:7]])
    assert torch.equal(seen.sort(dim=0).values, task.x.sort(dim=0).values)
    # batch order is reproducible
    assert all(torch.equal(a[0], b[0]) for a, b in zip(batches, task.batches()))


def test_from_config_matches_paper_setup(data):
    cfg = load_config(Path(__file__).parents[1] / "configs" / "permuted_mnist.yaml")
    stream = PermutedMNISTStream.from_config(*data, cfg["stream"], seed=cfg["seeds"][0])
    assert (stream.num_tasks, stream.batch_size, stream.epochs_per_task) == (500, 16, 1)
    assert cfg["data"]["num_images"] // cfg["stream"]["batch_size"] == 625
