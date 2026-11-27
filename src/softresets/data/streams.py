"""Non-stationary task streams built on top of a fixed image set.

A stream is an iterable of :class:`Task`. Each task holds the transformed images and
labels, and yields mini-batches for online training. Everything is derived from a
single integer ``seed``: the permutation of task ``k`` and the order of its batches
depend only on ``(seed, k)``, so a task can be rebuilt on its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

# Streams of random numbers derived from (seed, task_id, purpose).
_PERMUTATION, _SHUFFLE = 0, 1


def _generator(seed: int, task_id: int, purpose: int) -> torch.Generator:
    state = np.random.SeedSequence([seed, task_id, purpose]).generate_state(1, dtype=np.uint64)
    return torch.Generator().manual_seed(int(state[0]))


@dataclass
class Task:
    """One task of the stream: a fixed dataset seen for ``epochs`` epochs in batches."""

    task_id: int
    x: torch.Tensor
    y: torch.Tensor
    batch_size: int
    epochs: int
    seed: int
    permutation: torch.Tensor | None = None

    @property
    def num_steps(self) -> int:
        """Number of gradient steps (batches) in the task."""
        return self.epochs * -(-len(self.x) // self.batch_size)

    def batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(x, y)`` batches; data is reshuffled at each epoch."""
        g = _generator(self.seed, self.task_id, _SHUFFLE)
        n = len(self.x)
        for _ in range(self.epochs):
            order = torch.randperm(n, generator=g).to(self.x.device)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                yield self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return self.num_steps


class PermutedMNISTStream:
    """Permuted MNIST: each task applies a fixed random pixel permutation to all images.

    Labels are unchanged across tasks; only the input distribution shifts.

    Args:
        x: flattened images, shape ``(N, D)``.
        y: labels, shape ``(N,)``.
        num_tasks: number of tasks in the stream.
        batch_size: size of the incoming batches.
        epochs_per_task: passes over the task data before switching to the next task.
        seed: seed of the stream (permutations and batch order).
        permute_first_task: if False, task 0 uses the original pixel order.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_tasks: int,
        batch_size: int = 16,
        epochs_per_task: int = 1,
        seed: int = 0,
        permute_first_task: bool = True,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be flattened to (N, D), got shape {tuple(x.shape)}")
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples")
        self.x, self.y = x, y
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.epochs_per_task = epochs_per_task
        self.seed = seed
        self.permute_first_task = permute_first_task

    @classmethod
    def from_config(
        cls, x: torch.Tensor, y: torch.Tensor, stream_cfg: dict, seed: int
    ) -> PermutedMNISTStream:
        """Build the stream from the ``stream`` section of a config."""
        return cls(
            x,
            y,
            num_tasks=stream_cfg["num_tasks"],
            batch_size=stream_cfg["batch_size"],
            epochs_per_task=stream_cfg["epochs_per_task"],
            seed=seed,
            permute_first_task=stream_cfg.get("permute_first_task", True),
        )

    @property
    def steps_per_task(self) -> int:
        return self.epochs_per_task * -(-len(self.x) // self.batch_size)

    def permutation(self, task_id: int) -> torch.Tensor:
        """Pixel permutation of task ``task_id`` (on CPU)."""
        d = self.x.shape[1]
        if task_id == 0 and not self.permute_first_task:
            return torch.arange(d)
        return torch.randperm(d, generator=_generator(self.seed, task_id, _PERMUTATION))

    def task(self, task_id: int) -> Task:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")
        perm = self.permutation(task_id)
        return Task(
            task_id=task_id,
            x=self.x[:, perm.to(self.x.device)],
            y=self.y,
            batch_size=self.batch_size,
            epochs=self.epochs_per_task,
            seed=self.seed,
            permutation=perm,
        )

    def __iter__(self) -> Iterator[Task]:
        for k in range(self.num_tasks):
            yield self.task(k)

    def __len__(self) -> int:
        return self.num_tasks
