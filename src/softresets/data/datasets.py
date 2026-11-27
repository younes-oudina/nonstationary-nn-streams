"""Loading of the fixed image sets shared by every task of a stream."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets

_DATASETS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
}


def load_image_dataset(
    name: str,
    root: str | Path = "data",
    split: str = "test",
    num_images: int | None = None,
    subset_seed: int = 0,
    normalize: bool = True,
    flatten: bool = False,
    device: str | torch.device = "cpu",
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X, y)`` as tensors on ``device``.

    ``X`` has shape ``(N, C, H, W)`` (or ``(N, C*H*W)`` if ``flatten``), float in [0, 1]
    if ``normalize``. If ``num_images`` is smaller than the split, a random subset is
    drawn with ``subset_seed`` (fixed for all tasks and all seeds of the stream).
    """
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(_DATASETS)}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    ds = _DATASETS[name](root=root, train=split == "train", download=download)
    data = torch.as_tensor(ds.data)
    X = data.unsqueeze(1) if data.ndim == 3 else data.permute(0, 3, 1, 2)  # (N, C, H, W)
    y = torch.as_tensor(ds.targets, dtype=torch.long)

    if num_images is not None and num_images < len(X):
        g = torch.Generator().manual_seed(subset_seed)
        idx = torch.randperm(len(X), generator=g)[:num_images]
        X, y = X[idx], y[idx]

    X = X.float()
    if normalize:
        X = X / 255.0
    if flatten:
        X = X.flatten(start_dim=1)
    return X.to(device), y.to(device)
