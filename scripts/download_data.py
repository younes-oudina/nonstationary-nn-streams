"""Download the test splits of MNIST and CIFAR-10 used by the plasticity benchmarks.

The official test split (10,000 images for each dataset) is the fixed image set
shared by every task of the stream.

Usage:
    python scripts/download_data.py                      # both datasets into data/
    python scripts/download_data.py --datasets mnist     # MNIST only
    python scripts/download_data.py --root /content/data # custom location (e.g. Colab)

Only the raw files are downloaded here. Building the task streams
(random labels, permutations) is done in src/softresets/data/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets

DATASETS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
}


def download(name: str, root: Path) -> None:
    ds = DATASETS[name](root=root, train=False, download=True)
    print(f"[{name}] test: {len(ds)} images, shape {tuple(ds.data.shape[1:])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data"), help="download directory (default: data/)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="datasets to download (default: all)",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        download(name, args.root)
    print(f"Done. Files in {args.root.resolve()}")


if __name__ == "__main__":
    main()
