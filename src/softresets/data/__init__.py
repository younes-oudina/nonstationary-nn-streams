"""Datasets and non-stationary task streams."""

from softresets.data.datasets import load_image_dataset
from softresets.data.streams import PermutedMNISTStream, Task

__all__ = ["PermutedMNISTStream", "Task", "load_image_dataset"]
