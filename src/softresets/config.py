"""Loading of the YAML experiment configs stored in configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file and return it as a nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)
