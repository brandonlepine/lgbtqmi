"""I/O utilities: atomic saves, resume checking, path helpers."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def atomic_save_json(data: Any, path: str | Path) -> None:
    """Write JSON atomically: write to .tmp then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.rename(tmp_path, path)


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save .npz atomically: write to .tmp then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    np.savez(tmp_path, **arrays)
    # np.savez appends .npz to the path if it doesn't end with it
    actual_tmp = tmp_path if tmp_path.suffix == ".npz" else tmp_path.with_suffix(".tmp.npz")
    if actual_tmp.exists():
        os.rename(actual_tmp, path)
    elif tmp_path.exists():
        os.rename(tmp_path, path)


def item_exists(output_dir: str | Path, item_idx: int) -> bool:
    """Check if an item's .npz file already exists (for resume safety)."""
    path = Path(output_dir) / f"item_{item_idx:04d}.npz"
    return path.exists()


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(base: str | Path, model_id: str, date: str) -> Path:
    """Construct the standard run directory path."""
    return Path(base) / "runs" / model_id / date
