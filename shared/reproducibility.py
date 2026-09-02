"""Reproducibility helpers shared by the ACIIDS 2027 experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int = 42, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch (when installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by deterministic CUDA matrix multiplications on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    state: dict[str, Any] = {
        "seed": seed,
        "python_hash_seed": str(seed),
        "deterministic_requested": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_available": False,
    }
    try:
        import torch

        state["torch_available"] = True
        state["torch_version"] = torch.__version__
        state["cuda_available"] = torch.cuda.is_available()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    return state


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
