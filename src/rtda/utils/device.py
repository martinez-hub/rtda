from __future__ import annotations

import torch


def resolve_device(config_device: str | None) -> torch.device:
    requested = (config_device or "").strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested in {"cpu", "cuda"}:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
