"""Select the torch device. CUDA is optional. MPS is preferred on Apple Silicon."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("kynovar.device")


def select_device(requested: str = "auto") -> torch.device:
    """Return a torch device.

    `auto` uses MPS, then CUDA, then CPU. An explicit device name is used as
    given and raises if that backend is unavailable.
    """
    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")
        device = torch.device("mps")
    elif choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        device = torch.device("cuda")
    elif choice == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unknown device {requested!r}. Use auto, cpu, mps, or cuda.")
    logger.info("selected device %s", device)
    print(f"device={device}", flush=True)
    return device
