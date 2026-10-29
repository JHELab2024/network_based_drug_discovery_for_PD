"""
Device policy: auto-detect with CPU fallback, never a hard-coded cuda.

`resolve_device(None)`   -> cuda if actually usable, else cpu
`resolve_device("cuda")` -> cuda if usable; otherwise warn and fall back to cpu
`resolve_device("cpu")`  -> cpu

Asking for cuda on a machine without it is a warning, not an error: a screen that
runs slowly is more useful than one that refuses to run, and the R caller is not
in a position to probe the GPU before choosing.
"""
import warnings

import torch


def cuda_available():
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:                      # driver/library mismatch -> treat as absent
        return False


def resolve_device(device=None):
    """str|torch.device|None -> torch.device, honouring availability."""
    if device is None:
        return torch.device("cuda" if cuda_available() else "cpu")

    dev = torch.device(device)
    if dev.type == "cuda" and not cuda_available():
        warnings.warn("cuda requested but not available; falling back to cpu", RuntimeWarning)
        return torch.device("cpu")
    return dev


def describe(device):
    """Human-readable device label for run reports."""
    dev = torch.device(device)
    if dev.type == "cuda":
        try:
            return "cuda (%s)" % torch.cuda.get_device_name(dev.index or 0)
        except Exception:
            return "cuda"
    return "cpu"
