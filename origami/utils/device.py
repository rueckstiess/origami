"""Device utilities for ORIGAMI.

Provides consistent device detection across CPU, CUDA, and MPS.
"""

import torch


def auto_device() -> torch.device:
    """Auto-detect the best available device.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU

    Returns:
        torch.device for the best available accelerator
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device(device: torch.device | str | None = None) -> torch.device:
    """Get device, with auto-detection if not specified.

    Args:
        device: Device specification. Can be:
            - torch.device: Used as-is
            - str: Converted to torch.device (e.g., "cuda", "mps", "cpu")
            - None: Auto-detect best available device

    Returns:
        torch.device instance
    """
    if device is None:
        return auto_device()
    if isinstance(device, str):
        return torch.device(device)
    return device


def available_devices() -> list[torch.device]:
    """Get list of all available devices for testing.

    Always includes CPU. Adds CUDA and/or MPS if available.

    Returns:
        List of available torch.device instances
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices
