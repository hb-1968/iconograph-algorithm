"""Shared tensor geometry for the optional differentiable refiners."""

import torch


def _torch_determinism(seed: int = 0) -> None:
    """The determinism trio (H8): seeded, deterministic algorithms, single
    thread — nothing downstream may draw RNG."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
