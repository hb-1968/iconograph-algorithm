"""Torch-side utilities shared by diff_fit and render_fit (H5 lift).

Both refiners are lazy/optional — this module imports torch at module level
and must only ever be imported from inside their lazy paths (never from
render.py / decompose.py top level)."""

import torch


def _torch_determinism(seed: int = 0) -> None:
    """The determinism trio (H8): seeded, deterministic algorithms, single
    thread — nothing downstream may draw RNG."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
