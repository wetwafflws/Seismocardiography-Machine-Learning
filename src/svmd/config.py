"""SVMDConfig — all algorithm parameters in one frozen dataclass."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SVMDConfig:
    """All SVMD hyperparameters.  Immutable so it can be shared safely."""

    max_alpha: float
    tau: float
    tol: float
    stopc: int = 4
    init_omega: int = 0
    min_alpha: float = 10.0
    max_iter: int = 300
    max_modes: int | None = None
    use_sgolay: bool = True
    fixed_iterations: bool = False
    save_prev_mode: bool = False
    use_sum_h: bool = True
    tol_primal: float = 1e-3
    # debug flags
    debug_save: bool = False
    debug_admm: bool = False
    debug_denom: bool = False
    verbose: bool = False
