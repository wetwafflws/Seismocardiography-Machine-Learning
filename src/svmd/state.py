"""Data-carrier classes that flow between the three solver loops.

These are plain dataclasses (mutable where needed).  The backend receives
them and returns updated copies — no hidden state in the backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Output of NumpyBackend.preprocess()
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """Everything the solver needs after signal preparation."""
    T: int                       # length of mirrored signal (= 2 * save_T)
    save_T: int                  # original signal length
    omega_freqs: np.ndarray      # (T,) normalised frequency grid
    f_hat_onesided: np.ndarray   # (T,) one-sided FFT of mirrored signal (complex)
    noisepe: float               # ||f_hat_noise_onesided||^2  (for stopc==1)
    fs: float                    # 1 / save_T


# ---------------------------------------------------------------------------
# State that lives inside the ADMM inner loop
# ---------------------------------------------------------------------------

@dataclass
class InnerState:
    """Mutable state for one (mode, alpha) ADMM optimisation run."""
    n: int                       # current iteration index
    T: int                       # mirrored signal length
    N: int                       # max_iter
    u_hat_L: np.ndarray          # (N, T) complex – mode spectrum iterates
    omega_L: np.ndarray          # (N,) float   – centre frequency iterates
    lambda_mat: np.ndarray       # (N, T) complex – dual variable iterates
    udiff: float                 # convergence residual from last step
    primal_res: float = 0.0


# ---------------------------------------------------------------------------
# Accumulated solver state (outer loop, one entry per accepted mode)
# ---------------------------------------------------------------------------

@dataclass
class SVMDState:
    """Grows by one entry each time a mode is accepted by the outer loop."""
    # per-mode history
    u_hat_temp: list[np.ndarray] = field(default_factory=list)   # accepted one-sided spectra
    u_hat_i: list[np.ndarray] = field(default_factory=list)      # same — used for sum_u_i
    omega_d_temp: list[float] = field(default_factory=list)      # accepted centre frequencies
    alpha_hist: list[float] = field(default_factory=list)        # alpha at acceptance
    h_hat_rows: list[np.ndarray] = field(default_factory=list)   # mode interaction rows
    gamma_hist: list[float] = field(default_factory=list)

    # stopping-criterion histories (not all are used simultaneously)
    sigerror_hist: list[float] = field(default_factory=list)
    normind_hist: list[float] = field(default_factory=list)
    bic_hist: list[float] = field(default_factory=list)
    polm_hist: list[float] = field(default_factory=list)
    polm_temp: float | None = None

    # misc counters
    l: int = 0        # modes accepted so far (incremented after each acceptance)
    n2: int = 0       # random-omega retry counter (init_omega > 0 path)

    def sum_u_i(self) -> np.ndarray | float:
        """Sum of accepted mode spectra (0.0 scalar when empty)."""
        return np.sum(np.vstack(self.u_hat_i), axis=0) if self.u_hat_i else 0.0

    def sum_h(self, use_sum_h: bool) -> np.ndarray | float:
        """Accumulated mode-interaction term."""
        if self.h_hat_rows and use_sum_h:
            return np.sum(np.vstack(self.h_hat_rows), axis=0)
        return 0.0


# ---------------------------------------------------------------------------
# Final output of NumpyBackend.reconstruct_modes()
# ---------------------------------------------------------------------------

@dataclass
class ReconstructResult:
    u_time: np.ndarray       # (L, save_T) float
    u_hat_final: np.ndarray  # (save_T, L) complex
    omega_sorted: np.ndarray # (L,) float
