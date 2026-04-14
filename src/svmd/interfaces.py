"""Abstract interfaces (Strategy contracts) for the SVMD package.

Three swappable strategies:
  IComputeBackend    -- pure math (preprocess / ADMM step / reconstruct)
  IStoppingCriterion -- decide when to stop extracting modes
  IAlphaSchedule     -- manage the alpha growth schedule inside the middle loop
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import SVMDConfig
from .state import InnerState, PreprocessResult, ReconstructResult, SVMDState


# ---------------------------------------------------------------------------
# IComputeBackend
# ---------------------------------------------------------------------------

class IComputeBackend(ABC):
    """All numerical work lives here.  Swap NumpyBackend -> CppBackend freely."""

    @abstractmethod
    def preprocess(self, signal: np.ndarray, cfg: SVMDConfig) -> PreprocessResult:
        """Mirror-extend, FFT, build one-sided spectrum, noise power estimate."""
        ...

    @abstractmethod
    def admm_step(self, inner: InnerState, pre: PreprocessResult,
                  svmd_state: SVMDState, cfg: SVMDConfig,
                  Alpha: float) -> InnerState:
        """Advance the ADMM iteration by one step (n -> n+1).

        Returns a new InnerState (or mutates and returns the same object).
        The returned state's .n is incremented.
        """
        ...

    @abstractmethod
    def reconstruct_modes(self, svmd_state: SVMDState,
                          pre: PreprocessResult) -> ReconstructResult:
        """Build final time-domain modes from accepted one-sided spectra."""
        ...

    # Optional: mode-interaction row (h_row) computation.
    # Default implementation is in NumpyBackend; CppBackend may override.
    def compute_h_row(self, omega_mode: float, alpha: float,
                      omega_freqs: np.ndarray) -> np.ndarray:
        eps = np.finfo(np.float64).eps
        return 1.0 / ((alpha ** 2) * (omega_freqs - omega_mode) ** 4 + eps)


# ---------------------------------------------------------------------------
# IStoppingCriterion
# ---------------------------------------------------------------------------

class IStoppingCriterion(ABC):
    """Decide after each accepted mode whether to stop the outer loop."""

    @abstractmethod
    def should_stop(self, svmd_state: SVMDState,
                    pre: PreprocessResult,
                    cfg: SVMDConfig) -> bool:
        """Return True to stop the outer extraction loop."""
        ...


# ---------------------------------------------------------------------------
# IAlphaSchedule
# ---------------------------------------------------------------------------

class IAlphaSchedule(ABC):
    """Control the alpha growth schedule in the middle loop."""

    @abstractmethod
    def reset(self, cfg: SVMDConfig) -> None:
        """Called at the start of each new mode extraction."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Called once each time the inner ADMM loop converges."""
        ...

    @abstractmethod
    def current_alpha(self) -> float:
        """Return the current alpha value."""
        ...

    @abstractmethod
    def is_done(self, cfg: SVMDConfig) -> bool:
        """Return True when alpha has reached max_alpha (middle loop exits)."""
        ...

    @abstractmethod
    def needs_warm_restart(self, cfg: SVMDConfig) -> bool:
        """True when alpha was just updated and the inner loop must restart."""
        ...
