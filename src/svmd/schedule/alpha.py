"""LinearAlphaSchedule — the MATLAB-aligned alpha growth schedule.

Logic extracted verbatim from svmd_prototype.py lines 265-289.

The schedule lives entirely in Python control code.  The backend never
touches Alpha directly, so swapping to CppBackend needs no changes here.
"""
from __future__ import annotations

import numpy as np

from ..config import SVMDConfig
from ..interfaces import IAlphaSchedule
from ..state import SVMDState


class LinearAlphaSchedule(IAlphaSchedule):
    """Mimics the MATLAB alpha-growth loop exactly.

    Internal state:
      _alpha   -- current alpha value
      _m       -- log-scale counter (starts at 0.0 per mode)
      _bf      -- near-max counter (triggers integer steps)
      _dirty   -- True when alpha was just changed (warm-restart needed)
    """

    def __init__(self) -> None:
        self._alpha: float = 10.0
        self._m: float = 0.0
        self._bf: int = 0
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # IAlphaSchedule interface
    # ------------------------------------------------------------------

    def reset(self, cfg: SVMDConfig) -> None:
        """Called at the start of each new mode extraction."""
        self._alpha = float(cfg.min_alpha)
        self._m = 0.0
        self._bf = 0
        self._dirty = False

    def step(self) -> None:
        """Called once each time the inner ADMM loop converges at current alpha."""
        self._dirty = False  # clear flag from previous step

    def current_alpha(self) -> float:
        return self._alpha

    def is_done(self, cfg: SVMDConfig) -> bool:
        return self._alpha >= cfg.max_alpha + 1

    def needs_warm_restart(self, cfg: SVMDConfig) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # Core schedule logic (called by SVMDSolver after ADMM convergence)
    # ------------------------------------------------------------------

    def advance(self, cfg: SVMDConfig) -> None:
        """Advance the alpha schedule by one step (post-convergence).

        This is the translation of the original prototype's middle-loop body:

            if abs(m - log(max_alpha)) > 1: m += 1
            else: m += 0.05; bf += 1
            if bf >= 2: Alpha += 1
            if Alpha <= max_alpha - 1:
                if bf == 1: Alpha = max_alpha - 1
                else: Alpha = exp(m)
        """
        max_alpha = cfg.max_alpha

        if abs(self._m - np.log(max_alpha)) > 1:
            self._m += 1
        else:
            self._m += 0.05
            self._bf += 1

        if self._bf >= 2:
            self._alpha += 1

        if self._alpha <= max_alpha - 1:
            if self._bf == 1:
                self._alpha = max_alpha - 1
            else:
                self._alpha = float(np.exp(self._m))
            self._dirty = True   # warm restart required
        else:
            self._dirty = False  # alpha reached ceiling, no restart
