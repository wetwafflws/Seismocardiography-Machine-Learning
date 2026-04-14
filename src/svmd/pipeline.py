"""SVMDPipeline — top-level API for callers.

Usage (drop-in for svmd_prototype.svmd()):

    from svmd import SVMDConfig, SVMDPipeline

    cfg = SVMDConfig(max_alpha=2000, tau=0.0, tol=1e-7)
    u_time, u_hat_final, omega = SVMDPipeline(cfg).run(signal)

For feature extraction inside a rolling walkforward:

    features = SVMDPipeline(cfg).extract_features(window_slice)
    # -> np.ndarray shape (n_modes, n_features_per_mode)
"""
from __future__ import annotations

import numpy as np

from .config import SVMDConfig
from .backends.numpy_backend import NumpyBackend
from .interfaces import IComputeBackend, IStoppingCriterion, IAlphaSchedule
from .schedule.alpha import LinearAlphaSchedule
from .solver import SVMDSolver
from .state import ReconstructResult
from .stopping.criteria import make_stopping_criterion


class SVMDPipeline:
    """Wires together backend + stopping + schedule + solver."""

    def __init__(
        self,
        cfg: SVMDConfig,
        backend: IComputeBackend | None = None,
        stopping: IStoppingCriterion | None = None,
        schedule: IAlphaSchedule | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend: IComputeBackend = backend or NumpyBackend()
        self.stopping: IStoppingCriterion = stopping or make_stopping_criterion(cfg.stopc)
        self.schedule: IAlphaSchedule = schedule or LinearAlphaSchedule()
        self._solver = SVMDSolver(self.backend, self.stopping, self.schedule)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def run(
        self, signal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SVMD and return (u_time, u_hat_final, omega_sorted).

        Identical return signature to svmd_prototype.svmd().
        """
        pre = self.backend.preprocess(signal, self.cfg)
        result: ReconstructResult = self._solver.solve(pre, self.cfg)
        return result.u_time, result.u_hat_final, result.omega_sorted

    # ------------------------------------------------------------------
    # Feature extraction API (for rolling walkforward)
    # ------------------------------------------------------------------

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Decompose a causal window and return a feature matrix.

        Returns
        -------
        np.ndarray, shape (n_modes, 4)
            Each row: [centre_omega, mode_energy, mode_std, mode_kurtosis]
            Rows are sorted by centre_omega (lowest frequency first).
        """
        u_time, _, omega = self.run(window)
        if u_time.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float64)

        rows = []
        for i in range(u_time.shape[0]):
            mode = u_time[i]
            energy = float(np.sum(mode ** 2))
            std = float(np.std(mode))
            # kurtosis (Fisher, excess)
            if std > 0:
                kurt = float(np.mean(((mode - np.mean(mode)) / std) ** 4) - 3.0)
            else:
                kurt = 0.0
            rows.append([float(omega[i]), energy, std, kurt])

        return np.array(rows, dtype=np.float64)
