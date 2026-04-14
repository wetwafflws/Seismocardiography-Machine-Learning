"""Concrete stopping criteria — one class per stopc value.

stopc=1  StopNoise    noise-residual power threshold
stopc=2  StopRecon    normalised reconstruction index
stopc=3  StopBIC      Bayesian Information Criterion
stopc=4  StopPolm     mode-power ratio (MATLAB default)
"""
from __future__ import annotations

import numpy as np

from ..config import SVMDConfig
from ..interfaces import IStoppingCriterion
from ..state import PreprocessResult, SVMDState


class StopNoise(IStoppingCriterion):
    """stopc == 1: stop when residual spectral power <= noise power."""

    def should_stop(self, svmd_state: SVMDState,
                    pre: PreprocessResult, cfg: SVMDConfig) -> bool:
        if svmd_state.n2 >= 300:
            return True
        sum_u_i = svmd_state.sum_u_i()
        sigerror = float(np.linalg.norm(pre.f_hat_onesided - sum_u_i) ** 2)
        svmd_state.sigerror_hist.append(sigerror)
        return sigerror <= round(pre.noisepe)


class StopRecon(IStoppingCriterion):
    """stopc == 2: stop when normalised reconstruction index < 0.005."""

    def should_stop(self, svmd_state: SVMDState,
                    pre: PreprocessResult, cfg: SVMDConfig) -> bool:
        if svmd_state.n2 >= 300:
            return True
        eps = np.finfo(np.float64).eps
        T = pre.T
        sum_u = (
            np.sum(np.vstack(svmd_state.u_hat_temp), axis=0)
            if svmd_state.u_hat_temp
            else np.zeros(T, dtype=np.complex128)
        )
        normind = float(
            ((1.0 / T) * (np.linalg.norm(sum_u - pre.f_hat_onesided) ** 2))
            / (((1.0 / T) * (np.linalg.norm(pre.f_hat_onesided) ** 2)) + eps)
        )
        svmd_state.normind_hist.append(normind)
        return normind < 0.005


class StopBIC(IStoppingCriterion):
    """stopc == 3: stop when BIC increases (over-decomposed)."""

    def should_stop(self, svmd_state: SVMDState,
                    pre: PreprocessResult, cfg: SVMDConfig) -> bool:
        eps = np.finfo(np.float64).eps
        T = pre.T
        l = svmd_state.l
        sum_u_i = svmd_state.sum_u_i()
        sigerror = float(np.linalg.norm(pre.f_hat_onesided - sum_u_i) ** 2)
        svmd_state.sigerror_hist.append(sigerror)
        bic = float(2 * T * np.log(sigerror + eps) + (3 * (l + 1)) * np.log(2 * T))
        svmd_state.bic_hist.append(bic)
        return l > 0 and svmd_state.bic_hist[-1] > svmd_state.bic_hist[-2]


class StopPolm(IStoppingCriterion):
    """stopc == 4 (default): stop when mode-power ratio stabilises."""

    def should_stop(self, svmd_state: SVMDState,
                    pre: PreprocessResult, cfg: SVMDConfig) -> bool:
        eps = np.finfo(np.float64).eps
        l = svmd_state.l
        cur_mode = svmd_state.u_hat_i[-1]
        omega_mode = svmd_state.omega_d_temp[-1]
        # Prototype resets Alpha to min_alpha BEFORE this check (prototype line 321).
        # So the polm formula uses min_alpha, not the accepted alpha.
        alpha = cfg.min_alpha
        omega_freqs = pre.omega_freqs

        cur_polm = float(
            np.linalg.norm(
                (4.0 * alpha * cur_mode / (1.0 + 2.0 * alpha * (omega_freqs - omega_mode) ** 2))
                @ np.conj(cur_mode).T
            )
        )

        if l < 1:
            svmd_state.polm_temp = cur_polm if cur_polm != 0 else 1.0
            svmd_state.polm_hist.append(cur_polm / max(cur_polm, eps))
        else:
            svmd_state.polm_hist.append(cur_polm / ((svmd_state.polm_temp or 1.0) + eps))

        return l > 0 and abs(svmd_state.polm_hist[-1] - svmd_state.polm_hist[-2]) < 0.001


def make_stopping_criterion(stopc: int) -> IStoppingCriterion:
    """Factory: create the right stopping criterion from the stopc integer."""
    mapping = {1: StopNoise, 2: StopRecon, 3: StopBIC, 4: StopPolm}
    cls = mapping.get(stopc)
    if cls is None:
        raise ValueError(f"Unknown stopc={stopc}. Choose from {{1, 2, 3, 4}}.")
    return cls()
