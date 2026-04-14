"""SVMDSolver — pure control logic (the three nested loops).

This class owns no mathematics.  Every numerical operation is delegated to
an IComputeBackend.  Stopping and alpha-schedule decisions are delegated to
their respective strategy objects.

Loop structure (mirrors svmd_prototype.py exactly):
  outer: while sc2 != 1          -- extract one mode per iteration
    middle: while not schedule.is_done()  -- grow alpha
      inner: while n < N-1 and udiff > tol  -- ADMM iterations
    accept mode, update interaction term, check stopping
"""
from __future__ import annotations

import numpy as np

from .config import SVMDConfig
from .interfaces import IAlphaSchedule, IComputeBackend, IStoppingCriterion
from .schedule.alpha import LinearAlphaSchedule
from .state import InnerState, PreprocessResult, ReconstructResult, SVMDState


class SVMDSolver:
    """Orchestrates the three nested loops using injected strategies."""

    def __init__(
        self,
        backend: IComputeBackend,
        stopping: IStoppingCriterion,
        schedule: IAlphaSchedule | None = None,
    ) -> None:
        self.backend = backend
        self.stopping = stopping
        self.schedule: IAlphaSchedule = schedule or LinearAlphaSchedule()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, pre: PreprocessResult, cfg: SVMDConfig) -> ReconstructResult:
        """Run SVMD on an already-preprocessed signal and return modes."""
        eps = np.finfo(np.float64).eps
        T = pre.T
        N = int(cfg.max_iter)
        fs = pre.fs

        svmd_state = SVMDState()
        sc2 = False

        # ================================================================
        # OUTER LOOP — extract one mode per iteration
        # ================================================================
        while not sc2:
            if cfg.max_modes is not None and svmd_state.l >= cfg.max_modes:
                if cfg.verbose:
                    print(f"[svmd] hit max_modes={cfg.max_modes}; forcing stop")
                break

            # Initialise InnerState for this mode
            inner = self._init_inner(N, T, cfg, fs, svmd_state)
            self.schedule.reset(cfg)

            # ============================================================
            # MIDDLE LOOP — alpha growth schedule
            # ============================================================
            while not self.schedule.is_done(cfg):
                # ----------------------------------------------------------
                # INNER LOOP — ADMM iterations at current alpha
                # ----------------------------------------------------------
                Alpha = self.schedule.current_alpha()
                inner = self._run_admm(inner, pre, svmd_state, cfg, Alpha)

                # Advance alpha schedule (may trigger warm restart)
                self.schedule.advance(cfg)

                if self.schedule.needs_warm_restart(cfg):
                    inner = self._warm_restart(inner, N, T, cfg, self.schedule.current_alpha())

            # ============================================================
            # Accept mode
            # ============================================================
            Alpha_final = self.schedule.current_alpha()
            mode_hat, omega_mode = self._extract_mode(inner, N, cfg)

            if cfg.debug_save:
                self._print_save_debug(inner, N, svmd_state, pre, cfg, mode_hat, omega_mode, Alpha_final)

            svmd_state.u_hat_temp.append(mode_hat)
            svmd_state.omega_d_temp.append(omega_mode)
            svmd_state.alpha_hist.append(float(Alpha_final))
            svmd_state.gamma_hist.append(1.0)

            # Mode interaction row
            h_row = self.backend.compute_h_row(omega_mode, Alpha_final, pre.omega_freqs)
            svmd_state.h_hat_rows.append(h_row)
            svmd_state.u_hat_i.append(mode_hat)

            # ============================================================
            # Check stopping criterion
            # ============================================================
            sc2 = self.stopping.should_stop(svmd_state, pre, cfg)

            # Reset for next mode
            svmd_state.l += 1
            svmd_state.n2 = 0

            if cfg.verbose:
                print(f"[svmd] accepted mode={svmd_state.l} omega={omega_mode:.6f} sc2={sc2}")

            # Re-initialise omega seed for next mode
            if cfg.init_omega > 0:
                self._pick_random_omega(svmd_state, fs)

        # ================================================================
        # Reconstruct
        # ================================================================
        return self.backend.reconstruct_modes(svmd_state, pre)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_inner(
        self, N: int, T: int, cfg: SVMDConfig, fs: float, svmd_state: SVMDState
    ) -> InnerState:
        eps = np.finfo(np.float64).eps
        omega_L = np.zeros(N, dtype=np.float64)
        if cfg.init_omega == 0:
            omega_L[0] = 0.0
        else:
            omega_L[0] = float(
                np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand()))[0]
            )
        return InnerState(
            n=0,
            T=T,
            N=N,
            u_hat_L=np.zeros((N, T), dtype=np.complex128),
            omega_L=omega_L,
            lambda_mat=np.zeros((N, T), dtype=np.complex128),
            udiff=cfg.tol + eps,
        )

    def _run_admm(
        self,
        inner: InnerState,
        pre: PreprocessResult,
        svmd_state: SVMDState,
        cfg: SVMDConfig,
        Alpha: float,
    ) -> InnerState:
        """Run inner ADMM loop until convergence or max_iter."""
        while inner.n < (inner.N - 1) and (cfg.fixed_iterations or inner.udiff > cfg.tol):
            inner = self.backend.admm_step(inner, pre, svmd_state, cfg, Alpha)
            # Early stopping when both residuals satisfied
            if (
                not cfg.fixed_iterations
                and inner.udiff <= cfg.tol
                and inner.primal_res <= cfg.tol_primal
            ):
                break
        return inner

    def _warm_restart(
        self, inner: InnerState, N: int, T: int, cfg: SVMDConfig, new_alpha: float
    ) -> InnerState:
        """Warm-restart the ADMM state after an alpha increase."""
        eps = np.finfo(np.float64).eps
        n = inner.n
        omega_seed = inner.omega_L[n] if n < N else inner.omega_L[N - 1]
        temp_ud = inner.u_hat_L[n, :].copy() if n < N else inner.u_hat_L[N - 1, :].copy()

        new_omega_L = np.zeros(N, dtype=np.float64)
        new_omega_L[0] = omega_seed
        new_u_hat_L = np.zeros((N, T), dtype=np.complex128)
        new_u_hat_L[0, :] = temp_ud

        return InnerState(
            n=0,
            T=T,
            N=N,
            u_hat_L=new_u_hat_L,
            omega_L=new_omega_L,
            lambda_mat=np.zeros((N, T), dtype=np.complex128),
            udiff=cfg.tol + np.finfo(np.float64).eps,
        )

    def _extract_mode(
        self, inner: InnerState, N: int, cfg: SVMDConfig
    ) -> tuple[np.ndarray, float]:
        """Pick the accepted mode spectrum and centre frequency."""
        n = inner.n
        if cfg.save_prev_mode and n > 0:
            idx_mode = max(n - 1, 0)
            idx_omega = max(n - 1, 0)
        else:
            idx_mode = n if n < N else N - 1
            idx_omega = max(n - 1, 0)

        mode_hat = inner.u_hat_L[idx_mode, :].copy()
        omega_mode = float(
            inner.omega_L[idx_omega]
            if idx_omega < inner.omega_L.shape[0]
            else inner.omega_L[-1]
        )
        return mode_hat, omega_mode

    def _pick_random_omega(self, svmd_state: SVMDState, fs: float) -> None:
        """Pick a non-overlapping random centre frequency for the next mode."""
        ii = 0
        n2 = 0
        rand_omega = 0.0
        while ii < 1 and n2 < 300:
            rand_omega = float(
                np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand()))[0]
            )
            checkp = np.abs(np.asarray(svmd_state.omega_d_temp, dtype=np.float64) - rand_omega)
            if np.sum(checkp < 0.02) <= 0:
                ii = 1
            n2 += 1
        svmd_state.n2 = n2
        # Store seed for next _init_inner (init_omega > 0 path picks it up)
        # We write it into omega_d_temp temporarily; solver will overwrite after.
        # Actually we keep it as an attribute so _init_inner can read it.
        self._next_omega_seed: float = rand_omega

    def _print_save_debug(
        self,
        inner: InnerState,
        N: int,
        svmd_state: SVMDState,
        pre: PreprocessResult,
        cfg: SVMDConfig,
        mode_hat: np.ndarray,
        omega_mode: float,
        Alpha: float,
    ) -> None:
        eps = np.finfo(np.float64).eps
        T = pre.T
        n = inner.n
        candA_hat = inner.u_hat_L[n] if n < N else inner.u_hat_L[N - 1]
        candB_hat = inner.u_hat_L[n - 1] if n - 1 >= 0 else candA_hat
        candA_omega = inner.omega_L[n] if n < inner.omega_L.shape[0] else inner.omega_L[-1]
        candB_omega = inner.omega_L[n - 1] if n - 1 >= 0 else candA_omega
        base_sum = (
            np.sum(np.stack(svmd_state.u_hat_temp, axis=0), axis=0)
            if svmd_state.u_hat_temp
            else np.zeros_like(pre.f_hat_onesided)
        )

        def _freq_err(candidate: np.ndarray) -> float:
            pos = slice(T // 2, T)
            recon = base_sum + candidate
            return float(
                np.linalg.norm(pre.f_hat_onesided[pos] - recon[pos])
                / (np.linalg.norm(pre.f_hat_onesided[pos]) + eps)
            )

        errA = _freq_err(candA_hat)
        errB = _freq_err(candB_hat)
        policy = "B" if cfg.save_prev_mode else "A"
        idx_mode = (max(n - 1, 0) if cfg.save_prev_mode and n > 0 else (n if n < N else N - 1))
        idx_omega = max(n - 1, 0)
        print(f"[save-debug] n={n} A|omega={candA_omega:.6f} err={errA:.3e} ||A||^2={np.linalg.norm(candA_hat)**2:.3e}")
        print(f"[save-debug] n={n} B|omega={candB_omega:.6f} err={errB:.3e} ||B||^2={np.linalg.norm(candB_hat)**2:.3e}")
        print(f"[save-debug] policy={policy} idx_mode={idx_mode} idx_omega={idx_omega}")
