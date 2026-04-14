"""NumpyBackend — pure-numpy implementation of IComputeBackend.

All mathematics is extracted verbatim from svmd_prototype.py.
The only change is that functions are methods and state is passed in/out
via the dataclasses defined in state.py.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..config import SVMDConfig
from ..interfaces import IComputeBackend
from ..state import InnerState, PreprocessResult, ReconstructResult, SVMDState


class NumpyBackend(IComputeBackend):
    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    def preprocess(self, signal: np.ndarray, cfg: SVMDConfig) -> PreprocessResult:
        eps = np.finfo(np.float64).eps
        signal = np.asarray(signal, dtype=np.float64)
        if signal.shape[0] % 2:
            signal = signal[:-1]
        save_T = signal.shape[0]

        if cfg.use_sgolay:
            y = savgol_filter(signal, window_length=25, polyorder=8, mode="interp")
            signoise = signal - y
        else:
            signoise = np.zeros_like(signal, dtype=np.float64)

        f = self._mirror_signal(signal)
        fnoise = self._mirror_signal(signoise)
        T = f.shape[0]
        fs = 1.0 / save_T

        t = np.arange(1, T + 1, dtype=np.float64) / T
        omega_freqs = t - 0.5 - (1.0 / T)

        f_hat = np.fft.fftshift(np.fft.fft(f))
        f_hat_onesided = f_hat.copy()
        f_hat_onesided[: T // 2] = 0

        f_hat_n = np.fft.fftshift(np.fft.fft(fnoise))
        f_hat_n_onesided = f_hat_n.copy()
        f_hat_n_onesided[: T // 2] = 0
        noisepe = float(np.linalg.norm(f_hat_n_onesided) ** 2)

        return PreprocessResult(
            T=T,
            save_T=save_T,
            omega_freqs=omega_freqs,
            f_hat_onesided=f_hat_onesided,
            noisepe=noisepe,
            fs=fs,
        )

    # ------------------------------------------------------------------
    # admm_step
    # ------------------------------------------------------------------

    def admm_step(
        self,
        inner: InnerState,
        pre: PreprocessResult,
        svmd_state: SVMDState,
        cfg: SVMDConfig,
        Alpha: float,
    ) -> InnerState:
        """One ADMM iteration.  Mutates inner in-place and returns it."""
        eps = np.finfo(np.float64).eps
        n = inner.n
        T = inner.T
        omega_freqs = pre.omega_freqs
        f_hat_onesided = pre.f_hat_onesided

        sum_h = svmd_state.sum_h(cfg.use_sum_h)
        sum_u_i = svmd_state.sum_u_i()

        # 1. Mode spectrum update
        freq_term = (Alpha ** 2) * (omega_freqs - inner.omega_L[n]) ** 4
        denom = 1.0 + freq_term * (1.0 + 2.0 * Alpha * (omega_freqs - inner.omega_L[n]) ** 2) + sum_h

        if cfg.debug_denom:
            sum_h_norm = float(np.linalg.norm(sum_h)) if isinstance(sum_h, np.ndarray) else float(abs(sum_h))
            print(
                f"[debug-denom] mode={svmd_state.l + 1} alpha={Alpha:.6g} n={n} "
                f"|sum_h|={sum_h_norm:.6e} |freq_term|={float(np.linalg.norm(freq_term)):.6e}"
            )

        numer = f_hat_onesided + freq_term * inner.u_hat_L[n, :] + inner.lambda_mat[n, :] / 2.0
        inner.u_hat_L[n + 1, :] = numer / denom

        # 2. Centre frequency update (energy-weighted)
        pos = slice(T // 2, T)
        den_omega = np.sum(np.abs(inner.u_hat_L[n + 1, pos]) ** 2)
        if den_omega > 0:
            inner.omega_L[n + 1] = float(
                np.dot(omega_freqs[pos], np.abs(inner.u_hat_L[n + 1, pos]) ** 2) / den_omega
            )
        else:
            inner.omega_L[n + 1] = inner.omega_L[n]

        # 3. Dual ascent (lambda update)
        inner_num = (
            freq_term * (f_hat_onesided - inner.u_hat_L[n + 1, :] - sum_u_i + inner.lambda_mat[n, :] / 2.0)
            - sum_u_i
        )
        inner_den = 1.0 + freq_term
        recon_hat = inner.u_hat_L[n + 1, :] + (inner_num / inner_den) + sum_u_i
        inner.lambda_mat[n + 1, :] = inner.lambda_mat[n, :] + cfg.tau * (f_hat_onesided - recon_hat)

        # 4. Convergence residuals
        diff = inner.u_hat_L[n + 1, :] - inner.u_hat_L[n, :]
        top = (1.0 / T) * np.vdot(diff, diff)
        bottom = (1.0 / T) * np.vdot(inner.u_hat_L[n, :], inner.u_hat_L[n, :])
        udiff = float(np.abs(eps + top / (bottom + eps))) if np.abs(bottom) > 0 else float(cfg.tol + eps)
        primal_vec = f_hat_onesided - recon_hat
        primal_res = float(np.linalg.norm(primal_vec) / (np.linalg.norm(f_hat_onesided) + eps))

        if cfg.debug_admm:
            dual_res = float(np.linalg.norm(diff) / (np.linalg.norm(inner.u_hat_L[n, :]) + eps))
            converging = (not cfg.fixed_iterations) and udiff <= cfg.tol and primal_res <= cfg.tol_primal
            tag = " (primal-stop)" if converging else ""
            print(
                f"[debug-admm] mode={svmd_state.l + 1} alpha={Alpha:.6g} n={n + 1} "
                f"udiff={udiff:.3e} primal={primal_res:.3e} dual={dual_res:.3e} "
                f"|lambda|={np.linalg.norm(inner.lambda_mat[n + 1, :]):.3e} "
                f"|u|={np.linalg.norm(inner.u_hat_L[n + 1, :]):.3e} "
                f"omega={inner.omega_L[n + 1]:.6f}{tag}"
            )

        inner.n += 1
        inner.udiff = udiff
        inner.primal_res = primal_res
        return inner

    # ------------------------------------------------------------------
    # reconstruct_modes
    # ------------------------------------------------------------------

    def reconstruct_modes(
        self, svmd_state: SVMDState, pre: PreprocessResult
    ) -> ReconstructResult:
        T = pre.T
        save_T = pre.save_T
        omega_arr = np.asarray(svmd_state.omega_d_temp, dtype=np.float64)

        if omega_arr.size == 0:
            return ReconstructResult(
                u_time=np.zeros((0, save_T), dtype=np.float64),
                u_hat_final=np.zeros((save_T, 0), dtype=np.complex128),
                omega_sorted=omega_arr,
            )

        L = omega_arr.shape[0]
        u_hat = np.zeros((T, L), dtype=np.complex128)
        modes_stack = np.stack(svmd_state.u_hat_temp, axis=1)

        # Positive frequencies (including shifted DC at index T//2)
        u_hat[T // 2 :, :] = modes_stack[T // 2 :, :]
        # Negative frequencies via Hermitian symmetry
        u_hat[1 : T // 2, :] = np.conj(modes_stack[T - 1 : T // 2 : -1, :])
        # Enforce real-valued Nyquist/DC bins in shifted domain
        u_hat[T // 2, :] = np.real(u_hat[T // 2, :])
        u_hat[0, :] = np.conj(u_hat[-1, :])

        u_time = np.real(np.fft.ifft(np.fft.ifftshift(u_hat, axes=0), axis=0)).T
        u_time = u_time[:, T // 4 : 3 * T // 4]

        idx = np.argsort(omega_arr)
        omega_sorted = omega_arr[idx]
        u_time = u_time[idx, :]

        u_hat_final = np.zeros((save_T, L), dtype=np.complex128)
        for i in range(L):
            u_hat_final[:, i] = np.fft.fftshift(np.fft.fft(u_time[i, :]))

        return ReconstructResult(u_time=u_time, u_hat_final=u_hat_final, omega_sorted=omega_sorted)

    # ------------------------------------------------------------------
    # Private helpers (identical to svmd_prototype.py functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _mirror_signal(sig: np.ndarray) -> np.ndarray:
        T = sig.shape[0]
        first = sig[T // 2 - 1 :: -1]
        middle = sig
        last = sig[: T // 2 - 1 : -1]
        mirrored = np.concatenate([first, middle, last], axis=0)
        assert mirrored.shape[0] == 2 * T
        return mirrored
