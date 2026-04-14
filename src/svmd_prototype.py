"""
SVMD prototype (MATLAB-aligned engineering version) with validation metrics.

Goals:
- Keep the algorithm path close to `third_party/svmd_original_demo/svmd.m`.
- Provide practical runtime guards for local testing.
- Provide structural validation beyond plain RMSE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO_DIR = ROOT.parent / "third_party" / "svmd_original_demo"


@dataclass
class ValidationReport:
    rel_recon_err: float
    freq_recon_err: float
    energy_ratio: float
    orthogonality_index: float
    residual_peak_ratio: float
    matched_modes: int
    mean_abs_corr: float
    mean_nrmse: float
    mean_spec_corr: float
    mean_spec_overlap: float
    mean_omega_diff: float
    uhat_rmse: float | None
    uhat_nrmse: float | None
    uhat_abs_nrmse: float | None
    uhat_nrmse_matlab_bug: float | None
    uhat_abs_nrmse_matlab_bug: float | None
    uhat_nrmse_corrected: float | None
    uhat_abs_nrmse_corrected: float | None


def _mirror_signal(sig: np.ndarray) -> np.ndarray:
    """Mirror extension exactly following the MATLAB index pattern."""
    T = sig.shape[0]
    first = sig[T // 2 - 1 :: -1]
    middle = sig
    last = sig[: T // 2 - 1 : -1]
    mirrored = np.concatenate([first, middle, last], axis=0)
    assert mirrored.shape[0] == 2 * T, "Mirror length mismatch"
    return mirrored


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom <= 0:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    if denom <= 0:
        return np.inf
    return float(np.linalg.norm(a - b) / denom)


def _spectral_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    sa = np.abs(np.fft.fftshift(np.fft.fft(a))) ** 2
    sb = np.abs(np.fft.fftshift(np.fft.fft(b))) ** 2
    spec_corr = _corr(sa, sb)
    overlap = float(np.sum(np.minimum(sa, sb)) / (np.sum(np.maximum(sa, sb)) + 1e-12))
    return spec_corr, overlap


def _dominant_omega(mode: np.ndarray) -> float:
    T = mode.shape[0]
    grid = np.arange(1, T + 1, dtype=np.float64) / T - 0.5 - 1.0 / T
    s = np.abs(np.fft.fftshift(np.fft.fft(mode))) ** 2
    half = s[T // 2 :]
    idx = int(np.argmax(half))
    return float(grid[T // 2 + idx])


def _match_modes_by_corr(u_calc: np.ndarray, u_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching by max absolute time-domain correlation."""
    c = np.zeros((u_calc.shape[0], u_ref.shape[0]), dtype=np.float64)
    for i in range(u_calc.shape[0]):
        for j in range(u_ref.shape[0]):
            c[i, j] = abs(_corr(u_calc[i], u_ref[j]))
    row_ind, col_ind = linear_sum_assignment(-c)
    return row_ind, col_ind


def _align_modes(u_calc: np.ndarray, u_ref: np.ndarray, ridx: np.ndarray, cidx: np.ndarray) -> np.ndarray:
    """Phase/sign alignment for validation only."""
    aligned = u_calc.astype(np.complex128).copy()
    for i, j in zip(ridx, cidx):
        ref = u_ref[j].astype(np.complex128)
        calc = aligned[i]
        phase = np.vdot(ref, calc)
        if np.abs(phase) > 0:
            aligned[i] = calc * np.exp(-1j * np.angle(phase))
    return np.real(aligned)


def svmd(
    signal: np.ndarray,
    max_alpha: float,
    tau: float,
    tol: float,
    stopc: int = 4,
    init_omega: int = 0,
    min_alpha: float = 10.0,
    max_iter: int = 300,
    max_modes: int | None = None,
    use_sgolay: bool = True,
    fixed_iterations: bool = False,
    save_prev_mode: bool = False,
    debug_save: bool = False,
    debug_admm: bool = False,
    debug_denom: bool = False,
    use_sum_h: bool = True,
    tol_primal: float = 1e-3,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-aligned SVMD translation.

    Returns:
    - u_time: (L, save_T)
    - u_hat_final: (save_T, L)
    - omega_sorted: (L,)
    """
    eps = np.finfo(np.float64).eps
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape[0] % 2:
        signal = signal[:-1]
    save_T = signal.shape[0]

    if use_sgolay:
        y = savgol_filter(signal, window_length=25, polyorder=8, mode="interp")
        signoise = signal - y
    else:
        signoise = np.zeros_like(signal, dtype=np.float64)

    # Part 1: mirror extension
    f = _mirror_signal(signal)
    fnoise = _mirror_signal(signoise)
    if verbose or debug_save:
        print(f"[svmd] mirror lengths: first={save_T//2}, middle={save_T}, last={save_T//2}, total={f.shape[0]}")
        print(f"[svmd] T={2*save_T} positive bins={(2*save_T)-(2*save_T)//2} zeroed bins={(2*save_T)//2}")

    T = f.shape[0]
    fs = 1.0 / save_T
    t = np.arange(1, T + 1, dtype=np.float64) / T
    udiff = tol + eps
    omega_freqs = t - 0.5 - (1.0 / T)

    # Part 1: FFT and one-sided spectra
    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_onesided = f_hat.copy()
    f_hat_onesided[: T // 2] = 0

    f_hat_n = np.fft.fftshift(np.fft.fft(fnoise))
    f_hat_n_onesided = f_hat_n.copy()
    f_hat_n_onesided[: T // 2] = 0
    noisepe = np.linalg.norm(f_hat_n_onesided) ** 2

    # MATLAB variable initialization
    N = int(max_iter)
    if init_omega == 0:
        omega_L = np.zeros(N, dtype=np.float64)
        omega_L[0] = 0.0
    else:
        omega_L = np.zeros(N, dtype=np.float64)
        omega_L[0] = float(np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand()))[0])

    Alpha = float(min_alpha)
    alpha_hist: list[float] = []
    lambda_mat = np.zeros((N, T), dtype=np.complex128)
    u_hat_L = np.zeros((N, T), dtype=np.complex128)
    n = 0
    m = 0.0
    sc2 = 0
    l = 0
    bf = 0
    bic_hist: list[float] = []

    h_hat_rows: list[np.ndarray] = []
    u_hat_temp: list[np.ndarray] = []
    u_hat_i: list[np.ndarray] = []
    omega_d_temp: list[float] = []
    sigerror_hist: list[float] = []
    gamma_hist: list[float] = []
    normind_hist: list[float] = []
    polm_hist: list[float] = []
    polm_temp: float | None = None
    n2 = 0

    while sc2 != 1:
        if max_modes is not None and l >= max_modes:
            if verbose:
                print(f"[svmd] hit max_modes={max_modes}; forcing stop")
            break

        while Alpha < (max_alpha + 1):
            while n < (N - 1) and (fixed_iterations or udiff > tol):
                sum_h = np.sum(np.vstack(h_hat_rows), axis=0) if (h_hat_rows and use_sum_h) else 0.0
                sum_u_i = np.sum(np.vstack(u_hat_i), axis=0) if u_hat_i else 0.0

                freq_term = (Alpha**2) * (omega_freqs - omega_L[n]) ** 4
                denom = 1.0 + freq_term * (1.0 + (2.0 * Alpha) * (omega_freqs - omega_L[n]) ** 2) + sum_h
                if debug_denom:
                    sum_h_norm = float(np.linalg.norm(sum_h)) if isinstance(sum_h, np.ndarray) else float(abs(sum_h))
                    freq_term_norm = float(np.linalg.norm(freq_term))
                    print(f"[debug-denom] mode={l+1} alpha={Alpha:.6g} n={n} |sum_h|={sum_h_norm:.6e} |freq_term|={freq_term_norm:.6e}")
                numer = f_hat_onesided + freq_term * u_hat_L[n, :] + lambda_mat[n, :] / 2.0
                u_hat_L[n + 1, :] = numer / denom

                pos = slice(T // 2, T)
                den_omega = np.sum(np.abs(u_hat_L[n + 1, pos]) ** 2)
                if den_omega > 0:
                    omega_L[n + 1] = float(np.dot(omega_freqs[pos], np.abs(u_hat_L[n + 1, pos]) ** 2) / den_omega)
                else:
                    omega_L[n + 1] = omega_L[n]

                inner_num = freq_term * (f_hat_onesided - u_hat_L[n + 1, :] - sum_u_i + lambda_mat[n, :] / 2.0) - sum_u_i
                inner_den = 1.0 + freq_term
                recon_hat = u_hat_L[n + 1, :] + (inner_num / inner_den) + sum_u_i
                lambda_mat[n + 1, :] = lambda_mat[n, :] + tau * (f_hat_onesided - recon_hat)

                diff = u_hat_L[n + 1, :] - u_hat_L[n, :]
                top = (1.0 / T) * np.vdot(diff, diff)
                bottom = (1.0 / T) * np.vdot(u_hat_L[n, :], u_hat_L[n, :])
                udiff = float(np.abs(eps + top / (bottom + eps))) if np.abs(bottom) > 0 else float(tol + eps)
                primal_vec = f_hat_onesided - recon_hat
                primal_res = float(np.linalg.norm(primal_vec) / (np.linalg.norm(f_hat_onesided) + eps))
                if not fixed_iterations and (udiff <= tol) and (primal_res <= tol_primal):
                    if debug_admm:
                        print(
                            f"[debug-admm] mode={l+1} alpha={Alpha:.6g} n={n+1} "
                            f"udiff={udiff:.3e} primal={primal_res:.3e} dual=NA "
                            f"|lambda|={np.linalg.norm(lambda_mat[n + 1, :]):.3e} |u|={np.linalg.norm(u_hat_L[n + 1, :]):.3e} "
                            f"omega={omega_L[n + 1]:.6f} (primal-stop)"
                        )
                    n += 1
                    break
                if debug_admm:
                    dual_res = float(np.linalg.norm(diff) / (np.linalg.norm(u_hat_L[n, :]) + eps))
                    lam_norm = float(np.linalg.norm(lambda_mat[n + 1, :]))
                    u_norm = float(np.linalg.norm(u_hat_L[n + 1, :]))
                    print(
                        f"[debug-admm] mode={l+1} alpha={Alpha:.6g} n={n+1} "
                        f"udiff={udiff:.3e} primal={primal_res:.3e} dual={dual_res:.3e} "
                        f"|lambda|={lam_norm:.3e} |u|={u_norm:.3e} omega={omega_L[n + 1]:.6f}"
                    )
                n += 1

            if abs(m - np.log(max_alpha)) > 1:
                m = m + 1
            else:
                m = m + 0.05
                bf = bf + 1
            if bf >= 2:
                Alpha = Alpha + 1

            if Alpha <= (max_alpha - 1):
                if bf == 1:
                    Alpha = max_alpha - 1
                else:
                    Alpha = float(np.exp(m))

                # MATLAB: omega_L = omega_L(n,1); then n reset to 1 and arrays reinitialized.
                omega_seed = omega_L[n] if n < N else omega_L[N - 1]
                temp_ud = u_hat_L[n, :].copy() if n < N else u_hat_L[N - 1, :].copy()

                udiff = tol + eps
                n = 0
                lambda_mat = np.zeros((N, T), dtype=np.complex128)
                u_hat_L = np.zeros((N, T), dtype=np.complex128)
                omega_L = np.zeros(N, dtype=np.float64)
                omega_L[0] = omega_seed
                u_hat_L[0, :] = temp_ud

        # Part 4: Save current mode
        idx_mode = n if n < N else N - 1
        idx_omega = max(n - 1, 0)
        if save_prev_mode and n > 0:
            idx_mode = max(n - 1, 0)
            idx_omega = max(n - 1, 0)
        mode_hat = u_hat_L[idx_mode, :].copy()
        omega_mode = float(omega_L[idx_omega] if idx_omega < omega_L.shape[0] else omega_L[-1])

        # Debug comparison of candidates before committing
        if debug_save:
            candA_hat = u_hat_L[n, :] if n < N else u_hat_L[N - 1, :]
            candB_hat = u_hat_L[n - 1, :] if n - 1 >= 0 else candA_hat
            candA_omega = omega_L[n] if n < omega_L.shape[0] else omega_L[omega_L.shape[0] - 1]
            candB_omega = omega_L[n - 1] if n - 1 >= 0 else candA_omega
            base_sum = np.sum(np.stack(u_hat_temp, axis=0), axis=0) if u_hat_temp else np.zeros_like(f_hat_onesided)
            def _freq_err(candidate: np.ndarray) -> float:
                pos = slice(T // 2, T)
                recon = base_sum + candidate
                return float(np.linalg.norm(f_hat_onesided[pos] - recon[pos]) / (np.linalg.norm(f_hat_onesided[pos]) + eps))
            errA = _freq_err(candA_hat)
            errB = _freq_err(candB_hat)
            print(f"[save-debug] n={n} A|omega={candA_omega:.6f} err={errA:.3e} ||A||^2={np.linalg.norm(candA_hat)**2:.3e}")
            print(f"[save-debug] n={n} B|omega={candB_omega:.6f} err={errB:.3e} ||B||^2={np.linalg.norm(candB_hat)**2:.3e}")
            print(f"[save-debug] policy={'B' if save_prev_mode else 'A'} idx_mode={idx_mode} idx_omega={idx_omega}")

        u_hat_temp.append(mode_hat)
        omega_d_temp.append(omega_mode)
        alpha_hist.append(float(Alpha))

        Alpha = float(min_alpha)
        bf = 0

        if init_omega > 0:
            ii = 0
            while ii < 1 and n2 < 300:
                rand_omega = float(np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand()))[0])
                checkp = np.abs(np.asarray(omega_d_temp, dtype=np.float64) - rand_omega)
                if np.sum(checkp < 0.02) <= 0:
                    ii = 1
                n2 = n2 + 1
            omega_L = np.zeros(N, dtype=np.float64)
            omega_L[0] = rand_omega
        else:
            omega_L = np.zeros(N, dtype=np.float64)
            omega_L[0] = 0.0

        udiff = tol + eps
        lambda_mat = np.zeros((N, T), dtype=np.complex128)

        gamma_hist.append(1.0)
        h_row = gamma_hist[-1] / ((alpha_hist[-1] ** 2) * (omega_freqs - omega_d_temp[-1]) ** 4 + eps)
        h_hat_rows.append(h_row)
        u_hat_i.append(mode_hat)

        # Part 5: stopping criteria
        sum_u_i = np.sum(np.vstack(u_hat_i), axis=0) if u_hat_i else np.zeros(T, dtype=np.complex128)
        if stopc == 1:
            sigerror = float(np.linalg.norm(f_hat_onesided - sum_u_i) ** 2)
            sigerror_hist.append(sigerror)
            if n2 >= 300 or sigerror <= np.round(noisepe):
                sc2 = 1
        elif stopc == 2:
            sum_u = np.sum(np.vstack(u_hat_temp), axis=0) if u_hat_temp else np.zeros(T, dtype=np.complex128)
            normind = float(((1.0 / T) * (np.linalg.norm(sum_u - f_hat_onesided) ** 2)) / (((1.0 / T) * (np.linalg.norm(f_hat_onesided) ** 2)) + eps))
            normind_hist.append(normind)
            if n2 >= 300 or normind < 0.005:
                sc2 = 1
        elif stopc == 3:
            sigerror = float(np.linalg.norm(f_hat_onesided - sum_u_i) ** 2)
            sigerror_hist.append(sigerror)
            bic = float(2 * T * np.log(sigerror + eps) + (3 * (l + 1)) * np.log(2 * T))
            bic_hist.append(bic)
            if l > 0 and bic_hist[-1] > bic_hist[-2]:
                sc2 = 1
        else:
            cur_mode = u_hat_i[-1]
            cur_polm = float(
                np.linalg.norm(
                    (4.0 * Alpha * cur_mode / (1.0 + 2.0 * Alpha * (omega_freqs - omega_d_temp[-1]) ** 2))
                    @ np.conj(cur_mode).T
                )
            )
            if l < 1:
                polm_temp = cur_polm if cur_polm != 0 else 1.0
                polm_hist.append(cur_polm / max(cur_polm, eps))
            else:
                polm_hist.append(cur_polm / ((polm_temp if polm_temp is not None else 1.0) + eps))
            if l > 0 and abs(polm_hist[-1] - polm_hist[-2]) < 0.001:
                sc2 = 1

        # Part 6: reset counters
        u_hat_L = np.zeros((N, T), dtype=np.complex128)
        n = 0
        l = l + 1
        m = 0.0
        n2 = 0

        if debug_save:
            sum_u_i_debug = np.sum(np.vstack(u_hat_i), axis=0) if u_hat_i else np.zeros(T, dtype=np.complex128)
            freq_err_onesided = float(
                np.linalg.norm(f_hat_onesided - sum_u_i_debug) / (np.linalg.norm(f_hat_onesided) + eps)
            )
            mode_energy = float(np.linalg.norm(u_hat_i[-1]) ** 2)
            sum_energy = float(np.linalg.norm(sum_u_i_debug) ** 2)
            print(
                f"[debug-save] mode={l} freq_err_onesided={freq_err_onesided:.6e} "
                f"mode_energy={mode_energy:.6e} sum_energy={sum_energy:.6e}"
            )

        if verbose:
            print(f"[svmd] accepted mode={l} omega={omega_d_temp[-1]:.6f} sc2={sc2}")

    omega_arr = np.asarray(omega_d_temp, dtype=np.float64)
    if omega_arr.size == 0:
        return np.zeros((0, save_T), dtype=np.float64), np.zeros((save_T, 0), dtype=np.complex128), omega_arr

    # Part 7: reconstruction
    L = omega_arr.shape[0]
    u_hat = np.zeros((T, L), dtype=np.complex128)
    modes_stack = np.stack(u_hat_temp, axis=1)
    # 1) positive frequencies (including DC at shifted index T//2)
    u_hat[T // 2 :, :] = modes_stack[T // 2 :, :]
    # 2) negative frequencies via Hermitian symmetry
    u_hat[1 : T // 2, :] = np.conj(modes_stack[T - 1 : T // 2 : -1, :])
    # 3) enforce real-valued Nyquist/DC bins in shifted domain
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

    return u_time, u_hat_final, omega_sorted


def validate_outputs(
    signal: np.ndarray,
    u_calc: np.ndarray,
    omega_calc: np.ndarray,
    uhat_calc: np.ndarray | None = None,
    u_ref: np.ndarray | None = None,
    uhat_ref: np.ndarray | None = None,
) -> ValidationReport:
    x = signal.astype(np.float64)
    x_hat = np.sum(u_calc, axis=0) if u_calc.size else np.zeros_like(x)
    rel_recon_err = float(np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-12))
    f_hat = np.fft.fftshift(np.fft.fft(x))
    if uhat_calc is not None and uhat_calc.size:
        recon_hat = np.sum(uhat_calc, axis=1)
    else:
        recon_hat = np.fft.fftshift(np.fft.fft(x_hat))
    freq_recon_err = float(np.linalg.norm(f_hat - recon_hat) / (np.linalg.norm(f_hat) + 1e-12))
    energy_ratio = float(np.sum(u_calc**2) / (np.sum(x**2) + 1e-12)) if u_calc.size else 0.0

    # Orthogonality Index
    if u_calc.shape[0] <= 1:
        oi = 0.0
    else:
        num = 0.0
        den = 0.0
        for i in range(u_calc.shape[0]):
            den += float(np.dot(u_calc[i], u_calc[i]))
            for j in range(u_calc.shape[0]):
                if i != j:
                    num += float(np.dot(u_calc[i], u_calc[j]))
        oi = float(num / (den + 1e-12))

    r = x - x_hat
    r_spec = np.abs(np.fft.fftshift(np.fft.fft(r)))
    residual_peak_ratio = float(np.max(r_spec) / (np.mean(r_spec) + 1e-12))

    matched_modes = 0
    mean_abs_corr = np.nan
    mean_nrmse = np.nan
    mean_spec_corr = np.nan
    mean_spec_overlap = np.nan
    mean_omega_diff = np.nan
    ridx: np.ndarray | None = None
    cidx: np.ndarray | None = None
    u_metric = u_calc

    if u_ref is not None and u_ref.size and u_calc.size:
        ridx, cidx = _match_modes_by_corr(u_calc, u_ref)
        matched_modes = len(ridx)
        if matched_modes > 0:
            u_metric = _align_modes(u_calc, u_ref, ridx, cidx)
        corr_vals: list[float] = []
        nrmse_vals: list[float] = []
        spec_corr_vals: list[float] = []
        spec_overlap_vals: list[float] = []
        omega_diff_vals: list[float] = []
        for i, j in zip(ridx, cidx):
            corr_vals.append(abs(_corr(u_metric[i], u_ref[j])))
            nrmse_vals.append(_nrmse(u_metric[i], u_ref[j]))
            sc, so = _spectral_metrics(u_metric[i], u_ref[j])
            spec_corr_vals.append(sc)
            spec_overlap_vals.append(so)
            omega_diff_vals.append(abs(_dominant_omega(u_metric[i]) - _dominant_omega(u_ref[j])))
        mean_abs_corr = float(np.mean(corr_vals))
        mean_nrmse = float(np.mean(nrmse_vals))
        mean_spec_corr = float(np.mean(spec_corr_vals))
        mean_spec_overlap = float(np.mean(spec_overlap_vals))
        mean_omega_diff = float(np.mean(omega_diff_vals))

    uhat_rmse: float | None = None
    uhat_nrmse: float | None = None
    uhat_abs_nrmse: float | None = None
    uhat_nrmse_matlab_bug: float | None = None
    uhat_abs_nrmse_matlab_bug: float | None = None
    uhat_nrmse_corrected: float | None = None
    uhat_abs_nrmse_corrected: float | None = None
    if uhat_ref is not None and uhat_ref.size:
        if ridx is not None and cidx is not None and len(ridx) > 0:
            sq = 0.0
            sq_ref = 0.0
            sq_abs = 0.0
            sq_abs_ref = 0.0
            cnt = 0
            for i, j in zip(ridx, cidx):
                calc_col = np.fft.fftshift(np.fft.fft(u_metric[i, :]))
                ref_col = uhat_ref[:, j]
                d = calc_col - ref_col
                sq += float(np.sum(np.abs(d) ** 2))
                sq_ref += float(np.sum(np.abs(ref_col) ** 2))
                da = np.abs(calc_col) - np.abs(ref_col)
                sq_abs += float(np.sum(da**2))
                sq_abs_ref += float(np.sum(np.abs(ref_col) ** 2))
                cnt += calc_col.shape[0]
            if cnt > 0:
                uhat_rmse = float(np.sqrt(sq / cnt))
                uhat_nrmse = float(np.sqrt(sq / (sq_ref + 1e-12)))
                uhat_abs_nrmse = float(np.sqrt(sq_abs / (sq_abs_ref + 1e-12)))
        else:
            m = min(uhat_ref.shape[1], u_calc.shape[0])
            if m > 0:
                calc = np.fft.fftshift(np.fft.fft(u_metric[:m, :], axis=1), axes=1).T
                diff = calc - uhat_ref[:, :m]
                sq = float(np.sum(np.abs(diff) ** 2))
                sq_ref = float(np.sum(np.abs(uhat_ref[:, :m]) ** 2))
                cnt = diff.size
                uhat_rmse = float(np.sqrt(sq / cnt))
                uhat_nrmse = float(np.sqrt(sq / (sq_ref + 1e-12)))
                da = np.abs(calc) - np.abs(uhat_ref[:, :m])
                sq_abs = float(np.sum(da**2))
                sq_abs_ref = float(np.sum(np.abs(uhat_ref[:, :m]) ** 2))
                uhat_abs_nrmse = float(np.sqrt(sq_abs / (sq_abs_ref + 1e-12)))

    uhat_nrmse_matlab_bug = uhat_abs_nrmse
    uhat_abs_nrmse_matlab_bug = uhat_abs_nrmse
    uhat_nrmse_corrected = uhat_nrmse
    uhat_abs_nrmse_corrected = uhat_abs_nrmse

    return ValidationReport(
        rel_recon_err=rel_recon_err,
        freq_recon_err=freq_recon_err,
        energy_ratio=energy_ratio,
        orthogonality_index=oi,
        residual_peak_ratio=residual_peak_ratio,
        matched_modes=matched_modes,
        mean_abs_corr=mean_abs_corr,
        mean_nrmse=mean_nrmse,
        mean_spec_corr=mean_spec_corr,
        mean_spec_overlap=mean_spec_overlap,
        mean_omega_diff=mean_omega_diff,
        uhat_rmse=uhat_rmse,
        uhat_nrmse=uhat_nrmse,
        uhat_abs_nrmse=uhat_abs_nrmse,
        uhat_nrmse_matlab_bug=uhat_nrmse_matlab_bug,
        uhat_abs_nrmse_matlab_bug=uhat_abs_nrmse_matlab_bug,
        uhat_nrmse_corrected=uhat_nrmse_corrected,
        uhat_abs_nrmse_corrected=uhat_abs_nrmse_corrected,
    )


def load_csv_vector(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def print_validation(report: ValidationReport) -> None:
    print("[validation] Reconstruction")
    print(f"  rel_recon_err        : {report.rel_recon_err:.6e}")
    print(f"  freq_recon_err       : {report.freq_recon_err:.6e}")
    print(f"  energy_ratio         : {report.energy_ratio:.6e}")
    print(f"  orthogonality_index  : {report.orthogonality_index:.6e}")
    print(f"  residual_peak_ratio  : {report.residual_peak_ratio:.6e}")
    print("[validation] Mode Matching")
    print(f"  matched_modes        : {report.matched_modes}")
    print(f"  mean_abs_corr        : {report.mean_abs_corr:.6e}")
    print(f"  mean_nrmse           : {report.mean_nrmse:.6e}")
    print(f"  mean_spec_corr       : {report.mean_spec_corr:.6e}")
    print(f"  mean_spec_overlap    : {report.mean_spec_overlap:.6e}")
    print(f"  mean_omega_diff      : {report.mean_omega_diff:.6e}")
    if report.uhat_rmse is not None:
        print("[validation] Spectrum")
        print("  -- [MATLAB Bug (Simulated)] --")
        print(f"  uhat_nrmse_bug       : {report.uhat_nrmse_matlab_bug:.6e}")
        print(f"  uhat_abs_nrmse_bug   : {report.uhat_abs_nrmse_matlab_bug:.6e}")
        print("  -- [Corrected] --")
        print(f"  uhat_rmse            : {report.uhat_rmse:.6e}")
        print(f"  uhat_nrmse_corr      : {report.uhat_nrmse_corrected:.6e}")
        print(f"  uhat_abs_nrmse_corr  : {report.uhat_abs_nrmse_corrected:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=Path, default=DEFAULT_DEMO_DIR / "signal.csv")
    parser.add_argument("--params", type=Path, default=DEFAULT_DEMO_DIR / "params.csv")
    parser.add_argument("--u-ref", type=Path, default=DEFAULT_DEMO_DIR / "u.csv")
    parser.add_argument("--omega-ref", type=Path, default=DEFAULT_DEMO_DIR / "omega.csv")
    parser.add_argument("--uhat-real-ref", type=Path, default=DEFAULT_DEMO_DIR / "uhat_real.csv")
    parser.add_argument("--uhat-imag-ref", type=Path, default=DEFAULT_DEMO_DIR / "uhat_imag.csv")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-modes", type=int, default=0, help="0 means no mode cap (MATLAB-like).")
    parser.add_argument("--disable-sgolay", action="store_true")
    parser.add_argument("--fixed-iter", action="store_true")
    parser.add_argument("--force-stopc2", action="store_true", help="Force stop criterion to reconstruction-based (stopc=2).")
    parser.add_argument("--save-prev-mode", action="store_true", help="Use u_hat_L[n-1] / omega_L[n-1] when saving modes.")
    parser.add_argument("--debug-save", action="store_true", help="Print mode-save candidates and one-sided freq errors.")
    parser.add_argument("--debug-admm", action="store_true", help="Print primal/dual residuals each ADMM iteration.")
    parser.add_argument("--debug-denom", action="store_true", help="Print denominator component norms per ADMM step.")
    parser.add_argument("--disable-sum-h", action="store_true", help="Disable accumulated h_hat (sum_h) in denominator for ablation.")
    parser.add_argument("--tol-primal", type=float, default=1e-3, help="Primal residual threshold for ADMM stopping.")
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Validation mode: disable sgolay + fixed iterations.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    signal = load_csv_vector(args.signal)
    params = load_csv_vector(args.params)
    if params.shape[0] < 5:
        raise ValueError("params.csv must have at least 5 entries: fs,maxAlpha,stopc,tau,tol")
    _fs, max_alpha, stopc, tau, tol = params[:5]

    use_sgolay = not args.disable_sgolay
    fixed_iter = bool(args.fixed_iter)
    if args.strict_validation:
        use_sgolay = False
        fixed_iter = True
    if args.force_stopc2:
        stopc = 2
    mode_cap = None if int(args.max_modes) <= 0 else int(args.max_modes)

    u, u_hat, omega = svmd(
        signal=signal,
        max_alpha=float(max_alpha),
        tau=float(tau),
        tol=float(tol),
        stopc=int(stopc),
        init_omega=0,
        max_iter=int(args.max_iter),
        max_modes=mode_cap,
        use_sgolay=use_sgolay,
        fixed_iterations=fixed_iter,
        save_prev_mode=bool(args.save_prev_mode),
        debug_save=bool(args.debug_save),
        debug_admm=bool(args.debug_admm),
        debug_denom=bool(args.debug_denom),
        use_sum_h=not bool(args.disable_sum_h),
        tol_primal=float(args.tol_primal),
        verbose=not args.quiet,
    )
    print(f"Computed modes: {u.shape[0]}, samples per mode: {u.shape[1] if u.size else 0}")

    if args.no_verify:
        return

    u_ref = None
    if args.u_ref and args.u_ref.exists():
        u_ref = np.loadtxt(args.u_ref, delimiter=",", dtype=np.float64)

    omega_ref = None
    if args.omega_ref and args.omega_ref.exists():
        omega_ref = load_csv_vector(args.omega_ref)
        # The demo omega.csv is usually the frequency axis, not center frequencies.
        if omega_ref.ndim == 1 and omega_ref.shape[0] > 100:
            print("[note] omega_ref looks like frequency axis, skip direct center-frequency RMSE.")
            omega_ref = None

    uhat_ref = None
    if args.uhat_real_ref and args.uhat_imag_ref and args.uhat_real_ref.exists() and args.uhat_imag_ref.exists():
        uhat_ref = np.loadtxt(args.uhat_real_ref, delimiter=",", dtype=np.float64) + 1j * np.loadtxt(
            args.uhat_imag_ref, delimiter=",", dtype=np.float64
        )

    report = validate_outputs(signal=signal, u_calc=u, omega_calc=omega, uhat_calc=u_hat, u_ref=u_ref, uhat_ref=uhat_ref)
    print_validation(report)

    if omega_ref is not None and omega_ref.size and omega.size:
        m = min(len(omega_ref), len(omega))
        rmse_omega = float(np.sqrt(np.mean((omega[:m] - omega_ref[:m]) ** 2)))
        print(f"[validation] omega_rmse_direct: {rmse_omega:.6e}")


if __name__ == "__main__":
    main()
