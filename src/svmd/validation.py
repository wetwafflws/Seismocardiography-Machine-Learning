"""SVMDValidator — lifted from svmd_prototype.validate_outputs().

All helper functions (_corr, _nrmse, etc.) are now private methods so the
class is self-contained.  The ValidationReport dataclass is also defined here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Report dataclass (identical fields to svmd_prototype.ValidationReport)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class SVMDValidator:
    """Compute validation metrics for a set of extracted modes."""

    def validate(
        self,
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

        energy_ratio = float(np.sum(u_calc ** 2) / (np.sum(x ** 2) + 1e-12)) if u_calc.size else 0.0

        oi = self._orthogonality_index(u_calc)

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
            ridx, cidx = self._match_modes_by_corr(u_calc, u_ref)
            matched_modes = len(ridx)
            if matched_modes > 0:
                u_metric = self._align_modes(u_calc, u_ref, ridx, cidx)
            corr_vals, nrmse_vals, spec_corr_vals, spec_overlap_vals, omega_diff_vals = [], [], [], [], []
            for i, j in zip(ridx, cidx):
                corr_vals.append(abs(self._corr(u_metric[i], u_ref[j])))
                nrmse_vals.append(self._nrmse(u_metric[i], u_ref[j]))
                sc, so = self._spectral_metrics(u_metric[i], u_ref[j])
                spec_corr_vals.append(sc)
                spec_overlap_vals.append(so)
                omega_diff_vals.append(
                    abs(self._dominant_omega(u_metric[i]) - self._dominant_omega(u_ref[j]))
                )
            mean_abs_corr = float(np.mean(corr_vals))
            mean_nrmse = float(np.mean(nrmse_vals))
            mean_spec_corr = float(np.mean(spec_corr_vals))
            mean_spec_overlap = float(np.mean(spec_overlap_vals))
            mean_omega_diff = float(np.mean(omega_diff_vals))

        uhat_rmse = uhat_nrmse = uhat_abs_nrmse = None
        if uhat_ref is not None and uhat_ref.size:
            uhat_rmse, uhat_nrmse, uhat_abs_nrmse = self._uhat_metrics(
                u_metric, uhat_ref, ridx, cidx
            )

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

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: ValidationReport) -> None:
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

    # ------------------------------------------------------------------
    # Private math helpers (identical to svmd_prototype.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        aa = a - np.mean(a)
        bb = b - np.mean(b)
        denom = np.linalg.norm(aa) * np.linalg.norm(bb)
        if denom <= 0:
            return 0.0
        return float(np.dot(aa, bb) / denom)

    @staticmethod
    def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(b)
        if denom <= 0:
            return np.inf
        return float(np.linalg.norm(a - b) / denom)

    @staticmethod
    def _spectral_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        sa = np.abs(np.fft.fftshift(np.fft.fft(a))) ** 2
        sb = np.abs(np.fft.fftshift(np.fft.fft(b))) ** 2
        corr_val = SVMDValidator._corr(sa, sb)
        overlap = float(np.sum(np.minimum(sa, sb)) / (np.sum(np.maximum(sa, sb)) + 1e-12))
        return corr_val, overlap

    @staticmethod
    def _dominant_omega(mode: np.ndarray) -> float:
        T = mode.shape[0]
        grid = np.arange(1, T + 1, dtype=np.float64) / T - 0.5 - 1.0 / T
        s = np.abs(np.fft.fftshift(np.fft.fft(mode))) ** 2
        half = s[T // 2 :]
        idx = int(np.argmax(half))
        return float(grid[T // 2 + idx])

    @staticmethod
    def _match_modes_by_corr(
        u_calc: np.ndarray, u_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        c = np.zeros((u_calc.shape[0], u_ref.shape[0]), dtype=np.float64)
        for i in range(u_calc.shape[0]):
            for j in range(u_ref.shape[0]):
                c[i, j] = abs(SVMDValidator._corr(u_calc[i], u_ref[j]))
        row_ind, col_ind = linear_sum_assignment(-c)
        return row_ind, col_ind

    @staticmethod
    def _align_modes(
        u_calc: np.ndarray, u_ref: np.ndarray,
        ridx: np.ndarray, cidx: np.ndarray
    ) -> np.ndarray:
        aligned = u_calc.astype(np.complex128).copy()
        for i, j in zip(ridx, cidx):
            ref = u_ref[j].astype(np.complex128)
            calc = aligned[i]
            phase = np.vdot(ref, calc)
            if np.abs(phase) > 0:
                aligned[i] = calc * np.exp(-1j * np.angle(phase))
        return np.real(aligned)

    @staticmethod
    def _orthogonality_index(u_calc: np.ndarray) -> float:
        if u_calc.shape[0] <= 1:
            return 0.0
        num = 0.0
        den = 0.0
        for i in range(u_calc.shape[0]):
            den += float(np.dot(u_calc[i], u_calc[i]))
            for j in range(u_calc.shape[0]):
                if i != j:
                    num += float(np.dot(u_calc[i], u_calc[j]))
        return float(num / (den + 1e-12))

    @staticmethod
    def _uhat_metrics(
        u_metric: np.ndarray,
        uhat_ref: np.ndarray,
        ridx: np.ndarray | None,
        cidx: np.ndarray | None,
    ) -> tuple[float | None, float | None, float | None]:
        if ridx is not None and cidx is not None and len(ridx) > 0:
            sq = sq_ref = sq_abs = sq_abs_ref = 0.0
            cnt = 0
            for i, j in zip(ridx, cidx):
                calc_col = np.fft.fftshift(np.fft.fft(u_metric[i, :]))
                ref_col = uhat_ref[:, j]
                d = calc_col - ref_col
                sq += float(np.sum(np.abs(d) ** 2))
                sq_ref += float(np.sum(np.abs(ref_col) ** 2))
                da = np.abs(calc_col) - np.abs(ref_col)
                sq_abs += float(np.sum(da ** 2))
                sq_abs_ref += float(np.sum(np.abs(ref_col) ** 2))
                cnt += calc_col.shape[0]
            if cnt > 0:
                return (
                    float(np.sqrt(sq / cnt)),
                    float(np.sqrt(sq / (sq_ref + 1e-12))),
                    float(np.sqrt(sq_abs / (sq_abs_ref + 1e-12))),
                )
        else:
            m = min(uhat_ref.shape[1], u_metric.shape[0])
            if m > 0:
                calc = np.fft.fftshift(np.fft.fft(u_metric[:m, :], axis=1), axes=1).T
                diff = calc - uhat_ref[:, :m]
                sq = float(np.sum(np.abs(diff) ** 2))
                sq_ref = float(np.sum(np.abs(uhat_ref[:, :m]) ** 2))
                cnt = diff.size
                da = np.abs(calc) - np.abs(uhat_ref[:, :m])
                sq_abs = float(np.sum(da ** 2))
                sq_abs_ref = float(np.sum(np.abs(uhat_ref[:, :m]) ** 2))
                return (
                    float(np.sqrt(sq / cnt)),
                    float(np.sqrt(sq / (sq_ref + 1e-12))),
                    float(np.sqrt(sq_abs / (sq_abs_ref + 1e-12))),
                )
        return None, None, None
