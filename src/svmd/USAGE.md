# svmd Package — Usage Guide

## Quick Start

```python
from svmd import SVMDConfig, SVMDPipeline

cfg = SVMDConfig(max_alpha=2000, tau=0.0, tol=1e-7)
u_time, u_hat_final, omega = SVMDPipeline(cfg).run(signal)
# u_time:      (n_modes, n_samples) float   — time-domain modes
# u_hat_final: (n_samples, n_modes) complex — spectral modes
# omega:       (n_modes,) float              — centre frequencies (sorted)
```

Identical return signature to `svmd_prototype.svmd()`.

---

## SVMDConfig — all parameters

```python
SVMDConfig(
    max_alpha    = 2000,    # compactness upper bound (required)
    tau          = 0.0,     # ADMM dual ascent step (0 for noisy signals)
    tol          = 1e-7,    # ADMM convergence tolerance
    stopc        = 4,       # stopping criterion: 1=noise, 2=recon, 3=BIC, 4=polm (default)
    init_omega   = 0,       # 0=DC init, else random
    min_alpha    = 10.0,    # compactness starting value
    max_iter     = 300,     # max ADMM iterations per (mode, alpha)
    max_modes    = None,    # cap number of modes (None = unlimited)
    use_sgolay   = True,    # Savitzky-Golay noise estimation
    fixed_iterations = False,  # ignore convergence, run max_iter steps
    save_prev_mode   = False,  # use u_hat[n-1] instead of u_hat[n]
    use_sum_h    = True,    # include mode-interaction term in ADMM denominator
    tol_primal   = 1e-3,    # primal residual threshold for ADMM stopping
    # debug flags
    debug_save   = False,
    debug_admm   = False,
    debug_denom  = False,
    verbose      = False,
)
```

---

## Stopping Criteria

| stopc | Class       | Stops when |
|-------|-------------|------------|
| 1     | StopNoise   | residual spectral power ≤ noise power |
| 2     | StopRecon   | normalised reconstruction index < 0.005 |
| 3     | StopBIC     | BIC increases (over-decomposed) |
| 4     | StopPolm    | mode-power ratio stabilises (MATLAB default) |

---

## Validation

```python
from svmd import SVMDValidator
import numpy as np

validator = SVMDValidator()
report = validator.validate(
    signal=signal,
    u_calc=u_time,
    omega_calc=omega,
    uhat_calc=u_hat_final,
    u_ref=u_ref,        # optional MATLAB reference
    uhat_ref=uhat_ref,  # optional MATLAB reference
)
SVMDValidator.print_report(report)
```

---

## Feature Extraction (rolling walkforward)

```python
features = SVMDPipeline(cfg).extract_features(window_slice)
# returns np.ndarray, shape (n_modes, 4)
# columns: [centre_omega, mode_energy, mode_std, mode_kurtosis]
# rows sorted by centre_omega (low frequency first)
```

---

## Swapping to C++ Backend (future)

```python
from svmd import SVMDConfig, SVMDPipeline
from svmd.backends import CppBackend

cfg = SVMDConfig(max_alpha=2000, tau=0.0, tol=1e-7)
pipeline = SVMDPipeline(cfg, backend=CppBackend("path/to/svmd_core.dll"))
u_time, u_hat_final, omega = pipeline.run(signal)
# All control logic (loops, stopping, alpha schedule) unchanged.
```

---

## Parity with svmd_prototype.py

Verified bit-for-bit identical output (u_diff = 0.00e+00) across:

| Test case | Modes |
|-----------|-------|
| stopc=4 (default), full signal | 17 |
| stopc=4, short 64-sample signal | 17 |
| stopc=1 | 8 |
| stopc=2 | 15 |
| stopc=3 | 35 |
| use_sgolay=False | 17 |
| save_prev_mode=True | 17 |
| use_sum_h=False | 2 |
| max_modes=5 | 5 |
| min_alpha=50 | 17 |
| strict_validation (sgolay=False + fixed_iter=True) | 17 |
