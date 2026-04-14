"""CppBackend — ctypes wrapper stub (future C++ acceleration).

This file implements IComputeBackend using the same ctypes pattern shown in
svmd_experiment.py.  Currently it raises NotImplementedError so the interface
is wire-complete but safe to import before the DLL is built.

When the C++ DLL is ready:
  1. Set DLL_PATH to the built svmd_core.dll location.
  2. Implement _call_preprocess / _call_admm_step / _call_reconstruct.
  3. All control logic (SVMDSolver, stopping, schedule) stays unchanged.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from ..config import SVMDConfig
from ..interfaces import IComputeBackend
from ..state import InnerState, PreprocessResult, ReconstructResult, SVMDState

# Default DLL location (update once the C++ build is done)
# DLL lives outside this repo — set the path before instantiating CppBackend.
_DEFAULT_DLL = Path(__file__).resolve().parent / "svmd_core.dll"


class CppBackend(IComputeBackend):
    """ctypes-based backend that delegates math to a C++ shared library.

    The DLL is expected to export:
        int svmd_preprocess(...)
        int svmd_admm_step(...)
        int svmd_reconstruct(...)

    Until those symbols exist this class raises NotImplementedError on every
    method so callers get a clear message rather than a cryptic linker error.
    """

    def __init__(self, dll_path: Path | str | None = None) -> None:
        self._lib: ctypes.CDLL | None = None
        path = Path(dll_path) if dll_path is not None else _DEFAULT_DLL
        if path.exists():
            os.add_dll_directory(str(path.parent))
            self._lib = ctypes.CDLL(str(path))
            self._setup_signatures()

    def _setup_signatures(self) -> None:
        """Declare argtypes / restype for each exported C++ function."""
        # TODO: fill in once C++ API is finalised.
        # Example pattern from svmd_experiment.py:
        #   self._lib.svmd_admm_step.argtypes = [
        #       ctypes.c_int,           # T
        #       ctypes.c_int,           # n
        #       ctypes.c_double,        # Alpha
        #       ctypes.POINTER(ctypes.c_double),  # u_hat_real (in/out)
        #       ...
        #   ]
        #   self._lib.svmd_admm_step.restype = ctypes.c_int
        pass

    # ------------------------------------------------------------------
    # IComputeBackend interface — all raise until DLL is built
    # ------------------------------------------------------------------

    def preprocess(self, signal: np.ndarray, cfg: SVMDConfig) -> PreprocessResult:
        self._require_lib("preprocess")
        raise NotImplementedError("CppBackend.preprocess not yet implemented")

    def admm_step(
        self,
        inner: InnerState,
        pre: PreprocessResult,
        svmd_state: SVMDState,
        cfg: SVMDConfig,
        Alpha: float,
    ) -> InnerState:
        self._require_lib("admm_step")
        raise NotImplementedError("CppBackend.admm_step not yet implemented")

    def reconstruct_modes(
        self, svmd_state: SVMDState, pre: PreprocessResult
    ) -> ReconstructResult:
        self._require_lib("reconstruct_modes")
        raise NotImplementedError("CppBackend.reconstruct_modes not yet implemented")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_lib(self, method: str) -> None:
        if self._lib is None:
            raise FileNotFoundError(
                f"CppBackend.{method}: DLL not found at {_DEFAULT_DLL}. "
                "Build the C++ library first (see native_svmd/build_demo.ps1)."
            )

    @staticmethod
    def _to_c_double_array(arr: np.ndarray) -> ctypes.Array:
        flat = arr.ravel()
        return (ctypes.c_double * len(flat))(*flat)
