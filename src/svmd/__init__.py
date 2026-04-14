"""svmd — Strategy-Pattern SVMD package.

Public API
----------
SVMDConfig      -- all hyperparameters (frozen dataclass)
SVMDPipeline    -- top-level runner; drop-in for svmd_prototype.svmd()
SVMDValidator   -- validation metrics
ValidationReport

NumpyBackend    -- default backend (pure numpy)
CppBackend      -- future C++ backend (stub until DLL is built)

Stopping criteria (IStoppingCriterion):
  StopNoise, StopRecon, StopBIC, StopPolm, make_stopping_criterion

Alpha schedule (IAlphaSchedule):
  LinearAlphaSchedule

Quick start
-----------
    from svmd import SVMDConfig, SVMDPipeline

    cfg = SVMDConfig(max_alpha=2000, tau=0.0, tol=1e-7)
    u_time, u_hat_final, omega = SVMDPipeline(cfg).run(signal)
"""
from .config import SVMDConfig
from .pipeline import SVMDPipeline
from .validation import SVMDValidator, ValidationReport
from .interfaces import IComputeBackend, IStoppingCriterion, IAlphaSchedule
from .backends import NumpyBackend, CppBackend
from .stopping import StopNoise, StopRecon, StopBIC, StopPolm, make_stopping_criterion
from .schedule import LinearAlphaSchedule
from .solver import SVMDSolver
from .state import SVMDState, InnerState, PreprocessResult, ReconstructResult

__all__ = [
    "SVMDConfig",
    "SVMDPipeline",
    "SVMDValidator",
    "ValidationReport",
    "IComputeBackend",
    "IStoppingCriterion",
    "IAlphaSchedule",
    "NumpyBackend",
    "CppBackend",
    "StopNoise",
    "StopRecon",
    "StopBIC",
    "StopPolm",
    "make_stopping_criterion",
    "LinearAlphaSchedule",
    "SVMDSolver",
    "SVMDState",
    "InnerState",
    "PreprocessResult",
    "ReconstructResult",
]
