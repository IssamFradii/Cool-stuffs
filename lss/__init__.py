"""
lss -- Longstaff, Santa-Clara & Schwartz (2001),
"The Relative Valuation of Caps and Swaptions: Theory and Empirical Evidence"

A full implementation of the paper's String Market Model (SMM) pipeline:

    mathutil   -> numpy-only numerics (normal CDF, cubic spline, LM solver)
    conventions-> the semiannual grid, the 34 swaption structures, cap maturities
    paper_data -> Tables 1-10 transcribed from the paper
    curve      -> par curve spline + bootstrap -> 19 six-month Libor forwards
    instruments-> Black (1976) caps, caplets, swaptions; swap values
    historical -> ex-ante correlation matrix H = U Lambda U'
    smm        -> string market model simulation  dD = rDdt + J^-1 sigma F dZ   (eq 11)
    approx     -> fast closed-form swaption approximation (footnote 18)
    calibrate  -> solve for implied eigenvalues Psi,  Sigma = U Psi U'
    caps       -> cap prices from the main diagonal of the implied Sigma
    empirics   -> Tables 5-10 of the paper
    strategy   -> the cap/swaption "wedge" relative-value trading strategy
    synthetic  -> paper-calibrated synthetic data (so the pipeline runs with no data)
    data       -> loaders for real market data

Equation and table numbers in docstrings refer to the published paper.

Submodules are imported lazily, so `import lss` is cheap and a problem in one
module does not break the rest.
"""

from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

__version__ = "1.0.0"

_SUBMODULES = (
    "mathutil", "conventions", "paper_data", "curve", "instruments",
    "historical", "smm", "approx", "calibrate", "caps", "empirics",
    "strategy", "synthetic", "data",
)

__all__ = list(_SUBMODULES)

if TYPE_CHECKING:  # pragma: no cover
    from . import (  # noqa: F401
        mathutil, conventions, paper_data, curve, instruments, historical,
        smm, approx, calibrate, caps, empirics, strategy, synthetic, data,
    )


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_SUBMODULES))
