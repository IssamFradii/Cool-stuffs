"""Rich/cheap analysis of the cap/swaption wedge, per expiry bucket.

Based on Brigo & Mercurio, "Interest Rate Models - Theory and Practice",
Sections 6.20 (Connecting Caplet and S x 1-Swaption Volatilities) and
7.9 (Cascade Calibration and the cap market).
"""

from .annualize import (
    annual_forward,
    nesting_weights,
    cap_implied_swaption_vol,
    implied_infra_correlation,
)
from .analysis import analyze_wedges

__all__ = [
    "annual_forward",
    "nesting_weights",
    "cap_implied_swaption_vol",
    "implied_infra_correlation",
    "analyze_wedges",
]
