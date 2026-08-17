"""
FVA - Forward Volatility Agreement
Implementation of "The Convexity Maven - The FVA: Forward Volatility Agreement"
Harley S. Bassman / Danielle N. Mohney, Credit Suisse US Rates Trading, 27-Aug-2013.

Timeline convention of the paper (Appendix):

    T0 ------------ T1 ------------ T2 -------------------------- T3
    today        strike date     expiry date                 swap maturity

The FVA is a straddle *struck* at T1 (at the then ATM forward rate) and *expiring*
at T2 into a swap that matures at T3.  Example "6m1y10y":
    T1 = 0.5, T2 = 1.5, T3 = 11.5.

Replication (paper, p.3-5): buy the (T2-T0) x (T3-T2) vanilla straddle
(18m10y) and sell the mid-curve struck today, expiring T1, on the forward
swap T2->T3 (the 6m1y10y mid-curve).

Everything is normal (bp) vol: r_t - r_{t-1} ~ N(mu, sigma^2), sigma in bp/year.

Only ONE market vol object is loaded: the ATM normal vol matrix (expiry x tenor).
Every vol used below (sigma_{T2,T3}, sigma_{T1,T3}, sigma_{T1,T2}) is read off
that single surface by interpolation.  DV01s/BPVs of the forward swaps and of
the mid-curve legs are computed from the discount curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

import numpy as np

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # straddle price = sqrt(2/pi) * sigma * sqrt(T) * A

InterpMode = Literal["vol", "variance"]


class BpvSource(Protocol):
    """Anything that returns the BPV (= DV01, no notional) of the forward swap
    start -> end, in years.  `Curve.bpv` is one; see bpv_sources.py for a table
    of externally computed BPVs and for a SwapLegsValueCalculator adapter."""

    def __call__(self, start: float, end: float) -> float: ...


# --------------------------------------------------------------------------------------
# Discount curve  ->  BPV / DV01 of forward swaps
# --------------------------------------------------------------------------------------
@dataclass
class Curve:
    """Zero curve, continuously compounded zeros, linear in z(t), flat extrapolation.

    `freq` is the fixed-leg frequency used for annuities (2 = semi-annual, USD style).
    """

    times: np.ndarray
    zeros: np.ndarray  # continuously compounded
    freq: int = 2

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.zeros = np.asarray(self.zeros, dtype=float)
        order = np.argsort(self.times)
        self.times, self.zeros = self.times[order], self.zeros[order]

    # -- constructors ------------------------------------------------------------------
    @classmethod
    def flat(cls, par_rate: float, freq: int = 2, max_t: float = 60.0) -> "Curve":
        """Flat par swap curve.  DF(t) = (1 + r/f)^(-f t) reprices every par swap at par,
        and makes the paper's (1 + r_{T0,T1}/2)^(2(T1-T0)) factor exactly 1/DF(T1)."""
        z = freq * math.log1p(par_rate / freq)  # cont-comp equivalent
        return cls(np.array([0.0, max_t]), np.array([z, z]), freq)

    @classmethod
    def from_zeros(
        cls, times: Sequence[float], zeros: Sequence[float], freq: int = 2, comp: str = "cont"
    ) -> "Curve":
        t = np.asarray(times, float)
        z = np.asarray(zeros, float)
        if comp == "annual":
            z = np.log1p(z)
        elif comp == "semi":
            z = 2.0 * np.log1p(z / 2.0)
        return cls(t, z, freq)

    @classmethod
    def from_par_swaps(
        cls, maturities: Sequence[float], par_rates: Sequence[float], freq: int = 2
    ) -> "Curve":
        """Bootstrap a par swap curve (sequential, zeros linear in t between nodes)."""
        mats = np.asarray(maturities, float)
        rates = np.asarray(par_rates, float)
        order = np.argsort(mats)
        mats, rates = mats[order], rates[order]

        crv = cls(np.array([0.0, mats[0]]), np.array([rates[0], rates[0]]), freq)
        known_t: list[float] = [0.0]
        known_z: list[float] = [float(rates[0])]

        for mat, par in zip(mats, rates):
            lo, hi = -0.5, 1.0
            for _ in range(200):  # bisection: par error is monotone in the node zero
                mid = 0.5 * (lo + hi)
                crv.times = np.array(known_t + [mat])
                crv.zeros = np.array(known_z + [mid])
                err = par * crv.annuity(0.0, mat) + crv.df(mat) - 1.0
                if err > 0.0:  # rate too rich -> need higher zero (lower DF)
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-14:
                    break
            known_t.append(float(mat))
            known_z.append(0.5 * (lo + hi))

        crv.times = np.array(known_t)
        crv.zeros = np.array(known_z)
        return crv

    # -- pricing -----------------------------------------------------------------------
    def zero(self, t: float) -> float:
        if t <= 0:
            return float(self.zeros[0])
        return float(np.interp(t, self.times, self.zeros))

    def df(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return math.exp(-self.zero(t) * t)

    def schedule(self, start: float, end: float) -> np.ndarray:
        """Fixed leg payment dates, `freq` per year, from `start` (exclusive) to `end`."""
        n = max(1, int(round((end - start) * self.freq)))
        return start + np.arange(1, n + 1) / self.freq

    def bpv(self, start: float, end: float) -> float:
        """BPV = DV01 of the forward swap start->end: sum(tau_i * DF(t_i)).

        Same single quantity as the paper's DV01 (bp of price per bp of rate,
        i.e. the annuity in years: DV01_18m10y = 8.6).  Multiply by 100 to read it
        as $ per 1bp per $1mm of notional ($860/bp), see `dv01_usd`.
        """
        if end <= start:
            return 0.0
        pay = self.schedule(start, end)
        tau = 1.0 / self.freq
        return float(sum(tau * self.df(t) for t in pay))

    # DV01 and BPV are the same number - kept as an alias so either name works.
    annuity = bpv

    def dv01_usd(self, start: float, end: float, notional: float = 1e6) -> float:
        """The same BPV/DV01 expressed in $ per 1bp of rate for a given notional."""
        return self.bpv(start, end) * notional * 1e-4

    def fwd_swap_rate(self, start: float, end: float) -> float:
        a = self.annuity(start, end)
        if a == 0.0:
            return float("nan")
        return (self.df(start) - self.df(end)) / a

    def fwd_compounding(self, t: float) -> float:
        """(1 + r_{T0,t}/2)^(2t) of the paper = 1/DF(t): premium paid at strike date."""
        return 1.0 / self.df(t)

    def semi_rate(self, t: float) -> float:
        """Semi-annually compounded zero rate to t (the paper's r_{T0,T1})."""
        if t <= 0:
            return 2.0 * math.expm1(self.zero(0.0) / 2.0)
        return 2.0 * ((1.0 / self.df(t)) ** (1.0 / (2.0 * t)) - 1.0)


# --------------------------------------------------------------------------------------
# ATM normal vol surface (the single market input)
# --------------------------------------------------------------------------------------
@dataclass
class VolSurface:
    """ATM normal (bp) vol matrix: rows = option expiry (y), cols = swap tenor (y)."""

    expiries: np.ndarray
    tenors: np.ndarray
    vols: np.ndarray  # shape (n_expiry, n_tenor), in bp
    mode: InterpMode = "vol"

    def __post_init__(self) -> None:
        self.expiries = np.asarray(self.expiries, float)
        self.tenors = np.asarray(self.tenors, float)
        self.vols = np.asarray(self.vols, float)
        if self.vols.shape != (len(self.expiries), len(self.tenors)):
            raise ValueError(
                f"vol matrix {self.vols.shape} does not match "
                f"{len(self.expiries)} expiries x {len(self.tenors)} tenors"
            )

    def _expiry_slice(self, expiry: float, col: int) -> float:
        """Interpolate along the expiry axis for a fixed tenor column."""
        x, y = self.expiries, self.vols[:, col]
        if self.mode == "variance" and expiry > 0:
            # linear in total variance sigma^2 * T (flat vol extrapolation at the ends)
            w = np.interp(expiry, x, y**2 * x)
            return float(math.sqrt(max(w, 0.0) / expiry))
        return float(np.interp(expiry, x, y))

    def __call__(self, expiry: float, tenor: float) -> float:
        """Bilinear interpolation with flat extrapolation, in bp."""
        t = self.tenors
        if tenor <= t[0]:
            return self._expiry_slice(expiry, 0)
        if tenor >= t[-1]:
            return self._expiry_slice(expiry, len(t) - 1)
        j = int(np.searchsorted(t, tenor)) - 1
        j = min(max(j, 0), len(t) - 2)
        w = (tenor - t[j]) / (t[j + 1] - t[j])
        return (1.0 - w) * self._expiry_slice(expiry, j) + w * self._expiry_slice(expiry, j + 1)


# --------------------------------------------------------------------------------------
# The FVA itself
# --------------------------------------------------------------------------------------
@dataclass
class FVAResult:
    # timeline
    t1: float  # strike date
    t2: float  # expiry date
    t3: float  # swap maturity
    expiry: float  # T2 - T1  (option life once struck)
    tenor: float  # T3 - T2  (swap tail)

    # vanilla vols read off the single ATM surface (bp)
    sig_23: float  # expiry T2-T0, tenor T3-T2      (e.g. 18m10y)
    sig_13: float  # expiry T1-T0, tenor T3-T1      (e.g. 6m11y)
    sig_12: float  # expiry T1-T0, tenor T2-T1      (e.g. 6m1y)
    sig_spot: float  # expiry T2-T1, tenor T3-T2    (e.g. spot 1y10y, for the discount)

    # BPVs (years, = paper's DV01)
    dv01_23: float
    dv01_13: float
    dv01_12: float

    # forward swap rates
    fwd_23: float
    fwd_13: float
    fwd_12: float

    rho: float
    sig_mc: float  # mid-curve normal vol (bp)
    sig_fva: float  # FVA forward normal vol (bp), nan if the variance spread is negative

    # sensitivities
    d_fva_d23: float  # d sigma_FVA / d sigma_{T2,T3}
    d_fva_dmc: float  # d sigma_FVA / d sigma_MC          (negative)
    d_mc_d13: float  # d sigma_MC  / d sigma_{T1,T3}
    d_mc_d12: float  # d sigma_MC  / d sigma_{T1,T2}

    # prices, bp of notional
    price_fva: float  # premium paid at the strike date T1
    price_fva_pv: float  # discounted to T0
    price_van23: float  # the 18m10y vanilla straddle
    price_mc: float  # the mid-curve straddle

    # normal vega, bp of notional per 1bp of vol
    vega_fva: float  # per 1bp of sigma_FVA
    vega_van23: float  # per 1bp of sigma_{T2,T3}
    vega_mc: float  # per 1bp of sigma_MC

    compounding: float  # (1 + r/2)^(2 T1) = 1/DF(T1)
    r_t1: float  # semi-annual zero to T1

    @property
    def discount_vs_spot(self) -> float:
        """FVA vol vs the spot vol of the same expiry/tenor, e.g. -5% = '95nv vs 100nv'."""
        if not (self.sig_spot > 0) or math.isnan(self.sig_fva):
            return float("nan")
        return self.sig_fva / self.sig_spot - 1.0


def midcurve_vol(
    dv01_13: float, sig_13: float, dv01_12: float, sig_12: float, dv01_23: float, rho: float
) -> float:
    """Mid-curve normal vol via the DV01-weighted spread option formula (paper, p.6).

    sigma_MC = sqrt[ (DV01_13^2 s13^2 + DV01_12^2 s12^2
                      - 2 rho DV01_13 s13 DV01_12 s12) / DV01_23^2 ]

    i.e. the vol of the forward swap T2->T3 seen as the DV01-weighted spread between
    the T1->T3 and T1->T2 swaps, both observed at T1.
    """
    var = (
        dv01_13**2 * sig_13**2
        + dv01_12**2 * sig_12**2
        - 2.0 * rho * dv01_13 * sig_13 * dv01_12 * sig_12
    ) / dv01_23**2
    return math.sqrt(max(var, 0.0))


def fva_vol(t1: float, t2: float, sig_23: float, sig_mc: float) -> float:
    """(T2-T1) s_FVA^2 = (T2-T0) s_{T2,T3}^2 - (T1-T0) s_MC^2   (paper, p.5).

    Returns nan when the variance spread is negative (surface/correlation inconsistent
    with a positive forward variance).
    """
    var = (t2 * sig_23**2 - t1 * sig_mc**2) / (t2 - t1)
    if var <= 0.0:
        return float("nan")
    return math.sqrt(var)


def price_fva(
    curve: Curve, t1: float, t2: float, t3: float, sig_fva: float
) -> tuple[float, float]:
    """P_FVA = sqrt(2/pi) sqrt(T2-T1) s_FVA DV01_{T2,T3} (1 + r_{T0,T1}/2)^(2(T1-T0)).

    Returns (premium at T1, PV today), both in bp of notional.
    """
    a23 = curve.annuity(t2, t3)
    cmp_ = curve.fwd_compounding(t1)
    fwd_prem = SQRT_2_OVER_PI * math.sqrt(t2 - t1) * sig_fva * a23 * cmp_
    return fwd_prem, fwd_prem / cmp_


def quote(
    curve: Curve,
    surface: VolSurface,
    strike: float,
    expiry: float,
    tenor: float,
    rho: float = 1.0,
    bpv: "BpvSource | None" = None,
) -> FVAResult:
    """Price one FVA.

    strike : T1 - T0, when the straddle gets struck at the then-ATM forward (e.g. 0.5)
    expiry : T2 - T1, life of the straddle once struck (e.g. 1.0)
    tenor  : T3 - T2, swap tail (e.g. 10.0)

    -> the '6m1y10y' FVA of the paper.

    bpv : optional callable (start, end) -> BPV/DV01 (no notional), overriding the
    curve's own.  Use it to plug an external calculator; the three calls map 1:1 onto
    the legs of the replication, e.g. for 6m1y10y:
        bpv(T2, T3) = 18M10Y forward swap
        bpv(T1, T3) = 6M11Y   (long leg of the mid-curve decomposition)
        bpv(T1, T2) = 6M1Y    (short leg of the mid-curve decomposition)
    """
    t1 = float(strike)
    t2 = t1 + float(expiry)
    t3 = t2 + float(tenor)

    # the three vanilla vols, all read off the single ATM matrix
    sig_23 = surface(t2, t3 - t2)  # 18m10y
    sig_13 = surface(t1, t3 - t1)  # 6m11y
    sig_12 = surface(t1, t2 - t1)  # 6m1y
    sig_spot = surface(t2 - t1, t3 - t2)  # spot 1y10y, reference for the fwd discount

    # BPV = DV01 of the forward swap and of the two mid-curve replication legs
    src = curve.bpv if bpv is None else bpv
    a23 = src(t2, t3)
    a13 = src(t1, t3)
    a12 = src(t1, t2)

    sig_mc = midcurve_vol(a13, sig_13, a12, sig_12, a23, rho)
    sig_f = fva_vol(t1, t2, sig_23, sig_mc)

    # sensitivities
    if math.isnan(sig_f) or sig_f <= 0:
        d_fva_d23 = d_fva_dmc = float("nan")
    else:
        d_fva_d23 = t2 * sig_23 / ((t2 - t1) * sig_f)
        d_fva_dmc = -t1 * sig_mc / ((t2 - t1) * sig_f)

    if sig_mc > 0:
        denom = sig_mc * a23**2
        d_mc_d13 = (a13**2 * sig_13 - rho * a13 * a12 * sig_12) / denom
        d_mc_d12 = (a12**2 * sig_12 - rho * a13 * a12 * sig_13) / denom
    else:
        d_mc_d13 = d_mc_d12 = float("nan")

    cmp_ = curve.fwd_compounding(t1)
    p_fva, p_fva_pv = (
        price_fva(curve, t1, t2, t3, sig_f) if not math.isnan(sig_f) else (float("nan"),) * 2
    )

    # vegas: d(straddle bp) / d(vol bp)
    vega_fva = SQRT_2_OVER_PI * math.sqrt(t2 - t1) * a23 * cmp_
    vega_van23 = SQRT_2_OVER_PI * math.sqrt(t2) * a23
    vega_mc = SQRT_2_OVER_PI * math.sqrt(t1) * a23

    return FVAResult(
        t1=t1,
        t2=t2,
        t3=t3,
        expiry=float(expiry),
        tenor=float(tenor),
        sig_23=sig_23,
        sig_13=sig_13,
        sig_12=sig_12,
        sig_spot=sig_spot,
        dv01_23=a23,
        dv01_13=a13,
        dv01_12=a12,
        fwd_23=curve.fwd_swap_rate(t2, t3),
        fwd_13=curve.fwd_swap_rate(t1, t3),
        fwd_12=curve.fwd_swap_rate(t1, t2),
        rho=float(rho),
        sig_mc=sig_mc,
        sig_fva=sig_f,
        d_fva_d23=d_fva_d23,
        d_fva_dmc=d_fva_dmc,
        d_mc_d13=d_mc_d13,
        d_mc_d12=d_mc_d12,
        price_fva=p_fva,
        price_fva_pv=p_fva_pv,
        price_van23=SQRT_2_OVER_PI * math.sqrt(t2) * sig_23 * a23,
        price_mc=SQRT_2_OVER_PI * math.sqrt(t1) * sig_mc * a23,
        vega_fva=vega_fva,
        vega_van23=vega_van23,
        vega_mc=vega_mc,
        compounding=cmp_,
        r_t1=curve.semi_rate(t1),
    )


# --------------------------------------------------------------------------------------
# Replication / hedge book
# --------------------------------------------------------------------------------------
@dataclass
class Replication:
    """Hedge for a target FVA vega (in $ per 1nv), following pp.3-5 and p.8.

    Positive notional = buy, negative = sell.  Vegas in $ per 1bp of the relevant vol.
    """

    target_vega: float
    notional_fva: float

    vega_van23: float
    notional_van23: float

    vega_mc: float
    notional_mc: float

    vega_13: float
    notional_13: float

    vega_12: float
    notional_12: float

    res: FVAResult = field(repr=False)


def replicate(res: FVAResult, target_vega: float = 100_000.0) -> Replication:
    """target_vega: $ of FVA vega per 1nv (e.g. 100_000 = '100k/nv').

    Vega amounts follow the paper (buy 160k 18m10y / sell 60k mid-curve, then
    sell 66k 6m11y / buy 6k 6m1y).  Notionals convert each vega into the notional
    of the straddle that carries it, using that straddle's own expiry and BPV.
    """
    bp_to_usd = 1e-4  # bp of notional -> $ per $1 of notional

    n_fva = target_vega / (res.vega_fva * bp_to_usd)

    # step 1: long the (T2-T0)x(T3-T2) vanilla, short the mid-curve
    v_van23 = target_vega * res.d_fva_d23
    n_van23 = v_van23 / (res.vega_van23 * bp_to_usd)

    v_mc = target_vega * res.d_fva_dmc  # negative -> sell the mid-curve
    n_mc = v_mc / (res.vega_mc * bp_to_usd)

    # step 2: the mid-curve itself replicated with two T1-expiry vanillas.
    # Each leg is a straddle expiring T1 on its own swap, so its vega uses its own BPV.
    vega_13_unit = SQRT_2_OVER_PI * math.sqrt(res.t1) * res.dv01_13  # T1 x (T3-T1), e.g. 6m11y
    vega_12_unit = SQRT_2_OVER_PI * math.sqrt(res.t1) * res.dv01_12  # T1 x (T2-T1), e.g. 6m1y
    v_13 = v_mc * res.d_mc_d13
    v_12 = v_mc * res.d_mc_d12
    n_13 = v_13 / (vega_13_unit * bp_to_usd) if vega_13_unit else float("nan")
    n_12 = v_12 / (vega_12_unit * bp_to_usd) if vega_12_unit else float("nan")

    return Replication(
        target_vega=target_vega,
        notional_fva=n_fva,
        vega_van23=v_van23,
        notional_van23=n_van23,
        vega_mc=v_mc,
        notional_mc=n_mc,
        vega_13=v_13,
        notional_13=n_13,
        vega_12=v_12,
        notional_12=n_12,
        res=res,
    )


# --------------------------------------------------------------------------------------
# Grids: for a fixed strike date, expiry x tenor over the input matrix axes
# --------------------------------------------------------------------------------------
def grid(
    curve: Curve,
    surface: VolSurface,
    strike: float,
    rho: float = 1.0,
    expiries: Sequence[float] | None = None,
    tenors: Sequence[float] | None = None,
    bpv: "BpvSource | None" = None,
) -> dict[str, np.ndarray]:
    """Run `quote` over the (expiry x tenor) axes of the loaded ATM matrix.

    Row i / col j = FVA struck in `strike`, expiring `expiries[i]` after the strike
    date, into a `tenors[j]` swap.
    """
    exps = np.asarray(surface.expiries if expiries is None else expiries, float)
    tens = np.asarray(surface.tenors if tenors is None else tenors, float)

    keys = (
        "sig_fva",
        "sig_mc",
        "sig_23",
        "sig_13",
        "sig_12",
        "sig_spot",
        "discount",
        "price_fva",
        "price_fva_pv",
        "vega_fva",
        "d_fva_d23",
        "d_fva_dmc",
        "dv01_23",
        "dv01_13",
        "dv01_12",
        "fwd_23",
        "ratio_13",
        "ratio_12",
    )
    out = {k: np.full((len(exps), len(tens)), np.nan) for k in keys}

    for i, e in enumerate(exps):
        for j, n in enumerate(tens):
            r = quote(curve, surface, strike, e, n, rho, bpv=bpv)
            out["sig_fva"][i, j] = r.sig_fva
            out["sig_mc"][i, j] = r.sig_mc
            out["sig_23"][i, j] = r.sig_23
            out["sig_13"][i, j] = r.sig_13
            out["sig_12"][i, j] = r.sig_12
            out["sig_spot"][i, j] = r.sig_spot
            out["discount"][i, j] = r.discount_vs_spot * 100.0
            out["price_fva"][i, j] = r.price_fva
            out["price_fva_pv"][i, j] = r.price_fva_pv
            out["vega_fva"][i, j] = r.vega_fva
            out["d_fva_d23"][i, j] = r.d_fva_d23
            out["d_fva_dmc"][i, j] = r.d_fva_dmc
            out["dv01_23"][i, j] = r.dv01_23
            out["dv01_13"][i, j] = r.dv01_13
            out["dv01_12"][i, j] = r.dv01_12
            out["fwd_23"][i, j] = r.fwd_23
            out["ratio_13"][i, j] = r.dv01_13 / r.dv01_23 if r.dv01_23 else np.nan
            out["ratio_12"][i, j] = r.dv01_12 / r.dv01_23 if r.dv01_23 else np.nan

    out["expiries"] = exps
    out["tenors"] = tens
    return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def label_years(y: float) -> str:
    """0.5 -> '6m', 1.5 -> '18m', 10 -> '10y'."""
    if y < 1.0:
        m = y * 12.0
        return f"{m:g}m"
    if abs(y - round(y)) > 1e-9:
        return f"{y * 12:g}m"
    return f"{y:g}y"


def trade_name(strike: float, expiry: float, tenor: float) -> str:
    return f"{label_years(strike)}{label_years(expiry)}{label_years(tenor)}"


def parse_period(s: object) -> float:
    """'6m' -> 0.5, '18m' -> 1.5, '10y' -> 10.0, '2' -> 2.0, 'W1' -> 1/52."""
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    txt = str(s).strip().lower().replace(" ", "")
    if not txt:
        raise ValueError("empty period")
    unit, num = txt[-1], txt[:-1]
    try:
        if unit == "m":
            return float(num) / 12.0
        if unit == "y":
            return float(num)
        if unit == "w":
            return float(num) / 52.0
        if unit == "d":
            return float(num) / 365.0
        return float(txt)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"cannot parse period {s!r}") from exc
