# Vol surface PCA + hierarchical Nelson-Siegel-Svensson — reading list

Compiled 2026-06-25. Organized by the steps of the proposed pipeline (see chat / project notes
for the methodology writeup). All links verified live.

## 1. PCA / level-slope-curvature factor structure (curves and vol surfaces)

- Litterman, R. & Scheinkman, J. (1991), "Common Factors Affecting Bond Returns," *Journal of
  Fixed Income* 1(1), 54-61. The original level/slope/curvature decomposition — every PCA-on-a-curve
  exercise since traces back to this.
  https://jfi.pm-research.com/content/1/1/54
- Cont, R. & Da Fonseca, J. (2002), "Dynamics of Implied Volatility Surfaces," *Quantitative
  Finance* 2(1), 45-60. The direct vol-surface analogue: Karhunen-Loeve (continuous PCA) decomposition
  of SPX/FTSE implied vol into level / term-structure / skew factors, with a factor model for their
  dynamics. Closest existing academic match to what you're doing.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0300.00090 (companion paper with Durrleman,
  "Stochastic Models of Implied Volatility Surfaces")
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=295859
- Alexander, C., "Principal Component Analysis of Volatility Smiles and Skews." Same idea applied
  across strike/moneyness instead of tenor — relevant if you extend this to the full surface later.
  https://www.researchgate.net/publication/24121132_Principal_Component_Analysis_of_Volatility_Smiles_and_Skews
- Papanicolaou, A., "PCA for Implied Volatility Surfaces," *Journal of Financial Data Science*
  (2020). Recent, quant-practitioner-oriented treatment.
  http://math.stanford.edu/~papanico/pubftp/jfds.2020.1.032.full.pdf
- Practitioner-level (non-academic), same logic applied to swap curves — quick reads:
  https://www.clarusft.com/principal-component-analysis-of-the-swap-curve-an-introduction/
  https://www.nber.org/system/files/working_papers/w16549/w16549.pdf ("An Empirical Analysis of
  the Swaption Cube," NBER WP 16549 — full expiry x tenor x strike cube, not just term structure)

## 2. Nelson-Siegel / Svensson, dynamic & hierarchical/Bayesian extensions

- Svensson, L.E.O. (1994), "Estimating and Interpreting Forward Interest Rates: Sweden 1992-1994,"
  NBER WP 4871. The original 4-factor extension of Nelson-Siegel (adds the second hump term
  beta3/lambda2) — this is the "Svensson" in your question.
  https://www.nber.org/papers/w4871
  Practical companion: BIS Papers No. 25, "A technical note on the Svensson model."
  https://www.bis.org/publ/bppdf/bispap25l.pdf
- Diebold, F.X. & Li, C. (2006), "Forecasting the Term Structure of Government Bond Yields,"
  *Journal of Econometrics* 130(2), 337-364. Reinterprets NS parameters as latent level/slope/curvature
  factors and models their dynamics — the template for treating curve-fitting as a factor model
  rather than a daily snapshot.
  https://www.sas.upenn.edu/~fdiebold/papers/paper49/Diebold-Li.pdf
- Christensen, J.H., Diebold, F.X. & Rudebusch, G.D. (2011), "The Affine Arbitrage-Free Class of
  Nelson-Siegel Term Structure Models," *Journal of Econometrics* 164(1), 4-20. The state-space /
  Kalman-filter version with no-arbitrage restrictions — the template for the "dynamic" extension
  below.
  https://www.nber.org/papers/w13611
- Das, S., "Modeling Nelson-Siegel Yield Curve using Bayesian Approach" (arXiv:1809.06077). Closest
  existing match to your "hierarchical Svensson" idea — Bayesian shrinkage on NS factors.
  https://arxiv.org/abs/1809.06077
- "Term structure shapes and their consistent dynamics in the Svensson family" (arXiv:2410.08808,
  Oct 2024). New. Characterizes which curve shapes the Svensson family can/can't produce and how
  they evolve consistently without arbitrage — directly relevant to the "outliers that can't be fit
  nicely" question, since it tells you which shapes are structurally outside what NSS can represent.
  https://arxiv.org/abs/2410.08808
- "A Comparative Analysis of Parsimonious Yield Curve Models... Nelson-Siegel, Svensson and Bliss,"
  *Computational Economics* (2021). Practical comparison of which variant fits best and when.
  https://link.springer.com/article/10.1007/s10614-021-10113-w
- "Forecasting the Term Structure of Interest Rates with SPDE-Based Models" (arXiv, Dec 2025). New —
  stochastic-PDE alternative to NSS for term-structure dynamics.
  https://arxiv.org/html/2512.23910v1

## 3. Calibration: global optimizers for the nonlinear decay parameters (the "cross-entropy" question)

- Gilli, M. & Schumann, E. et al. (NMOF package), "Fitting the Nelson-Siegel-Svensson model with
  Differential Evolution." The standard reference for *why* gradient methods get stuck on NSS
  (multimodal loss surface in lambda1/lambda2) and how a population-based global optimizer fixes
  it. The Cross-Entropy Method is a direct cousin of this approach.
  https://cran.r-project.org/web/packages/NMOF/vignettes/DEnss.pdf
- Lakhany, A., "Calibrating the Nelson-Siegel-Svensson model by Genetic Algorithm" (arXiv:2108.01760).
  Same problem, GA instead of DE — good comparison point for a CEM implementation.
  https://arxiv.org/pdf/2108.01760
- Rubinstein, R.Y. (1999), "The Cross-Entropy Method for Combinatorial and Continuous Optimization,"
  *Methodology and Computing in Applied Probability* 1, 127-190. CEM itself — what you'd actually
  implement as the calibration engine.
  https://link.springer.com/article/10.1023/A:1010091220143

## 4. Beyond NSS — ML approaches to the surface (the "smarter than CEM" extension)

- Bergeron, M., Fung, N., Hull, J. & Poulos, Z. (2021), "Variational Autoencoders: A Hands-Off
  Approach to Volatility" (arXiv:2102.03945). VAE learns the surface's low-dimensional latent
  factors directly from data with no parametric form imposed — competes with / extends the PCA
  approach.
  https://ideas.repec.org/p/arx/papers/2102.03945.html
- "Arbitrage-Free Implied Volatility Surface Generation with Variational Autoencoders," *SIAM J.
  Financial Mathematics* (arXiv:2108.04941). Same group, adds no-arbitrage constraints.
  https://arxiv.org/abs/2108.04941
- "Controllable Generation of Implied Volatility Surfaces with Variational Autoencoders"
  (arXiv:2509.01743, 2025). New — lets you steer the generated surface toward a target
  level/slope/skew, i.e. a generative version of "what does my surface look like if PC2 moves 2
  std devs."
  https://arxiv.org/pdf/2509.01743
- "Robust Yield Curve Estimation for Mortgage Bonds Using Neural Networks" (arXiv:2510.21347, Oct
  2025). New — robust/outlier-resistant neural curve fitting, relevant to the residual/outlier step.
  https://arxiv.org/pdf/2510.21347
- Gatheral, J. & Jacquier, A. (2014), "Arbitrage-Free SVI Volatility Surfaces," *Quantitative
  Finance* 14(1), 59-71. If you extend this to the strike/skew axis (the "vol add-on"), SVI/SSVI is
  the standard parametric form — same spirit as NSS but for the smile.
  https://arxiv.org/abs/1204.0646

## 5. Hierarchical clustering of regimes / outlier patterns (the "1 -> 2 -> 3 pattern" step)

- Marti, G., Nielsen, F., Binkowski, M. & Donnat, P. (2021), "A Review of Two Decades of
  Correlations, Hierarchies, Networks and Clustering in Financial Markets." Survey of hierarchical
  clustering methods on financial data — good map of options beyond plain Ward linkage.
  https://arxiv.org/abs/1703.00485
- "Representation Learning for Regime Detection in Block Hierarchical Financial Markets"
  (arXiv:2410.22346, Oct 2024). New — learns regime structure directly; comparable to clustering
  residual shapes into level-like / slope-like / idiosyncratic regimes.
  https://arxiv.org/html/2410.22346v1
- "Improving S&P 500 Volatility Forecasting through Regime-Switching Methods" (arXiv:2510.03236,
  Oct 2025). New — soft Markov-switching + spectral clustering for vol regimes, a statistical
  alternative to hierarchical clustering for the same goal.
  https://arxiv.org/abs/2510.03236
