# USU / Clean-Basis Investigation — Handoff Report

Date: 2026-07-30. Repo: `neutron-standards-evaluation`, branch
`eval_exact_hessian` (off `eval_recommend`). gmapy: submodule `gmapy/` and
development clone `/home/gschnabel/Seafile/OmegaSpace/gmapy`, both on branch
`feature_linear_constraints` at commit `060e1b5` (submodule `origin` = the
clone; sync by committing in the clone and `git fetch && git merge --ff-only`
in the submodule).

Everything in `evaluation/` and `data/` described below is **uncommitted**.
gmapy additions are committed (`bb43e70` SoftplusHalfNormal, `060e1b5`
SoftplusHalfCauchy, both with unit tests; suite
`tests/test_distribution_utility_classes.py` 17/17).

## 1. The problem

The evaluated uncertainties of U5(n,f)/Pu9(n,f) around 1–5 MeV were deemed
too small in STD2017; USU components (per-dataset energy-knot bands) were
added but did not prevent uncertainty shrinking. This investigation asked
why, and what a defensible USU methodology looks like. Two threads emerged:

1. **The database carries hidden uncertainty inflation** introduced by the
   DATP preprocessing (2017): points deviating more than ULI=3 sigma from
   the a-priori get uncertainty component 3 (`CO[:,2]` in the JSON, `F(3)`
   in Fortran; `DATP.FOR:466-482`, `datpy/reduction.py:90-106`,
   `ULI=3.0` in `datpy/constants.py`) enlarged until the deviation is
   exactly 3 sigma. The inflation also changes merged *values* (it feeds
   the point-merge weights). Column 12 (total unc) is all zeros in
   `data.json` — never exported.
2. **The statistical model lacked an arbitration tier**: shared channel
   bands cannot distinguish which of two conflicting datasets of the same
   observable is wrong; per-dataset bands average away and provide no
   floor. Both tiers are needed.

## 2. Database provenance and the ULI de-inflation

Provenance chain of `data/data.json` (this branch): `data2017.gma`
(post-DATP, inflation baked in) → +NIFFTE TPC update (commit `f3784a8`,
prepared from `vladimir/email*/GMDATA_TPC*.CRD`) → that's it *on this
branch*. The 2025 US/EU package (LA-UR-25-32229-1, datasets 6000-6014 +
8002) was merged via `data/merge_crd_datasets.py` only on branch
`PrepStdRelease2026_IncludingNewEuropeanUSExperiments` (commit `9fb607c`
holds originals in `data/orig_crd_files/` incl. `DAT.INP`).

**Reduction reproduction**: the manual workflow (per gschnabel) is
`python -m datpy.datpy --legacy --no-reduce --output <unred.json>` (cwd
holds `DAT.INP` + `GMDATA.CRD`) then
`python -m datpy.datpy --input <unred.json> --output <red.json>`.
This reproduces Denise's 2025 reduced json **byte-identically**. ULI=0
variant: `usu_analysis/uli_reduce.py` (patches `datpy.reduction.ULI`
before running step 2). Best reproduction of `data2017.gma` from CRD lives
in `/home/gschnabel/Seafile/OmegaSpace/std2017-repro` (scenario S16,
0.01% on all reactions; needs DATP-initial + bin-boundary fix + hand-edit
scripts + priors injected from the target — see its `reports/`).

**Database variants built** (in `data/`, uncommitted):
- `data_newds_uli3.json` — data.json with 6000/6001/6002/8002 swapped to
  the 2025 vintage, clamping ON (isolates vintage effect).
- `data_newds_uli0.json` — same, clamping OFF (**the working uli0 basis**).
  ULI=0 vs ULI=3 on these datasets: 173 comp3 changes (exp_6007 +517pp max,
  8002 +44pp, NIFFTE +31pp), values shift up to 43% (exp_6000).
- Failed approaches (do NOT retry): zeroing `CO[:,2]` wholesale → 449
  zero-uncertainty points → singular covariance; capping at dataset median
  → near-singular blocks (comp3 carries the per-point uncorrelated
  variance). A disabled `DESPIKE_COMP3` block remains in
  `01_model_preparation.py` as documentation.

**U5-network finding**: direct U5 datasets (abs MT:1, shape MT:2) *never*
trigger ULI. The U5-related ratio datasets in data.json mostly carry
uninflated comp3 already (json comp3 == our ULI=0 output for
805/821/828/1030/8016-8018; values differ 5-46% because they came through
other reduction paths/iterated priors). The de-inflation lever for U5 is
exhausted by Tier-1. Residual unknowns: 54 hand-edited TOF datasets
(NS 5xx, values+unc edited by the 2017 team) and 49 grid-mismatched.
Tier-2 (legacy bulk) needs the std2017-repro S16 apparatus with ULI=0 on
stdin (DATP-initial reads ULI interactively).

## 3. Statistical model evolution (all on the uli0 basis)

Pipeline scripts take `EVAL_DB_PATH` and `EVAL_OUTPUT_DIR` env vars
(defaults `../data/data.json`, `output/` — baseline behavior unchanged).
Stage-02 drivers for the variants live in `usu_analysis/run02_*_warmstart.py`
(run from `evaluation/`, they hardcode their `OUTDIR`).

| variant | stage-01 script | output dir | model |
|---|---|---|---|
| baseline | `01_model_preparation.py` | `output/` | shared channel USU (53 knots), obj −19393.63 |
| v1 | `01_model_preparation_dsdisc_sp.py` | `output_dsdisc_sp_uli0/` | per-dataset Legendre bands only, softplus |
| v2 | `01_model_preparation_dsdisc_v2.py` | `output_dsdisc_v2_uli0/` | two-tier: shared floor + per-dataset (no constant mode for shape MT:2/4/8/9), half-normal(0.2) restraint |
| v3 | `01_model_preparation_dsdisc_v3.py` | `output_dsdisc_v3_uli0/` | regularized horseshoe, tau fitted w/ half-Cauchy — tau RAN AWAY (1596%) |
| v3b | `01_model_preparation_dsdisc_v3b.py` | `output_dsdisc_v3b_uli0/` | **reference model**: horseshoe w/ tau anchored by half-normal(0.05); slab c=0.3 |

Model form: `C(u) = C_exp + S diag(d(u)) S^T` via `LowRankCovarianceModel`
(Woodbury closed forms; `diag_fun` is the only place the parameterization
lives). Per-dataset columns: Legendre (1, t, P2(t)) on the dataset's
log-energy span, tied to one magnitude via gather. v3b `diag_fun`:
shared cols `softplus(u)^2`; per-dataset cols
`c^2 t^2 l^2/(c^2 + t^2 l^2)` with `l=softplus(u_d)`,
`t=softplus(u[-1])` (global scale, the LAST covpar), priors
`SoftplusHalfCauchy(1.0)` on locals, `SoftplusHalfNormal(0.05)` on tau.

**Hard-won lessons** (do not relearn):
- NEVER parameterize variance components as raw `u` (`u^2` in diag):
  every u=0 is a stationary saddle, the objective is quartic → the
  optimizer crawls for 2500+ iterations without converging. Softplus
  converges in ~650-1050 iterations (~30 min total). Moment-based warm
  start (GLS projection of residuals onto each dataset's Legendre modes,
  see the drivers) is the multimodality insurance for horseshoe MAP.
- MAP + plain half-Cauchy on a global scale = runaway. Anchor tau
  (half-normal) for MAP; a real tau posterior needs the sampling stage.
- The base covariance operator in the pickles is
  `Composition(BlockDiag(L), BlockDiag(L^T))` — Cholesky-factored. Do NOT
  eigendecompose the first factor thinking it is C (mistake made once:
  produced phantom "indefiniteness").
- AutoGraph reads source lazily: never edit scripts/gmapy source while a
  TF run that references them is alive.
- Without any robustness tier, the uli0 basis MELTS DOWN (GLS meltdown,
  obj −11868 vs −19216; warm start confirms it is the true optimum of
  that posterior, not a bad basin): sub-percent-uncertainty discrepant
  datasets drag the whole ratio network. ULI=3 was the 2017 substitute
  for a robustness model.

## 4. Key results (v3b unless stated)

- chi2/dof with bands: 1.699 (v2 1.89, v1 2.0). Training chi2 rejects
  joint calibration; see LODO below for the decomposition.
- Culprit ranking: Pu9/U5 ratio complex carries ~10 datasets with 14-29%
  bands (8002, 536, 602, 600, 631, 6001, 6002, 1012, 1029, ...); many
  crowd the slab cap 0.3, i.e. cap-limited, not measured.
- **Exonerations by refinement**: exp_6000 (v1: 31% band) → 0% band,
  1.2% median residual once shape datasets lose the constant Legendre
  mode (normalization degeneracy artifact). Same for exp_128 (111%→4%)
  and exp_124 (85%→0%).
- Shared floors concentrate in Pu9 channels: shape Pu9 24%@1MeV, abs
  Pu9 12%@0.1MeV, Pu9/U5 shape 16%@10keV, U8/U5 ~12-19% at grid edges.
  Known artifact: 30% (v2: 104%) shared knot @1keV abs Pu9/U5 anchored
  by a single datapoint below 100 keV (boundary-knot support issue).
- **Variance decomposition @2 MeV** (`usu_analysis/variance_decomposition2.py <outdir>`):
  Pu9(n,f) sd 0.924% (baseline 0.576%): ratio-bridge info route
  collapsed 85%→7%, SACS ratio now dominant (49%); U5(n,f) sd 0.569%
  (baseline 0.535%): composition unchanged.
- **Two-route U5 test** (`usu_analysis/u5_route_tension.py`): direct-U5
  (719 rows) vs ratio-borne U5: raw covariance tension −2.3σ@0.5MeV,
  −4.8σ@1MeV, −1.5σ@2MeV (indirect wants U5 1-2.5% higher); with
  per-dataset bands ≤1σ in 0.5-2 MeV (a −6σ@5MeV artifact appears,
  undiagnosed). Attribution ambiguity is fundamental: same residuals
  explainable by ~10 defective ratio datasets OR a 0.5-1% shared U5 band;
  MAP picks the former (sparsity economics). This is THE methodological
  decision to surface.
- **Variant spread of central values** (beyond-model, model-form term):
  Pu9@2MeV 3.9% across 5 defensible variants (1.950-2.028 b) — 4x the
  in-model sd; U5@2MeV 0.41% (< in-model sd).
- **LODO predictive calibration** (`usu_analysis/lodo_calibration.py`):
  claimed (own band incl.) global chi2/n 1.345 → x1.16; out-of-sample
  (own band excl.) 13.2 → x3.6, but entirely Pu9-complex (abs Pu9 43,
  Pu9/U5 ratios 47-78, Pu9 sums 57-121); all non-Pu9 channels ≤2.1;
  **U5 direct channels 0.79/0.83/0.28 — overconservative, no inflation
  needed**.
- Bottom line for the original question: U5@2MeV extra uncertainty is
  assertion-only (data actively refuse to demand it); Pu9@2MeV should
  not be quoted below ~1.5-2% (variant spread) regardless of variant.

## 5. In flight / next steps

1. **MCMC on v3b is RUNNING** (`03_mcmc_sampling_v3b.py`,
   `EVAL_OUTPUT_DIR=output_dsdisc_v3b_uli0`, HMC with exact-Hessian
   bijector, 3k burnin + 15k samples, launched 2026-07-29 23:37, est.
   3-6 h; writes `output_dsdisc_v3b_uli0/03_mcmc_sampling_output.pkl`).
   First checks: acceptance/step-size diagnostics in `tracing_info`
   (flat lambda-directions may have troubled adaptation), tau marginal,
   band polarization/bimodality, sampled Pu9/U5@2MeV sd vs Laplace.
2. Fix the 1keV boundary-knot support (prune/merge knots vs data support)
   and diagnose the −6σ@5MeV route-tension artifact.
3. Tier-2: replay std2017-repro S16 with ULI=0 on stdin; separately
   review the 54 hand-edited TOF datasets (NS 5xx) — the main place
   hidden inflation could still lurk, entangled with 2017 manual edits.
4. Reporting framework: in-model sd x CV calibration + variant-spread
   model-form term (numbers above); decide the route-tension attribution
   (dataset blame vs shared U5 band) as an explicit prior choice.
5. Merge decisions: gmapy feature branches into dev; commit the
   evaluation-repo scripts/data variants (user's call; nothing pushed
   anywhere).

## 6. File inventory

- `evaluation/usu_analysis/` — all analysis scripts (chi2 consistency,
  variance decomposition, route tension, LODO, warm-start drivers,
  ULI-patched reduction, DATP forensics, crd matching). Most take the
  output dir as argv[1]. Run from `evaluation/` with
  `PYTHONPATH=/home/gschnabel/Seafile/OmegaSpace/gmapy ../venv/bin/python`.
- `evaluation/output*` dirs — per-variant pickles (01 model, 02 fit).
  `output/` = restored baseline (obj −19393.6313). `output_deep_mode_backup/`,
  `output_baseline_eval_recommend/` = older states.
- Session memory (auto-loaded for the LLM): `std2017-provenance-uli-inflation`,
  `two-tier-usu-model` memory files hold condensed versions of this report.
- Earlier infrastructure (previous sessions): VectorizedCompoundMap, exact
  Hessians for all likelihood classes, `determine_MAP_estimate_precond_lbfgs`
  (two-phase bold/saddle-free, batched line search, checkpointing),
  `LowRankCovarianceModel`, `LogBarrier` — all on `feature_*` branches of
  gmapy, tests green.
