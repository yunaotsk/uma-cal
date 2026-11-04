# Cal-only benchmark, label check, and sizing proposal

Real UMA inference on **20 Cal structures only** (`scripts/benchmark_cal.py`; outputs `data/benchmark/cal_benchmark.json`, `data/benchmark/size_options.json`).
No structure from A0, A, B or C was predicted: the 20 structures come from `data/splits/cal.csv`, are asserted disjoint from the other split files, and were passed through `predict_cached(..., allowed_sids=<Cal sids>)`.
Afterwards the prediction cache holds exactly 20 rows, all Cal.

## Setup
| | |
|---|---|
| Machine | Apple M2 (Mac14,2), 8 logical CPUs, **8.0 GB RAM**, macOS 15.7.2 (Darwin 24.6.0), arm64, Python 3.11.14, torch threads = 4 |
| Config | `uma-s-1p2p1`, task `omat`, inference setting `batch`, `device="cpu"`; torch 2.13.0, fairchem-core 2.25.0, ase 3.29.0 |
| Selection | 20 Cal structures, seed 20241220, purposive spread over cell size (4 at 10–11 atoms, 4 at 12–14, 6 at 15–16, 3 at 17–20, 3 at 21–24) so the cost-vs-atoms fit is anchored at both ends. Not a representative sample. |

## a. Timing
| Quantity | Value |
|---|---|
| Model load (weights already cached) | 13.3 s |
| First call (includes one-off warm-up) | 1.084 s |
| Steady state, 19 calls: mean / median / max | **0.350 s / 0.346 s / 0.578 s per structure** |
| Steady state per atom | 0.0227 s per atom |
| All 20 calls, mean | 0.387 s per structure (0.025 s per atom) |

Seconds by cell size are nearly flat in the 10–24-atom range (0.25 s at 10 atoms, 0.34 s at 12–13, 0.49–0.51 s at 16–18, 0.24 s at 22, 0.34 s at 24). The linear fit of seconds on atoms has slope 0.0072 s/atom (SE 0.0052) and r = 0.31 (n = 19), i.e. **no significant dependence on atom count in this range**.
With 19 calls per fit, per-structure cost is better described as roughly 0.22–0.58 s regardless of size than by a per-atom law.

### Cost extrapolation (EXTRAPOLATION, not measured)
Basis: 19 steady-state calls on this one machine, scaled to each split's atom counts from the split files (metadata only).

| Scenario | Structures | Hours at mean rate (0.35 s) | Hours at slowest observed call (0.58 s) |
|---|---:|---:|---:|
| Current split files (Cal 2,000, A0 1,000, A 473, B 1,000, C 1,000) | 5,473 | 0.53 | 0.88 |
| **About 6,000 structures** | 6,000 | 0.58 | 0.96 |

The linear-fit and per-atom-rate variants give 0.47–0.53 h for the current files. Not accounted for: thermal throttling over a long run, other load on the Mac, memory pressure on an 8 GB machine, and cells larger than 24 atoms (not measured).
Predictions are cached per structure, so an interrupted run resumes.

## b. Label check (20 Cal structures, no test data)
UMA total energy per atom compared with the raw DFT label (the stored calculator energy) and with `energy_corrected_mp2020`.

| | n | mean signed (eV/atom) | MAE | median abs | RMSE | max abs |
|---|---:|---:|---:|---:|---:|---:|
| UMA − raw DFT | 20 | −0.0150 | 0.0174 | **0.0033** | 0.0606 | 0.2697 |
| UMA − MP2020-corrected | 20 | +0.0798 | 0.1089 | 0.0049 | 0.1846 | 0.4248 |

- **UMA `omat` tracks the raw label, not the corrected one.** 7 of the 20 structures carry a nonzero MP2020 correction (offset 0.07–0.43 eV/atom, all negative). For those 7, MAE is 0.0063 eV/atom against raw and 0.2675 against corrected, and UMA is closer to raw in 7 of 7.
  The regression of (UMA − raw) on the MP2020 offset has slope −0.04 (r = −0.10, p = 0.66; slope ≈ 0 means "tracks raw"). This agrees with the OMat24 model card ("trained with DFT and DFT+U total energy labels") and satisfies the pre-specified rule in `analysis_plan.md` (no flag: median error to raw 0.0033 is below that to corrected 0.0049).
  Raw DFT energy stays the primary label. All 20 structures have an MP2020 value.
- **Errors are small except for one outlier.** 19 of 20 structures have |error| < 0.02 eV/atom; MAE excluding the worst structure is 0.0042. The outlier is `Co1Mn1Sr3` (10 atoms, intermetallic, 3d-only) at −0.270 eV/atom. A plausible but **untested** explanation is magnetism (OMat24 starts every structure ferromagnetic); this benchmark cannot show that.
- **Composition dependence cannot be established from 20 structures.** Descriptively: MAE is 0.0296 for intermetallics (n = 10, driven by the outlier) versus 0.0039–0.0096 for pnictide, halide, chalcogenide, oxide (n = 1–3 each); 0.0554 for 3d-containing without 4d (n = 5) versus 0.0048 for 4d-only (n = 9) and for neither (n = 6).
  Spearman correlation of |error| with the number of elements is −0.56 (p = 0.011, n = 20), with atom count −0.23 and with max force −0.32; these are exploratory, not pre-specified, and uncorrected for multiple looks. The error distribution is heavy-tailed, which matters for the conformal quantile.

## c. Sizing proposal (not decided; awaiting your approval)
Compute is not the binding constraint: even 13,000 structures is about 1.3 h at the mean rate. Statistical resolution is. Sizes below come from `build_splits` run in memory (no split files written); A is limited by system overlap with Cal and shrinks slightly as A0 grows (it draws from the same leftover pool).

| Option (Cal / A0 / B / C) | Cal | A0 | A (capacity) | B | C | Total | Composition-level SE (A0, B, C) | A SE | Cal spread | Combined SE (A0) | Hours (0.35 s / 0.58 s) | A0 gate false-alarm rate (±2 pts) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 2,000 | 1,000 | 473 | 1,000 | 1,000 | 5,473 | 0.0095 | 0.0138 | 0.0067 | 0.0116 | 0.53 / 0.88 | 8.5% |
| mid | 3,000 | 1,500 | 609 | 1,500 | 1,500 | 8,109 | 0.0077 | 0.0122 | 0.0055 | 0.0095 | 0.79 / 1.30 | 3.5% |
| **proposed** | 4,000 | 2,000 | 713 | 2,000 | 2,000 | 10,713 | 0.0067 | 0.0112 | 0.0047 | 0.0082 | 1.04 / 1.72 | 1.5% |
| large | 4,000 | 3,000 | 643 | 3,000 | 3,000 | 13,643 | 0.0055 | 0.0118 | 0.0047 | 0.0072 | 1.33 / 2.19 | 0.6% |

**Proposed values** (for your approval; hours are EXTRAPOLATIONS from 19 calls): `max_atoms = 24`, `n_cal = 4000`, `n_a0 = 2000`, `n_b = 2000`, `n_c = 2000`, A = all available (≈713).
- `max_atoms = 24`: cost is nearly flat in atom count over the measured range, so the cap is not a cost decision. 24 stays inside the measured range and keeps 98.1% of structures (39,044 of 39,785). Raising it to 32 adds about 455 structures and falls outside what was timed.
- `n_cal = 4000`: halves the finite-Cal spread of the coverage estimate (0.0067 → 0.0047) and increases A's capacity (473 → 713).
- `n = 2000` per test split: composition-level SE 0.0067 (95% half-width ±1.3 points; a two-split difference ±1.9 points), against ±1.9 and ±2.6 now. This resolves differences of about 2 points; it still cannot resolve 1 point.
  A0 with n = 2000 and Cal 4000 lowers the false-alarm rate of the ±2-point gate from 8.5% to 1.5%.
- Expected cost ≈ 1.0 h at the mean rate (1.7 h if every call were as slow as the slowest observed), plus 13 s load.
- Caveats: A stays small (≈713 compositions, SE 0.0112, ±2.2 points) however large the others get; the 8 GB RAM was not stressed by ≤24-atom cells but longer runs were not observed; the sizes cover only this one machine and one seed.

If you approve, regenerate with `python scripts/propose_splits.py --heldout-elements f-block --max-atoms 24 --n-cal 4000 --n-a0 2000 --n-b 2000 --n-c 2000 --seed 20241220`, then re-run `scripts/compare_splits.py` and `scripts/family_sensitivity.py`. Nothing has been regenerated.
