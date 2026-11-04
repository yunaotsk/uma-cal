# UMA-Cal analysis plan (pre-specified)

**Status:** written and committed **before any UMA prediction exists on any test split (A0, A, B, C)**. No test-split prediction has been made. The only inference permitted before the user approves sizes is a 20-structure Cal-only benchmark, which runs after this commit. Any change after results exist must be recorded in the "Deviations" section at the end, with a date and the reason. Nothing here is tuned on results.

## 1. Question and design
Do split-conformal intervals around UMA (`uma-s-1p2p1`, task `omat`, `batch`, CPU) energy predictions keep their nominal coverage as the test chemistry moves away from the calibration set (Cal)?
A finding that coverage fails under shift is a valid result. "Shift" is relative to **Cal only**; UMA may have seen these structures (OMat24 validation is a random split of the training pool), so no claim is made about UMA's training distribution.

Data: OMat24 validation `rattled-1000-subsampled`; splits in `data/splits/{cal,A0,A,B,C}.csv` (definitions in `reports/split_proposal.md` §4; sizes may change after the timing benchmark and your approval).
**Unit of analysis = reduced composition.** Each split has one structure per composition, and A, B, C at most one composition per chemical system. The five splits are pairwise disjoint on composition, `parent_id` and `sid`.

## 2. Primary metric
- **Score:** `s_i = |ŷ_i − y_i|` in **eV/atom**, with `ŷ_i` the UMA total energy divided by the atom count and `y_i` the raw DFT total energy divided by the atom count.
- **Interval (split conformal):** with `n` Cal scores and `α = 0.10`, `q` = the `k`-th smallest Cal score, `k = ⌈(n+1)(1−α)⌉` (infinite if `k > n`). Interval = `ŷ ± q`.
  A single `q` from the **entire** Cal set serves every test split (marginal calibration; no per-stratum recalibration).
- **Primary metric: marginal coverage at 90%** on each of A0, A, B, C: `cov = mean(s_i ≤ q)`.
- **Headline uncertainty: composition-level standard error** `√(cov(1−cov)/n_compositions)` with a Wilson 95% interval, since compositions are the independent units. Structure-level SEs are not reported as results.
  Supplementary: a Cal bootstrap (1,000 resamples of Cal, recomputing `q` and coverage each time) to include finite-Cal spread, and a composition bootstrap for each test split. The planning value of the finite-Cal spread is √(0.09/(n_cal+2)) ≈ 0.0067 at n_cal = 2,000.
- **Interval width:** every test split shares one `q`, so width = `2q` is the same by construction. Report `q` and each split's median absolute error instead of comparing widths.
- **Label:** raw DFT total energy (the stored calculator energy), which the OMat24 model card says is the training label. The label check on Cal structures only (`reports/cal_benchmark.md`) is informational. If it shows a gross inconsistency
  (median absolute error to the raw label more than twice that to the MP2020-corrected label), flag it to the user and do not switch labels unilaterally. Test structures are never used for this choice.

## 3. Pipeline gate and execution order
1. Predict Cal (cached). 2. Predict **A0** and apply the gate below. 3. Only if the gate passes, predict A, B, C. Predictions are cached by `sid` with the settings and library versions recorded; nothing is re-predicted under different settings.

**A0 pass criterion:** A0 coverage within **about 2 points of 90% (0.88 to 0.92)**. Otherwise treat it as a **pipeline bug** and do not interpret A, B or C until it is explained.
Honest false-alarm rate: with a combined SE ≈ 0.0116 (n_A0 = 1,000, n_cal = 2,000), a correct pipeline lands outside ±2 points about 8.5% of the time (z ≈ 1.72). A failure therefore triggers, in order:
(a) the diagnostic checklist (units and per-atom normalisation, label column, cache/version mixing, duplicated `sid`s, Cal/A0 disjointness, NaNs), then (b) an independent second A0 draw (`A0b`, different seed, same rules), before concluding anything. The user decides on (b).

## 4. Stratifications (descriptive diagnostics)
Marginal conformal coverage does not imply conditional coverage, so strata are reported as diagnostics, with `n` and a Wilson interval each. Strata with `n < 30` are pooled or marked unreliable. No multiplicity correction; strata beyond the primary metric are exploratory.
1. **Ln vs An (split C):** lanthanide-only, actinide-only, both; and by number of f-block elements (1, 2, 3 or more).
2. **Composition order** (binary, ternary, quaternary or more), within every split.
3. **Chemical family under both rule sets** (rules used in the splits, and the alternative rules; `scripts/family_sensitivity.py`); families with `n < 30` in a split are pooled into "other".
4. **Similarity to Cal** (`max_jaccard_to_cal`): bins `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 0.75)`, `[0.75, 1)`, `1.0`, pooled over A0, A, B, C and within each split; plus a descriptive logistic fit of the coverage indicator on similarity.

## 5. Order-matched C vs B comparison
C has more quaternary compounds than B, so the raw C − B gap mixes shift with order. **Matched difference:** `Δ = Σ_k w_k (cov_C,k − cov_B,k)` over order strata `k` (binary, ternary, quaternary or more) with weights `w_k` equal to **C's order distribution**
(B standardised to C's mix). If any stratum has fewer than 30 compositions in either split, binary and ternary are merged. Report the raw difference too. Uncertainty: composition bootstrap within strata (2,000 resamples), 95% percentile interval.
Secondary matching on atom-count bins (≤12, 13–16, >16 atoms). No equivalence margin is pre-specified for Δ: report the estimate with its interval and interpret it against the ±2-point band used for A0.
The f-block label-noise confounder (plain PBE, no +U) cannot be removed by matching and must be stated alongside Δ.

## 6. Sensitivity analyses (pre-specified)
Each is applied **identically to Cal and all test splits**, recomputing `q`, and compared with the primary result. Report the number of structures removed per split.
- **S1:** exclude structures with max |force| > 50 eV/Å.
- **S2:** exclude structures with raw energy/atom > 0.
- **S3:** exclude both.
- **S4:** MP2020-corrected energy as the label (structures lacking it, about 3%, dropped); descriptive only.
- **S5:** family stratification under the alternative rule set (already part of §4.3).
No filter is applied in the primary analysis.

## 7. Interpretation rules
- Coverage "deviates from nominal" when its 95% interval (including finite-Cal spread) excludes 0.90. Report the size of the deviation in points; there is no pass/fail verdict for A, B or C.
- A conclusion about shift needs the same direction in the primary result and in S1 to S3.
- Expected resolution: at n = 1,000 a split is resolved to about ±1.9 points (composition-level, 95%) and a two-split difference to about ±2.6 points; A (n ≈ 473) to about ±2.7 points. Differences of 1 to 2 points cannot be distinguished.
- Report every split, including ones that look unremarkable. Report failures, aborted runs and deviations.

## 8. Known limitations (stated up front)
- UMA may have been trained on these structures; shift is relative to Cal only.
- A0 is nearly as system-novel as B (about 5% of A0 systems occur in Cal). B's contrast with A0 is small by construction.
- A is small (473 compositions) and drawn from multi-composition systems, so its mix differs from Cal.
- C differs from Cal in more than element family: order mix, cell size, and noisier f-block labels.
- Compositions that share elements or prototypes are correlated, so the composition-level SE is a lower bound on the true SE. The design effect is not estimated.
- One seed for the splits; split-to-split variability is not measured.

## Deviations
(none recorded)

## Amendment 1 (2026-11-03): final sizes, secondary levels, execution order, A0 gate
Made **before any prediction on A0, A, B or C**. Sections 1 to 8 above are unchanged; where they quote planning sizes (n = 1,000, n_cal = 2,000, an 8.5% gate false-alarm rate), this amendment supersedes those numbers.

**1. Final sizes (frozen).** Splits are frozen in `reports/split_freeze.md` (commit `4de8962`, SHA256 per file): **Cal 4,000; A0 2,000; A 713 (all available); B 2,000; C 2,000** compositions (one structure each), `max_atoms = 24`, seed 20241220.
Composition-level SE at nominal 90% coverage: A0, B, C 0.0067 (95% half-width ±1.3 points); A 0.0112 (±2.2 points). Finite-Cal spread √(0.09/4002) = 0.0047; combined SE 0.0082 for A0, B, C and 0.0122 for A.
A pairwise difference between two n = 2,000 splits has SE ≈ 0.0095 (±1.9 points). With a combined SE of 0.0082 the ±2-point A0 gate has a false-alarm rate of about 1.5% for a correct pipeline (it was 8.5% at the earlier planning sizes).
These resolve differences of about 2 points; they cannot resolve differences of about 1 point.

**2. Secondary coverage levels.** In addition to the primary 90% level (α = 0.10), report marginal coverage at **80%** (α = 0.20) and **95%** (α = 0.05), each from its own split-conformal quantile on the Cal absolute errors, with `k = ⌈(n+1)(1−α)⌉` for n = 4,000: k = 3,201 (80%), 3,601 (90%), 3,801 (95%).
Same procedure, same composition-level SEs and Wilson intervals (planning SE at n = 2,000: 0.0089 at 80%, 0.0067 at 90%, 0.0049 at 95%). The 80% and 95% levels are reported for A0, A, B and C overall and, descriptively, in the §4 strata and the §5 order-matched comparison.
They are secondary: the primary metric, the A0 gate and the §7 interpretation rules stay on 90%. No multiplicity correction; a deviation is interpreted only when it is consistent across the three levels or clearly stated otherwise.

**3. Execution order and the A0 gate.** Predictions run in this order: **Cal, then A0 (gate check), then A, B, C.**
The A0 gate decision (|coverage₉₀(A0) − 0.90| ≤ 0.02, with `q` from Cal) is computed from Cal and A0 predictions only and **is made, and recorded with `record_a0_gate` in `src/pred_cache.py`, before any A, B or C prediction exists or is looked at.**
This is enforced in code: `predict_cached` refuses any structure outside Cal and A0 until the gate record says `passed`. The 80% and 95% coverage on A0 may be inspected as diagnostics but cannot change the gate decision.
If the gate fails, A, B and C are not predicted. The §3 diagnostics run first, and a second A0 draw (`A0b`) requires its own amendment and freeze before it is used. Editing the gate record by hand to unlock A, B or C would be a protocol violation and would have to be reported as one.
