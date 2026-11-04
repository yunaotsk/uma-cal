# UMA-Cal: OMat24 audit and proposed distribution-shift splits (for review)

Status: **proposal only.** No UMA inference, calibration or training has been run. All numbers below come from
`results/dataset_audit.json` (`scripts/audit_dataset.py`), `data/splits/split_summary.json` (`scripts/propose_splits.py`, seed 20241220) and
`data/splits/family_sensitivity.json` (`scripts/family_sensitivity.py`) unless marked otherwise.

Reproduce (archive already in `data/raw/`; the loader finds the nested `.aselmdb` recursively):
```
python scripts/audit_dataset.py --archive data/raw/rattled-1000-subsampled.tar.gz --source-url <URL in §1>
python scripts/propose_splits.py --heldout-elements f-block --max-atoms 24 --seed 20241220
python scripts/family_sensitivity.py
```

## 1. Dataset audit

| Item | Value |
|---|---|
| Source | `https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz` (corrected validation sets, uploaded 2024-12-20 per the HF dataset card) |
| Archive size / SHA256 | 79,870,946 bytes (~76 MiB; docs-expected "~77 MB") / `a5a06b29…c9fc1` (full hash in the audit JSON) |
| `du -sh data/raw` | **199 MB** (archive 76 MiB + extracted LMDB) |
| Loader | `fairchem.core.datasets.AseDBDataset` (fairchem-core 2.25.0), as shown on the HF dataset card |
| Structures | **39,785**, which matches the official count (39,785) |
| Unique reduced compositions | 31,630 (my gcd reduction) |
| Unique chemical systems (exact element sets) | 26,414 |
| Unique `parent_id`s | 32,977 |
| Unique elements | 87 (UMA paper Table 11 lists 89 for all of OMat24; the validation subset has no representatives of the other two) |
| Atoms per cell | min 10, median 14, p95 20, p99 31, max 136 (≤16 atoms: 35,095 structures) |
| Structures per composition | 1: 24,822, 2: 5,758, 3: 841, ≥4: 209 |

**Labels and units.** The dataset card says: "labeled with total energy (eV), forces (eV/A) and stress (eV/A^3)".
I use the raw DFT total energy from the ASE calculator results and divide by `natoms` for eV/atom. The
`atoms.info` field `energy_corrected_mp2020` is a *different* quantity (median offset 0, mean −1.28 eV per cell, range
−63.9 to +3.3 eV; absent for 1,234 structures). It is not the label. The fairchem `data/omat` README says "Do not use MP2020 corrections or use the MP references compounds when using OMat24 trained models". I cannot confirm that UMA's `omat`
head predicts the raw energy without running it. Verifying this is a first check for the inference stage.

Unit sanity (not a UMA comparison): energy/atom ranges from −12.1 to +0.58 eV with median −4.0; relaxed-ish unary entries are
plausible in magnitude for PBE (Th −6.4, Re −9.7, Li −1.4, Co −5.6 eV/atom; these are rattled cells, so not formation references). No duplicate (composition, natoms, energy) rows.

**Label-quality flags (not filtered, see §5):** 10 structures with energy/atom > 0; 274 with max |force| > 50 eV/Å (max 751);
4 stored `composition_reduced` strings are unreduced peroxides (e.g. `Li2 O2`); 32 compositions carry more than one prototype label.

## 2. Chemical diversity

Classification rules are in `src/dataset.py` (`chemical_family`, `tm_class`). "Strong anions" = F, Cl, Br, I, O, S, Se, Te, N.
One strong-anion class → that family (H/B/C/Si/P/As then act as cations, so phosphates are oxides); two or more → **mixed-anion**
(not forced into a single family); none → P/As pnictide, H hydride, C carbide, B boride, Si silicide, several of these →
mixed-light-element; otherwise intermetallic (includes Ge, Sn, Sb, Bi, Pb, Tl). Transition metals: 3d = Sc–Zn, 4d = Y–Cd.
The classification is by anion chemistry only; ambiguity remains for e.g. Zintl-like Sb/Bi compounds (counted as intermetallic).

**Why these rules.** The family is meant to describe the *bonding anion*, because that is what changes the chemistry a model sees. Phosphates, carbonates and
hydroxides are oxides, not "mixed-anion" compounds: P, C and H sit in the oxoanion or cation network and O is the anion. So only halogens, O, chalcogens and N
count as anions when deciding mixed-anion status. Si and B in a compound with no strong anion are reported as silicide / boride rather than lumped with intermetallics.
Genuinely multi-anion compounds (oxyhalides, oxynitrides, oxysulfides) are kept as their own **mixed-anion** class and not forced into either parent.

**Sensitivity to the rules** (`family_sensitivity.json`). The alternative rules (the `claude/vibrant-rubin-1v8122` branch: H, C, N, P and As also count as anions; B/Si/Ge/Sb form one
"metalloid-containing" bucket; noble-gas compounds are "other") change the label of **4,357 of 31,630 compositions (13.8%)**; 4,212 of those are in the eligible pool.
- 2,781 are metalloid-bucket relabels (1,575 of them intermetallic under my rules, 902 silicide, 290 boride, 14 mixed-light-element). These are mostly renames, though my "intermetallic" loses its Ge/Sb members.
- 1,576 come from the different anion set. 1,401 become mixed-anion under the alternative (485 halide, 473 chalcogenide, 230 oxide, 114 nitride, 99 mixed-light-element), i.e. compounds with an extra H/C/N/P/As.
  Total mixed-anion: 1,718 compositions under my rules, 3,119 under the alternative.
- **Strong-shift tier C does not change.** C is defined by element identity (≥1 f-block element), so its membership is independent of the family rules (0 compositions change; same 2,000 structures).
  Only its *family breakdown* shifts: 250 of the 2,000 structures get a different label (e.g. intermetallic 1,232 → 1,120, "metalloid-containing" 198 appears, mixed-anion 54 → 99).
- **A family-defined strong-shift tier would change materially.** For a hold-out of one family on the eligible non-f-block pool: mixed-anion is 1,322 compositions under my rules and 2,377 under the alternative;
  oxide 581 vs 447; chalcogenide 2,210 vs 1,846; halide 2,516 vs 2,105. A family hold-out is therefore rule-dependent, which is one more reason the proposed tier is element-based.
  Family-stratified coverage results should be reported under both rule sets, or described as rule-dependent.

| Family | Structures (all) | Compositions (all) | Structures (eligible*) | Compositions (eligible*) |
|---|---:|---:|---:|---:|
| intermetallic | 21,188 | 16,898 | 20,943 | 16,686 |
| chalcogenide | 4,429 | 3,488 | 4,350 | 3,426 |
| halide | 4,113 | 3,247 | 3,841 | 3,016 |
| mixed-anion | 2,125 | 1,718 | 1,922 | 1,554 |
| pnictide | 1,847 | 1,432 | 1,828 | 1,418 |
| oxide | 1,787 | 1,451 | 1,189 | 971 |
| silicide | 1,174 | 902 | 1,153 | 885 |
| nitride | 989 | 791 | 972 | 778 |
| hydride | 894 | 713 | 866 | 693 |
| carbide | 490 | 401 | 486 | 397 |
| mixed-light-element | 374 | 284 | 366 | 278 |
| boride | 355 | 290 | 351 | 286 |
| elemental | 20 | 15 | 0 | 0 |

\*Eligible = ≤24 atoms, no Hubbard-U treatment (rule in §3), ≥2 elements, no noble gas: 38,267 structures, 30,388 compositions, 25,417 systems.

| Transition-metal content | Structures (all) | Compositions (all) |
|---|---:|---:|
| contains 3d TM | 13,592 | 10,788 |
| contains 4d TM | 15,341 | 12,126 |
| contains both 3d and 4d | 4,372 | 3,446 |
| 3d only / 4d only / neither | 9,220 / 10,969 / 15,224 | 7,342 / 8,680 / 12,162 |

| Order | Structures | Compositions |
|---|---:|---:|
| unary | 20 | 15 |
| binary | 2,126 | 1,627 |
| ternary | 27,158 | 21,378 |
| quaternary or higher | 10,481 | 8,610 |

| Atom limit | Structures | Compositions |
|---|---:|---:|
| ≤16 | 35,095 | 27,729 |
| ≤20 | 38,721 | 30,750 |
| ≤24 | 39,044 | 31,021 |
| ≤32 | 39,499 | 31,400 |
| ≤64 | 39,600 | 31,480 |

Key observations: the set is dominated by ternary intermetallics; oxides are few (1,451 compositions) because
OMat24 chemistry is sampled from a broad hypothetical-structure pool, not from experimentally studied oxide space.
**Most chemical systems appear once**: 26,414 systems for 31,630 compositions (see §4A).
Rattled-1000 structures are strongly distorted (median max-force 7.2 eV/Å), so this subset is not a near-equilibrium benchmark.

## 3. DFT compatibility (what is and is not documented)

Verified by reading the OMat24 paper (arXiv:2410.12771, HTML) directly:

- **Functional / PAW:** PBE with PAW pseudopotentials (VASP). *"Exchange and correlation effects were calculated using the
  generalized gradient approximation and the Perdew-Burke-Ernzerhof (PBE) with Hubbard U corrections for oxide and fluoride
  materials containing Co, Cr, Fe, Mn, Mo, Ni, V, or W, following Materials Project defaults."*
- **Inputs:** pymatgen `MPRelaxSet`, VASP PBE pseudopotential **version 54** (not MP's legacy set); `Yb_3` and `W_sv` for Yb and W;
  `ALGO = Normal`. All structures are initialised **ferromagnetic**, so final magnetic states can differ from MP.
  Source: arXiv:2410.12771, §4.2 "OMat24 DFT calculation settings and details" (HTML version, read directly). This is the only statement of the Hubbard-U rule I found; it names the **elements**
  (Co, Cr, Fe, Mn, Mo, Ni, V, W) and the **anion condition** (oxide and fluoride materials).
- **Not documented in sources I could read:** the numerical U values (the paper says only "following Materials Project defaults"), how "oxide and fluoride materials" is decided (any O or F in the
  composition, or O/F as the anion; the paper and the fairchem `data/omat` README do not say, and I did not verify pymatgen's `MPRelaxSet` logic), cutoff and k-points, and per-structure flags marking which entries were run with +U. The data records contain no such flag, so my
  Hubbard rule (oxide/fluoride containing Co, Cr, Fe, Mn, Mo, Ni, V or W) is a **composition-based inference** of the documented rule and
  is not checked against actual INCAR settings. It marks 831 structures / 674 compositions.
- **Energy conventions:** the paper's Appendix C fits OMat24-specific elemental references and MP2020-style anion / GGA(+U)
  mixing corrections for *formation-energy* analysis. Training labels are raw total energies (OMAT24 model card, read via authenticated `huggingface_hub`: "Models trained using OMat24 are trained with DFT and DFT+U total energy labels."). The dataset's `energy_corrected_mp2020` is a corrected variant, not used here.

**Do the proposed comparisons change the DFT treatment?** Within the proposed splits, no structure falls under the Hubbard-U rule
(all three test sets and Cal have 0 such structures, see audit JSON), so treatment is uniform PBE, up to the inference caveat above. Residual
confounders: (i) `Yb_3` pseudopotential: 10 Yb compositions are in the f-block pool, one in split C; (ii) f-electron compounds are plain PBE
without +U (my reading of the rule above; no statement on f-element correlation treatment found), likely noisier labels. Any family comparison *across*
the U boundary (e.g. 3d oxides vs 3d sulfides) would mix PBE and PBE+U; I excluded those structures instead of comparing across.

**May `uma-s-1p2p1` have seen these structures? Possibly yes; this is not resolved by the docs.**
- OMat24 paper: train/validation/ID-test were a *random* split of the same pool (100M/5M/5M), so validation compositions and
  elements overlap with training by construction. (Separately, "OOD composition" and "OOD element" test sets exist but are not this archive.)
- UMA model card (retrieved via authenticated `huggingface_hub`, exact text): checkpoints "trained on the entirety of OC20, ODAC23, OMat24, OMC25, and OMol25 datasets".
  "Entirety" could include the validation split.
- UMA paper Table 11 lists OMat24 *training size* 100,824,585, which equals the OMat24 *train* split total on the dataset card
  (validation = 1,025,361), which points the other way. The paper concerns UMA-1/1.1; I found no statement specific to `uma-s-1p2p1` (UMA 1.2.x training is
  described only at high level in A.7).
- **Treat all of this data as possibly seen by UMA.** A split's "shift" is therefore relative to Cal only. It is not a claim about UMA's training distribution.
  Note: the model card's r²SCAN row applies to `uma-s-1p2p1-ft-r2scan`, a different checkpoint from the one we use.

(An earlier web-summary of the UMA card misattributed that r²SCAN row to the `omat` task generally; I rechecked against the primary text.)

## 4. Splits (frozen sizes; see `reports/split_freeze.md`)

**Common design.** Eligible pool: ≤24 atoms, no Hubbard-U rule, ≥2 elements, no noble gas (38,267 structures / 30,388 compositions). **One randomly chosen
structure per reduced composition** (siblings and polymorphs of a composition are dropped). A, B and C also keep **at most one composition per chemical system**.
Seed 20241220 drives all randomness (`src/dataset.py: build_splits`). Held-out set H = f-block (lanthanides + actinides, 21 elements present). No selection uses energies or forces.
Files `data/splits/{cal,A0,A,B,C}.csv` list `sid`, `parent_id`, `comp_key` (reduced composition), elements, metadata and `max_jaccard_to_cal` (max Jaccard similarity of the
structure's element set to any Cal element set; leave-one-composition-out for Cal itself). All five splits are pairwise disjoint on reduced composition, `parent_id` and `sid`
(asserted in `check_split_invariants` and re-checked on the frozen files by `tests/test_split_files.py`). [split_comparison.md](split_comparison.md) compares family, order, atom-count and similarity distributions.

**Construction.** Cal and A0 are drawn *jointly*: one random permutation of the non-B-reserved familiar pool, first n_cal → Cal, next n_a0 → A0, rest → leftover. Cal is therefore a uniform random sample and exchangeable with A0.
(Forcing A's same-system mates into Cal would have broken that exchangeability.) The 20-structure Cal benchmark set (`reports/cal_benchmark.md`) is still inside the frozen Cal, because the 4,000-composition Cal extends the earlier 2,000-composition Cal drawn from the same permutation.

**Targets and actual capacity (frozen run).** Every requested size was reached; nothing was lowered.

| Split | Definition | Target | Realized | Capacity with this Cal and A0 |
|---|---|---:|---:|---:|
| Cal | uniform random draw from the non-B-reserved familiar pool (no f-block) | 4,000 | 4,000 | n/a |
| **A0** exchangeable control | drawn jointly with Cal from the same pool; composition-disjoint | 2,000 | 2,000 | n/a (fixed draw) |
| **A** system-matched control | leftover composition whose chemical system is **present** in Cal | all available | 713 | 713 |
| **B** mild shift | system **absent** from Cal; all elements familiar; every element pair co-occurs in Cal | 2,000 | 2,000 | 3,941 |
| **C** strong shift | ≥1 held-out f-block element, absent from Cal, A0, A, B | 2,000 | 2,000 | 9,454 (11,273 compositions) |

A is the only split limited by capacity (713 compositions). Its capacity grows slowly with Cal size and shrinks as A0 grows, since both draw from the same pool; with this Cal and A0 it is exactly 713.

### A0. Exchangeable control (pipeline check)
Random composition-disjoint holdout from the same familiar pool as Cal, exchangeable with Cal by construction, so split-conformal coverage on A0 should be ≈90% (pre-specified gate in `reports/analysis_plan.md`).
It looks like Cal on the frozen files: family chi-square p = 0.56 (alternative rules p = 0.43), order p = 0.17, atom-count KS p = 0.58, similarity KS p ≈ 1.0; the largest family-share difference is 1.3 points and the largest order-share difference 2.2 points.
**Interpretation caveat:** A0 is *not* system-matched. Only 10.5% of A0 systems occur in Cal, so by system overlap A0 is nearly as "novel" as B (0%): mean similarity to Cal 0.632 for A0 vs 0.587 for B, and 96.5% vs 94.9% of structures share ≥50% of their elements with some Cal system.
If A0 and B have similar coverage, that is expected: B's shift is confined to "unseen exact element set with familiar elements and pairs", which a random split already largely delivers. B is a clean *definition* of mild shift rather than a large *amount* of it.

### A. System-matched control
A random composition-disjoint split gives only about 10% system overlap with Cal, so A0 alone cannot separate "new stoichiometry of a seen system" from "new system". A restricts to leftover compositions whose system is in Cal (same element set, different stoichiometry).
Cost: selection bias. A comes from multi-composition systems, so its mix differs from Cal (intermetallic 63.4% vs 47.5% for Cal; ternary 83.9% vs 68.4%; quaternary+ 2.4% vs 24.4%) and it has only 713 compositions.
- *Leakage risks:* same-system, different-stoichiometry compositions are by design related (e.g. Li1O1 vs Li2O1). `parent_id` is disjoint (checked). 71.7% of A's structural prototypes appear in Cal.
- *DFT confounders:* none beyond the baseline; no Hubbard-U structures.

### B. Mild shift (unseen element combinations)
25% of familiar systems are reserved for B before Cal and A0 are drawn. B keeps those whose every element pair appears in some Cal composition and whose every element appears in ≥5 Cal compositions (`--min-comps-per-element`; not binding here: capacity is 3,941 with or without it).
A "familiar pair" can come from any Cal composition containing both elements, so B includes 101 binaries whose two elements co-occur only inside Cal ternaries (B: 101 binary, 1,326 ternary, 567 quaternary, 6 quinary).
- *Leakage risks:* B systems are new, but **related**: mean max Jaccard similarity to any Cal system is 0.587 and 94.9% of B structures share ≥50% of their elements with some Cal system (A: 100% by construction; C: 54.7%).
  "Unseen combination" does not mean "unrelated chemistry". 79.7% of B structural prototypes appear in Cal. A stricter variant (e.g. a Jaccard cap) would shrink B and remains an open decision.
- *DFT confounders:* none beyond baseline.

### C. Strong shift (held-out f-block elements)
Cal, A0, A and B contain no lanthanide or actinide. C has 1,319 compositions with exactly one f-element, 496 with two, 146 with three and 39 with four; 1,353 are lanthanide-only, 404 actinide-only and 243 contain both.
Eligible pool: 11,273 compositions in 9,454 systems. Alternatives considered (exploratory counts on the eligible pool): 5d metals (~9,700 compositions, overlaps heavily with Ln chemistry), actinides alone (~3,300; few compositions per element), element-by-element holdouts.
- *Leakage risks:* none at composition/element level. Mean max Jaccard to Cal is 0.419 (54.7% ≥ 0.5, because f-block compounds share their non-f elements with Cal). 53.8% of structural prototypes appear in Cal, so C tests **element substitution into known structure types** as much as new chemistry. f-block elements are also not 'chemically novel' for UMA, which trained on OMat24 (possibly including these very structures; §3).
- *DFT confounders:* `Yb_3` pseudopotential (2 structures in C); plain-PBE treatment of f electrons implies noisier labels, which would lower coverage for reasons unrelated to model shift.
  **Composition confounders (observed):** C has more quaternaries (36.4% vs 22.2% in A0, 28.6% in B), larger cells (85.1% ≤16 atoms vs 92–94% elsewhere), fewer 3d+4d compounds (114 of 2,000 vs 304 in A0, 296 in B), a different energy range (mean −4.69 vs −4.02 eV/atom in A0), and 61.6% intermetallic (A0 48.8%).
  Any coverage gap between B and C partly reflects these, hence the order-matched comparison and stratifications pre-specified in `reports/analysis_plan.md`.

### Sample size and the 90% coverage estimate (actual n)
**Headline: composition-level SE.** Structures that share a reduced composition (rattled copies, polymorphs) are not independent, so the number of distinct compositions is the effective sample size.
The split files hold one structure per composition, so the structure count equals the composition count. SE = √(0.9·0.1/n) from `data/splits/split_summary.json`:

| Split | Compositions n | **SE (composition level)** | 95% half-width | + finite-Cal spread (n_cal=4,000: 0.0047) → combined SE |
|---|---:|---:|---:|---:|
| A0 | 2,000 | **0.0067** | ±0.0132 | 0.0082 |
| A | 713 | **0.0112** | ±0.0220 | 0.0122 |
| B | 2,000 | **0.0067** | ±0.0132 | 0.0082 |
| C | 2,000 | **0.0067** | ±0.0132 | 0.0082 |

**Secondary: structure level.** If *every* eligible structure of each split's compositions were used (rattled copies included) the pools would be 2,595 (A0), 956 (A), 2,555 (B) and 2,438 (C) structures, giving a naive
structure-level SE of 0.0059, 0.0097, 0.0059 and 0.0061. These are **optimistic**: the extra structures are copies of the same compositions and add almost no independent information. I keep one structure per composition and do not use the structure-level figure to judge adequacy.

At n = 2,000 one split's coverage is resolved to about ±1.3 points (95%), a pairwise difference between two n = 2,000 splits (SE ≈ 0.0095) to about ±1.9 points, and A (n = 713) to about ±2.2 points on its own. This will resolve coverage
differences of about 2 points or more but **cannot distinguish differences of ~1 point**. Even the composition-level count is an upper bound on the effective n: compositions that share elements or structure prototypes remain correlated, so the true SE is likely larger; I have not estimated that design effect.
The finite-Cal spread, √(0.09/(4000+2)) = 0.0047, is a standard split-conformal result that I did not re-derive here. With these sizes a correct pipeline falls outside the ±2-point A0 gate about 1.5% of the time (combined SE 0.0082). No SE analysis exists yet for interval *width* comparisons.

### CPU feasibility
Measured on 20 **Cal** structures (`reports/cal_benchmark.md`; Apple M2, 8 GB RAM, `batch`, CPU): model load 13.3 s with cached weights, 1.08 s for the first call, then **0.35 s per structure** in steady state (0.22–0.58 s), with no significant dependence on atom count over 10–24 atoms.
Frozen sizes total 10,713 structures (4,000 + 2,000 + 713 + 2,000 + 2,000). **Extrapolation (not measured):** about 1.0 h at the mean rate, 1.7 h if every call were as slow as the slowest observed. Memory is not a concern at ≤24 atoms.

## 5. Decisions

**Settled**
1. **A:** system-matched A **plus an exchangeable control A0** (§4).
2. **C:** f-block held out. Report Ln-vs-An and order-matched stratifications (`reports/analysis_plan.md`).
3. **Label filters:** none by default. Pre-specified sensitivity analyses exclude (a) max force > 50 eV/Å and (b) energy/atom > 0.
4. **Label check:** done on 20 Cal structures only; raw DFT energy is the label (`reports/cal_benchmark.md`).
5. **Sizes (approved, frozen):** `max_atoms = 24`, `n_cal = 4,000`, `n_a0 = 2,000`, `n_b = 2,000`, `n_c = 2,000`, A = all available (713). Hashes and command: `reports/split_freeze.md`.

**Still open**
6. Matching on family / atom count across B and C beyond the pre-specified order-matched comparison.
7. A stricter B (Jaccard cap).
