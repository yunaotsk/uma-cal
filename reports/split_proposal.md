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
- **Strong-shift tier C does not change.** C is defined by element identity (≥1 f-block element), so its membership is independent of the family rules (0 compositions change; same 1,000 structures).
  Only its *family breakdown* shifts: 106 of the 1,000 structures get a different label (e.g. intermetallic 620 → 565, "metalloid-containing" 86 appears, mixed-anion 26 → 44).
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

## 4. Proposed splits

**Common design.** Eligible pool: ≤24 atoms, no Hubbard-U rule, ≥2 elements, no noble gas (38,267 structures / 30,388 compositions). **One randomly chosen
structure per reduced composition** (siblings and polymorphs of a composition are dropped). Test sets also keep **at most one composition per chemical system**.
Cal is shared. `Cal` = 2,000 compositions. Seeds: all RNG derives from seed 20241220 (`src/dataset.py: build_splits`).
Held-out set H = f-block (lanthanides + actinides, 21 elements present). No selection uses energies or forces.
Files (`data/splits/{cal,A,B,C}.csv`) list `sid`, `parent_id`, `comp_key` (reduced composition), elements and metadata.

| Split | Definition | Structures = compositions | Max available with this Cal |
|---|---|---:|---:|
| Cal | familiar pool (no f-block), ≥1 comp per A-system plus random fill | 2,000 | n/a |
| **A** ID control | composition not in Cal; chemical system **present** in Cal | 1,000 | 1,898 |
| **B** mild shift | system **absent** from Cal; all elements familiar; every element pair co-occurs in Cal | 1,000 | 2,440 |
| **C** strong shift | ≥1 held-out f-block element, absent from Cal, A, B | 1,000 | 9,454 (11,273 compositions) |

### A. In-distribution control
**Design decision flagged for review.** A plain random composition-disjoint split would not be an in-distribution control here: because most systems
occur once, only **5.3%** of test systems appear in Cal in that case (audit `random_split_system_overlap`; pool 19,115 compositions, 2,000 Cal, 1,000 test).
It would therefore already be a mild-shift test. I made A system-matched (same element set, different stoichiometry). The cost is selection
bias: A is drawn only from systems with ≥2 compositions, so its mix differs from Cal (intermetallic 61% vs 57%; ternary 84.5% vs 77.2%, quaternary+ 2.1% vs 11.4%; see audit JSON).
- *Leakage risks:* same-system, different-stoichiometry compositions are by design related (e.g. Li1O1 vs Li2O1). `parent_id` is disjoint (checked). 71.8% of A's structural prototypes
  appear in Cal, which is inherent to an ID design.
- *DFT confounders:* none beyond the baseline; no Hubbard-U structures.

### B. Mild shift (unseen element combinations)
System-level holdout (25% of familiar systems reserved before Cal is drawn), then keep only systems whose every element pair appears in Cal. Each element must also appear in ≥5 Cal compositions (`--min-comps-per-element`; ported from the branch). With this Cal the floor is not binding: capacity stays at 2,440 compositions. This implies
that a "familiar pair" can come from any Cal composition containing both elements, so B includes 66 binaries whose two elements co-occur only inside Cal ternaries (B: 66 binary, 725 ternary, 209 quaternary+).
- *Leakage risks:* B systems are new, but **related**: mean max Jaccard similarity to any Cal system is 0.56, and 92.9% of B structures share ≥50% of their elements
  with some Cal system (versus 100% by construction for A and 43.8% for C). "Unseen combination" does not mean "unrelated chemistry". 77.7% of B structural prototypes appear in Cal. A stricter variant
  (e.g. require Jaccard ≤ 0.5) is easy to add; I have not, because it would shrink B and I want your view first.
- *DFT confounders:* none beyond baseline.

### C. Strong shift (held-out f-block elements)
Cal, A and B contain no lanthanide or actinide. C has 650 compositions with exactly one f-element, 261 with two, 73 with three, 16 with four (stratify analysis by this count).
Eligible pool: 11,273 compositions in 9,454 systems (7,959 Ln-only, 1,986 An-only). Alternatives I considered but did not choose (exploratory counts on the eligible pool): 5d metals (~9,700 compositions, overlaps heavily with
Ln chemistry), actinides alone (~3,300; includes Ac/Pa/Np/U with few compositions each), and element-by-element holdouts.
- *Leakage risks:* none at composition/element level. Max Jaccard to Cal is 0.40 (43.8% ≥ 0.5, because f-block compounds share their non-f elements with Cal). 46.5% of structural
  prototypes appear in Cal, so C tests **element substitution into known structure types** as much as new chemistry. f-block elements are also far from
  'chemically novel' for UMA, which trained on OMat24 (possibly including these very structures; §3).
- *DFT confounders:* `Yb_3` pseudopotential (1 structure in C); plain-PBE treatment of f electrons implies noisier labels, which would lower coverage for reasons unrelated to model shift.
  **Composition confounders (observed):** C has more quaternaries (37% vs 2% in A, 21% in B), fewer 3d/4d TM (52 vs 156 3d+4d), a different energy range (mean −4.56 vs −3.81 eV/atom), and 62% intermetallic.
  Any coverage gap between B and C partly reflects these. I recommend reporting stratified coverage (by family, order) and deciding on matching after review.

### Sample size and the 90% coverage estimate
**Headline: composition-level SE.** Structures that share a reduced composition (rattled copies, polymorphs) are not independent, so the number of distinct compositions is the effective sample size.
The split files already hold one structure per composition and one composition per system, so for A, B and C the file-level structure count equals the composition count (1,000 each).
SE = √(0.9·0.1/n) from `data/splits/split_summary.json`:

| Split | Compositions n | **SE (composition level)** | 95% half-width | + finite-Cal spread (n_cal=2000) → combined SE |
|---|---:|---:|---:|---:|
| A | 1,000 | **0.0095** | ±0.0186 | 0.0116 |
| B | 1,000 | **0.0095** | ±0.0186 | 0.0116 |
| C | 1,000 | **0.0095** | ±0.0186 | 0.0116 |

Other sizes (composition level): n=100 → 0.0300; 200 → 0.0212; 500 → 0.0134; 1,898 (max A) → 0.0069; 2,440 (max B) → 0.0061.

**Secondary: structure level.** If *every* eligible structure of each split's compositions were used (rattled copies included) the pool would be 1,342 (A), 1,285 (B) and 1,203 (C) structures, giving a naive
structure-level SE of 0.0082, 0.0084 and 0.0087. These are **optimistic**: the extra structures are copies of the same compositions, so they add almost no independent information. I therefore keep one structure per composition
and do not use the structure-level figure to judge adequacy.

At n=1,000 one split's coverage is resolved to about ±1.9 points (95%), and a pairwise A-vs-B or B-vs-C difference (independent samples, SE ≈ 0.0134) to about ±2.6 points. This will resolve coverage
drops of ≳3 points but **cannot distinguish differences of ~1–2 points**. Even the composition-level count is an upper bound on the effective n: compositions that share elements or structure prototypes remain correlated
(one composition per system removes only exact-system repeats), so the true SE is likely larger; I have not estimated that design effect. The finite-Cal spread, √(0.09/(2000+2)) = 0.0067, is a standard split-conformal result that I did not re-derive here.
No SE analysis exists yet for interval *width* comparisons.

### CPU feasibility
Total proposed structures: 2,000 + 3×1,000 = 5,000, median 14 atoms (A 16, B 14, C 16; max 24). The only timing I have is the smoke test (0.225 s per call for a 4-atom cell, uma-s-1p2p1, `batch`, CPU).
**I have no timing for 10–24-atom cells and have not run any; cost per structure on this machine is unmeasured.** A short timing benchmark on ~20 held-out-eligible structures (before the full run) is the first
inference-stage task. Memory is not a concern at ≤24 atoms.

## 5. Open decisions for you

1. **A definition:** system-matched (proposed) vs plain random composition-disjoint (5% system overlap).
2. **Held-out family for C:** f-block (proposed) vs actinides-only / lanthanides-only / 5d.
3. **Label-quality filters:** I did *not* drop structures with positive energy/atom (10 total) or max force > 50 eV/Å (274 total; 2 in each test split). Filtering on labels or
   DFT outputs can bias coverage; a sensitivity analysis excluding force > 50 seems more defensible than a default exclusion.
4. **Matching on order / family / atom count** across B and C (not done).
5. **Stricter B** (Jaccard cap) and **stratified analysis** plans.
6. Whether `max_atoms=24`, `n_cal=2000`, `n=1000` suit your compute budget once timed.
7. Confirming the raw-energy label against UMA `omat` behaviour at the first inference step. Docs do not settle it.
