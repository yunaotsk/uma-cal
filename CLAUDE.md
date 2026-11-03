# CLAUDE.md — UMA-Cal

## Research goal
Investigate whether post-hoc conformal prediction intervals around Meta UMA energy predictions remain reliable
across chemical compositions, particularly under out-of-distribution (OOD) chemical shifts. A finding that
calibration fails under distribution shift is a valid result.

## Constraints
- CPU only; always pass `device="cpu"` explicitly. (Stage 1-2 ran on an Apple Silicon Mac; see Handoff status.)
- Use the existing `.venv` (Python 3.11), never global Python.
- No model retraining, no GPU dependencies.
- Do not reinstall/upgrade dependencies unless necessary; explain compatibility issues before changing the environment.
- Never read, print, expose, store, or commit the Hugging Face token or any credential.
- No UI, no unnecessary infrastructure.
- Scope is staged: do not start a stage (e.g. UMA inference on splits, calibration experiments) until the user says so.

## Research standards
- Scientific correctness and reproducibility outrank impressive metrics.
- Never fabricate results, metrics, or test outcomes; never claim tests passed unless they ran.
- Never silently use mocks, random predictions, or another model. Test doubles are allowed only in unit tests, explicitly injected and labelled.
- Unit tests must run offline without downloading weights; real-model runs live in `scripts/`, not in pytest.
- Diagnose blockers rather than applying speculative fixes.
- Keep code modular, readable, pinned (`requirements.txt`) and reproducible.

## Decisions so far
| Decision | Value |
|---|---|
| Checkpoint | `uma-s-1p2p1` |
| Task | `omat` |
| Device | `cpu` |
| Inference settings | `"batch"` |
| Stack | torch 2.13.0, fairchem-core 2.25.0, ase 3.29.0 |

Defaults live in `src/inference.py` (`UMAPredictor`).

## Stage status
- **Stage 1 (CPU inference verified): complete.**
- **Stage 2 (OMat24 audit and shift-split design): complete.** The validation `rattled-1000-subsampled` archive is in `data/raw/`
  (git-ignored); the audit is in `results/`; the proposed splits (`cal`, `A`, `B`, `C`) are in `data/splits/` **awaiting design review**.
  No UMA inference has been run on Cal or test structures.
- **Stage 3 prep: done on branch `stage3-prep`** (A0 control, similarity scores, `reports/analysis_plan.md`, resumable cache `src/pred_cache.py`, Cal-only benchmark `reports/cal_benchmark.md`).
  Raw DFT energy confirmed as the label on 20 Cal structures. **Sizes are pending the user's approval; no test-split prediction exists.**
- **Stage 3 (inference on splits, calibration): not started.** Do not begin until the user approves sizes. Order: Cal, then the A0 gate, then A, B, C.

### Justification for `inference_settings="batch"`
Sources are the installed fairchem-core 2.25.0 files, under
`.venv/lib/python3.11/site-packages/fairchem/core/`:

- `units/mlip_unit/api/inference.py:269-273` — "Batch mode is the stable entry point for heterogeneous inputs. It
  currently uses the general unmerged and uncompiled path…"; the preset sets `tf32=False`, `merge_mole=False`,
  `compile=False` (lines 274-282).
- `units/mlip_unit/predict.py:653` — "Use inference_settings='batch' for heterogeneous batched evaluations."
  (the warning emitted when the merge_mole fast path is broken by a composition change).
- `units/mlip_unit/predict.py:~648-652` — the fast path (merge_mole + compile) "is only available for fixed
  composition, task, charge, and spin. This is optimized for MD applications."
- `calculate/pretrained_mlip.py:156-157` and `calculate/ase_calculator.py:143-144` — docstrings: `"batch"` keeps MOLE
  unmerged for heterogeneous inputs.
- `units/mlip_unit/api/inference.py:259-267` vs `311-312` — the library's `"default"` is `nitro` (multigpu,
  reduced-precision); `high-precision` (merge_mole + compile, FP32) is a fast path for fixed inputs.

Our reasoning (inference, not stated by the library): calibration data spans many compositions, so the
fixed-composition fast paths do not apply, and `batch` is full FP32 (`tf32=False`) with no reduced-precision
variants.

Caveats: I did not find a separate official docs page for this setting; the justification is from source
comments and docstrings only. I have not measured numerical or speed differences between `batch` and other presets
on this machine. The docstring says `default` falls back to `multigpu` when Nitro cannot run (it targets CUDA), so
on CPU the practical alternative would be `multigpu`/`high-precision`; this was not benchmarked.

## UMA-Cal research design
UMA-Cal investigates whether post-hoc conformal prediction intervals for Meta UMA's energy predictions remain reliable under chemical distribution shift. The primary evaluation uses OMat24 reference DFT energies and UMA's omat task, with energy error measured in eV/atom.
We compare three conditions:
1. In-distribution control: different reduced compositions sampled from the same chemical population.
2. Mild chemical shift: unseen element combinations composed of familiar elements.
3. Strong chemical shift: compositions containing held-out elements or chemical families.
All splits must prevent reduced-composition overlap between calibration and test sets. Preserve source identifiers and consider structural duplicates (e.g. polymorphs) and related chemical systems.
The exact shift definitions must be selected only after inspecting chemical diversity, sample counts, atom counts, DFT settings, and potential label inconsistencies.
A shift relative to the calibration set does not automatically establish that a structure is out of distribution relative to UMA's original training data.
Do not perform expensive model inference until the dataset audit and experimental splits have been reviewed.
Scientific correctness, reproducibility, and transparent failure reporting take priority over positive results.

## Handoff status (moving to Claude Code on the web)
- **Different machine.** Different OS/CPU: `requirements.txt` is a macOS arm64 `pip freeze` and may not install as is. Diagnose and explain any
  incompatibility before changing the environment; do not silently loosen pins.
- **No cached weights.** The `uma-s-1p2p1` checkpoint (~GBs) must be re-downloaded on first use; the first model load will be slow.
- **Credentials.** `HF_TOKEN` comes from an environment secret. Never print, log, echo, write or commit it (including in tracebacks and `env` dumps).
- **First action: re-run `python scripts/smoke_test.py` and check the FCC Al4 energy is -14.964057 eV** (value from the Stage 1 macOS arm64 run,
  `batch` preset). Report any difference rather than adjusting code or tolerances to match.
- **Timings do not transfer.** Stage 1 numbers (load 127 s incl. download, 0.645 s first call, 0.225 s second, 4-atom cell) came from the Mac. Re-measure on the new machine.
- Run `python -m pytest -q` first; the tests are offline and need no weights.

## Decisions (settled; sizes still pending, see last item)
- **A: keep the system-matched A and add an exchangeable control A0.** A0 is a random composition-disjoint holdout drawn *jointly* with Cal from one random split of the familiar pool
  (same filters), so Cal and A0 are exchangeable. Cal, A0, A, B, C stay pairwise disjoint on reduced composition, `parent_id` and `sid`.
- **C = f-block held out** (lanthanides + actinides). Report Ln-vs-An and order-matched stratifications alongside the pooled result.
- **Label filters: none by default.** Pre-specified sensitivity analyses exclude max force > 50 eV/Å and, separately, energy/atom > 0 (274 and 10 structures in the full set).
- **Check the `omat` energy label on Cal structures only, never on test structures** (raw DFT energy vs MP2020-corrected), so test data never guide the choice.
- **Shift is relative to Cal only.** UMA may have seen these structures (OMat24 validation was a random split of the training pool; model card says trained on the "entirety" of OMat24).
- **Label noise on f-block compounds (plain PBE, no +U) is a confounder.** Report it; lower coverage on C is not by itself evidence of model-driven shift.
- **Analysis is pre-specified** in `reports/analysis_plan.md`, committed before any test-split prediction exists. Primary metric: marginal 90% coverage (split conformal, absolute error in eV/atom); headline SE at composition level.
- **`max_atoms`, `n_cal` and `n` per split are NOT decided.** Decide only after the Cal-only timing benchmark: propose values with a labelled cost extrapolation and wait for user approval.
- **Hard rule until approval:** no UMA inference on any test split (A, A0, B, C). Permitted: Cal structures only (timing benchmark and label check).
