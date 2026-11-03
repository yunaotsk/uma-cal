# CLAUDE.md — UMA-Cal

## Research goal
Investigate whether post-hoc conformal prediction intervals around Meta UMA energy predictions remain reliable
across chemical compositions, particularly under out-of-distribution (OOD) chemical shifts. A finding that
calibration fails under distribution shift is a valid result.

## Constraints
- Apple Silicon Mac, CPU only; always pass `device="cpu"` explicitly.
- Use the existing `.venv` (Python 3.11), never global Python.
- No model retraining, no GPU dependencies.
- Do not reinstall/upgrade dependencies unless necessary; explain compatibility issues before changing the environment.
- Never read, print, expose, store, or commit the Hugging Face token or any credential.
- No UI, no unnecessary infrastructure.
- Scope is staged: do not start a stage (e.g. OMat24 download, calibration experiments) until the user says so.

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

Defaults live in `src/inference.py` (`UMAPredictor`). Stage 1 (CPU inference verified) is complete; see README.

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
