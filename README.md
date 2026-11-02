# UMA-Cal

Reproducible research project on **uncertainty calibration for Meta's Universal Models for Atoms (UMA)**.
Question: do post-hoc conformal prediction intervals around UMA energy predictions stay reliable across
chemical compositions, especially under out-of-distribution (OOD) chemical shifts? A finding that calibration
fails under shift is a valid result.

**Status:** Stage 1 only (CPU inference verified). No datasets and no calibration experiments yet.

## Setup

- Apple Silicon Mac, CPU only, Python 3.11.
- Checkpoint `uma-s-1p2p1`, task `omat`, `device="cpu"`, inference preset `batch` (FP32, unmerged MOLE).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The UMA weights are gated on Hugging Face. Request access, then authenticate with `huggingface-cli login`
(or `hf auth login`). Never put tokens in this repository; `.env` is git-ignored.

## Usage

```python
from ase.build import bulk
from src.inference import UMAPredictor

predictor = UMAPredictor()            # loads checkpoint once
energy_ev = predictor.predict_energy(bulk("Al", "fcc", a=4.05, cubic=True))
```

```bash
python scripts/smoke_test.py   # real model; downloads weights on first run (~minutes)
python -m pytest -q            # offline unit tests; no weights, no network
```

## Acknowledgment

This project uses Meta FAIR Chemistry's [UMA models](https://github.com/facebookresearch/fairchem)
(`fairchem-core`). Please follow the UMA license and cite the UMA paper when using these results.
