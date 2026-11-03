# UMA-Cal

Reproducible research project on **uncertainty calibration for Meta's Universal Models for Atoms (UMA)**.
Question: do post-hoc conformal prediction intervals around UMA energy predictions stay reliable across
chemical compositions, especially under out-of-distribution (OOD) chemical shifts? A finding that calibration
fails under shift is a valid result.

**Status:** Stage 2 complete: OMat24 validation subset audited and chemical-shift splits proposed (awaiting design
review). UMA inference has been run only for the Stage 1 smoke test (FCC Al). **No UMA inference has been run on
the Cal or test splits yet, and no calibration experiments exist.** See `reports/split_proposal.md`.

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

## OMat24 data (Stage 2)

Only the validation `rattled-1000-subsampled` archive is used (39,785 structures). The exact URL, size and SHA256
are recorded in `results/dataset_audit.json`.

```bash
mkdir -p data/raw
curl -L -o data/raw/rattled-1000-subsampled.tar.gz \
  https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz
shasum -a 256 data/raw/rattled-1000-subsampled.tar.gz      # macOS (Linux: sha256sum)
# expected: a5a06b296acb0cfd4f6d436f7022db4a4319915f003f47ee69dac4189c6d9fc1   (79,870,946 bytes)
```

`data/raw/` is git-ignored; never commit archives or `.aselmdb` files.

Reproduce the audit and the proposed splits (metadata only; no UMA inference):

```bash
python scripts/audit_dataset.py --archive data/raw/rattled-1000-subsampled.tar.gz \
  --source-url https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz
python scripts/propose_splits.py --heldout-elements f-block --max-atoms 24 --seed 20241220
python scripts/family_sensitivity.py
```

Outputs: `results/dataset_audit.json`, `results/composition_counts.csv`, `data/splits/{cal,A,B,C}.csv`,
`data/splits/split_summary.json`, `data/splits/family_sensitivity.json`.

## Acknowledgment and citation

This project uses Meta FAIR Chemistry's [UMA models](https://github.com/facebookresearch/fairchem)
(`fairchem-core`), released under Meta's FAIR Chemistry license (see the model card on Hugging Face), and the
[OMat24 dataset](https://huggingface.co/datasets/facebook/OMAT24) (Meta FAIR), licensed CC-BY-4.0.
Please cite both:

- B. M. Wood et al., *UMA: A Family of Universal Models for Atoms*, arXiv:2506.23971.
- L. Barroso-Luque et al., *Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models*, arXiv:2410.12771.
