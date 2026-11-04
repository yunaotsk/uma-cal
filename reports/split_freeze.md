# Split freeze

Frozen 2026-11-03, **before any prediction on A0, A, B or C** (the only predictions in the cache are the 20 Cal benchmark structures).
Verify with `shasum -a 256 data/splits/*.csv`; `tests/test_split_files.py` checks every hash below. Any change to a split file after this commit is a protocol deviation and must be recorded in `reports/analysis_plan.md`.

## Split files

| File | Rows (compositions) | SHA256 |
|---|---:|---|
| `data/splits/cal.csv` | 4,000 | `82dda29cb2f711b896aca87bc77957af67be8d6b3f883b1952420bb7f591bbbb` |
| `data/splits/A0.csv` | 2,000 | `46efcacea24170790f4ff9a6a8123ad352ac2fe1fc070bc1ec403b2957d1d081` |
| `data/splits/A.csv` | 713 | `7df71a516c3df83eb2717fdc7163b609b2e791dbd42a33a0db82f626f867249e` |
| `data/splits/B.csv` | 2,000 | `98144b5689a918b2449b47c41ee57287b853b69b317b9cb12d4faedcb1228e6b` |
| `data/splits/C.csv` | 2,000 | `47587f5dc60e74c743145c23a4bfab000c9d2f295c8342cc393e39f03553d44f` |

## Parameters

- Seed: **20241220**
- Targets: n_cal 4,000, n_a0 2,000, n_b 2,000, n_c 2,000, A = all available; max_atoms 24; b_holdout_frac 0.25; min_comps_per_element 5; held-out elements = f-block (30 symbols incl. unobserved ones)
- Realized: Cal 4,000, A0 2,000, A 713, B 2,000, C 2,000
- Capacity with this Cal and A0: A0 2,000, A 713, B 3,941, C 9,454

## Command

```
python scripts/propose_splits.py --heldout-elements f-block --max-atoms 24 --seed 20241220 --n-cal 4000 --n-a0 2000 --n-b 2000 --n-c 2000 --n-a all
```

Followed by `python scripts/compare_splits.py`, `python scripts/family_sensitivity.py` and `python scripts/freeze_splits.py`.

## Provenance

- Code commit when generated: `368cbba752486a578e3f078bcfb2fe837ffe03eb` (src/ and scripts/ unmodified at generation: **True**)
- OMat24 archive: `https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz`
- Archive SHA256: `a5a06b296acb0cfd4f6d436f7022db4a4319915f003f47ee69dac4189c6d9fc1` (79,870,946 bytes)
- `results/dataset_audit.json` SHA256: `acac602ca53e09aabda19e71424208bef11322fab3258362f002a56f7441ca42`
- `results/composition_counts.csv` SHA256: `2769ac75c683308e47a80cd58ec42b1d3f94541aa0183e88c96bd1643b3dacf9`

## Derived files (regenerable from the split files)

| File | SHA256 |
|---|---|
| `data/splits/split_summary.json` | `580782c12e240428b14f918886d0ab06e04ad516acb1da9d512a1fae8279b827` |
| `data/splits/split_comparison.json` | `8adfa3accb06ba0530fa6ee12a5476263d8adeeda826bcf403a524e4528cc695` |
| `data/splits/family_sensitivity.json` | `7306627fc2bdfc426cfb11171929d3ee53c288eb780956b9d079afe093645567` |

## Library versions

- python: 3.11.14
- numpy: 2.4.6
- pandas: 3.1.0
- scipy: 1.17.1
- ase: 3.29.0
- fairchem-core: 2.25.0
- torch: 2.13.0
- platform: Darwin 24.6.0 arm64
