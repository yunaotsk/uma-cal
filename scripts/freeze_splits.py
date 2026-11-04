"""Write reports/split_freeze.md: SHA256 of every split file, seed, command, library versions.

Run after propose_splits.py / compare_splits.py / family_sensitivity.py. Commit the split files and this
document together, before any prediction is made. No UMA inference.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ("python scripts/propose_splits.py --heldout-elements f-block --max-atoms 24 --seed 20241220 "
           "--n-cal 4000 --n-a0 2000 --n-b 2000 --n-c 2000 --n-a all")
SPLITS = ["cal", "A0", "A", "B", "C"]
DERIVED = ["data/splits/split_summary.json", "data/splits/split_comparison.json", "data/splits/family_sensitivity.json"]
INPUTS = ["results/dataset_audit.json", "results/composition_counts.csv"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def versions() -> dict:
    import ase
    import fairchem.core
    import numpy
    import pandas
    import scipy
    import torch

    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "ase": ase.__version__, "fairchem-core": fairchem.core.__version__, "torch": torch.__version__,
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}"}


def main():
    summary = json.loads((ROOT / "data/splits/split_summary.json").read_text())
    audit = json.loads((ROOT / "results/dataset_audit.json").read_text())
    cfg = summary["config"]
    code_clean = git("status", "--porcelain", "--", "src", "scripts") == ""
    rows = []
    for n in SPLITS:
        p = ROOT / f"data/splits/{n}.csv"
        rows.append((f"data/splits/{n}.csv", sum(1 for _ in open(p)) - 1, sha(p)))
    L = [
        "# Split freeze", "",
        f"Frozen {date.today().isoformat()}, **before any prediction on A0, A, B or C** (the only predictions in the cache are the 20 Cal benchmark structures).",
        "Verify with `shasum -a 256 data/splits/*.csv`; `tests/test_split_files.py` checks every hash below. Any change to a split file after this commit is a protocol deviation and must be recorded in `reports/analysis_plan.md`.", "",
        "## Split files", "", "| File | Rows (compositions) | SHA256 |", "|---|---:|---|",
        *[f"| `{f}` | {r:,} | `{h}` |" for f, r, h in rows], "",
        "## Parameters", "",
        f"- Seed: **{cfg['seed']}**",
        f"- Targets: n_cal {cfg['n_cal']:,}, n_a0 {cfg['n_a0']:,}, n_b {cfg['n_b']:,}, n_c {cfg['n_c']:,}, A = all available; max_atoms {cfg['max_atoms']}; "
        f"b_holdout_frac {cfg['b_holdout_frac']}; min_comps_per_element {cfg['min_comps_per_element']}; held-out elements = f-block ({len(cfg['held_out_elements'])} symbols incl. unobserved ones)",
        f"- Realized: " + ", ".join(f"{k} {v:,}" for k, v in {"Cal": len(open(ROOT / 'data/splits/cal.csv').readlines()) - 1, **{n: summary[n]['compositions'] for n in ('A0', 'A', 'B', 'C')}}.items()),
        f"- Capacity with this Cal and A0: " + ", ".join(f"{k} {v:,}" for k, v in summary["capacity_with_same_cal"].items()),
        "", "## Command", "", "```", COMMAND, "```", "",
        "Followed by `python scripts/compare_splits.py`, `python scripts/family_sensitivity.py` and `python scripts/freeze_splits.py`.", "",
        "## Provenance", "",
        f"- Code commit when generated: `{git('rev-parse', 'HEAD')}` (src/ and scripts/ unmodified at generation: **{code_clean}**)",
        f"- OMat24 archive: `{audit['provenance']['source_url']}`",
        f"- Archive SHA256: `{audit['provenance']['archive_sha256']}` ({audit['provenance']['archive_bytes']:,} bytes)",
        *[f"- `{f}` SHA256: `{sha(ROOT / f)}`" for f in INPUTS], "",
        "## Derived files (regenerable from the split files)", "", "| File | SHA256 |", "|---|---|",
        *[f"| `{f}` | `{sha(ROOT / f)}` |" for f in DERIVED], "",
        "## Library versions", "", *[f"- {k}: {v}" for k, v in versions().items()], "",
    ]
    (ROOT / "reports/split_freeze.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
