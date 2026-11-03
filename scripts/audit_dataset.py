"""Audit an OMat24 archive: counts, labels, chemical diversity, provenance.

    python scripts/audit_dataset.py --archive data/raw/<name>.tar.gz --source-url <exact URL>

Writes results/dataset_audit.json and results/composition_counts.csv. Splits are generated separately by
scripts/propose_splits.py, after the audit has been reviewed. Metadata operations only: no UMA inference.
"""

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import dataset as D

EXPECTED_N = 39785  # official dataset card, validation / rattled-1000-subsampled


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table(df: pd.DataFrame, col: str) -> dict:
    g = df.groupby(col).agg(structures=("sid", "size"), compositions=("comp_key", "nunique"))
    return g.sort_values("structures", ascending=False).to_dict("index")


def label_checks(df: pd.DataFrame) -> dict:
    corr = (df.energy_corrected_mp2020_eV - df.energy_total_eV).dropna()
    unary = df[df.n_elements == 1]
    dup = df.assign(e5=df.energy_total_eV.round(5)).duplicated(["comp_key", "natoms", "e5"]).sum()
    return {
        "label_used": "raw DFT total energy from calculator results (eV per cell); eV/atom = total / natoms",
        "docs_statement": "OMat24 dataset card: 'labeled with total energy (eV), forces (eV/A) and stress (eV/A^3)'",
        "energy_per_atom_eV": df.energy_per_atom_eV.describe().round(4).to_dict(),
        "n_energy_per_atom_positive": int((df.energy_per_atom_eV > 0).sum()),
        "n_max_force_gt_50_eV_per_A": int((df.max_abs_force_eV_A > 50).sum()),
        "max_force_eV_per_A": float(df.max_abs_force_eV_A.max()),
        "n_missing_mp2020_corrected_energy": int(df.energy_corrected_mp2020_eV.isna().sum()),
        "mp2020_minus_raw_total_eV": corr.describe().round(4).to_dict(),
        "n_unary_structures": len(unary),
        "unary_energy_per_atom_eV": unary.groupby("comp_key").energy_per_atom_eV.mean().round(3).to_dict(),
        "n_stored_composition_reduced_mismatch": int(
            (df.composition_reduced_stored.map(_stored_key) != df.comp_key).sum()
        ),
        "n_duplicate_comp_natoms_energy": int(dup),
        "n_compositions_with_multiple_prototypes": int((df.groupby("comp_key").prototype_label.nunique() > 1).sum()),
    }


def _stored_key(s: str) -> str:
    import re

    d = {}
    for tok in s.split():
        m = re.match(r"([A-Z][a-z]?)(\d*)$", tok)
        d[m.group(1)] = int(m.group(2) or 1)
    return D.composition_key(d)


def diversity(df: pd.DataFrame, limits=(16, 20, 24, 32, 64)) -> dict:
    n_el = df.n_elements.map(D.order_name)
    return {
        "by_family": table(df, "family"),
        "by_tm_class": table(df, "tm_class"),
        "by_order": table(df.assign(order=n_el), "order"),
        "has_3d": {"structures": int(df.has_3d.sum()), "compositions": int(df[df.has_3d].comp_key.nunique())},
        "has_4d": {"structures": int(df.has_4d.sum()), "compositions": int(df[df.has_4d].comp_key.nunique())},
        "has_both_3d_and_4d": {
            "structures": int((df.has_3d & df.has_4d).sum()),
            "compositions": int(df[df.has_3d & df.has_4d].comp_key.nunique()),
        },
        "hubbard_u_applies": {"structures": int(df.hubbard_u.sum()), "compositions": int(df[df.hubbard_u].comp_key.nunique())},
        "f_block": {
            "structures": int(df.elements.map(lambda s: bool(set(s.split()) & D.F_BLOCK)).sum()),
            "compositions": int(df[df.elements.map(lambda s: bool(set(s.split()) & D.F_BLOCK))].comp_key.nunique()),
        },
        "atom_limit_candidates": {
            str(L): {"structures": int((df.natoms <= L).sum()), "compositions": int(df[df.natoms <= L].comp_key.nunique())}
            for L in limits
        },
    }


def extract(archive: Path, dest: Path) -> None:
    """Extract into its own directory (tar 'data' filter rejects absolute/escaping paths)."""
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip extract] {dest} is not empty; reusing it")
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--source-url", required=True, help="URL the archive was downloaded from (recorded, not fetched)")
    ap.add_argument("--extract-dir", type=Path, default=ROOT / D.DEFAULT_LMDB_DIR)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--max-atoms", type=int, default=24, help="atom cap used for the eligibility funnel")
    ap.add_argument("--expected-structures", type=int, default=EXPECTED_N)
    ap.add_argument("--expected-archive-mb", type=float, default=77.0)
    args = ap.parse_args()

    import ase
    import fairchem.core

    archive = args.archive
    extract(archive, args.extract_dir)
    df = D.load_metadata(args.extract_dir)
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    df.to_pickle(ROOT / "data/processed/meta.pkl")  # git-ignored cache used by propose_splits.py

    base = D.eligible_structures(df, args.max_atoms)
    funnel = {
        "all": len(df),
        f"natoms<={args.max_atoms}": int((df.natoms <= args.max_atoms).sum()),
        "and no Hubbard-U treatment": int(((df.natoms <= args.max_atoms) & ~df.hubbard_u).sum()),
        "and >=2 elements, no noble gas (= eligible)": len(base),
        "eligible compositions": int(base.comp_key.nunique()),
        "eligible systems (element sets)": int(base.elements.nunique()),
    }
    size = archive.stat().st_size
    audit = {
        "provenance": {
            "date": date.today().isoformat(),
            "source_url": args.source_url,
            "archive_file": archive.name,
            "archive_bytes": size,
            "archive_mb": round(size / 1e6, 2),
            "archive_sha256": sha256(archive),
            "fairchem_core_version": fairchem.core.__version__,
            "ase_version": ase.__version__,
            "loader": "fairchem.core.datasets.AseDBDataset (files located recursively)",
            "expected_structures_official": args.expected_structures,
            "expected_archive_mb_official": args.expected_archive_mb,
            "observed_structures": len(df),
            "count_matches_official": len(df) == args.expected_structures,
        },
        "counts": {
            "structures": len(df),
            "unique_reduced_compositions": int(df.comp_key.nunique()),
            "unique_systems_element_sets": int(df.elements.nunique()),
            "unique_parent_ids": int(df.parent_id.nunique()),
            "unique_source_ids": int(df.sid.nunique()),
            "unique_elements": int(len({e for s in df.elements for e in s.split()})),
            "elements": sorted({e for s in df.elements for e in s.split()}),
            "structures_per_composition": {int(k): int(v) for k, v in df.groupby("comp_key").size().value_counts().sort_index().items()},
            "natoms": df.natoms.describe().round(2).to_dict(),
            "natoms_percentiles": {str(q): float(df.natoms.quantile(q / 100)) for q in (5, 25, 50, 75, 90, 95, 99)},
            "natoms_histogram": {int(k): int(v) for k, v in df.natoms.value_counts().sort_index().items()},
        },
        "label_checks": label_checks(df),
        "diversity_all": diversity(df),
        "eligibility_funnel": funnel,
        "diversity_eligible": diversity(base),
        "coverage_se_headline_note": "SEs for proposed splits are produced by scripts/propose_splits.py (composition level first).",
    }
    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str))

    comp = (
        df.groupby("comp_key")
        .agg(
            n_structures=("sid", "size"),
            n_parents=("parent_id", "nunique"),
            natoms_min=("natoms", "min"),
            natoms_max=("natoms", "max"),
            n_elements=("n_elements", "first"),
            elements=("elements", "first"),
            family=("family", "first"),
            tm_class=("tm_class", "first"),
            hubbard_u=("hubbard_u", "first"),
        )
        .reset_index()
    )
    comp["eligible"] = comp.comp_key.isin(set(base.comp_key))
    comp.to_csv(args.out_dir / "composition_counts.csv", index=False)
    print(json.dumps({"provenance": audit["provenance"], "funnel": funnel}, indent=2, default=str))
    if len(df) != args.expected_structures:
        print(f"WARNING: expected {args.expected_structures} structures per docs, got {len(df)}", file=sys.stderr)


if __name__ == "__main__":
    main()
