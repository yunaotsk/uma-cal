"""Audit the OMat24 validation rattled-1000-subsampled archive and write proposed splits.

Metadata operations only. This script never imports UMA / FAIRChemCalculator inference.
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import dataset as D

ARCHIVE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz"
EXPECTED_N = 39785  # official dataset card, validation / rattled-1000-subsampled
ALPHA = 0.10
SPLIT_COLS = ["index", "sid", "parent_id", "comp_key", "elements", "natoms", "family", "tm_class", "prototype_label"]


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


def se_block(n_test: int, n_cal: int) -> dict:
    se_test = D.binomial_se(1 - ALPHA, n_test)
    # sd of the true coverage given a calibration set of size n_cal (standard split-conformal result)
    sd_cal = D.binomial_se(1 - ALPHA, n_cal + 2)
    return {
        "n_test": n_test,
        "binomial_se_test_only": round(se_test, 5),
        "ci95_half_width_test_only": round(1.96 * se_test, 5),
        "approx_sd_of_coverage_given_cal(n_cal)": round(sd_cal, 5),
        "combined_se": round(float(np.hypot(se_test, sd_cal)), 5),
    }


def split_report(name: str, s: pd.DataFrame, cal: pd.DataFrame) -> dict:
    cal_protos = set(cal.parent_prototype_label.map(D.structural_prototype))
    protos = s.parent_prototype_label.map(D.structural_prototype)
    j = D.max_jaccard_to_cal(s, cal)
    return {
        "structures": len(s),
        "compositions": int(s.comp_key.nunique()),
        "systems": int(s.elements.nunique()),
        "natoms": s.natoms.describe().round(2).to_dict(),
        "family": s.family.value_counts().to_dict(),
        "tm_class": s.tm_class.value_counts().to_dict(),
        "n_elements": s.n_elements.value_counts().sort_index().to_dict(),
        "hubbard_u_structures": int(s.hubbard_u.sum()),
        "frac_structural_prototype_seen_in_cal": round(float(protos.isin(cal_protos).mean()), 4),
        "max_jaccard_to_cal_systems": {
            "mean": round(float(j.mean()), 4),
            "median": round(float(np.median(j)), 4),
            "frac_ge_0.5": round(float((j >= 0.5).mean()), 4),
            "frac_eq_1": round(float((j == 1).mean()), 4),
        },
        "descriptive_energy_per_atom_eV": s.energy_per_atom_eV.describe().round(3).to_dict(),
        "n_max_force_gt_50": int((s.max_abs_force_eV_A > 50).sum()),
        "coverage_se": se_block(len(s), len(cal)),
    }


def random_split_overlap(df: pd.DataFrame, seed: int, n_cal: int, n_test: int) -> dict:
    """How often does a plain composition-disjoint random split give test systems seen in Cal?"""
    rng = np.random.default_rng(seed)
    u = D.one_per_composition(D.eligible_structures(df), rng)
    u = u[~u.elements.map(lambda s: bool(set(s.split()) & D.F_BLOCK))]
    perm = rng.permutation(len(u))
    test, cal = u.iloc[perm[:n_test]], u.iloc[perm[n_test : n_test + n_cal]]
    seen = test.elements.isin(set(cal.elements))
    return {
        "pool_compositions": len(u),
        "n_cal": len(cal),
        "n_test": len(test),
        "frac_test_systems_seen_in_cal": round(float(seen.mean()), 4),
        "note": "A purely random composition split makes most test systems new, i.e. it is already 'mild shift'.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ROOT / "data/raw/rattled-1000-subsampled.tar.gz"))
    ap.add_argument("--lmdb-dir", default=str(ROOT / D.DEFAULT_LMDB_DIR))
    ap.add_argument("--seed", type=int, default=D.DEFAULT_SEED)
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-atoms", type=int, default=24)
    args = ap.parse_args()

    import ase
    import fairchem.core

    archive = Path(args.archive)
    df = D.load_metadata(args.lmdb_dir)
    df.to_pickle(ROOT / "data/processed/meta.pkl")

    base = D.eligible_structures(df, args.max_atoms)
    funnel = {
        "all": len(df),
        f"natoms<={args.max_atoms}": int((df.natoms <= args.max_atoms).sum()),
        "and no Hubbard-U treatment": int(((df.natoms <= args.max_atoms) & ~df.hubbard_u).sum()),
        "and >=2 elements, no noble gas (= eligible)": len(base),
        "eligible compositions": int(base.comp_key.nunique()),
        "eligible systems (element sets)": int(base.elements.nunique()),
    }

    splits = D.build_splits(
        df, seed=args.seed, n_cal=args.n_cal, n_a=args.n_test, n_b=args.n_test, n_c=args.n_test, max_atoms=args.max_atoms
    )
    out_dir = ROOT / "data/splits"
    for name, s in splits.items():
        cols = SPLIT_COLS + (["n_held_out_elements"] if name == "C" else [])
        s[cols].to_csv(out_dir / f"{name}.csv", index=False)

    # capacity (how large each test set could be with the same Cal)
    cap = D.build_splits(
        df, seed=args.seed, n_cal=args.n_cal, n_a=10**6, n_b=10**6, n_c=10**6, max_atoms=args.max_atoms
    )
    audit = {
        "provenance": {
            "date": date.today().isoformat(),
            "source_url": ARCHIVE_URL,
            "archive_file": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256(archive),
            "fairchem_core_version": fairchem.core.__version__,
            "ase_version": ase.__version__,
            "loader": "fairchem.core.datasets.AseDBDataset",
            "expected_structures_official": EXPECTED_N,
            "observed_structures": len(df),
            "count_matches_official": len(df) == EXPECTED_N,
        },
        "counts": {
            "structures": len(df),
            "unique_reduced_compositions": int(df.comp_key.nunique()),
            "unique_systems_element_sets": int(df.elements.nunique()),
            "unique_parent_ids": int(df.parent_id.nunique()),
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
        "splits": {
            "config": {
                "seed": args.seed,
                "n_cal": args.n_cal,
                "n_test_each": args.n_test,
                "max_atoms": args.max_atoms,
                "held_out_elements": sorted(D.F_BLOCK),
                "alpha": ALPHA,
            },
            **{name: split_report(name, s, splits["cal"]) for name, s in splits.items() if name != "cal"},
            "cal": {"structures": len(splits["cal"]), "natoms": splits["cal"].natoms.describe().round(2).to_dict(),
                    "family": splits["cal"].family.value_counts().to_dict()},
            "capacity_with_same_cal": {k: len(v) for k, v in cap.items() if k != "cal"},
            "random_split_system_overlap": random_split_overlap(df, args.seed, args.n_cal, args.n_test),
        },
    }
    (ROOT / "results/dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str))

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
    comp.to_csv(ROOT / "results/composition_counts.csv", index=False)
    print(json.dumps({"provenance": audit["provenance"], "funnel": funnel, "capacity": audit["splits"]["capacity_with_same_cal"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
