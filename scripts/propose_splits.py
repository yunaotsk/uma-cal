"""Generate the shared-calibration splits: Cal, A0 (exchangeable control), A (system-matched), B (mild shift), C (strong shift).

Run only after reviewing results/dataset_audit.json. Metadata operations only (no UMA inference).
``--heldout-elements`` and ``--max-atoms`` have no default on purpose: choose them from the audit.

    python scripts/propose_splits.py --heldout-elements f-block --max-atoms 24 --seed 20241220

Element tokens may be symbols (Ru Rh) and/or groups: f-block lanthanides actinides 3d 4d 5d.
Writes data/splits/{cal,A0,A,B,C}.csv (with a max_jaccard_to_cal similarity column) and data/splits/split_summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import dataset as D

ALPHA = 0.10
SPLIT_COLS = ["index", "sid", "parent_id", "comp_key", "elements", "natoms", "family", "tm_class", "prototype_label", "max_jaccard_to_cal"]


def se_levels(test: pd.DataFrame, base: pd.DataFrame, n_cal: int) -> dict:
    """Headline: composition-level SE. Secondary: structure-level SE if every eligible structure of the
    test compositions were used (rattled copies/polymorphs included), which is optimistic."""
    n_comp = test.comp_key.nunique()
    n_struct_all = int(base.comp_key.isin(set(test.comp_key)).sum())
    lv = D.coverage_se_levels(n_struct_all, n_comp, 1 - ALPHA)
    sd_cal = D.binomial_se(1 - ALPHA, n_cal + 2)
    return {
        "composition_level": {
            "n": n_comp,
            "se": round(lv["se_composition_level"], 5),
            "ci95_half_width": round(1.96 * lv["se_composition_level"], 5),
            "approx_sd_from_finite_cal": round(sd_cal, 5),
            "combined_se": round(float(np.hypot(lv["se_composition_level"], sd_cal)), 5),
        },
        "structure_level_secondary": {
            "n_structures_in_split_file": len(test),
            "n_all_eligible_structures_of_these_compositions": n_struct_all,
            "se_if_all_used_assuming_independence": round(lv["se_structure_level"], 5),
            "note": "optimistic: structures sharing a composition are correlated",
        },
    }


def split_report(s: pd.DataFrame, cal: pd.DataFrame, base: pd.DataFrame, n_cal: int) -> dict:
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
        "n_elements": {int(k): int(v) for k, v in s.n_elements.value_counts().sort_index().items()},
        "hubbard_u_structures": int(s.hubbard_u.sum()),
        "frac_structural_prototype_seen_in_cal": round(float(protos.isin(cal_protos).mean()), 4),
        "max_jaccard_to_cal_systems": {
            "mean": round(float(j.mean()), 4),
            "frac_ge_0.5": round(float((j >= 0.5).mean()), 4),
            "frac_eq_1": round(float((j == 1).mean()), 4),
        },
        "descriptive_energy_per_atom_eV": s.energy_per_atom_eV.describe().round(3).to_dict(),
        "n_max_force_gt_50": int((s.max_abs_force_eV_A > 50).sum()),
        "coverage_se": se_levels(s, base, n_cal),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout-elements", nargs="+", required=True)
    ap.add_argument("--max-atoms", type=int, required=True)
    ap.add_argument("--seed", type=int, default=D.DEFAULT_SEED)
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-a", type=int, default=1000, help="upper bound; A is limited by system overlap with Cal")
    ap.add_argument("--n-a0", type=int, default=1000)
    ap.add_argument("--n-b", type=int, default=1000)
    ap.add_argument("--n-c", type=int, default=1000)
    ap.add_argument("--b-holdout-frac", type=float, default=0.25, help="fraction of familiar systems reserved for B")
    ap.add_argument("--min-comps-per-element", type=int, default=5, help="B: each element needs this many Cal compositions")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/processed/meta.pkl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/splits")
    args = ap.parse_args()

    held = D.resolve_element_spec(args.heldout_elements)
    df = pd.read_pickle(args.metadata)
    kw = dict(seed=args.seed, n_cal=args.n_cal, n_a=args.n_a, n_a0=args.n_a0, n_b=args.n_b, n_c=args.n_c, max_atoms=args.max_atoms,
              b_holdout_frac=args.b_holdout_frac, held_out_elements=held, min_comps_per_element=args.min_comps_per_element)
    splits = D.build_splits(df, **kw)
    cap = D.build_splits(df, **{**kw, "n_a": 10**6, "n_a0": args.n_a0, "n_b": 10**6, "n_c": 10**6})
    base = D.eligible_structures(df, args.max_atoms)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        # similarity to Cal (max Jaccard on element sets); for Cal itself, leave-one-composition-out
        s["max_jaccard_to_cal"] = D.max_jaccard_to_cal(s, splits["cal"], exclude_self=(name == "cal")).round(4)
        cols = SPLIT_COLS + (["n_held_out_elements"] if name == "C" else [])
        s[cols].to_csv(args.out_dir / f"{name}.csv", index=False)

    summary = {
        "config": {**{k: v for k, v in kw.items() if k != "held_out_elements"}, "held_out_elements": sorted(held), "alpha": ALPHA},
        "cal": {"compositions": len(splits["cal"]), "natoms": splits["cal"].natoms.describe().round(2).to_dict(),
                "family": splits["cal"].family.value_counts().to_dict()},
        **{n: split_report(splits[n], splits["cal"], base, len(splits["cal"])) for n in ("A0", "A", "B", "C")},
        "capacity_with_same_cal": {k: len(v) for k, v in cap.items() if k != "cal"},
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({n: {"structures": summary[n]["structures"], "compositions": summary[n]["compositions"],
                          "se_composition_level": summary[n]["coverage_se"]["composition_level"]["se"],
                          "se_structure_level_secondary": summary[n]["coverage_se"]["structure_level_secondary"]["se_if_all_used_assuming_independence"]}
                      for n in ("A0", "A", "B", "C")}, indent=2))
    print("capacity:", summary["capacity_with_same_cal"])


if __name__ == "__main__":
    main()
