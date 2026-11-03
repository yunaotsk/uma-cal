"""How sensitive are family labels, and the strong-shift tier, to the choice of family rules?

Compares src.dataset.chemical_family (used) with family_alternative (the compare-branch rules).
Reads data/processed/meta.pkl and data/splits/C.csv; writes data/splits/family_sensitivity.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import dataset as D

df = pd.read_pickle(ROOT / "data/processed/meta.pkl")
comp = df.drop_duplicates("comp_key").copy()  # family depends only on the element set
comp["fam_alt"] = comp.elements.map(lambda s: D.family_alternative(s.split()))
comp["fam"] = comp.family
base = D.eligible_structures(df, 24)
comp["eligible"] = comp.comp_key.isin(set(base.comp_key))

changed = comp[comp.fam != comp.fam_alt]
pairs = changed.groupby(["fam", "fam_alt"]).size().sort_values(ascending=False)
metal_rename = changed.fam_alt == "metalloid-containing"
anion_set = changed[~metal_rename]

c = pd.read_csv(ROOT / "data/splits/C.csv")
c = c.merge(comp[["comp_key", "fam_alt"]], on="comp_key")
elig = comp[comp.eligible]
f_hold = elig.elements.map(lambda s: bool(set(s.split()) & D.F_BLOCK))


def fam_tier_sizes(col):
    return elig[~f_hold].groupby(col).size().sort_values(ascending=False).to_dict()


out = {
    "compositions_total": len(comp),
    "compositions_changed_label": int(len(changed)),
    "changed_due_to_metalloid_bucket (alt='metalloid-containing')": int(metal_rename.sum()),
    "changed_due_to_anion_set_differences": int(len(anion_set)),
    "changed_to_alt_mixed-anion": int((changed.fam_alt == "mixed-anion").sum()),
    "top_label_pairs_used_to_alt": {f"{a} -> {b}": int(n) for (a, b), n in pairs.head(12).items()},
    "eligible_compositions_changed": int((changed.eligible).sum()),
    "strong_shift_C": {
        "definition": "contains >=1 f-block element; independent of family rules",
        "membership_depends_on_family_rule": False,
        "pool_compositions_f_block": int(f_hold.sum()),
        "C_family_mix_used_rules": c.family.value_counts().to_dict(),
        "C_family_mix_alt_rules": c.fam_alt.value_counts().to_dict(),
        "C_structures_whose_label_changes": int((c.family != c.fam_alt).sum()),
    },
    "hypothetical_family_held_out_tier_sizes_eligible_non_f_block": {
        "used_rules": fam_tier_sizes("fam"),
        "alt_rules": fam_tier_sizes("fam_alt"),
    },
}
(ROOT / "data/splits/family_sensitivity.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
