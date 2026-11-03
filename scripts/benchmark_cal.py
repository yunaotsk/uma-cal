"""Cal-only timing benchmark and label check. REAL UMA inference on 20 *Cal* structures only.

Hard rule: no structure from A0, A, B or C is ever predicted. Enforced three ways: structures are
chosen from data/splits/cal.csv, asserted disjoint from the other split files, and passed to
``predict_cached`` with ``allowed_sids`` = the Cal sids.

Outputs data/benchmark/cal_benchmark.json. Predictions go to the git-ignored cache.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src import dataset as D
from src.inference import UMAPredictor
from src.pred_cache import CacheConfig, PredictionCache, predict_cached

SEED = 20241220
N_BENCH = 20
# (low, high, count): purposive spread over cell size so the cost-vs-atoms fit is anchored at both ends
BINS = [(10, 11, 4), (12, 14, 4), (15, 16, 6), (17, 20, 3), (21, 24, 3)]
SPLITS = ["cal", "A0", "A", "B", "C"]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unavailable"


def machine_info():
    import torch

    mem = sh(["sysctl", "-n", "hw.memsize"])
    return {
        "os": f"{platform.system()} {platform.release()} ({sh(['sw_vers', '-productVersion'])})",
        "machine": platform.machine(),
        "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "model": sh(["sysctl", "-n", "hw.model"]),
        "logical_cpus": sh(["sysctl", "-n", "hw.ncpu"]),
        "ram_gb": round(int(mem) / 2**30, 1) if mem.isdigit() else mem,
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
    }


def pick_bench(cal: pd.DataFrame, rng) -> pd.DataFrame:
    chosen = []
    for lo, hi, k in BINS:
        pool = cal[cal.natoms.between(lo, hi)]
        take = min(k, len(pool))
        chosen.append(pool.iloc[np.sort(rng.choice(len(pool), take, replace=False))])
    out = pd.concat(chosen)
    if len(out) < N_BENCH:  # a bin was short: top up with random remaining Cal structures
        rest = cal[~cal.sid.isin(out.sid)]
        out = pd.concat([out, rest.iloc[rng.choice(len(rest), N_BENCH - len(out), replace=False)]])
    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # random call order


def main():
    sp = {n: pd.read_csv(ROOT / f"data/splits/{n}.csv") for n in SPLITS}
    cal_sids = set(sp["cal"].sid)
    for n in SPLITS[1:]:
        assert not (cal_sids & set(sp[n].sid)), f"Cal overlaps {n}"
    meta = pd.read_pickle(ROOT / "data/processed/meta.pkl").set_index("sid")

    rng = np.random.default_rng(SEED)
    bench = pick_bench(sp["cal"].sort_values("index").reset_index(drop=True), rng)
    assert set(bench.sid) <= cal_sids and len(bench) == N_BENCH
    assert not any(set(bench.sid) & set(sp[n].sid) for n in SPLITS[1:])

    ds = D.open_dataset()
    atoms = {r.sid: ds.get_atoms(int(r["index"])) for _, r in bench.iterrows()}

    t0 = time.perf_counter()
    predictor = UMAPredictor()
    t_load = time.perf_counter() - t0
    cfg = CacheConfig.from_predictor(predictor)
    cache = PredictionCache(ROOT / "data/cache/predictions.sqlite")
    hits = [s for s in bench.sid if cache.get(s, cfg) is not None]
    if hits:
        sys.exit(f"{len(hits)} benchmark structures already cached; timing would be invalid. Use a fresh cache.")

    times, energies = [], {}
    for sid in bench.sid:
        t0 = time.perf_counter()
        energies.update(predict_cached(predictor, [(sid, atoms[sid])], cache, allowed_sids=cal_sids))
        times.append(time.perf_counter() - t0)
    bench["seconds"] = times
    bench["uma_total_eV"] = bench.sid.map(energies)
    bench["uma_epa"] = bench.uma_total_eV / bench.natoms
    bench["dft_raw_epa"] = bench.sid.map(meta.energy_per_atom_eV)
    bench["dft_mp2020_epa"] = bench.sid.map(meta.energy_corrected_mp2020_eV) / bench.natoms
    bench["err_raw"] = bench.uma_epa - bench.dft_raw_epa
    bench["err_mp2020"] = bench.uma_epa - bench.dft_mp2020_epa
    bench["mp2020_offset_epa"] = bench.dft_mp2020_epa - bench.dft_raw_epa
    bench["max_force"] = bench.sid.map(meta.max_abs_force_eV_A)

    # ---- timing ----
    steady = bench.iloc[1:]  # first call carries one-off warm-up cost
    X = steady.natoms.to_numpy(float)
    slope, intercept, r, _, se_slope = stats.linregress(X, steady.seconds.to_numpy())
    fit = lambda n: intercept + slope * np.asarray(n, float)  # noqa: E731
    per_split = {}
    for n in SPLITS:
        nat = sp[n].natoms.to_numpy()
        per_split[n] = {"n": len(nat), "mean_natoms": round(float(nat.mean()), 2),
                        "fit_seconds_total": round(float(fit(nat).sum()), 1),
                        "meanrate_seconds_total": round(float(steady.seconds.mean() * len(nat)), 1),
                        "per_atom_rate_seconds_total": round(float(steady.seconds.sum() / steady.natoms.sum() * nat.sum()), 1)}
    tot = {k: round(sum(v[k] for v in per_split.values()), 1) for k in ("fit_seconds_total", "meanrate_seconds_total", "per_atom_rate_seconds_total")}
    n_tot = sum(v["n"] for v in per_split.values())
    timing = {
        "model_load_seconds": round(t_load, 1),
        "first_call_seconds": round(times[0], 3),
        "all_20": {"mean_s_per_structure": round(float(np.mean(times)), 3), "s_per_atom": round(float(bench.seconds.sum() / bench.natoms.sum()), 4)},
        "steady_state_19": {"mean_s_per_structure": round(float(steady.seconds.mean()), 3), "median": round(float(steady.seconds.median()), 3),
                            "max": round(float(steady.seconds.max()), 3), "s_per_atom": round(float(steady.seconds.sum() / steady.natoms.sum()), 4)},
        "linear_fit_seconds_vs_natoms": {"intercept": round(float(intercept), 4), "slope_per_atom": round(float(slope), 5),
                                         "se_slope": round(float(se_slope), 5), "r": round(float(r), 3), "n": len(steady)},
        "seconds_by_natoms": {int(n): round(float(g.seconds.mean()), 3) for n, g in bench.groupby("natoms")},
        "EXTRAPOLATION_current_proposal": {"label": "EXTRAPOLATION from 19 steady-state calls on one machine; not measured",
                                           "n_structures": n_tot, "per_split": per_split, "total_seconds_by_method": tot,
                                           "total_hours_by_method": {k: round(v / 3600, 2) for k, v in tot.items()}},
    }
    mean_rate = float(steady.seconds.mean())
    timing["EXTRAPOLATION_6000_structures"] = {
        "label": "EXTRAPOLATION: 6,000 x mean steady-state seconds per structure (Cal-based mix of cell sizes)",
        "hours_at_mean_rate": round(6000 * mean_rate / 3600, 2),
        "hours_at_linear_fit_with_current_split_natoms": round(tot["fit_seconds_total"] / n_tot * 6000 / 3600, 2)}

    # ---- label check (Cal structures only) ----
    ok = bench.dropna(subset=["dft_mp2020_epa"])
    def stat(e):
        e = np.asarray(e, float)
        return {"n": len(e), "mean_signed_eV_per_atom": round(float(e.mean()), 4), "MAE": round(float(np.abs(e).mean()), 4),
                "median_abs": round(float(np.median(np.abs(e))), 4), "RMSE": round(float(np.sqrt((e**2).mean())), 4), "max_abs": round(float(np.abs(e).max()), 4)}
    reg = stats.linregress(ok.mp2020_offset_epa, ok.uma_epa - ok.dft_raw_epa) if ok.mp2020_offset_epa.std() > 0 else None
    comp = {}
    for col in ("family", "tm_class"):
        comp[col] = {k: {"n": len(g), "MAE_raw": round(float(g.err_raw.abs().mean()), 4)} for k, g in bench.assign(**{col: bench.sid.map(sp["cal"].set_index("sid")[col])}).groupby(col)}
    nel = bench.sid.map(meta.n_elements)
    label = {
        "note": "n=20 Cal structures only; descriptive. No test-split structure used.",
        "vs_raw_dft": stat(bench.err_raw),
        "vs_mp2020_corrected (structures with a correction)": stat(ok.err_mp2020),
        "vs_raw_dft (same subset)": stat(ok.err_raw),
        "n_without_mp2020": int(bench.dft_mp2020_epa.isna().sum()),
        "regression_of_(UMA-raw)_on_(mp2020_offset_per_atom)": None if reg is None else {
            "slope": round(float(reg.slope), 3), "r": round(float(reg.rvalue), 3), "p": round(float(reg.pvalue), 3), "n": len(ok),
            "reading": "slope ~0: UMA tracks the raw label; slope ~1: UMA tracks the MP2020-corrected label"},
        "n_with_nonzero_mp2020_offset": int((ok.mp2020_offset_epa.abs() > 1e-6).sum()),
        "spearman_abs_err_vs_natoms": round(float(stats.spearmanr(bench.err_raw.abs(), bench.natoms)[0]), 3),
        "spearman_abs_err_vs_n_elements": round(float(stats.spearmanr(bench.err_raw.abs(), nel)[0]), 3),
        "spearman_abs_err_vs_max_force": round(float(stats.spearmanr(bench.err_raw.abs(), bench.max_force)[0]), 3),
        "by_composition_group": comp,
    }
    cols = ["sid", "comp_key", "elements", "natoms", "seconds", "uma_epa", "dft_raw_epa", "dft_mp2020_epa", "err_raw", "err_mp2020", "mp2020_offset_epa", "max_force"]
    out = {"machine": machine_info(), "config": {"checkpoint": cfg.checkpoint, "task": cfg.task, "inference_settings": cfg.inference_settings,
                                                 "device": cfg.device, **cache.versions()},
           "selection": {"seed": SEED, "bins(natoms_lo,hi,count)": BINS, "note": "purposive spread over cell size, not a representative sample"},
           "timing": timing, "label_check": label, "per_structure": bench[cols].round(5).to_dict("records")}
    (ROOT / "data/benchmark/cal_benchmark.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"machine": out["machine"], "config": out["config"], "timing": {k: v for k, v in timing.items() if k != "EXTRAPOLATION_current_proposal"}, "label": {k: v for k, v in label.items() if k != "by_composition_group"}}, indent=1, default=str))
    cache.close()


if __name__ == "__main__":
    main()
