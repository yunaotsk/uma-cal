"""Resumable on-disk cache of UMA energy predictions, keyed by structure id (``sid``).

SQLite (stdlib), one commit per prediction, so an interrupted run resumes where it stopped.
The inference configuration (checkpoint, task, inference setting, device) is part of the key, so
predictions made under different settings are never mixed; library versions are stored alongside
every row and can be audited with ``distinct_versions``. The cache file is git-ignored (``data/cache/``).
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CACHE_PATH = Path("data/cache/predictions.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    sid TEXT NOT NULL,
    config_id TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    task TEXT NOT NULL,
    inference_settings TEXT NOT NULL,
    device TEXT NOT NULL,
    energy_total_eV REAL NOT NULL,
    natoms INTEGER NOT NULL,
    seconds REAL,
    torch_version TEXT,
    fairchem_core_version TEXT,
    ase_version TEXT,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (sid, config_id)
)
"""


@dataclass(frozen=True)
class CacheConfig:
    checkpoint: str
    task: str
    inference_settings: str
    device: str

    @property
    def config_id(self) -> str:
        raw = "|".join((self.checkpoint, self.task, self.inference_settings, self.device))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def from_predictor(cls, predictor) -> "CacheConfig":
        return cls(predictor.checkpoint, predictor.task, predictor.inference_settings, predictor.device)


def library_versions() -> dict[str, str]:
    import ase
    import fairchem.core
    import torch

    return {"torch_version": torch.__version__, "fairchem_core_version": fairchem.core.__version__, "ase_version": ase.__version__}


class PredictionCache:
    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH, versions: dict[str, str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._versions = versions  # injectable for tests; looked up lazily otherwise
        self._db = sqlite3.connect(self.path)
        self._db.execute(_SCHEMA)
        self._db.commit()

    def versions(self) -> dict[str, str]:
        if self._versions is None:
            self._versions = library_versions()
        return self._versions

    def get(self, sid: str, cfg: CacheConfig) -> float | None:
        row = self._db.execute(
            "SELECT energy_total_eV FROM predictions WHERE sid = ? AND config_id = ?", (sid, cfg.config_id)
        ).fetchone()
        return None if row is None else float(row[0])

    def put(self, sid: str, cfg: CacheConfig, energy_total_eV: float, natoms: int, seconds: float | None = None) -> None:
        v = self.versions()
        self._db.execute(
            "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, cfg.config_id, cfg.checkpoint, cfg.task, cfg.inference_settings, cfg.device, float(energy_total_eV), int(natoms),
             seconds, v.get("torch_version"), v.get("fairchem_core_version"), v.get("ase_version"),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def rows(self, cfg: CacheConfig | None = None) -> list[dict]:
        q, args = "SELECT * FROM predictions", ()
        if cfg is not None:
            q, args = q + " WHERE config_id = ?", (cfg.config_id,)
        cur = self._db.execute(q + " ORDER BY sid", args)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def distinct_versions(self) -> list[tuple]:
        """Distinct (config_id, torch, fairchem, ase) tuples: more than one per config means mixed library versions."""
        return self._db.execute(
            "SELECT DISTINCT config_id, torch_version, fairchem_core_version, ase_version FROM predictions"
        ).fetchall()

    def close(self) -> None:
        self._db.close()


GATE_PATH = Path("data/cache/a0_gate.json")
SPLITS_DIR = Path("data/splits")
GATE_BAND = 0.02  # A0 coverage must be within about 2 points of nominal (reports/analysis_plan.md)


class SplitGuard:
    """Decides which structures may be predicted.

    Cal and A0 are always allowed. A, B and C stay **locked until the A0 gate is recorded as passed**
    (``record_a0_gate``). The gate decision is made on Cal and A0 predictions only, before any A, B or C
    prediction exists, which is what keeps the analysis pre-specified. This is a procedural guard against
    accidents, not a security boundary: the gate file can be edited by hand, and doing so would be a protocol violation.
    """

    def __init__(self, splits_dir: str | Path = SPLITS_DIR, gate_path: str | Path = GATE_PATH):
        self.splits_dir, self.gate_path = Path(splits_dir), Path(gate_path)

    def _sids(self, name: str) -> set[str]:
        import csv

        with open(self.splits_dir / f"{name}.csv", newline="") as f:
            return {row["sid"] for row in csv.DictReader(f)}

    def gate_passed(self) -> bool:
        import json

        if not self.gate_path.exists():
            return False
        return json.loads(self.gate_path.read_text()).get("passed") is True

    def allowed_sids(self) -> set[str]:
        allowed = self._sids("cal") | self._sids("A0")
        if self.gate_passed():
            allowed |= self._sids("A") | self._sids("B") | self._sids("C")
        return allowed


def a0_gate_decision(coverage: float, nominal: float = 0.90, band: float = GATE_BAND) -> bool:
    """True iff A0 coverage is within ``band`` of nominal (inclusive, rounded to avoid float edge noise)."""
    return round(abs(coverage - nominal), 9) <= band


GATE_REPORT_PATH = Path("reports/a0_gate_result.md")
GATE_LEVELS = (0.80, 0.90, 0.95)  # 90% is the gate; 80% and 95% are reported as secondary diagnostics


def conformal_quantile(cal_scores: Iterable[float], alpha: float) -> float:
    """Split-conformal quantile: the k-th smallest score with k = ceil((n+1)(1-alpha)); +inf if k > n."""
    import math

    xs = sorted(float(x) for x in cal_scores)
    n = len(xs)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else xs[k - 1]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    import math

    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def a0_gate_summary(cal_scores: Iterable[float], a0_scores: Iterable[float], levels=GATE_LEVELS) -> dict:
    """Coverage of Cal-calibrated intervals on A0 at each level, from absolute-error scores (eV/atom)."""
    import math

    cal, a0 = [float(x) for x in cal_scores], [float(x) for x in a0_scores]
    if not cal or not a0:
        raise ValueError("cal_scores and a0_scores must be non-empty")
    rows = {}
    for level in levels:
        q = conformal_quantile(cal, 1 - level)
        covered = sum(1 for x in a0 if x <= q)
        p = covered / len(a0)
        lo, hi = wilson_interval(covered, len(a0))
        rows[level] = {"level": level, "alpha": round(1 - level, 10), "k": math.ceil((len(cal) + 1) * level), "q_eV_per_atom": q,
                       "n_covered": covered, "coverage": p, "se_composition_level": math.sqrt(p * (1 - p) / len(a0)),
                       "wilson95": [lo, hi]}
    return {"n_cal": len(cal), "n_a0": len(a0), "levels": rows}


def _git(*args: str) -> str:
    import subprocess

    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "unavailable"


def write_gate_report(summary: dict, passed: bool, decided_utc: str, config: "CacheConfig", versions: dict[str, str],
                      code_commit: str, code_dirty: bool, report_path: str | Path = GATE_REPORT_PATH,
                      nominal: float = 0.90, band: float = GATE_BAND, split_hashes: dict[str, str] | None = None) -> Path:
    """Write the committed, human-readable record of the A0 gate. Written whether the gate passes or fails."""
    lv = summary["levels"][0.90] if 0.90 in summary["levels"] else None
    L = [
        "# A0 gate result", "",
        f"**Result: {'PASS' if passed else 'FAIL'}**" + (f" (A0 coverage at 90% = {lv['coverage']:.4f}; rule: within {band} of {nominal})" if lv else ""), "",
        f"Decided {decided_utc} from Cal and A0 predictions only. A, B and C were locked (`SplitGuard`) when this was decided.",
        "Rule (`reports/analysis_plan.md`, Amendment 1): pass iff |coverage at 90% on A0 − 0.90| ≤ 0.02; 80% and 95% are diagnostics and do not change the decision.", "",
        f"- n_cal = {summary['n_cal']:,}; n_a0 = {summary['n_a0']:,} (one structure per composition, so n is the composition count)",
        "- Score: absolute error of UMA energy per atom against the raw DFT label, in eV/atom; intervals ŷ ± q from split conformal on Cal.", "",
        "| Level | k | q (eV/atom) | A0 covered | Coverage | SE (composition level) | Wilson 95% interval |", "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for level, r in sorted(summary["levels"].items()):
        L.append(f"| {level:.0%} | {r['k']:,} | {r['q_eV_per_atom']:.5f} | {r['n_covered']:,} | {r['coverage']:.4f} | {r['se_composition_level']:.4f} | "
                 f"{r['wilson95'][0]:.4f} to {r['wilson95'][1]:.4f} |")
    L += ["", "## Provenance", "",
          f"- Code commit: `{code_commit}`" + (" (**working tree had uncommitted changes in src, scripts, reports or data/splits**)" if code_dirty else " (clean)"),
          f"- Inference config: checkpoint `{config.checkpoint}`, task `{config.task}`, inference settings `{config.inference_settings}`, device `{config.device}`",
          *[f"- {k}: {v}" for k, v in sorted(versions.items())]]
    if split_hashes:
        L += ["", "Split files used (compare with `reports/split_freeze.md`):", "", *[f"- `{k}`: `{v}`" for k, v in split_hashes.items()]]
    L += [""]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L))
    return report_path


def record_a0_gate(cal_scores: Iterable[float], a0_scores: Iterable[float], config: "CacheConfig",
                   gate_path: str | Path = GATE_PATH, report_path: str | Path = GATE_REPORT_PATH,
                   nominal: float = 0.90, band: float = GATE_BAND, versions: dict[str, str] | None = None,
                   code_commit: str | None = None, code_dirty: bool | None = None, splits_dir: str | Path | None = SPLITS_DIR,
                   overwrite: bool = False) -> bool:
    """Compute and record the A0 gate decision. Returns ``passed``.

    Coverage is computed here from the absolute-error scores (not typed in by hand). Writes the machine-readable gate
    record (which unlocks or keeps locked A, B, C in ``SplitGuard``) and the committed report ``reports/a0_gate_result.md``;
    **the report is written even when the gate fails.** The decision is made once: an existing record is not
    overwritten unless ``overwrite=True`` (which would need an amendment to the analysis plan).
    """
    import json

    gate_path = Path(gate_path)
    if gate_path.exists() and not overwrite:
        raise FileExistsError(f"{gate_path} already exists: the A0 gate is decided once (see reports/analysis_plan.md)")
    summary = a0_gate_summary(cal_scores, a0_scores)
    coverage = summary["levels"][nominal]["coverage"]
    passed = a0_gate_decision(coverage, nominal, band)
    decided = datetime.now(timezone.utc).isoformat(timespec="seconds")
    versions = versions if versions is not None else library_versions()
    code_commit = code_commit if code_commit is not None else (_git("rev-parse", "HEAD") or "unavailable")
    if code_dirty is None:
        code_dirty = bool(_git("status", "--porcelain", "--", "src", "scripts", "reports", "data/splits"))
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps({
        "passed": passed, "a0_coverage": coverage, "nominal": nominal, "band": band, "n_a0": summary["n_a0"], "n_cal": summary["n_cal"],
        "decided_utc": decided, "code_commit": code_commit,
        "levels": {f"{k:.2f}": {"q": v["q_eV_per_atom"], "coverage": v["coverage"]} for k, v in summary["levels"].items()},
        "rule": "pass iff |coverage - nominal| <= band (reports/analysis_plan.md); decided before any A, B or C prediction",
    }, indent=2))
    hashes = None
    if splits_dir is not None:
        sd = Path(splits_dir)
        hashes = {f"{sd.as_posix()}/{n}.csv": _sha256_file(sd / f"{n}.csv") for n in ("cal", "A0")}
    write_gate_report(summary, passed, decided, config, versions, code_commit, code_dirty, report_path, nominal, band, hashes)
    return passed


def predict_cached(predictor, items: Iterable[tuple[str, object]], cache: PredictionCache,
                   allowed_sids: Iterable[str] | None = None) -> dict[str, float]:
    """Predict total energies (eV) for ``(sid, Atoms)`` pairs, skipping cache hits.

    Hard guard: any sid outside the allowed set raises ``PermissionError`` *before* any prediction is made.
    ``allowed_sids=None`` means the default ``SplitGuard()``: Cal and A0 only, with A, B and C locked until the
    A0 gate has been recorded as passed. Pass an explicit set to narrow it further (e.g. Cal only).
    Returns ``{sid: energy_total_eV}`` for every item (hits and new predictions).
    """
    items = list(items)
    allowed = SplitGuard().allowed_sids() if allowed_sids is None else set(allowed_sids)
    bad = [sid for sid, _ in items if sid not in allowed]
    if bad:
        raise PermissionError(f"{len(bad)} structure(s) not allowed, e.g. {bad[:3]}; refusing to predict "
                              "(A, B and C stay locked until the A0 gate is recorded as passed)")
    cfg = CacheConfig.from_predictor(predictor)
    out: dict[str, float] = {}
    for sid, atoms in items:
        hit = cache.get(sid, cfg)
        if hit is None:
            t0 = time.perf_counter()
            e = predictor.predict_energy(atoms)
            cache.put(sid, cfg, e, len(atoms), time.perf_counter() - t0)
            hit = e
        out[sid] = hit
    return out
