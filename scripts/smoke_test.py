"""Real-model smoke test: UMA energy of FCC Al on CPU. Not part of pytest."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ase
import fairchem.core
import torch
from ase.build import bulk

from src.inference import UMAPredictor


def main():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)  # 4-atom periodic cell
    print(f"torch {torch.__version__}, ase {ase.__version__}, fairchem-core {fairchem.core.__version__}")

    t0 = time.perf_counter()
    predictor = UMAPredictor()
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    e1 = predictor.predict_energy(atoms)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    e2 = predictor.predict_energy(atoms)
    t_second = time.perf_counter() - t0

    print(f"checkpoint: {predictor.checkpoint}  task: {predictor.task}  device: {predictor.device}")
    print(f"inference settings: {predictor.inference_settings}")
    print(f"structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms, pbc={atoms.pbc.tolist()})")
    print(f"model load time:        {t_load:.3f} s")
    print(f"first inference time:   {t_first:.3f} s")
    print(f"second inference time:  {t_second:.3f} s")
    print(f"energy (first):  {e1:.6f} eV")
    print(f"energy (second): {e2:.6f} eV")
    print(f"energy per atom: {e1 / len(atoms):.6f} eV/atom")


if __name__ == "__main__":
    main()
