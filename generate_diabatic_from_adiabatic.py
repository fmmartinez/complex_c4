#!/usr/bin/env python3
"""Build reduced diabatic Hamiltonians from adiabatic-state JSON output."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def hermite_phys(n: int, x: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(2, n + 1):
        h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def ho_basis(n: int, alpha: float, center: float, r_grid: np.ndarray) -> np.ndarray:
    xi = np.sqrt(alpha) * (r_grid - center)
    norm = (alpha / math.pi) ** 0.25 / math.sqrt((2.0 ** n) * math.factorial(n))
    return norm * hermite_phys(n, xi) * np.exp(-0.5 * xi * xi)


def build_basis_from_metadata(r_grid: np.ndarray, n_per_center: int, centers: Sequence[float], alphas: Sequence[float]) -> np.ndarray:
    basis: List[np.ndarray] = []
    for n in range(n_per_center):
        basis.append(ho_basis(n, float(alphas[0]), float(centers[0]), r_grid))
    for n in range(n_per_center):
        basis.append(ho_basis(n, float(alphas[1]), float(centers[1]), r_grid))
    return np.asarray(basis)


def parse_state_selection(raw: str, n_states: int) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--selected-states cannot be empty")
    idx: List[int] = []
    for p in parts:
        val = int(p)
        if val < 1 or val > n_states:
            raise ValueError(f"Selected state {val} out of valid range [1, {n_states}].")
        idx.append(val - 1)
    if len(set(idx)) != len(idx):
        raise ValueError("--selected-states contains duplicates.")
    return idx


def best_permutation_by_overlap(prev_states: np.ndarray, cur_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find permutation/sign for current diabatic states to maximize continuity with previous states.

    prev_states/cur_states: shape (n_states, n_grid)
    """
    n = prev_states.shape[0]
    overlap = prev_states @ cur_states.T

    if n <= 7:
        best_score = -1.0
        best_perm = tuple(range(n))
        for perm in itertools.permutations(range(n)):
            score = sum(abs(overlap[i, perm[i]]) for i in range(n))
            if score > best_score:
                best_score = score
                best_perm = perm
        perm = np.asarray(best_perm, dtype=int)
    else:
        # Greedy fallback for larger subspaces.
        remaining = set(range(n))
        perm_list = [-1] * n
        for i in range(n):
            j_best = max(remaining, key=lambda j: abs(overlap[i, j]))
            perm_list[i] = j_best
            remaining.remove(j_best)
        perm = np.asarray(perm_list, dtype=int)

    reordered = cur_states[perm, :]
    signs = np.ones(n)
    for i in range(n):
        ov = np.dot(prev_states[i], reordered[i])
        if ov < 0.0:
            signs[i] = -1.0
    reordered = (signs[:, None]) * reordered
    return perm, signs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reduced diabatic Hamiltonian JSON from adiabatic-state JSON")
    parser.add_argument("--adiabatic-json", type=Path, default=Path("adiabatic_states.json"))
    parser.add_argument("--selected-states", type=str, default="1,2,3", help="1-based comma-separated adiabatic state indices")
    parser.add_argument("--output-json", type=Path, default=Path("diabatic_matrices.json"))
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("diabatic_state_density_plots"),
        help="Directory for per-R_AB diabatic |phi|^2 plots",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for diabatic-state density plots. Install dependencies from requirements.txt") from exc

    ad = json.loads(args.adiabatic_json.read_text(encoding="utf-8"))

    r_values = [float(v) for v in ad["r_ab_values"]]
    evals_all = np.asarray(ad["eigenvalues"], dtype=float)  # (n_r, n_basis)
    coeffs_all = np.asarray(ad["coefficients"], dtype=float)  # (n_r, n_basis, n_basis)

    grid_meta = ad["grid"]
    r_grid = np.asarray(grid_meta["values"], dtype=float)
    dr = float(grid_meta["step"])
    weights = np.full(r_grid.shape[0], dr, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    basis_meta = ad["basis"]
    n_per_center = int(basis_meta["n_per_center"])
    basis = build_basis_from_metadata(r_grid, n_per_center, basis_meta["centers"], basis_meta["alphas"])

    n_basis = int(basis.shape[0])
    selected = parse_state_selection(args.selected_states, n_basis)
    n_sel = len(selected)

    plots_dir = args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    prev_diab_grid = None

    for ir, r_ab in enumerate(r_values):
        eigvals = evals_all[ir]
        coeffs = coeffs_all[ir]  # columns are adiabatic eigenvectors in HO basis

        # Build selected adiabatic wavefunctions on integration grid.
        psi_ad_all = coeffs.T @ basis  # (n_basis, n_grid)
        psi_sel = psi_ad_all[selected, :]  # (n_sel, n_grid)

        # Reduced adiabatic Hamiltonian is diagonal in selected adiabatic basis.
        h_adi = np.diag(eigvals[selected])

        # R operator matrix in selected adiabatic basis.
        r_mat = np.zeros((n_sel, n_sel), dtype=float)
        for i in range(n_sel):
            for j in range(n_sel):
                r_mat[i, j] = np.sum(weights * psi_sel[i] * r_grid * psi_sel[j])
        r_mat = 0.5 * (r_mat + r_mat.T)

        r_eval, u = np.linalg.eigh(r_mat)

        # Transform to diabatic basis.
        h_dia = u.T @ h_adi @ u
        h_dia = 0.5 * (h_dia + h_dia.T)

        # Diabatic states on grid.
        psi_dia = u.T @ psi_sel  # (n_sel, n_grid)

        # Continuity tracking over R_AB.
        if prev_diab_grid is not None:
            perm, signs = best_permutation_by_overlap(prev_diab_grid, psi_dia)
            psi_dia = psi_dia[perm, :]
            h_dia = h_dia[np.ix_(perm, perm)]
            h_adi = h_adi[np.ix_(perm, perm)]
            r_mat = r_mat[np.ix_(perm, perm)]
            r_eval = r_eval[perm]
            # apply sign gauge: |d_i> -> s_i |d_i> => H_ij -> s_i s_j H_ij
            h_dia = (signs[:, None] * h_dia) * signs[None, :]
            psi_dia = signs[:, None] * psi_dia

        prev_diab_grid = psi_dia.copy()

        state_densities = psi_dia * psi_dia
        r_tag = re.sub(r"[^0-9A-Za-z_.-]", "_", f"{r_ab:.4f}")
        plot_path = plots_dir / f"diabatic_state_densities_R_{r_tag}.png"

        plt.figure(figsize=(7, 5))
        for i_state in range(n_sel):
            plt.plot(r_grid, state_densities[i_state], linewidth=1.2, label=f"state {i_state + 1}")
        plt.xlabel(r"$r_{AH}$ ($\AA$)")
        plt.ylabel(r"$|\phi|^2$")
        plt.title(f"Diabatic state densities at R_AB={r_ab:.4f} A")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

        results.append(
            {
                "R": float(r_ab),
                "n_eigen": n_sel,
                "selected_state_indices_1based": [i + 1 for i in selected],
                "selected_eigenvalues": [float(v) for v in eigvals[selected]],
                "r_matrix_before_diagonalization": r_mat.tolist(),
                "r_matrix_eigenvalues": [float(v) for v in r_eval],
                "hamiltonian_reduced_adiabatic": h_adi.tolist(),
                "hamiltonian_reduced_diabatic": h_dia.tolist(),
                "r_diagonalized_eigenstates_grid": psi_dia.tolist(),
                "r_diagonalized_state_densities_grid": state_densities.tolist(),
                "r_diagonalized_states_plot": str(plot_path),
            }
        )

    out = {
        "description": "Reduced diabatic Hamiltonian from adiabatic-state subspace via r_AH operator diabatization",
        "input_file": str(args.adiabatic_json),
        "n_eigen": n_sel,
        "requested_R": r_values,
        "processed_R": r_values,
        "results": results,
    }

    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote diabatic JSON: {args.output_json}")
    print(f"Selected adiabatic states (1-based): {[i+1 for i in selected]}")
    print(f"Wrote diabatic density plots to: {plots_dir}")


if __name__ == "__main__":
    main()
