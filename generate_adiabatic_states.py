#!/usr/bin/env python3
"""Generate adiabatic proton-subsystem eigenstates for A-H-B and save intermediate JSON + eigenvalue plot."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List

import numpy as np

AMU_TO_KG = 1.66053906660e-27
AVOGADRO = 6.02214076e23
HBAR_J_S = 1.054571817e-34
KCAL_PER_J = 1.0 / 4184.0
ANG_PER_M = 1.0e10

# V_AHB constants (kcal/mol and Angstrom-consistent parameters)
A_CONST = 11.2
B_CONST = 7.1e13
C_CONST = 0.776
D_A = 110.0
D_AH = 0.95
D_BH = 0.97
N_A = 9.26
N_B = 11.42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate adiabatic eigenstates for the proton subsystem h_subsys = K_p + V_AHB "
            "using a two-center harmonic-oscillator basis."
        )
    )
    parser.add_argument(
        "--r-ab-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit R_AB values in Angstrom (overrides range options)",
    )
    parser.add_argument("--r-ab-start", type=float, default=None, help="Range start for R_AB (Angstrom)")
    parser.add_argument("--r-ab-stop", type=float, default=None, help="Range stop for R_AB (Angstrom, inclusive)")
    parser.add_argument("--r-ab-step", type=float, default=None, help="Range step for R_AB (Angstrom)")

    parser.add_argument("--n-ho-per-center", type=int, default=6, help="Number of HO basis functions per center")
    parser.add_argument("--center-a", type=float, default=1.0, help="Center for first HO set (Angstrom)")
    parser.add_argument("--center-b", type=float, default=1.6, help="Center for second HO set (Angstrom)")
    parser.add_argument("--alpha-a", type=float, default=9.9, help="HO alpha for first center")
    parser.add_argument("--alpha-b", type=float, default=10.1, help="HO alpha for second center")

    parser.add_argument("--grid-min", type=float, default=0.3, help="r_AH grid minimum (Angstrom)")
    parser.add_argument("--grid-step", type=float, default=0.02, help="r_AH grid spacing (Angstrom)")
    parser.add_argument("--grid-points", type=int, default=106, help="Number of r_AH grid points")

    parser.add_argument("--output-json", type=Path, default=Path("adiabatic_states.json"))
    parser.add_argument("--output-plot", type=Path, default=Path("adiabatic_eigenvalues.png"))
    parser.add_argument(
        "--density-plots-dir",
        type=Path,
        default=Path("adiabatic_state_density_plots"),
        help="Directory for per-R_AB adiabatic |phi|^2 plots",
    )
    return parser.parse_args()


def resolve_r_ab_values(args: argparse.Namespace) -> List[float]:
    if args.r_ab_values is not None:
        if len(args.r_ab_values) < 1:
            raise ValueError("--r-ab-values must contain at least one value.")
        return args.r_ab_values

    if args.r_ab_start is not None or args.r_ab_stop is not None or args.r_ab_step is not None:
        if args.r_ab_start is None or args.r_ab_stop is None or args.r_ab_step is None:
            raise ValueError("--r-ab-start, --r-ab-stop, and --r-ab-step must all be provided for range mode.")
        if args.r_ab_step <= 0.0:
            raise ValueError("--r-ab-step must be positive.")

        values: List[float] = []
        cur = args.r_ab_start
        while cur <= args.r_ab_stop + 1e-12:
            values.append(float(cur))
            cur += args.r_ab_step
        if not values:
            raise ValueError("R_AB range produced no values.")
        return values

    return [2.70]


def proton_kinetic_prefactor_kcal_mol_ang2() -> float:
    proton_mass_kg = 1.0 * AMU_TO_KG
    pref_j_ang2 = (HBAR_J_S * HBAR_J_S / (2.0 * proton_mass_kg)) * (ANG_PER_M * ANG_PER_M)
    return pref_j_ang2 * AVOGADRO * KCAL_PER_J


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


def second_derivative(values: np.ndarray, dx: float) -> np.ndarray:
    d2 = np.zeros_like(values)
    # 2nd-order one-sided at edges
    d2[0] = (2.0 * values[0] - 5.0 * values[1] + 4.0 * values[2] - values[3]) / (dx * dx)
    d2[-1] = (2.0 * values[-1] - 5.0 * values[-2] + 4.0 * values[-3] - values[-4]) / (dx * dx)
    # centered interior
    d2[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / (dx * dx)
    return d2


def v_ahb(r_ah: np.ndarray, r_ab: float) -> np.ndarray:
    eps = 1e-12
    r_ah_safe = np.maximum(r_ah, eps)
    r_bh = r_ab - r_ah
    r_bh_safe = np.maximum(r_bh, eps)

    term1 = B_CONST * np.exp(-A_CONST * r_ab)
    term2 = D_A * (1.0 - np.exp((-N_A * (r_ah - D_AH) ** 2) / (2.0 * r_ah_safe)))
    term3 = C_CONST * D_A * (1.0 - np.exp((-N_B * (r_bh - D_BH) ** 2) / (2.0 * r_bh_safe)))
    return term1 + term2 + term3


def build_basis(r_grid: np.ndarray, n_per_center: int, center_a: float, center_b: float, alpha_a: float, alpha_b: float) -> np.ndarray:
    basis = []
    for n in range(n_per_center):
        basis.append(ho_basis(n, alpha_a, center_a, r_grid))
    for n in range(n_per_center):
        basis.append(ho_basis(n, alpha_b, center_b, r_grid))
    return np.asarray(basis)


def generalized_symmetric_eigh(h_mat: np.ndarray, s_mat: np.ndarray, regularization_floor: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    s_vals, s_vecs = np.linalg.eigh(s_mat)
    max_eval = max(float(np.max(s_vals)), regularization_floor)
    floor = max(regularization_floor, 1e-14 * max_eval)

    # Keep full basis dimensionality by regularizing very small/negative overlap eigenvalues.
    s_vals_reg = np.copy(s_vals)
    s_vals_reg[s_vals_reg < floor] = floor

    x = s_vecs @ np.diag(1.0 / np.sqrt(s_vals_reg))
    h_ortho = x.T @ h_mat @ x
    h_ortho = 0.5 * (h_ortho + h_ortho.T)
    e_vals, y_vecs = np.linalg.eigh(h_ortho)
    coeffs = x @ y_vecs
    return e_vals, coeffs


def main() -> None:
    args = parse_args()
    r_ab_values = resolve_r_ab_values(args)

    if args.n_ho_per_center < 1:
        raise ValueError("--n-ho-per-center must be >= 1")

    r_grid = np.asarray([args.grid_min + i * args.grid_step for i in range(args.grid_points)], dtype=float)
    weights = np.full(args.grid_points, args.grid_step, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    basis = build_basis(
        r_grid=r_grid,
        n_per_center=args.n_ho_per_center,
        center_a=args.center_a,
        center_b=args.center_b,
        alpha_a=args.alpha_a,
        alpha_b=args.alpha_b,
    )
    n_basis = basis.shape[0]

    k_pref = proton_kinetic_prefactor_kcal_mol_ang2()
    basis_d2 = np.asarray([second_derivative(basis[i], args.grid_step) for i in range(n_basis)])

    s_mat = np.zeros((n_basis, n_basis), dtype=float)
    t_mat = np.zeros((n_basis, n_basis), dtype=float)
    for i in range(n_basis):
        for j in range(n_basis):
            s_mat[i, j] = np.sum(weights * basis[i] * basis[j])
            t_mat[i, j] = np.sum(weights * basis[i] * (-k_pref * basis_d2[j]))

    # numerical symmetry cleanup
    s_mat = 0.5 * (s_mat + s_mat.T)
    t_mat = 0.5 * (t_mat + t_mat.T)

    all_eigvals = []
    all_coeffs = []

    for r_ab in r_ab_values:
        v_grid = v_ahb(r_grid, r_ab)
        v_mat = np.zeros((n_basis, n_basis), dtype=float)
        for i in range(n_basis):
            for j in range(n_basis):
                v_mat[i, j] = np.sum(weights * basis[i] * v_grid * basis[j])
        v_mat = 0.5 * (v_mat + v_mat.T)

        h_mat = t_mat + v_mat
        evals, coeffs = generalized_symmetric_eigh(h_mat, s_mat)
        all_eigvals.append(evals.tolist())
        all_coeffs.append(coeffs.tolist())

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate the eigenvalue plot. Install dependencies from requirements.txt") from exc

    plots_dir = args.density_plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_r_results = []
    
    for i_r, r_ab in enumerate(r_ab_values):
        coeffs = np.asarray(all_coeffs[i_r], dtype=float)
        psi_ad = coeffs.T @ basis
        densities = psi_ad * psi_ad

        r_tag = re.sub(r"[^0-9A-Za-z_.-]", "_", f"{r_ab:.4f}")
        plot_path = plots_dir / f"adiabatic_state_densities_R_{r_tag}.png"

        plt.figure(figsize=(7, 5))
        for i_state in range(n_basis):
            plt.plot(r_grid, densities[i_state], linewidth=1.0, label=f"state {i_state + 1}")
        plt.xlabel(r"$r_{AH}$ ($\AA$)")
        plt.ylabel(r"$|\phi|^2$")
        plt.title(f"Adiabatic state densities at R_AB={r_ab:.4f} A")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

        per_r_results.append(
            {
                "R": float(r_ab),
                "eigenvalues": [float(v) for v in all_eigvals[i_r]],
                "adiabatic_eigenstates_grid": psi_ad.tolist(),
                "adiabatic_state_densities_grid": densities.tolist(),
                "adiabatic_states_plot": str(plot_path),
            }
        )

    output = {
        "model": "AHB proton subsystem adiabatic states",
        "units": {"energy": "kcal/mol", "distance": "angstrom", "time": "fs", "mass": "amu"},
        "constants": {
            "a": A_CONST,
            "b": B_CONST,
            "c": C_CONST,
            "d_A": D_AH,
            "d_B": D_BH,
            "D_A": D_A,
            "n_A": N_A,
            "n_B": N_B,
        },
        "basis": {
            "n_per_center": args.n_ho_per_center,
            "centers": [args.center_a, args.center_b],
            "alphas": [args.alpha_a, args.alpha_b],
            "total_basis_functions": n_basis,
        },
        "grid": {"min": args.grid_min, "step": args.grid_step, "points": args.grid_points, "values": r_grid.tolist()},
        "r_ab_values": r_ab_values,
        "eigenvalues": all_eigvals,
        "coefficients": all_coeffs,
        "state_labels": [f"state_{i+1}" for i in range(n_basis)],
        "results": per_r_results,
    }

    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    x = np.asarray(r_ab_values, dtype=float)
    eigvals_arr = np.asarray(all_eigvals, dtype=float)
    plt.figure(figsize=(7, 5))
    for state in range(eigvals_arr.shape[1]):
        plt.plot(x, eigvals_arr[:, state], linewidth=1.0, label=f"state {state + 1}")
    plt.xlabel(r"$R_{AB}$ ($\AA$)")
    plt.ylabel("Eigenvalue (kcal/mol)")
    plt.title("Adiabatic subsystem eigenvalues vs $R_{AB}$")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=200)

    print(f"Wrote intermediate JSON: {args.output_json}")
    print(f"Wrote eigenvalue plot: {args.output_plot}")
    print(f"Wrote adiabatic density plots to: {plots_dir}")


if __name__ == "__main__":
    main()
