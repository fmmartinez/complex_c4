#!/usr/bin/env python3
"""Generate and run NVE MD for coarse-grained chloromethane + AHB complex in vacuum."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

AMU_TO_KG = 1.66053906660e-27
KB = 1.380649e-23
COULOMB_KCAL_MOL_ANG_E2 = 332.063713299

# Conversion factors for internal units (angstrom, fs, amu, kcal/mol)
M_S_TO_ANG_FS = 1e-5
AMU_ANG2_FS2_TO_KCAL_MOL = 2390.05736055072
KCAL_MOL_ANG_TO_AMU_ANG_FS2 = 1.0 / AMU_ANG2_FS2_TO_KCAL_MOL

MASS = {
    "C1": 15.0,
    "C2": 35.5,
    "A": 93.0,
    "B": 59.0,
    "H": 1.0,
}
LABEL = {"C1": "C", "C2": "Cl", "A": "O", "B": "N", "H": "H"}
CHARGE = {"C1": +0.25, "C2": -0.25, "A": -0.5, "B": 0.0, "H": +0.5}

POLARIZATION = {
    "Q_A_cov": -0.5,
    "Q_H_cov": +0.5,
    "Q_B_cov": 0.0,
    "Q_A_ion": -1.0,
    "Q_H_ion": +0.5,
    "Q_B_ion": +0.5,
    "r0": 1.43,
    "l": 0.125,
}

# Solvent LJ parameters
SIGMA_SOLVENT = {"C1": 3.774, "C2": 3.481}
EPSILON_SOLVENT = {"C1": 0.238, "C2": 0.415}

# Explicit LJ pairs for complex interactions (kcal/mol, angstrom)
LJ_PARAMS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("A", "C1"): (3.5, 0.3974),
    ("A", "C2"): (3.5, 0.3974),
    ("B", "C1"): (3.5, 0.3974),
    ("B", "C2"): (3.5, 0.3974),
}

AHB_PARAMS = {
    "a": 11.2,
    "b": 7.1e13,
    "c": 0.776,
    "d_A": 0.95,
    "d_B": 0.97,
    "D_A": 110.0,
    "n_A": 9.26,
    "n_B": 11.42,
}


@dataclass
class Site:
    molecule_id: int
    site_type: str
    label: str
    mass_amu: float
    position_angstrom: List[float]
    velocity_ang_fs: List[float]


def dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: List[float]) -> float:
    return math.sqrt(dot(a, a))


def random_unit_vector(rng: random.Random) -> Tuple[float, float, float]:
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        mag = math.sqrt(x * x + y * y + z * z)
        if mag > 1e-12:
            return (x / mag, y / mag, z / mag)


def random_point_in_sphere(rng: random.Random, radius: float) -> Tuple[float, float, float]:
    ux, uy, uz = random_unit_vector(rng)
    r = radius * (rng.random() ** (1.0 / 3.0))
    return (r * ux, r * uy, r * uz)


def dist(a: Tuple[float, float, float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def sample_velocity(rng: random.Random, temperature_k: float, mass_amu: float) -> List[float]:
    sigma_m_s = math.sqrt(KB * temperature_k / (mass_amu * AMU_TO_KG))
    return [
        rng.gauss(0.0, sigma_m_s) * M_S_TO_ANG_FS,
        rng.gauss(0.0, sigma_m_s) * M_S_TO_ANG_FS,
        rng.gauss(0.0, sigma_m_s) * M_S_TO_ANG_FS,
    ]


def remove_net_linear_momentum(sites: List[Site]) -> None:
    total_mass = 0.0
    px = py = pz = 0.0
    for site in sites:
        vx, vy, vz = site.velocity_ang_fs
        total_mass += site.mass_amu
        px += site.mass_amu * vx
        py += site.mass_amu * vy
        pz += site.mass_amu * vz

    vcm = [px / total_mass, py / total_mass, pz / total_mass]
    for site in sites:
        site.velocity_ang_fs[0] -= vcm[0]
        site.velocity_ang_fs[1] -= vcm[1]
        site.velocity_ang_fs[2] -= vcm[2]


def kinetic_energy_kcal_mol(sites: List[Site]) -> float:
    kinetic_internal = 0.0
    for site in sites:
        vx, vy, vz = site.velocity_ang_fs
        kinetic_internal += 0.5 * site.mass_amu * (vx * vx + vy * vy + vz * vz)
    return kinetic_internal * AMU_ANG2_FS2_TO_KCAL_MOL


def instantaneous_temperature(sites: List[Site], remove_momentum_dof: bool = True) -> float:
    kinetic_energy_joule = 0.0
    for site in sites:
        vx, vy, vz = site.velocity_ang_fs
        v2_m_s2 = (vx / M_S_TO_ANG_FS) ** 2 + (vy / M_S_TO_ANG_FS) ** 2 + (vz / M_S_TO_ANG_FS) ** 2
        kinetic_energy_joule += 0.5 * (site.mass_amu * AMU_TO_KG) * v2_m_s2
    dof = 3 * len(sites)
    if remove_momentum_dof:
        dof -= 3
    return (2.0 * kinetic_energy_joule) / (dof * KB)


def get_lj_params(site_type_i: str, site_type_j: str) -> Optional[Tuple[float, float]]:
    a, b = sorted((site_type_i, site_type_j))
    if a in {"C1", "C2"} and b in {"C1", "C2"}:
        sigma_ij = 0.5 * (SIGMA_SOLVENT[a] + SIGMA_SOLVENT[b])
        epsilon_ij = math.sqrt(EPSILON_SOLVENT[a] * EPSILON_SOLVENT[b])
        return sigma_ij, epsilon_ij
    return LJ_PARAMS.get((a, b))


def coulomb_allowed(site_type_i: str, site_type_j: str) -> bool:
    pair = {site_type_i, site_type_j}
    if pair == {"A", "H"} or pair == {"B", "H"} or pair == {"A", "B"}:
        return False
    if pair == {"H"}:
        return False
    return True


def polarization_switch(r_ah: float) -> Tuple[float, float]:
    dr = r_ah - POLARIZATION["r0"]
    l = POLARIZATION["l"]
    denom = math.sqrt(dr * dr + l * l)
    f = 0.5 * (1.0 + dr / denom)
    # derivative of f wrt r_ah
    df_dr = 0.5 * (l * l) / (denom ** 3)
    return f, df_dr


def compute_complex_polarized_charges(sites: List[Site]) -> Tuple[float, float, float, float, float]:
    idx_a = idx_h = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "H":
            idx_h = idx

    if idx_a is None or idx_h is None:
        return CHARGE["A"], CHARGE["H"], CHARGE["B"], 0.0, 0.0

    ra = sites[idx_a].position_angstrom
    rh = sites[idx_h].position_angstrom
    r_ah = math.sqrt((rh[0]-ra[0])**2 + (rh[1]-ra[1])**2 + (rh[2]-ra[2])**2)

    f, _ = polarization_switch(r_ah)
    q_a = (1.0 - f) * POLARIZATION["Q_A_cov"] + f * POLARIZATION["Q_A_ion"]
    q_h = (1.0 - f) * POLARIZATION["Q_H_cov"] + f * POLARIZATION["Q_H_ion"]
    q_b = (1.0 - f) * POLARIZATION["Q_B_cov"] + f * POLARIZATION["Q_B_ion"]
    return q_a, q_h, q_b, f, r_ah


def compute_ahb_potential_and_derivatives(r_ah: float, r_ab: float) -> Tuple[float, float, float]:
    eps = 1e-12
    r = max(r_ah, eps)
    rab = max(r_ab, eps)
    s = max(rab - r, eps)

    a = AHB_PARAMS["a"]
    b = AHB_PARAMS["b"]
    c = AHB_PARAMS["c"]
    d_a = AHB_PARAMS["d_A"]
    d_b = AHB_PARAMS["d_B"]
    dcap = AHB_PARAMS["D_A"]
    n_a = AHB_PARAMS["n_A"]
    n_b = AHB_PARAMS["n_B"]

    term1 = b * math.exp(-a * rab)

    x = -n_a * ((r - d_a) ** 2) / (2.0 * r)
    term2 = dcap * (1.0 - math.exp(x))

    y = -n_b * ((s - d_b) ** 2) / (2.0 * s)
    term3 = c * dcap * (1.0 - math.exp(y))

    potential = term1 + term2 + term3

    dterm1_drab = -a * b * math.exp(-a * rab)

    dx_dr = -n_a * (r * r - d_a * d_a) / (2.0 * r * r)
    dterm2_dr = -dcap * math.exp(x) * dx_dr

    dy_ds = -n_b * (s * s - d_b * d_b) / (2.0 * s * s)
    dterm3_ds = -c * dcap * math.exp(y) * dy_ds

    dterm3_drab = dterm3_ds
    dterm3_dr = -dterm3_ds

    dV_drab = dterm1_drab + dterm3_drab
    dV_dr = dterm2_dr + dterm3_dr

    return potential, dV_dr, dV_drab


def compute_forces_and_potential(
    sites: List[Site], n_solvent_molecules: int
) -> Tuple[List[List[float]], float, float, float, float, float, float]:
    forces = [[0.0, 0.0, 0.0] for _ in sites]
    potential = 0.0

    idx_a = idx_h = idx_b = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "H":
            idx_h = idx
        elif site.site_type == "B":
            idx_b = idx

    q_a, q_h, q_b, f_pol, r_ah = compute_complex_polarized_charges(sites)

    site_charges = [CHARGE[site.site_type] for site in sites]
    if idx_a is not None:
        site_charges[idx_a] = q_a
    if idx_h is not None:
        site_charges[idx_h] = q_h
    if idx_b is not None:
        site_charges[idx_b] = q_b

    for i in range(len(sites)):
        si = sites[i]
        xi, yi, zi = si.position_angstrom
        for j in range(i + 1, len(sites)):
            sj = sites[j]
            if si.molecule_id == sj.molecule_id and si.molecule_id < n_solvent_molecules:
                continue

            dx = xi - sj.position_angstrom[0]
            dy = yi - sj.position_angstrom[1]
            dz = zi - sj.position_angstrom[2]
            r2 = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r2)
            inv_r = 1.0 / r

            pair_energy = 0.0
            dUdr = 0.0

            lj_params = get_lj_params(si.site_type, sj.site_type)
            if lj_params is not None:
                sigma_ij, epsilon_ij = lj_params
                sr = sigma_ij * inv_r
                sr2 = sr * sr
                sr6 = sr2 * sr2 * sr2
                sr12 = sr6 * sr6

                lj = 4.0 * epsilon_ij * (sr12 - sr6)
                pair_energy += lj
                dUdr += 4.0 * epsilon_ij * (-12.0 * sr12 * inv_r + 6.0 * sr6 * inv_r)

            if coulomb_allowed(si.site_type, sj.site_type):
                qi = site_charges[i]
                qj = site_charges[j]
                if abs(qi * qj) > 0.0:
                    coul = COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r
                    pair_energy += coul
                    dUdr += -COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r * inv_r

            if pair_energy != 0.0:
                potential += pair_energy
                scale = -dUdr * inv_r
                fx = scale * dx
                fy = scale * dy
                fz = scale * dz
                forces[i][0] += fx
                forces[i][1] += fy
                forces[i][2] += fz
                forces[j][0] -= fx
                forces[j][1] -= fy
                forces[j][2] -= fz

    if idx_a is not None and idx_h is not None and idx_b is not None:
        ra = sites[idx_a].position_angstrom
        rh = sites[idx_h].position_angstrom
        rb = sites[idx_b].position_angstrom

        v_ah = [rh[0] - ra[0], rh[1] - ra[1], rh[2] - ra[2]]
        v_ab = [rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]]
        r_ah_local = norm(v_ah)
        r_ab = norm(v_ab)

        v_ah_u = [v / max(r_ah_local, 1e-12) for v in v_ah]
        v_ab_u = [v / max(r_ab, 1e-12) for v in v_ab]

        v_ahb, dV_dr, dV_dR = compute_ahb_potential_and_derivatives(r_ah_local, r_ab)
        potential += v_ahb

        for k in range(3):
            f_h = -dV_dr * v_ah_u[k]
            forces[idx_h][k] += f_h
            forces[idx_a][k] -= f_h

        for k in range(3):
            f_b = -dV_dR * v_ab_u[k]
            forces[idx_b][k] += f_b
            forces[idx_a][k] -= f_b

    return forces, potential, q_a, q_h, q_b, f_pol, r_ah


def enforce_solvent_bond_constraints(sites: List[Site], n_solvent_molecules: int, bond_distance: float) -> None:
    for m in range(n_solvent_molecules):
        i = 2 * m
        j = i + 1
        si = sites[i]
        sj = sites[j]

        rij = [
            sj.position_angstrom[0] - si.position_angstrom[0],
            sj.position_angstrom[1] - si.position_angstrom[1],
            sj.position_angstrom[2] - si.position_angstrom[2],
        ]
        r = norm(rij)
        if r < 1e-12:
            continue

        unit = [rij[0] / r, rij[1] / r, rij[2] / r]
        delta = r - bond_distance
        mi = si.mass_amu
        mj = sj.mass_amu
        mt = mi + mj

        corr_i = (mj / mt) * delta
        corr_j = -(mi / mt) * delta

        for k in range(3):
            si.position_angstrom[k] += corr_i * unit[k]
            sj.position_angstrom[k] += corr_j * unit[k]


def enforce_solvent_velocity_constraints(sites: List[Site], n_solvent_molecules: int) -> None:
    for m in range(n_solvent_molecules):
        i = 2 * m
        j = i + 1
        si = sites[i]
        sj = sites[j]

        rij = [
            sj.position_angstrom[0] - si.position_angstrom[0],
            sj.position_angstrom[1] - si.position_angstrom[1],
            sj.position_angstrom[2] - si.position_angstrom[2],
        ]
        r = norm(rij)
        if r < 1e-12:
            continue
        unit = [rij[0] / r, rij[1] / r, rij[2] / r]

        vrel = [
            sj.velocity_ang_fs[0] - si.velocity_ang_fs[0],
            sj.velocity_ang_fs[1] - si.velocity_ang_fs[1],
            sj.velocity_ang_fs[2] - si.velocity_ang_fs[2],
        ]
        radial_rel = dot(vrel, unit)

        mi = si.mass_amu
        mj = sj.mass_amu
        mt = mi + mj

        for k in range(3):
            si.velocity_ang_fs[k] += (mj / mt) * radial_rel * unit[k]
            sj.velocity_ang_fs[k] -= (mi / mt) * radial_rel * unit[k]


def add_ahb_complex(
    sites: List[Site],
    n_solvent_molecules: int,
    temperature_k: float,
    radius_angstrom: float,
    min_inter_site_distance_angstrom: float,
    seed_rng: random.Random,
) -> None:
    complex_mol_id = n_solvent_molecules

    for _attempt in range(50000):
        a_pos = random_point_in_sphere(seed_rng, radius_angstrom)
        axis = random_unit_vector(seed_rng)
        h_pos = (
            a_pos[0] + 1.0 * axis[0],
            a_pos[1] + 1.0 * axis[1],
            a_pos[2] + 1.0 * axis[2],
        )
        b_pos = (
            a_pos[0] + 2.7 * axis[0],
            a_pos[1] + 2.7 * axis[1],
            a_pos[2] + 2.7 * axis[2],
        )

        if (
            norm([a_pos[0], a_pos[1], a_pos[2]]) > radius_angstrom
            or norm([h_pos[0], h_pos[1], h_pos[2]]) > radius_angstrom
            or norm([b_pos[0], b_pos[1], b_pos[2]]) > radius_angstrom
        ):
            continue

        if all(
            dist(a_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
            and dist(h_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
            and dist(b_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
            for existing in sites
        ):
            sites.append(
                Site(
                    molecule_id=complex_mol_id,
                    site_type="A",
                    label=LABEL["A"],
                    mass_amu=MASS["A"],
                    position_angstrom=[a_pos[0], a_pos[1], a_pos[2]],
                    velocity_ang_fs=sample_velocity(seed_rng, temperature_k, MASS["A"]),
                )
            )
            sites.append(
                Site(
                    molecule_id=complex_mol_id,
                    site_type="H",
                    label=LABEL["H"],
                    mass_amu=MASS["H"],
                    position_angstrom=[h_pos[0], h_pos[1], h_pos[2]],
                    velocity_ang_fs=sample_velocity(seed_rng, temperature_k, MASS["H"]),
                )
            )
            sites.append(
                Site(
                    molecule_id=complex_mol_id,
                    site_type="B",
                    label=LABEL["B"],
                    mass_amu=MASS["B"],
                    position_angstrom=[b_pos[0], b_pos[1], b_pos[2]],
                    velocity_ang_fs=sample_velocity(seed_rng, temperature_k, MASS["B"]),
                )
            )
            return

    raise RuntimeError("Could not place A-H-B complex inside placement radius without overlaps.")


def generate_configuration(
    n_molecules: int,
    temperature_k: float,
    radius_angstrom: float,
    c1_c2_distance_angstrom: float,
    min_inter_site_distance_angstrom: float,
    seed: int,
) -> List[Site]:
    rng = random.Random(seed)
    sites: List[Site] = []
    half_bond = 0.5 * c1_c2_distance_angstrom

    for mol_idx in range(n_molecules):
        placed = False
        for _attempt in range(20000):
            center = random_point_in_sphere(rng, radius_angstrom)
            direction = random_unit_vector(rng)

            c1_pos = (
                center[0] - half_bond * direction[0],
                center[1] - half_bond * direction[1],
                center[2] - half_bond * direction[2],
            )
            c2_pos = (
                center[0] + half_bond * direction[0],
                center[1] + half_bond * direction[1],
                center[2] + half_bond * direction[2],
            )

            if all(
                dist(c1_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
                and dist(c2_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
                for existing in sites
            ):
                sites.append(
                    Site(
                        molecule_id=mol_idx,
                        site_type="C1",
                        label=LABEL["C1"],
                        mass_amu=MASS["C1"],
                        position_angstrom=[c1_pos[0], c1_pos[1], c1_pos[2]],
                        velocity_ang_fs=sample_velocity(rng, temperature_k, MASS["C1"]),
                    )
                )
                sites.append(
                    Site(
                        molecule_id=mol_idx,
                        site_type="C2",
                        label=LABEL["C2"],
                        mass_amu=MASS["C2"],
                        position_angstrom=[c2_pos[0], c2_pos[1], c2_pos[2]],
                        velocity_ang_fs=sample_velocity(rng, temperature_k, MASS["C2"]),
                    )
                )
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place solvent molecule {mol_idx + 1} without overlaps after many attempts."
            )

    add_ahb_complex(
        sites=sites,
        n_solvent_molecules=n_molecules,
        temperature_k=temperature_k,
        radius_angstrom=radius_angstrom,
        min_inter_site_distance_angstrom=min_inter_site_distance_angstrom,
        seed_rng=rng,
    )

    remove_net_linear_momentum(sites)
    enforce_solvent_bond_constraints(sites, n_molecules, c1_c2_distance_angstrom)
    enforce_solvent_velocity_constraints(sites, n_molecules)
    return sites


def append_xyz_frame(path: Path, sites: List[Site], comment: str) -> None:
    lines = [str(len(sites)), comment]
    for site in sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_initial_xyz(path: Path, sites: List[Site], comment: str) -> None:
    lines = [str(len(sites)), comment]
    for site in sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_nve_md(
    sites: List[Site],
    n_solvent_molecules: int,
    steps: int,
    dt_fs: float,
    write_frequency: int,
    solvent_bond_distance: float,
    trajectory_path: Path,
    energy_log_path: Path,
) -> None:
    trajectory_path.write_text("", encoding="utf-8")
    energy_log_path.write_text("step time_fs KE_kcal_mol PE_kcal_mol TE_kcal_mol T_K Q_A Q_H Q_B f_pol r_AH\n", encoding="utf-8")

    forces, potential, q_a, q_h, q_b, f_pol, r_ah = compute_forces_and_potential(sites, n_solvent_molecules)

    for step in range(steps + 1):
        kinetic = kinetic_energy_kcal_mol(sites)
        temperature = instantaneous_temperature(sites)
        total = kinetic + potential

        with energy_log_path.open("a", encoding="utf-8") as flog:
            flog.write(
                f"{step} {step * dt_fs:.6f} {kinetic:.10f} {potential:.10f} {total:.10f} {temperature:.6f} "
                f"{q_a:.8f} {q_h:.8f} {q_b:.8f} {f_pol:.8f} {r_ah:.8f}\n"
            )

        if step % write_frequency == 0:
            append_xyz_frame(
                trajectory_path,
                sites,
                (
                    f"step={step} time_fs={step * dt_fs:.3f} "
                    f"KE={kinetic:.6f} PE={potential:.6f} TE={total:.6f} T={temperature:.3f} "
                    f"Q_A={q_a:.4f} Q_H={q_h:.4f} Q_B={q_b:.4f}"
                ),
            )

        if step == steps:
            break

        for idx, site in enumerate(sites):
            inv_mass = 1.0 / site.mass_amu
            ax = forces[idx][0] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            ay = forces[idx][1] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            az = forces[idx][2] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            site.velocity_ang_fs[0] += 0.5 * dt_fs * ax
            site.velocity_ang_fs[1] += 0.5 * dt_fs * ay
            site.velocity_ang_fs[2] += 0.5 * dt_fs * az

        for site in sites:
            site.position_angstrom[0] += dt_fs * site.velocity_ang_fs[0]
            site.position_angstrom[1] += dt_fs * site.velocity_ang_fs[1]
            site.position_angstrom[2] += dt_fs * site.velocity_ang_fs[2]

        enforce_solvent_bond_constraints(sites, n_solvent_molecules, solvent_bond_distance)

        new_forces, potential, q_a, q_h, q_b, f_pol, r_ah = compute_forces_and_potential(sites, n_solvent_molecules)

        for idx, site in enumerate(sites):
            inv_mass = 1.0 / site.mass_amu
            ax = new_forces[idx][0] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            ay = new_forces[idx][1] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            az = new_forces[idx][2] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            site.velocity_ang_fs[0] += 0.5 * dt_fs * ax
            site.velocity_ang_fs[1] += 0.5 * dt_fs * ay
            site.velocity_ang_fs[2] += 0.5 * dt_fs * az

        enforce_solvent_velocity_constraints(sites, n_solvent_molecules)

        forces = new_forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run NVE MD for coarse-grained chloromethane + AHB complex."
    )
    parser.add_argument("--n-molecules", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=50.0, help="Kelvin")
    parser.add_argument("--radius", type=float, default=7.0, help="Angstrom")
    parser.add_argument("--bond-distance", type=float, default=1.7, help="Angstrom (solvent C1-C2)")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Angstrom")
    parser.add_argument("--seed", type=int, default=20260218)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--write-frequency", type=int, default=100)

    parser.add_argument("--initial-output", type=Path, default=Path("solvent_initial.xyz"))
    parser.add_argument("--trajectory", type=Path, default=Path("solvent_nve.xyz"))
    parser.add_argument("--energy-log", type=Path, default=Path("solvent_energy.log"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sites = generate_configuration(
        n_molecules=args.n_molecules,
        temperature_k=args.temperature,
        radius_angstrom=args.radius,
        c1_c2_distance_angstrom=args.bond_distance,
        min_inter_site_distance_angstrom=args.min_distance,
        seed=args.seed,
    )

    initial_ke = kinetic_energy_kcal_mol(sites)
    _, initial_pe, q_a_i, q_h_i, q_b_i, f_i, r_i = compute_forces_and_potential(sites, args.n_molecules)
    initial_temp = instantaneous_temperature(sites)

    write_initial_xyz(
        args.initial_output,
        sites,
        (
            "initial frame chloromethane+AHB "
            f"molecules={args.n_molecules} T_target={args.temperature:.2f}K "
            f"T_inst={initial_temp:.2f}K seed={args.seed} TE={initial_ke + initial_pe:.3f}kcal/mol "
            f"Q_A={q_a_i:.4f} Q_H={q_h_i:.4f} Q_B={q_b_i:.4f}"
        ),
    )

    run_nve_md(
        sites=sites,
        n_solvent_molecules=args.n_molecules,
        steps=args.steps,
        dt_fs=args.dt_fs,
        write_frequency=args.write_frequency,
        solvent_bond_distance=args.bond_distance,
        trajectory_path=args.trajectory,
        energy_log_path=args.energy_log,
    )

    final_ke = kinetic_energy_kcal_mol(sites)
    _, final_pe, q_a_f, q_h_f, q_b_f, f_f, r_f = compute_forces_and_potential(sites, args.n_molecules)
    final_temp = instantaneous_temperature(sites)

    print(f"Initial KE: {initial_ke:.6f} kcal/mol")
    print(f"Initial PE: {initial_pe:.6f} kcal/mol")
    print(f"Initial TE: {initial_ke + initial_pe:.6f} kcal/mol")
    print(f"Initial temperature: {initial_temp:.3f} K")
    print(f"Initial charges: Q_A={q_a_i:.6f}, Q_H={q_h_i:.6f}, Q_B={q_b_i:.6f}, f={f_i:.6f}, r_AH={r_i:.6f} Å")
    print(f"Final KE: {final_ke:.6f} kcal/mol")
    print(f"Final PE: {final_pe:.6f} kcal/mol")
    print(f"Final TE: {final_ke + final_pe:.6f} kcal/mol")
    print(f"Final temperature: {final_temp:.3f} K")
    print(f"Final charges: Q_A={q_a_f:.6f}, Q_H={q_h_f:.6f}, Q_B={q_b_f:.6f}, f={f_f:.6f}, r_AH={r_f:.6f} Å")
    print(f"Initial frame written to: {args.initial_output}")
    print(f"Trajectory written to: {args.trajectory}")
    print(f"Energy log written to: {args.energy_log}")


if __name__ == "__main__":
    main()
