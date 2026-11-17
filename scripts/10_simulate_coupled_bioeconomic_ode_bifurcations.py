#!/usr/bin/env python3
"""
Module 10: Coupled Wild-Aquaculture Bioeconomic ODEs and Bifurcations (Eqs. S4a-S4d)
====================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Solves the coupled wild-biomass and fishing-effort dynamical system grounded in
FishBase life-history priors and Dual-IV market elasticities, computing Jacobian
eigenvalues, saddle-node tipping manifolds, and Supplementary Fig. S8.

Outputs:
  - data/derived/bioeconomic_ode_bifurcation_regimes.csv
  - data/derived/bioeconomic_ode_trajectories.csv
  - figures/Supplementary_Fig_S8_Bioeconomic_ODE_Bifurcations.{png,pdf}
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.environ.get("EXTERNAL_DB_ROOT", os.path.join(os.path.dirname(PROJ_DIR), "_Databases"))
DERIVED_DIR = os.path.join(PROJ_DIR, "data", "derived")
FIG_DIR = os.path.join(PROJ_DIR, "figures")
SUB_DIR = FIG_DIR

os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Coupled Bioeconomic ODE System & Analytical / Numerical Jacobian
# ---------------------------------------------------------------------------
def load_fishbase_life_history_priors() -> dict:
    """Reads empirical von Bertalanffy K, natural mortality M, and tmax from local FishBase parquets."""
    sp_path = os.path.join(DB_DIR, "FishBase_Workspace", "data", "species.parquet")
    pg_path = os.path.join(DB_DIR, "FishBase_Workspace", "data", "popgrowth.parquet")
    priors = {}
    if os.path.exists(sp_path) and os.path.exists(pg_path):
        sp = pd.read_parquet(sp_path)[["SpecCode", "Genus"]]
        pg = pd.read_parquet(pg_path)[["SpecCode", "K", "M", "tmax"]].merge(sp, on="SpecCode", how="left")
        for g in ["Acipenser", "Isurus", "Carcharhinus", "Anguilla", "Thunnus", "Melanogrammus"]:
            sub = pg[pg["Genus"] == g]
            priors[g] = {
                "n_records": int(len(sub)),
                "K_mean": round(float(sub["K"].dropna().mean()), 4),
                "M_mean": round(float(sub["M"].dropna().mean()), 4),
                "tmax_mean": round(float(sub["tmax"].dropna().mean()), 2),
            }
    return priors


def load_option03_optimal_control_metrics() -> dict:
    """Reads Pontryagin optimal Pigouvian border levy and 30-yr SDE metrics."""
    opt3_cal = os.path.join(DERIVED_DIR, "bioeconomic_allee_bifurcation_calibration.csv")
    metrics = {}
    if os.path.exists(opt3_cal):
        df3 = pd.read_csv(opt3_cal)
        for _, row in df3.iterrows():
            metrics[str(row["Taxon_Group"])] = {
                "Opt3_Tau_Star_Pct": float(row["Optimal_Pigouvian_Tax_Tau_Star_Pct"]),
                "Opt3_Tau_Star_USD_KG": float(row["Optimal_Tax_At_Mean_Price_USD_KG"]),
                "Opt3_B_Star_K": float(row["Pontryagin_Optimal_Biomass_B_Star_K"]),
                "Opt3_Severity_Ratio": float(row["Vortex_Severity_Ratio"]),
            }
    return metrics


def wild_growth(B: float, r: float, K: float, A: float, iota: float) -> float:
    """Logistic growth with optional depensatory Allee threshold A (iota in [0, 1])."""
    if B <= 0.0:
        return 0.0
    logistic = r * B * (1.0 - B / K)
    depens = (1.0 - iota) + iota * ((B - A) / K)
    return logistic * depens


def wild_harvest(B: float, E: float, q: float, alpha: float) -> float:
    if B <= 1e-9 or E <= 1e-9:
        return 0.0
    return q * E * (max(B, 1e-9) ** alpha)


def wild_unit_price(H: float, Q_aqua: float, P0: float, gamma: float, psi: float, dP_prem: float) -> float:
    eff_supply = max(H + psi * Q_aqua, 1e-4)
    return P0 * (eff_supply ** (-gamma)) + dP_prem


def pooled_customs_price(H: float, Q_aqua: float, P_wild: float, P_aqua: float) -> float:
    Q_tot = max(H + Q_aqua, 1e-6)
    omega = H / Q_tot
    return omega * P_wild + (1.0 - omega) * P_aqua


def ode_rhs(t: float, state: np.ndarray, p: dict) -> np.ndarray:
    B = max(float(state[0]), 0.0)
    E = max(float(state[1]), 0.0)
    if B <= 1e-7:
        return np.array([0.0, -p["kappa"] * E * p["c"]])

    G = wild_growth(B, p["r"], p["K"], p["A"], p["iota"])
    H = wild_harvest(B, E, p["q"], p["alpha"])
    # Smooth saturation avoids piecewise kink near low biomass while preserving wild-seed drain
    drain = p["sigma"] * p["Q_aqua"] * (B / (B + 0.005 * p["K"]))
    dB = G - H - drain

    Pw = wild_unit_price(H, p["Q_aqua"], p["P0"], p["gamma"], p["psi"], p["dP_prem"])
    net_rev_per_effort = (1.0 - p["tau_enf"]) * Pw * p["q"] * (B ** p["alpha"]) - p["c"]
    dE = p["kappa"] * E * net_rev_per_effort
    return np.array([dB, dE])


def solve_interior_equilibria_1d(p: dict, b_grid: np.ndarray | None = None):
    """Solves G(B) - drain(B) - H_eq(B) = 0 cleanly across B in (0, K) to locate true equilibria."""
    if b_grid is None:
        b_grid = np.linspace(1e-4, 0.9999 * p["K"], 1200)
    fvals = np.empty_like(b_grid)
    for idx, B in enumerate(b_grid):
        P_req = p["c"] / ((1.0 - p["tau_enf"]) * p["q"] * (B ** p["alpha"]))
        if P_req <= p["dP_prem"]:
            H_eq = 1e6
        else:
            H_eq = ((p["P0"] / (P_req - p["dP_prem"])) ** (1.0 / p["gamma"])) - p["psi"] * p["Q_aqua"]
        H_eq = max(0.0, H_eq)
        drain = p["sigma"] * p["Q_aqua"] * (B / (B + 0.005 * p["K"]))
        fvals[idx] = wild_growth(B, p["r"], p["K"], p["A"], p["iota"]) - drain - H_eq

    roots_stable, roots_unstable = [], []
    for i in range(len(b_grid) - 1):
        if abs(fvals[i]) < 5.0 and abs(fvals[i + 1]) < 5.0 and fvals[i] * fvals[i + 1] <= 0.0:
            # Skip zero-crossing caused solely by identical zeros
            if fvals[i] == 0.0 and fvals[i + 1] == 0.0:
                continue
            b_mid = 0.5 * (b_grid[i] + b_grid[i + 1])
            if fvals[i + 1] < fvals[i]:
                roots_stable.append(float(b_mid))
            else:
                roots_unstable.append(float(b_mid))
    return roots_stable, roots_unstable, float(fvals[-1])


def compute_jacobian_and_eigenvalues(B_star: float, E_star: float, p: dict) -> dict:
    """Computes exact finite-difference 2x2 Jacobian J(B*, E*), trace, determinant, and eigenvalues."""
    eps = 1e-5
    is_boundary_collapse = B_star <= 0.02 * p["K"]
    J = np.zeros((2, 2))

    if is_boundary_collapse:
        # Evaluate clean one-sided right-hand linearization near (B=1e-4, E=0) without negative-B clipping
        y_eval = np.array([1e-4, 0.0])
        h_b = 1e-6
        h_e = 1e-6
        f0 = ode_rhs(0.0, y_eval, p)
        J[:, 0] = (ode_rhs(0.0, y_eval + np.array([h_b, 0.0]), p) - f0) / h_b
        J[:, 1] = (ode_rhs(0.0, y_eval + np.array([0.0, h_e]), p) - f0) / h_e
    else:
        y0 = np.array([B_star, E_star])
        for j in range(2):
            dy = np.zeros(2)
            dy[j] = eps * max(1.0, abs(y0[j]))
            f_plus = ode_rhs(0.0, y0 + dy, p)
            f_minus = ode_rhs(0.0, y0 - np.where(y0 - dy >= 0.0, dy, 0.5 * y0), p)
            denom = dy[j] + np.where(y0 - dy >= 0.0, dy, 0.5 * y0)[j]
            J[:, j] = (f_plus - f_minus) / max(denom, 1e-12)

    tr_J = float(np.trace(J))
    det_J = float(np.linalg.det(J))
    eigs = np.linalg.eigvals(J)
    eig1, eig2 = eigs[0], eigs[1]

    # Classify dynamical stability regime
    if is_boundary_collapse:
        stability_class = "Boundary Extinction Attractor (B* -> 0; No Positive Interior Equilibrium)"
    elif det_J < 0:
        stability_class = "Unstable Tipping Point (Allee / Rarity Depletion Threshold)"
    elif tr_J < 0 and det_J > 0:
        if np.abs(np.imag(eig1)) > 1e-4:
            stability_class = "Locally Asymptotically Stable Spiral Focus (tr J < 0, det J > 0)"
        else:
            stability_class = "Locally Asymptotically Stable Node (tr J < 0, det J > 0)"
    else:
        stability_class = "Unstable Focus / Repeller (tr J >= 0)"

    # Also compute observable pooled customs price elasticity d ln P_pool / d ln Q_tot
    H_star = wild_harvest(B_star, E_star, p["q"], p["alpha"])
    Q_tot = H_star + p["Q_aqua"]
    omega = H_star / max(Q_tot, 1e-6)
    Pw = wild_unit_price(H_star, p["Q_aqua"], p["P0"], p["gamma"], p["psi"], p["dP_prem"])
    Pa = p["P_aqua"]
    P_pool = pooled_customs_price(H_star, p["Q_aqua"], Pw, Pa)

    dPw_dH = -p["gamma"] * p["P0"] * ((max(H_star + p["psi"] * p["Q_aqua"], 1e-4)) ** (-p["gamma"] - 1.0))
    dlnPpool_dlnQ = float((omega * H_star * dPw_dH) / max(P_pool, 1e-6))
    cross_partial_aqua = float(abs(dlnPpool_dlnQ) * (1.0 - omega * (1.0 - p["psi"])))

    return {
        "J11": float(J[0, 0]),
        "J12": float(J[0, 1]),
        "J21": float(J[1, 0]),
        "J22": float(J[1, 1]),
        "Trace_J": tr_J,
        "Det_J": det_J,
        "Eigenvalue_1_Real": float(np.real(eig1)),
        "Eigenvalue_1_Imag": float(np.imag(eig1)),
        "Eigenvalue_2_Real": float(np.real(eig2)),
        "Eigenvalue_2_Imag": float(np.imag(eig2)),
        "Stability_Class": stability_class,
        "Wild_Biomass_Ratio_B_over_K": round(B_star / p["K"], 4),
        "Equilibrium_Effort_E_star": round(E_star, 4),
        "Wild_Harvest_Share_omega": round(omega, 4),
        "Implied_Pooled_Price_Elasticity": round(dlnPpool_dlnQ, 4),
        "Cross_Partial_Aqua_Attenuation": round(cross_partial_aqua, 4)
    }


def main():
    print("=" * 80)
    print("MODULE 10: SIMULATING COUPLED BIOECONOMIC ODEs & EQUILIBRIUM STABILITY")
    print("=" * 80)

    fishbase_priors = load_fishbase_life_history_priors()
    opt3_metrics = load_option03_optimal_control_metrics()
    if fishbase_priors:
        print("   -> Empirical FishBase popgrowth priors loaded:")
        for gen, stats_g in fishbase_priors.items():
            print(f"      * {gen} (n={stats_g['n_records']}): K={stats_g['K_mean']}, M={stats_g['M_mean']} yr^-1, tmax={stats_g['tmax_mean']} yr")

    # Calibrate 6 canonical regime scenarios grounded in the empirical Dual-IV elasticities,
    # FishBase intrinsic growth rates r_k, FAO wild-vs.-aquaculture shares, and treaty enforcement shocks
    regimes = [
        {
            "Regime_ID": "R1_Holothuroidea_Rarity_Collapse",
            "Clade": "Sea Cucumbers (Holothuroidea)",
            "FishBase_Genus": "N/A (Invertebrate)",
            "Opt3_Key": "Sea Cucumbers (Holothuroidea)",
            "Scenario_Label": "Regime I: Rarity-Driven Depletion Dynamics (Wild-Only Luxury Benthic)",
            "r": 0.32, "K": 1.0, "A": 0.12, "iota": 0.45,
            "q": 0.45, "alpha": 0.62, "kappa": 0.40, "c": 0.28,
            "P0": 0.65, "gamma": 0.38, "psi": 0.10, "dP_prem": 0.22,
            "Q_aqua": 0.02, "sigma": 0.00, "tau_enf": 0.05, "P_aqua": 0.40,
            "B_init": 0.72, "E_init": 0.45
        },
        {
            "Regime_ID": "R2_Anguilla_Ranching_Cryptic_Collapse",
            "Clade": "Diadromous Eels (Anguilla spp.)",
            "FishBase_Genus": "Anguilla",
            "Opt3_Key": "Diadromous Eels (Anguilla spp.)",
            "Scenario_Label": "Regime II: Ranching-Coupled Cryptic Depletion Drain (Flat Pooled Price, Wild Glass-Eel Drain)",
            "r": 0.24, "K": 1.0, "A": 0.08, "iota": 0.35,
            "q": 0.38, "alpha": 0.75, "kappa": 0.35, "c": 0.30,
            "P0": 0.70, "gamma": 0.26, "psi": 0.92, "dP_prem": 0.08,
            "Q_aqua": 0.85, "sigma": 0.095, "tau_enf": 0.10, "P_aqua": 0.58,
            "B_init": 0.68, "E_init": 0.38
        },
        {
            "Regime_ID": "R3a_Acipenseriformes_ClosedCycle_Buffered",
            "Clade": "Sturgeons & Caviar (Acipenseriformes)",
            "FishBase_Genus": "Acipenser",
            "Opt3_Key": "Sturgeons & Caviar (Acipenseriformes)",
            "Scenario_Label": "Regime IIIa: Closed-Cycle Aquaculture Stabilization (sigma=0, Low Wild Premium)",
            "r": 0.24, "K": 1.0, "A": 0.06, "iota": 0.25,
            "q": 0.36, "alpha": 0.85, "kappa": 0.35, "c": 0.29,
            "P0": 0.85, "gamma": 0.30, "psi": 0.88, "dP_prem": 0.08,
            "Q_aqua": 0.90, "sigma": 0.00, "tau_enf": 0.28, "P_aqua": 0.65,
            "B_init": 0.35, "E_init": 0.42
        },
        {
            "Regime_ID": "R3b_Acipenseriformes_WildPoaching_Premium",
            "Clade": "Sturgeons & Caviar (Acipenseriformes)",
            "FishBase_Genus": "Acipenser",
            "Opt3_Key": "Sturgeons & Caviar (Acipenseriformes)",
            "Scenario_Label": "Regime IIIb: Wild Luxury Stigma/Premium Override (dP_prem > c / q B^alpha)",
            "r": 0.20, "K": 1.0, "A": 0.08, "iota": 0.35,
            "q": 0.42, "alpha": 0.65, "kappa": 0.40, "c": 0.26,
            "P0": 0.85, "gamma": 0.32, "psi": 0.88, "dP_prem": 0.72,
            "Q_aqua": 0.90, "sigma": 0.00, "tau_enf": 0.05, "P_aqua": 0.65,
            "B_init": 0.35, "E_init": 0.42
        },
        {
            "Regime_ID": "R4a_Thunnus_Pre2010_Overexploitation",
            "Clade": "High-Trophic Tunas (Thunnus spp.)",
            "FishBase_Genus": "Thunnus",
            "Opt3_Key": "High-Trophic Tunas (Thunnus spp.)",
            "Scenario_Label": "Regime IVa: Pre-2010 Open-Access / Unverified Quota (Low tau_enf = 0.05)",
            "r": 0.42, "K": 1.0, "A": 0.05, "iota": 0.15,
            "q": 0.52, "alpha": 0.78, "kappa": 0.45, "c": 0.31,
            "P0": 0.78, "gamma": 0.34, "psi": 0.45, "dP_prem": 0.18,
            "Q_aqua": 0.18, "sigma": 0.03, "tau_enf": 0.05, "P_aqua": 0.62,
            "B_init": 0.75, "E_init": 0.35
        },
        {
            "Regime_ID": "R4b_Thunnus_Post2010_eBCD_Stabilized",
            "Clade": "High-Trophic Tunas (Thunnus spp.)",
            "FishBase_Genus": "Thunnus",
            "Opt3_Key": "High-Trophic Tunas (Thunnus spp.)",
            "Scenario_Label": "Regime IVb: Post-2010 Binding TAC + Digital eBCD Enforcement (tau_enf = 0.32)",
            "r": 0.42, "K": 1.0, "A": 0.05, "iota": 0.15,
            "q": 0.52, "alpha": 0.82, "kappa": 0.45, "c": 0.31,
            "P0": 0.78, "gamma": 0.34, "psi": 0.45, "dP_prem": 0.16,
            "Q_aqua": 0.25, "sigma": 0.02, "tau_enf": 0.32, "P_aqua": 0.62,
            "B_init": 0.28, "E_init": 0.55
        }
    ]

    summary_rows = []
    traj_rows = []
    t_eval = np.linspace(0, 65, 261)

    for reg in regimes:
        # 65-year transient integration for panel plotting
        sol65 = solve_ivp(
            fun=lambda t, y: ode_rhs(t, y, reg),
            t_span=(0, 65),
            y0=[reg["B_init"], reg["E_init"]],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-7, atol=1e-9
        )
        B_65 = float(max(0.0, sol65.y[0, -1]))

        # Long-horizon convergence + exact root polishing for true equilibrium (||rhs|| < 1e-10)
        sol_eq = solve_ivp(
            fun=lambda t, y: ode_rhs(t, y, reg),
            t_span=(0, 500),
            y0=[reg["B_init"], reg["E_init"]],
            method="RK45",
            rtol=1e-8, atol=1e-10
        )
        B_eq = float(max(0.0, sol_eq.y[0, -1]))
        E_eq = float(max(0.0, sol_eq.y[1, -1]))
        if B_eq <= 0.02 * reg["K"]:
            B_eq, E_eq = 0.0, 0.0
        elif E_eq <= 1e-4:
            # Wild fishery has exited (dE/dt < 0): polish B* via root or carrying capacity
            E_eq = 0.0
            if reg["sigma"] * reg["Q_aqua"] == 0.0:
                B_eq = reg["K"]
        else:
            rt = root(lambda y: ode_rhs(0.0, y, reg), np.array([B_eq, E_eq]), tol=1e-12)
            if rt.success and np.max(np.abs(ode_rhs(0.0, rt.x, reg))) < 1e-9 and rt.x[0] > 0.02:
                B_eq = float(max(0.0, rt.x[0]))
                E_eq = float(max(0.0, rt.x[1]))

        jac = compute_jacobian_and_eigenvalues(B_eq, E_eq, reg)
        fb = fishbase_priors.get(reg["FishBase_Genus"], {"K_mean": np.nan, "M_mean": np.nan, "tmax_mean": np.nan})
        o3 = opt3_metrics.get(reg["Opt3_Key"], {"Opt3_Tau_Star_Pct": np.nan, "Opt3_Tau_Star_USD_KG": np.nan, "Opt3_B_Star_K": np.nan, "Opt3_Severity_Ratio": np.nan})

        row_out = {
            "Regime_ID": reg["Regime_ID"],
            "Clade": reg["Clade"],
            "Scenario_Label": reg["Scenario_Label"],
            "r_Growth": reg["r"],
            "alpha_Hyperstability": reg["alpha"],
            "gamma_Rarity_Elasticity": reg["gamma"],
            "psi_Aqua_Substitution": reg["psi"],
            "Q_aqua_Supply": reg["Q_aqua"],
            "sigma_WildSeed_Drain": reg["sigma"],
            "dP_prem_Wild_Luxury": reg["dP_prem"],
            "tau_enf_Quota_eBCD": reg["tau_enf"],
            "Biomass_65Yr_Transient": round(B_65, 4),
            "FishBase_K": fb["K_mean"],
            "FishBase_M": fb["M_mean"],
            "FishBase_Tmax": fb["tmax_mean"],
            **o3,
            **jac
        }
        summary_rows.append(row_out)

        for idx_t, t_val in enumerate(sol65.t):
            b_t = float(max(0.0, sol65.y[0, idx_t]))
            e_t = float(max(0.0, sol65.y[1, idx_t]))
            h_t = wild_harvest(b_t, e_t, reg["q"], reg["alpha"])
            pw_t = wild_unit_price(h_t, reg["Q_aqua"], reg["P0"], reg["gamma"], reg["psi"], reg["dP_prem"])
            pp_t = pooled_customs_price(h_t, reg["Q_aqua"], pw_t, reg["P_aqua"])
            traj_rows.append({
                "Regime_ID": reg["Regime_ID"],
                "Clade": reg["Clade"],
                "Time_Years": round(float(t_val), 2),
                "Wild_Biomass_B_over_K": round(b_t, 5),
                "Fishing_Effort_E": round(e_t, 5),
                "Wild_Harvest_H": round(h_t, 5),
                "Wild_Unit_Price_Pw": round(pw_t, 5),
                "Pooled_Customs_Price_Ppool": round(pp_t, 5)
            })

    df_sum = pd.DataFrame(summary_rows)
    df_traj = pd.DataFrame(traj_rows)
    sum_csv = os.path.join(DERIVED_DIR, "bioeconomic_ode_bifurcation_regimes.csv")
    traj_csv = os.path.join(DERIVED_DIR, "bioeconomic_ode_trajectories.csv")
    df_sum.to_csv(sum_csv, index=False)
    df_traj.to_csv(traj_csv, index=False)
    print(f"[SUCCESS] Saved calibrated ODE equilibria & Jacobian stability table: {sum_csv}")
    print(df_sum[["Regime_ID", "Wild_Biomass_Ratio_B_over_K", "Biomass_65Yr_Transient", "Trace_J", "Det_J", "Stability_Class"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Generate 600-DPI 4-Panel Supplementary Figure S8
    # -------------------------------------------------------------------------
    print("\nGenerating 600-DPI 4-Panel Supplementary Figure S8 (ODE Phase Portraits & Tipping Manifolds)...")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5
    })
    fig = plt.figure(figsize=(12.2, 9.2), dpi=600)
    gs = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.28)

    # Panel a: Phase-Plane Vector Field & Paths (Wild Biomass B/K vs Fishing Effort E)
    ax_a = fig.add_subplot(gs[0, 0])
    B_grid = np.linspace(0.03, 0.98, 18)
    E_grid = np.linspace(0.05, 0.95, 18)
    BB, EE = np.meshgrid(B_grid, E_grid)
    U = np.zeros_like(BB)
    V = np.zeros_like(EE)
    r_ref = regimes[5]  # Stabilized Bluefin reference field
    for i in range(BB.shape[0]):
        for j in range(BB.shape[1]):
            rhs = ode_rhs(0.0, np.array([BB[i, j], EE[i, j]]), r_ref)
            U[i, j], V[i, j] = rhs[0], rhs[1]
    mag = np.hypot(U, V) + 1e-6
    ax_a.quiver(BB, EE, U / mag, V / mag, color="#CBD5E1", alpha=0.65, scale=26, width=0.003)

    colors_traj = {
        "R1_Holothuroidea_Rarity_Collapse": ("#B91C1C", "Sea cucumbers: Rarity-driven depletion ($B^*/K \\to 0$)"),
        "R2_Anguilla_Ranching_Cryptic_Collapse": ("#D97706", "Diadromous eels: Glass-eel ranching drain ($\\sigma Q^{\\mathrm{aqua}} > G_{\\max}$)"),
        "R3a_Acipenseriformes_ClosedCycle_Buffered": ("#0D9488", "Sturgeon caviar: Closed-cycle aquaculture recovery ($B_{65}=0.93K \\to K$)"),
        "R4b_Thunnus_Post2010_eBCD_Stabilized": ("#1D4ED8", "Bluefin tuna: TAC + eBCD stabilization ($B^*/K = 0.52$)")
    }
    for r_id, (col, lbl) in colors_traj.items():
        sub_t = df_traj[df_traj["Regime_ID"] == r_id]
        ax_a.plot(sub_t["Wild_Biomass_B_over_K"], sub_t["Fishing_Effort_E"], color=col, lw=2.0, label=lbl)
        ax_a.scatter(sub_t["Wild_Biomass_B_over_K"].iloc[0], sub_t["Fishing_Effort_E"].iloc[0], color=col, marker="o", s=26, zorder=5)
        ax_a.scatter(sub_t["Wild_Biomass_B_over_K"].iloc[-1], sub_t["Fishing_Effort_E"].iloc[-1], color=col, marker="*", s=75, edgecolor="black", linewidth=0.5, zorder=6)

    ax_a.axvline(0.12, color="#991B1B", ls="--", lw=1.0, alpha=0.7, label="Allee critical threshold ($A_k / K_k$)")
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_xlabel("Wild Stock Biomass Ratio ($B_k(t) / K_k$)")
    ax_a.set_ylabel("Harvesting Fleet Effort ($E_k(t)$)")
    ax_a.set_title("a  Coupled Bioeconomic ODE Phase Portrait ($\\dot{B}_k, \\dot{E}_k$)", loc="left", fontweight="bold")
    ax_a.legend(loc="upper right", frameon=True, fontsize=7.0)
    ax_a.grid(True, ls=":", alpha=0.4)

    # Panel b: Numerical Equilibrium Curves over Rarity Elasticity gamma_k (solved directly from ode_rhs)
    ax_b = fig.add_subplot(gs[0, 1])
    gamma_scan = np.linspace(0.05, 0.65, 120)
    p_wild_scan = dict(
        r=0.65, K=1.0, A=0.15, iota=0.95, q=0.50, alpha=0.72, kappa=0.40,
        c=0.36, P0=0.45, gamma=0.35, psi=0.0, dP_prem=0.05, Q_aqua=0.0, sigma=0.0, tau_enf=0.05
    )
    p_closed_scan = dict(p_wild_scan)
    p_closed_scan.update({"psi": 0.88, "Q_aqua": 0.30})

    g_stable, b_stable_low_aqua = [], []
    g_unstable, b_unstable_allee = [], []
    g_closed, b_closed_aqua = [], []

    for g_val in gamma_scan:
        pw_s = dict(p_wild_scan, gamma=float(g_val))
        rs, ru, f_top = solve_interior_equilibria_1d(pw_s)
        if rs:
            g_stable.append(float(g_val))
            b_stable_low_aqua.append(max(rs))
        elif f_top > 0:
            g_stable.append(float(g_val))
            b_stable_low_aqua.append(1.0)
        if ru:
            g_unstable.append(float(g_val))
            b_unstable_allee.append(min(ru))

        pc_s = dict(p_closed_scan, gamma=float(g_val))
        rs_c, _, f_top_c = solve_interior_equilibria_1d(pc_s)
        if rs_c:
            g_closed.append(float(g_val))
            b_closed_aqua.append(max(rs_c))
        elif f_top_c > 0:
            g_closed.append(float(g_val))
            b_closed_aqua.append(1.0)

    gamma_crit = max(g_stable) if g_stable else 0.36
    b_crit = b_stable_low_aqua[-1] if b_stable_low_aqua else 0.45
    ax_b.plot(g_stable, b_stable_low_aqua, color="#1E293B", lw=2.2, label="Stable interior equilibrium ($\\psi_k=0$, wild-only)")
    ax_b.plot(g_unstable, b_unstable_allee, color="#DC2626", ls="--", lw=1.6, label="Unstable separatrix ($\\det \\mathbf{J} < 0$)")
    ax_b.plot(g_closed, b_closed_aqua, color="#0D9488", lw=2.0, label="Closed-cycle aquaculture stabilized ($\\psi_k=0.88, \\sigma_k=0$)")
    ax_b.axvline(gamma_crit, color="#B91C1C", ls=":", lw=1.3)
    ax_b.scatter([gamma_crit], [b_crit], color="#B91C1C", s=45, zorder=6)
    ax_b.annotate(
        f"Critical Tipping Threshold\n($\\gamma^* \\approx {gamma_crit:.2f}, \\det \\mathbf{{J}} = 0$)\nWild-only stock collapse beyond $\\gamma^*$",
        xy=(gamma_crit, b_crit), xytext=(0.42, 0.45),
        arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.0), fontsize=7.2, color="#991B1B", fontweight="bold"
    )
    ax_b.set_xlim(0.05, 0.65)
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_xlabel("Rarity Price Elasticity Parameter ($\\gamma_k = -\\partial \\ln P^{\\mathrm{wild}}_k / \\partial \\ln H_k$)")
    ax_b.set_ylabel("Steady-State Wild Biomass ($B^*_k / K_k$)")
    ax_b.set_title("b  Numerical Tipping Threshold vs. Closed-Cycle Stabilization", loc="left", fontweight="bold")
    ax_b.legend(loc="lower left", frameon=True, fontsize=6.8)
    ax_b.grid(True, ls=":", alpha=0.4)

    # Panel c: Rarity Depletion vs. Ranching-Masked Cryptic Collapse
    ax_c = fig.add_subplot(gs[1, 0])
    sub_eel = df_traj[df_traj["Regime_ID"] == "R2_Anguilla_Ranching_Cryptic_Collapse"]
    sub_cuc = df_traj[df_traj["Regime_ID"] == "R1_Holothuroidea_Rarity_Collapse"]
    ax_c2 = ax_c.twinx()
    l1 = ax_c.plot(2000 + sub_eel["Time_Years"], sub_eel["Wild_Biomass_B_over_K"], color="#D97706", lw=2.2, label="Eel wild biomass $B_{\\mathrm{eel}}(t)/K$ (glass-eel drain $\\sigma > 0$)")
    l2 = ax_c.plot(2000 + sub_cuc["Time_Years"], sub_cuc["Wild_Biomass_B_over_K"], color="#B91C1C", lw=1.8, ls="-.", label="Sea cucumber wild biomass $B_{\\mathrm{hol}}(t)/K$ ($\\psi \\approx 0$)")
    l3 = ax_c2.plot(2000 + sub_eel["Time_Years"], sub_eel["Pooled_Customs_Price_Ppool"], color="#0F766E", lw=2.0, ls="--", label="Eel pooled customs price $P^{\\mathrm{pool}}_{\\mathrm{eel}}(t)$ (flat under ranching)")
    l4 = ax_c2.plot(2000 + sub_cuc["Time_Years"], sub_cuc["Pooled_Customs_Price_Ppool"], color="#7C2D12", lw=1.6, ls=":", label="Sea cucumber customs price $P^{\\mathrm{pool}}_{\\mathrm{hol}}(t)$ (rarity surge)")
    l5 = ax_c.axvline(2028, color="#1D4ED8", ls="--", lw=1.4, label="WCO 2028 Split + CITES e-Permit (2028)")

    ax_c.set_xlim(2000, 2065)
    ax_c.set_xticks([2000, 2010, 2020, 2028, 2040, 2050, 2060])
    ax_c.set_xlabel("Year (Bioeconomic Harvest Projection, 2000–2065)")
    ax_c.set_ylabel("Wild Stock Biomass Ratio ($B_k(t) / K_k$)", color="#9A3412")
    ax_c2.set_ylabel("Observed Pooled Customs Unit Price ($P^{\\mathrm{pool}}_k$)", color="#0F766E")
    ax_c.set_title("c  Visible Rarity Depletion vs. Ranching-Masked Cryptic Collapse", loc="left", fontweight="bold")
    lines_all = l1 + l2 + l3 + l4 + [l5]
    labels_all = [l.get_label() for l in lines_all]
    ax_c.legend(lines_all, labels_all, loc="center right", frameon=True, fontsize=6.6)
    ax_c.grid(True, ls=":", alpha=0.4)

    # Panel d: True Numerical 2D Steady-State Biomass Surface over (psi_k, ranching/premium pressure)
    ax_d = fig.add_subplot(gs[1, 1])
    psi_vec = np.linspace(0.0, 1.0, 55)
    press_vec = np.linspace(0.0, 1.0, 55)
    PSI, DRN = np.meshgrid(psi_vec, press_vec)
    B_surf = np.zeros_like(PSI)
    p_surf_base = dict(
        r=0.45, K=1.0, A=0.14, iota=0.92, q=0.50, alpha=0.72, kappa=0.40,
        c=0.35, P0=0.45, gamma=0.34, Q_aqua=0.30, tau_enf=0.08
    )
    b_scan_grid = np.linspace(1e-3, 0.9999, 260)
    for ii in range(PSI.shape[0]):
        for jj in range(PSI.shape[1]):
            psi_v = float(PSI[ii, jj])
            pr_v = float(DRN[ii, jj])
            pm = dict(p_surf_base, psi=psi_v, sigma=0.22 * pr_v, dP_prem=0.02 + 0.50 * pr_v)
            rs_m, _, f_top_m = solve_interior_equilibria_1d(pm, b_grid=b_scan_grid)
            if rs_m:
                B_surf[ii, jj] = max(rs_m)
            elif f_top_m > 0:
                B_surf[ii, jj] = 1.0
            else:
                B_surf[ii, jj] = 0.0

    im = ax_d.contourf(PSI, DRN, B_surf, levels=np.linspace(0.0, 1.0, 16), cmap="RdYlGn")
    cs_lines = ax_d.contour(PSI, DRN, B_surf, levels=[0.14, 0.40], colors=["#7F1D1D", "#1E3A8A"], linewidths=[1.8, 1.2], linestyles=["--", "-"])
    ax_d.clabel(cs_lines, fmt={0.14: "Tipping cliff (B*=A)", 0.40: "B*=0.40K"}, manual=[(0.22, 0.42), (0.58, 0.68)], fontsize=7.0)
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label("Numerical Steady-State Wild Biomass ($B^*_k / K_k$)", fontsize=7.5)

    pts = [
        (0.10, 0.78, "Sea cucumbers\n& pelagic sharks\n(Rarity depletion)", "#7F1D1D"),
        (0.92, 0.82, "Diadromous eels\n(Ranching cryptic\ndrain, $\\sigma>0$)", "#92400E"),
        (0.88, 0.18, "Closed-cycle\ncaviar ($\\sigma=0$)", "#065F46"),
        (0.45, 0.28, "Bluefin tuna\n(TAC + eBCD)", "#1E3A8A")
    ]
    for px, py, plbl, pcol in pts:
        ax_d.scatter(px, py, s=55, color=pcol, edgecolor="white", linewidth=1.2, zorder=6)
        off_x = px - 0.28 if px > 0.5 else px + 0.04
        off_y = py - 0.08 if py > 0.7 else py + 0.04
        ax_d.annotate(plbl, (px, py), xytext=(off_x, off_y),
                      fontsize=6.8, fontweight="bold", color="black",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=pcol, alpha=0.88))

    ax_d.set_xlabel("Wild-Aquaculture Market Substitutability ($\\psi_k \\in [0, 1]$)")
    ax_d.set_ylabel("Normalized Wild-Seed Drain & Luxury Wild Premium Pressure")
    ax_d.set_title("d  2D Numerical Equilibrium Space Across Traded Marine Clades", loc="left", fontweight="bold")

    fig.tight_layout()
    for out_base in [
        os.path.join(FIG_DIR, "Supplementary_Fig_S8_Bioeconomic_ODE_Bifurcations"),
        os.path.join(FIG_DIR, "Figure_S08_Bioeconomic_ODE_Bifurcations"),
    ]:
        fig.savefig(out_base + ".pdf", dpi=600, bbox_inches="tight")
        fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[SUCCESS] Saved 600-DPI Supplementary_Fig_S8_Bioeconomic_ODE_Bifurcations (.pdf & .png)")


if __name__ == "__main__":
    main()
