#!/usr/bin/env python3
"""
Module 09: Comparative Interrupted Time Series and Structural Gravity PPML (Eqs. 6-6c)
======================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Evaluates CITES Appendices I-II listings, ICCAT catch-documentation reforms, and
European Union trade bans using Newey-West HAC segmented ITS (Eq. 6; Eq. S2),
log-link Quasi-Poisson GLM sensitivity (Eq. S3; Supplementary Table S5),
differenced Comparative ITS with Prais-Winsten AR(1) correction (Eq. 6b),
zero-retaining route-level Structural Gravity PPML (Eq. 6c; Table 1; Supplementary
Table S15), and eel cross-chapter displacement (Supplementary Table S16).
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyfixest as pf

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"

ANNUAL_CONSERVATION_CSV = DATA_DERIVED / "annual_global_conservation_trade_summary.csv"
ANNUAL_LONGITUDINAL_CSV = DATA_DERIVED / "annual_taxa_allee_time_series.csv"
THREATENED_PARQUET = DATA_DERIVED / "global_threatened_taxa_bilateral_panel.parquet"
BACI_PARQUET = DATA_RAW / "baci_hs92_seafood_1995_2024.parquet"

EU_EEA_GBR_ISO = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR", "NOR", "ISL"
}
EAST_ASIA_ISO = {"CHN", "JPN", "TWN", "KOR", "HKG"}


def estimate_hac_its(df_sub, intervention_year, label, demand_map):
    """
    Estimates a full-rank segmented regression with Newey-West HAC errors:
    ln(Y_t + c) = beta_0 + beta_1 * T_t + beta_2 * D_t + beta_3 * (T_t - T_event) * D_t + gamma * Demand_Index_t + eps_t
    """
    df_reg = df_sub.sort_values("Year").copy()
    c = 0.01

    min_year = int(df_reg["Year"].min())
    df_reg["T"] = df_reg["Year"] - min_year + 1
    t_event = intervention_year - min_year + 1

    df_reg["D"] = (df_reg["Year"] >= intervention_year).astype(int)
    df_reg["T_post"] = np.where(df_reg["D"] == 1, df_reg["T"] - t_event, 0)

    raw_demand = np.array([demand_map.get(int(y), 1000.0) for y in df_reg["Year"]])
    df_reg["Demand_Index"] = np.log(raw_demand) - np.mean(np.log(raw_demand))

    df_reg["ln_Volume"] = np.log(df_reg["Total_Weight_MT"] + c)
    df_reg["ln_Real_USD"] = np.log(df_reg["Total_Real_USD_M"] + c)

    X = sm.add_constant(df_reg[["T", "D", "T_post", "Demand_Index"]])
    y = df_reg["ln_Volume"]

    T_len = len(df_reg)
    bandwidth = max(int(np.floor(4 * (T_len / 100) ** (2 / 9))), 1)

    ols_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": bandwidth})
    glm_model = sm.GLM(df_reg["Total_Weight_MT"], X, family=sm.families.Poisson()).fit(
        cov_type="HAC", cov_kwds={"maxlags": bandwidth}
    )

    df_reg["pred_ln_Volume"] = ols_model.predict(X)
    df_reg["pred_Volume_MT"] = np.exp(df_reg["pred_ln_Volume"])

    X_cf = X.copy()
    X_cf["D"] = 0
    X_cf["T_post"] = 0
    df_reg["cf_Volume_MT"] = np.exp(ols_model.predict(X_cf))

    b0 = float(ols_model.params["const"])
    b1 = float(ols_model.params["T"])
    b2 = float(ols_model.params["D"])
    b3 = float(ols_model.params["T_post"])
    gamma = float(ols_model.params["Demand_Index"])

    p_b1 = float(ols_model.pvalues["T"])
    p_b2 = float(ols_model.pvalues["D"])
    p_b3 = float(ols_model.pvalues["T_post"])
    p_gamma = float(ols_model.pvalues["Demand_Index"])

    res_dict = {
        "Intervention_Label": label,
        "Intervention_Year": int(intervention_year),
        "Observations": int(T_len),
        "Matrix_Rank": int(np.linalg.matrix_rank(X)),
        "R_Squared": np.round(float(ols_model.rsquared), 4),
        "Baseline_Level_beta0": np.round(b0, 4),
        "Pre_Trend_beta1": np.round(b1, 4),
        "Pre_Trend_Pval": np.round(p_b1, 5),
        "Immediate_Shift_beta2": np.round(b2, 4),
        "Immediate_Shift_Pval": np.round(p_b2, 5),
        "Post_Slope_Change_beta3": np.round(b3, 4),
        "Post_Slope_Change_Pval": np.round(p_b3, 5),
        "Macro_Demand_gamma": np.round(gamma, 4),
        "Macro_Demand_Pval": np.round(p_gamma, 5),
        "Immediate_Shift_Percent": np.round((np.exp(b2) - 1.0) * 100.0, 2),
        "Stat_Sig_Reduction": bool((p_b2 < 0.05 and b2 < 0) or (p_b3 < 0.05 and b3 < 0))
    }

    glm_dict = {
        "Intervention_Label": label,
        "Intervention_Year": int(intervention_year),
        "Observations": int(T_len),
        "GLM_Pre_Trend_beta1": np.round(float(glm_model.params["T"]), 4),
        "GLM_Immediate_Shift_beta2": np.round(float(glm_model.params["D"]), 4),
        "GLM_Immediate_Shift_Pval": np.round(float(glm_model.pvalues["D"]), 5),
        "GLM_Post_Slope_Change_beta3": np.round(float(glm_model.params["T_post"]), 4),
        "GLM_Post_Slope_Change_Pval": np.round(float(glm_model.pvalues["T_post"]), 5),
        "GLM_Immediate_Shift_Percent": np.round((np.exp(float(glm_model.params["D"])) - 1.0) * 100.0, 2)
    }

    df_reg["Real_Unit_Value_USD_per_KG"] = df_reg["Weighted_Unit_Price_USD_per_KG"]
    return res_dict, glm_dict, df_reg[[
        "Year", "T", "D", "Total_Weight_MT", "pred_Volume_MT", "cf_Volume_MT", "Real_Unit_Value_USD_per_KG"
    ]]


def estimate_differenced_cits(sub_treat, dem_ctrl, intervention_year, cat, label, demand_map):
    """
    Estimates exact annual differenced Comparative ITS (SI Eq. S1b):
      Delta ln(Q_t) = ln(Q_{treat,t}) - ln(Q_{demersal,t})
                    = delta_0 + delta_1 * T_t + delta_2 * D_t + delta_3 * T_post_t + theta * Demand_Index_t + nu_t
    Eliminates the stacked-series seam bug and respects exact annual time ordering for Newey-West HAC
    and Cochrane-Orcutt / Prais-Winsten AR(1) serial-correlation checks.
    """
    m = pd.merge(
        sub_treat[["Year", "Total_Weight_MT"]].rename(columns={"Total_Weight_MT": "Q_treat"}),
        dem_ctrl[["Year", "Total_Weight_MT"]].rename(columns={"Total_Weight_MT": "Q_ctrl"}),
        on="Year", how="inner"
    ).sort_values("Year").reset_index(drop=True)

    min_yr = int(m["Year"].min())
    m["T"] = m["Year"] - min_yr + 1
    t_ev = intervention_year - min_yr + 1
    m["D"] = (m["Year"] >= intervention_year).astype(int)
    m["T_post"] = np.where(m["D"] == 1, m["T"] - t_ev, 0)
    raw_dem = np.array([demand_map.get(int(y), 1000.0) for y in m["Year"]])
    m["Demand_Index"] = np.log(raw_dem) - np.mean(np.log(raw_dem))
    m["Delta_ln_Q"] = np.log(m["Q_treat"] + 0.01) - np.log(m["Q_ctrl"] + 0.01)

    bw = max(int(np.floor(4 * (len(m) / 100) ** (2 / 9))), 1)
    X_c = sm.add_constant(m[["T", "D", "T_post", "Demand_Index"]])
    fit_c = sm.OLS(m["Delta_ln_Q"], X_c).fit(cov_type="HAC", cov_kwds={"maxlags": bw})

    # Prais-Winsten / GLSAR(1) check
    glsar_mod = sm.GLSAR(m["Delta_ln_Q"], X_c, rho=1)
    glsar_res = glsar_mod.iterative_fit(maxiter=15)

    d1 = float(fit_c.params["T"])
    d2 = float(fit_c.params["D"])
    d3 = float(fit_c.params["T_post"])
    return {
        "Intervention_Label": label,
        "Intervention_Year": int(intervention_year),
        "Conservation_Category": cat,
        "Benchmark_Control": "Demersal Commercial Taxonomic Groups",
        "Years_Overlapped": int(len(m)),
        "CITS_Pre_Trend_Diff_delta1": np.round(d1, 4),
        "CITS_Pre_Trend_Diff_Pval": np.round(float(fit_c.pvalues["T"]), 5),
        "CITS_Level_Shift_delta2": np.round(d2, 4),
        "CITS_Level_Shift_SE": np.round(float(fit_c.bse["D"]), 4),
        "CITS_Level_Shift_Pval": np.round(float(fit_c.pvalues["D"]), 5),
        "CITS_Level_Shift_Percent": np.round((np.exp(d2) - 1.0) * 100.0, 2),
        "CITS_Slope_Change_delta3": np.round(d3, 4),
        "CITS_Slope_Change_SE": np.round(float(fit_c.bse["T_post"]), 4),
        "CITS_Slope_Change_Pval": np.round(float(fit_c.pvalues["T_post"]), 5),
        "GLSAR1_Level_Shift_delta2": np.round(float(glsar_res.params["D"]), 4),
        "GLSAR1_Slope_Change_delta3": np.round(float(glsar_res.params["T_post"]), 4),
        "AR1_Rho": np.round(float(glsar_mod.rho[0]), 3),
        "CITS_R_Squared": np.round(float(fit_c.rsquared), 4),
    }


def estimate_bilateral_gravity_ppml(df_routes, intervention_year, cat, label, source_tag):
    """
    Estimates bilateral route-level Structural Gravity PPML (Santos Silva & Tenreyro 2006)
    comparing treated clade corridors against bulk demersal control corridors, explicitly
    expanding active routes (>= 4 active years) into a balanced route-by-year grid so
    extensive-margin trade zeros (Quantity_MT = 0.0) are retained without log(Q+c) bias.
    """
    years = sorted(df_routes["Year"].unique())
    min_yr = int(min(years))

    # Aggregate by (route_id, Treated, Year)
    grp = df_routes.groupby(["route_id", "Treated", "Year"], as_index=False).agg(
        Quantity_MT=("Quantity_MT", "sum")
    )
    active_cnt = grp.groupby("route_id")["Year"].nunique()
    keep_routes = active_cnt[active_cnt >= 4].index
    grp = grp[grp["route_id"].isin(keep_routes)].copy()

    # Cap demersal control routes to top 1,500 highest-volume corridors if huge, keeping all treated routes
    treat_routes = grp[grp["Treated"] == 1]["route_id"].unique()
    ctrl_routes = (
        grp[grp["Treated"] == 0]
        .groupby("route_id")["Quantity_MT"].sum()
        .sort_values(ascending=False)
        .head(1500)
        .index
    )
    sel_routes = np.concatenate([treat_routes, ctrl_routes])
    grp = grp[grp["route_id"].isin(sel_routes)].copy()

    # Expand balanced grid over [min_yr, max_yr] to retain true zero flows!
    route_meta = grp[["route_id", "Treated"]].drop_duplicates("route_id")
    grid = route_meta.assign(_k=1).merge(
        pd.DataFrame({"Year": years, "_k": 1}), on="_k"
    ).drop(columns="_k")
    bal = grid.merge(grp[["route_id", "Year", "Quantity_MT"]], on=["route_id", "Year"], how="left")
    bal["Quantity_MT"] = bal["Quantity_MT"].fillna(0.0)

    bal["Post"] = (bal["Year"] >= intervention_year).astype(int)
    bal["Treat_PreTrend"] = bal["Treated"] * (bal["Year"] - min_yr)
    bal["Treat_Post"] = bal["Treated"] * bal["Post"]
    bal["Treat_TrendPost"] = bal["Treated"] * bal["Post"] * (bal["Year"] - intervention_year)

    fit_did = pf.fepois(
        "Quantity_MT ~ Treat_Post | route_id + Year",
        data=bal, vcov={"CRV1": "route_id"}
    )
    fit_cits = pf.fepois(
        "Quantity_MT ~ Treat_PreTrend + Treat_Post + Treat_TrendPost | route_id + Year",
        data=bal, vcov={"CRV1": "route_id"}
    )

    # Also estimate Destination-Demand-Controlled Gravity PPML (with leave-one-clade-out importer demand ln_Dest_Demand)
    bal["Importer_ISO3"] = bal["route_id"].str.split("_").str[1]
    imp_yr_q = bal.groupby(["Importer_ISO3", "Year", "Treated"])["Quantity_MT"].transform("sum")
    imp_yr_tot = bal.groupby(["Importer_ISO3", "Year"])["Quantity_MT"].transform("sum")
    bal["ln_Dest_Demand"] = np.log(np.maximum(imp_yr_tot - imp_yr_q * bal["Treated"], 1.0))
    fit_cits_dem = pf.fepois(
        "Quantity_MT ~ Treat_PreTrend + Treat_Post + Treat_TrendPost + ln_Dest_Demand | route_id + Year",
        data=bal, vcov={"CRV1": "route_id"}
    )
    b_lvl_dem = float(fit_cits_dem.coef()["Treat_Post"])
    pv_lvl_dem = float(fit_cits_dem.pvalue()["Treat_Post"])

    b_did = float(fit_did.coef()["Treat_Post"])
    se_did = float(fit_did.se()["Treat_Post"])
    pv_did = float(fit_did.pvalue()["Treat_Post"])

    b_pre = float(fit_cits.coef()["Treat_PreTrend"])
    pv_pre = float(fit_cits.pvalue()["Treat_PreTrend"])
    b_lvl = float(fit_cits.coef()["Treat_Post"])
    se_lvl = float(fit_cits.se()["Treat_Post"])
    pv_lvl = float(fit_cits.pvalue()["Treat_Post"])
    b_slp = float(fit_cits.coef()["Treat_TrendPost"])
    se_slp = float(fit_cits.se()["Treat_TrendPost"])
    pv_slp = float(fit_cits.pvalue()["Treat_TrendPost"])

    zero_share = float((bal["Quantity_MT"] == 0.0).mean() * 100.0)
    return {
        "Conservation_Category": cat,
        "Intervention_Label": label,
        "Intervention_Year": int(intervention_year),
        "Panel_Source": source_tag,
        "Total_Route_Year_Cells": int(fit_cits._N),
        "Zero_Flow_Share_Pct": np.round(zero_share, 1),
        "PPML_Simple_DiD_Beta": np.round(b_did, 4),
        "PPML_Simple_DiD_SE": np.round(se_did, 4),
        "PPML_Simple_DiD_Pval": np.round(pv_did, 5),
        "PPML_Simple_DiD_Pct": np.round((np.exp(b_did) - 1.0) * 100.0, 2),
        "PPML_PreTrend_Diff_Beta": np.round(b_pre, 4),
        "PPML_PreTrend_Diff_Pval": np.round(pv_pre, 5),
        "PPML_Level_Shift_Beta": np.round(b_lvl, 4),
        "PPML_Level_Shift_SE": np.round(se_lvl, 4),
        "PPML_Level_Shift_Pval": np.round(pv_lvl, 5),
        "PPML_Level_Shift_Pct": np.round((np.exp(b_lvl) - 1.0) * 100.0, 2),
        "PPML_Slope_Change_Beta": np.round(b_slp, 4),
        "PPML_Slope_Change_SE": np.round(se_slp, 4),
        "PPML_Slope_Change_Pval": np.round(pv_slp, 5),
        "PPML_DemandCtrl_Level_Shift_Beta": np.round(b_lvl_dem, 4),
        "PPML_DemandCtrl_Level_Shift_Pval": np.round(pv_lvl_dem, 5),
    }


def main():
    print("=" * 80)
    print("MODULE 09: QUASI-EXPERIMENTAL ITS, DIFFERENCED CITS & GRAVITY PPML MODELS")
    print("=" * 80)

    if not ANNUAL_CONSERVATION_CSV.exists() or not ANNUAL_LONGITUDINAL_CSV.exists():
        print("Error: Required input CSVs do not exist.")
        sys.exit(1)

    df_ann = pd.read_csv(ANNUAL_CONSERVATION_CSV)
    df_long = pd.read_csv(ANNUAL_LONGITUDINAL_CSV)
    imp_all = df_ann[df_ann["Trade_Flow"] == "Import"].groupby("Year")["Total_Real_USD_M"].sum().to_dict()

    interventions = [
        ("High-Trophic Tunas (Thunnus spp.)", 2008, "ICCAT Multi-Annual Bluefin Recovery Plan & eBCD (2007/2008)"),
        ("Diadromous Eels (Anguilla spp.)", 2009, "CITES Appendix II Listing (2009) & EU Zero-Quota Export Ban (2010)"),
        ("Chondrichthyans (Sharks & Rays)", 2014, "CITES CoP16 Pelagic Shark & Hammerhead Listings (2013/2014)"),
        ("Sturgeons & Caviar (Acipenseriformes)", 2006, "CITES Wild Caspian Export Suspension & Universal Caviar Labelling (2006)"),
        ("Sea Cucumbers (Holothuroidea)", 2020, "CITES CoP18 Teatfish Listing (2019/2020)"),
    ]

    all_results = []
    all_glm = []
    all_trajectories = []
    cits_results = []

    dem_sub = df_long[
        (df_long["Conservation_Category"] == "Demersal Commercial Taxonomic Groups") &
        (df_long["Year"] <= 2024)
    ].sort_values("Year").copy()

    for cat, year, label in interventions:
        sub = df_long[(df_long["Conservation_Category"] == cat) & (df_long["Year"] <= 2024)].sort_values("Year").copy()
        if len(sub) >= 12:
            cat_imp = df_ann[
                (df_ann["Trade_Flow"] == "Import") & (df_ann["Conservation_Category"] == cat)
            ].set_index("Year")["Total_Real_USD_M"].to_dict()
            demand_map = {
                int(y): max(imp_all.get(y, 1000.0) - cat_imp.get(y, 0.0), 1.0)
                for y in sub["Year"]
            }
            print(f"1. Estimating Aggregate ITS & Exact Differenced CITS: {cat} (Shock: {year}, T={len(sub)})...")
            res, glm_res, traj = estimate_hac_its(sub, year, label, demand_map)
            traj["Conservation_Category"] = cat
            all_results.append(res)
            all_glm.append(glm_res)
            all_trajectories.append(traj)

            cits_row = estimate_differenced_cits(sub, dem_sub, year, cat, label, demand_map)
            cits_results.append(cits_row)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(DATA_DERIVED / "global_interrupted_time_series_results.csv", index=False)

    df_cits = pd.DataFrame(cits_results)
    df_cits.to_csv(DATA_DERIVED / "global_comparative_its_results.csv", index=False)

    df_glm = pd.DataFrame(all_glm)
    df_glm.to_csv(DATA_DERIVED / "global_its_glm_sensitivity_results.csv", index=False)

    df_all_traj = pd.concat(all_trajectories, ignore_index=True)
    df_all_traj.to_csv(DATA_DERIVED / "its_model_predicted_trajectories.csv", index=False)

    print("\n--- Exact Differenced CITS Summary (vs. Unregulated Demersal Control) ---")
    print(df_cits[[
        "Conservation_Category", "Intervention_Year", "Years_Overlapped",
        "CITS_Level_Shift_delta2", "CITS_Level_Shift_Pval",
        "CITS_Slope_Change_delta3", "CITS_Slope_Change_Pval", "AR1_Rho"
    ]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Bilateral Route-Level Structural Gravity PPML (Retaining Zero Flows)
    # -------------------------------------------------------------------------
    print("\n2. Estimating Bilateral Route-Level Structural Gravity PPML Models (with Zero Flows)...")
    th_df = pd.read_parquet(THREATENED_PARQUET)
    th_exp = th_df[
        (th_df["Trade_Flow"] == "Export") &
        (th_df["Year"] <= 2024) &
        (th_df["Reporter_ISO3"] != th_df["Partner_ISO3"]) &
        (~th_df["HS6_Code"].astype(str).str.zfill(6).isin(["160417", "030812"]))
    ].copy()
    th_exp["HS6_Code"] = th_exp["HS6_Code"].astype(str).str.zfill(6)
    th_exp["Quantity_MT"] = th_exp["Weight_KG"] / 1000.0
    th_exp["route_id"] = th_exp["Reporter_ISO3"] + "_" + th_exp["Partner_ISO3"] + "_" + th_exp["HS6_Code"]

    baci = pd.read_parquet(BACI_PARQUET)
    baci["HS92_Code"] = baci["HS92_Code"].astype(str).str.zfill(6)
    baci = baci[(baci["Year"] >= 1995) & (baci["Year"] <= 2024) & (baci["Exporter_ISO3"] != baci["Importer_ISO3"])].copy()
    baci["Quantity_MT"] = baci["Quantity_t"]
    baci["route_id"] = baci["Exporter_ISO3"] + "_" + baci["Importer_ISO3"] + "_" + baci["HS92_Code"]

    ppml_rows = []
    for cat, year, label in interventions:
        if cat in ["Chondrichthyans (Sharks & Rays)", "Sturgeons & Caviar (Acipenseriformes)"]:
            t_codes = ["030265", "030375"] if "Chondrichthyans" in cat else ["160430"]
            c_codes = ["030250", "030262", "030263", "030360", "030372", "030373", "030378"]
            sub_p = baci[baci["HS92_Code"].isin(t_codes + c_codes)][["Year", "route_id", "HS92_Code", "Quantity_MT"]].copy()
            sub_p["Treated"] = sub_p["HS92_Code"].isin(t_codes).astype(int)
            src_tag = "CEPII BACI HS92 (1995-2024)"
        else:
            min_cat_yr = int(th_exp[th_exp["Conservation_Category"] == cat]["Year"].min())
            sub_p = th_exp[
                (th_exp["Conservation_Category"].isin([cat, "Demersal Commercial Taxonomic Groups"])) &
                (th_exp["Year"] >= min_cat_yr)
            ][["Year", "route_id", "Conservation_Category", "Quantity_MT"]].copy()
            sub_p["Treated"] = (sub_p["Conservation_Category"] == cat).astype(int)
            src_tag = f"UN Comtrade ({min_cat_yr}-2024)"

        ppml_res = estimate_bilateral_gravity_ppml(sub_p, year, cat, label, src_tag)
        ppml_rows.append(ppml_res)

    df_ppml = pd.DataFrame(ppml_rows)
    out_ppml = DATA_DERIVED / "global_gravity_ppml_policy_results.csv"
    df_ppml.to_csv(out_ppml, index=False)
    print(f"   -> Saved Structural Gravity PPML results: {out_ppml}")
    print(df_ppml[[
        "Conservation_Category", "Intervention_Year", "Total_Route_Year_Cells", "Zero_Flow_Share_Pct",
        "PPML_Simple_DiD_Beta", "PPML_Simple_DiD_Pval",
        "PPML_Level_Shift_Beta", "PPML_Level_Shift_Pval",
        "PPML_Slope_Change_Beta", "PPML_Slope_Change_Pval"
    ]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. Formal Eel Jurisdictional & Cross-Chapter Substitution Test (H3a)
    # -------------------------------------------------------------------------
    print("\n3. Quantifying Eel Jurisdictional & Cross-Chapter Processing Displacement (H3a)...")
    eel_all = th_df[
        (th_df["Trade_Flow"] == "Export") &
        (th_df["Conservation_Category"] == "Diadromous Eels (Anguilla spp.)") &
        (th_df["Year"] <= 2024) &
        (th_df["Reporter_ISO3"] != th_df["Partner_ISO3"])
    ].copy()
    eel_all["HS6_Code"] = eel_all["HS6_Code"].astype(str).str.zfill(6)
    eel_all["Quantity_MT"] = eel_all["Weight_KG"] / 1000.0
    eel_all["Real_USD_M"] = eel_all["Real_Trade_Value_USD"] / 1e6

    def classify_eel_corridor(row):
        rep, par, hs = row["Reporter_ISO3"], row["Partner_ISO3"], row["HS6_Code"]
        if hs == "160417":
            if rep in EAST_ASIA_ISO:
                return "East Asia Prepared Eel Fillets (Ch. 16: HS 160417)"
            return "Rest of World Prepared Eel Fillets (Ch. 16: HS 160417)"
        if rep in EU_EEA_GBR_ISO and par not in EU_EEA_GBR_ISO:
            return "Extra-European Raw/Live Eel Exports (Ch. 03: Banned 2010)"
        if rep in EU_EEA_GBR_ISO and par in EU_EEA_GBR_ISO:
            return "Intra-European Single-Market Raw/Live Eel Flows (Ch. 03)"
        return "Non-European Congeneric Raw/Live Eel Exports (Ch. 03: Americas/Asia/Africa)"

    eel_all["Corridor_Regime"] = eel_all.apply(classify_eel_corridor, axis=1)
    ann_eel = eel_all.groupby(["Year", "Corridor_Regime"], as_index=False).agg(
        Total_MT=("Quantity_MT", "sum"),
        Total_Real_USD_M=("Real_USD_M", "sum"),
        Route_Count=("HS6_Code", "count")
    )

    disp_rows = []
    for reg in sorted(ann_eel["Corridor_Regime"].unique()):
        g = ann_eel[ann_eel["Corridor_Regime"] == reg]
        pre_mt = float(g[g["Year"] <= 2008]["Total_MT"].mean()) if (g["Year"] <= 2008).any() else np.nan
        post_mt = float(g[g["Year"] >= 2012]["Total_MT"].mean()) if (g["Year"] >= 2012).any() else np.nan
        pre_usd = float(g[g["Year"] <= 2008]["Total_Real_USD_M"].mean()) if (g["Year"] <= 2008).any() else np.nan
        post_usd = float(g[g["Year"] >= 2012]["Total_Real_USD_M"].mean()) if (g["Year"] >= 2012).any() else np.nan
        pct_chg_mt = ((post_mt - pre_mt) / pre_mt * 100.0) if (not np.isnan(pre_mt) and pre_mt > 0) else np.nan
        disp_rows.append({
            "Corridor_Regime": reg,
            "Pre_2009_Mean_Annual_MT": np.round(pre_mt, 1) if not np.isnan(pre_mt) else np.nan,
            "Post_2012_Mean_Annual_MT": np.round(post_mt, 1) if not np.isnan(post_mt) else np.nan,
            "Volume_Change_Pct": np.round(pct_chg_mt, 1) if not np.isnan(pct_chg_mt) else np.nan,
            "Pre_2009_Mean_Real_USD_M": np.round(pre_usd, 2) if not np.isnan(pre_usd) else np.nan,
            "Post_2012_Mean_Real_USD_M": np.round(post_usd, 2) if not np.isnan(post_usd) else np.nan,
        })

    # Also run route-level PPML DiD comparing Extra-European Ch. 03 exports vs Non-European Congeneric Ch. 03 exports (1988-2024)
    eel_ch03 = eel_all[eel_all["HS6_Code"] != "160417"].copy()
    eel_ch03["route_id"] = eel_ch03["Reporter_ISO3"] + "_" + eel_ch03["Partner_ISO3"] + "_" + eel_ch03["HS6_Code"]
    eel_ch03["Is_ExtraEU"] = (
        eel_ch03["Reporter_ISO3"].isin(EU_EEA_GBR_ISO) & (~eel_ch03["Partner_ISO3"].isin(EU_EEA_GBR_ISO))
    ).astype(int)
    grp_e = eel_ch03.groupby(["route_id", "Is_ExtraEU", "Year"], as_index=False).agg(Quantity_MT=("Quantity_MT", "sum"))
    keep_e = grp_e.groupby("route_id")["Year"].nunique()
    grp_e = grp_e[grp_e["route_id"].isin(keep_e[keep_e >= 3].index)].copy()
    yrs_e = sorted(eel_ch03["Year"].unique())
    grid_e = grp_e[["route_id", "Is_ExtraEU"]].drop_duplicates("route_id").assign(_k=1).merge(
        pd.DataFrame({"Year": yrs_e, "_k": 1}), on="_k"
    ).drop(columns="_k")
    bal_e = grid_e.merge(grp_e[["route_id", "Year", "Quantity_MT"]], on=["route_id", "Year"], how="left")
    bal_e["Quantity_MT"] = bal_e["Quantity_MT"].fillna(0.0)
    bal_e["ExtraEU_Post2010"] = bal_e["Is_ExtraEU"] * (bal_e["Year"] >= 2010).astype(int)
    fit_eel_ppml = pf.fepois("Quantity_MT ~ ExtraEU_Post2010 | route_id + Year", data=bal_e, vcov={"CRV1": "route_id"})
    b_eu = float(fit_eel_ppml.coef()["ExtraEU_Post2010"])
    se_eu = float(fit_eel_ppml.se()["ExtraEU_Post2010"])
    pv_eu = float(fit_eel_ppml.pvalue()["ExtraEU_Post2010"])

    df_disp = pd.DataFrame(disp_rows)
    df_disp["ExtraEU_vs_NonEU_PPML_DiD_Beta"] = np.round(b_eu, 4)
    df_disp["ExtraEU_vs_NonEU_PPML_DiD_SE"] = np.round(se_eu, 4)
    df_disp["ExtraEU_vs_NonEU_PPML_DiD_Pval"] = np.round(pv_eu, 6)
    df_disp["ExtraEU_vs_NonEU_PPML_Change_Pct"] = np.round((np.exp(b_eu) - 1.0) * 100.0, 2)

    out_disp = DATA_DERIVED / "global_eel_processing_displacement_results.csv"
    df_disp.to_csv(out_disp, index=False)
    print(f"   -> Saved Eel displacement & substitution decomposition: {out_disp}")
    print(df_disp.to_string(index=False))
    print("=" * 80)
    print("MODULE 09 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
