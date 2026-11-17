#!/usr/bin/env python3
"""
Module 15: Sovereign Governance, Tariff-vs.-CITES Evasion, and SRECI (Eqs. 2c, 4c, 6d, 9d, 11c, 13b)
====================================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Estimates cross-sovereign Shapley-Owen R2 decompositions of customs pooling,
2WFE Dual-IV aquaculture and income interactions, governance-moderated Gravity PPML
displacement models, 2012-2024 dyad-HS6-year WTO applied tariff versus CITES
evasion models, natural-capital wealth regressions, and SRECI reform tracks.

Outputs:
  - data/derived/sovereign_customs_consolidation_decomposition.csv
  - data/derived/sovereign_allee_income_aquaculture_moderation.csv
  - data/derived/governance_moderated_gravity_ppml_summary.csv
  - data/derived/governance_moderated_ppml_and_evasion_results.csv
  - data/derived/brander_taylor_cwon_natural_capital_results.csv
  - data/derived/sovereign_cbdr_reform_exposure_sreci.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import pyfixest as pf

warnings.filterwarnings("ignore")

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.environ.get("EXTERNAL_DB_ROOT", os.path.join(os.path.dirname(PROJ_DIR), "_Databases"))
DERIVED_DIR = os.path.join(PROJ_DIR, "data", "derived")
MASTER_PARQUET = os.path.join(PROJ_DIR, "data", "raw", "master_global_bilateral_panel.parquet")

CPI_DEFLATOR = {
    1988: 118.3, 1989: 124.0, 1990: 130.7, 1991: 136.2, 1992: 140.3,
    1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9, 1997: 160.5,
    1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9,
    2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.3,
    2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6,
    2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1,
    2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0, 2022: 292.7,
    2023: 304.7, 2024: 313.0, 2025: 320.0
}
BASE_CPI = 313.0  # canonical US BLS CPI-U base (aligned with Scripts 02-07)


def compute_shapley_r2(df: pd.DataFrame, y_col: str, x_groups: dict, weights_col: str = None) -> dict:
    """Computes exact Shapley / Owen R^2 decomposition across grouped predictors."""
    import itertools
    keys = list(x_groups.keys())
    k = len(keys)
    w = df[weights_col].values if weights_col else np.ones(len(df))
    y = df[y_col].values

    def fit_r2(subset_keys):
        if not subset_keys:
            return 0.0
        cols = [c for g in subset_keys for c in x_groups[g]]
        X = sm.add_constant(df[cols].values)
        res = sm.WLS(y, X, weights=w).fit()
        return max(0.0, float(res.rsquared))

    r2_cache = {}
    for r in range(k + 1):
        for comb in itertools.combinations(keys, r):
            r2_cache[frozenset(comb)] = fit_r2(comb)

    shapley = {g: 0.0 for g in keys}
    import math
    fact_k = math.factorial(k)
    for g in keys:
        others = [x for x in keys if x != g]
        for r in range(k):
            weight = (math.factorial(r) * math.factorial(k - r - 1)) / math.factorial(k)
            for S in itertools.combinations(others, r):
                s_set = frozenset(S)
                s_with_g = frozenset(list(S) + [g])
                shapley[g] += weight * (r2_cache[s_with_g] - r2_cache[s_set])

    tot_r2 = r2_cache[frozenset(keys)]
    return {
        "Total_R2": tot_r2,
        "Shapley_R2": shapley,
        "Shapley_Share_Pct": {g: 100.0 * v / max(tot_r2, 1e-9) for g, v in shapley.items()}
    }


def main():
    print("=" * 80)
    print("MODULE 15: ESTIMATING SOVEREIGN, BLOC & GOVERNANCE PARTIAL-DERIVATIVE SUITES")
    print("=" * 80)

    cs = pd.read_csv(os.path.join(DERIVED_DIR, "country_cross_section_governance_concordance.csv"))
    if "TBT" in cs.columns:
        cs = cs.rename(columns={"TBT": "WTO_TBT_Count", "SPS": "WTO_SPS_Count"})
    panel_gov = pd.read_parquet(os.path.join(DERIVED_DIR, "country_socioeconomic_governance_master.parquet"))
    sov_tar_raw = pd.read_csv(os.path.join(DERIVED_DIR, "sovereign_customs_aggregation_tar.csv"))
    sov_tar = sov_tar_raw.groupby("Reporter_ISO3", as_index=False).agg(
        Total_Export_Real_USD=("Total_Export_Real_USD", "sum"),
        Aggregated_Export_Real_USD=("Aggregated_Export_Real_USD", "sum"),
        Specific_Export_Real_USD=("Specific_Export_Real_USD", "sum"),
        Total_Export_Weight_KG=("Total_Export_Weight_KG", "sum"),
        Aggregated_Export_Weight_KG=("Aggregated_Export_Weight_KG", "sum"),
        Specific_Export_Weight_KG=("Specific_Export_Weight_KG", "sum"),
    )
    sov_tar["TAR_Value_Percent"] = 100.0 * sov_tar["Aggregated_Export_Real_USD"] / np.maximum(sov_tar["Total_Export_Real_USD"], 1.0)
    sov_tar["TAR_Volume_Percent"] = 100.0 * sov_tar["Aggregated_Export_Weight_KG"] / np.maximum(sov_tar["Total_Export_Weight_KG"], 1.0)
    sov_tar["Total_Export_Real_USD_B"] = sov_tar["Total_Export_Real_USD"] / 1e9
    sov_tar["Total_Export_Weight_MMT"] = sov_tar["Total_Export_Weight_KG"] / 1e9

    sov_tbt_raw = pd.read_csv(os.path.join(DERIVED_DIR, "sovereign_trophic_balance_tbt.csv"))
    sov_tbt_raw["_wx"] = sov_tbt_raw["Total_Weight_KG_Export"] * sov_tbt_raw["MTL_Export"]
    sov_tbt_raw["_wm"] = sov_tbt_raw["Total_Weight_KG_Import"] * sov_tbt_raw["MTL_Import"]
    sov_tbt = sov_tbt_raw.groupby("Reporter_ISO3", as_index=False).agg(
        _wx=("_wx", "sum"),
        _wm=("_wm", "sum"),
        Total_Weight_KG_Export=("Total_Weight_KG_Export", "sum"),
        Total_Weight_KG_Import=("Total_Weight_KG_Import", "sum"),
        Total_Real_USD_Export=("Total_Real_USD_Export", "sum"),
        Total_Real_USD_Import=("Total_Real_USD_Import", "sum"),
    )
    sov_tbt["MTL_Export"] = sov_tbt["_wx"] / np.maximum(sov_tbt["Total_Weight_KG_Export"], 1.0)
    sov_tbt["MTL_Import"] = sov_tbt["_wm"] / np.maximum(sov_tbt["Total_Weight_KG_Import"], 1.0)
    sov_tbt["Unit_Price_USD_per_KG_Export"] = sov_tbt["Total_Real_USD_Export"] / np.maximum(sov_tbt["Total_Weight_KG_Export"], 1.0)
    sov_tbt["Unit_Price_USD_per_KG_Import"] = sov_tbt["Total_Real_USD_Import"] / np.maximum(sov_tbt["Total_Weight_KG_Import"], 1.0)
    sov_tbt["TBT"] = sov_tbt["MTL_Export"] - sov_tbt["MTL_Import"]
    sov_tbt["Net_Biomass_Export_MT"] = (sov_tbt["Total_Weight_KG_Export"] - sov_tbt["Total_Weight_KG_Import"]) / 1e3
    sov_tbt["Net_Value_USD_B"] = (sov_tbt["Total_Real_USD_Export"] - sov_tbt["Total_Real_USD_Import"]) / 1e9
    sov_tbt = sov_tbt.drop(columns=["_wx", "_wm"])
    scri_audit = pd.read_csv(os.path.join(DERIVED_DIR, "global_hs6_customs_scri_audit.csv"))
    scri_audit["HS6_Code"] = scri_audit["HS6_Code"].astype(str).str.zfill(6)
    scri_audit["SCRI"] = scri_audit["Subheading_SCRI"]
    scri_audit["Taxonomic_Pooling_Index_Tau"] = 1.0 - scri_audit["SCRI"]

    # -------------------------------------------------------------------------
    # SUITE A: Multi-Scale Group Summaries & Sovereign Customs Decomposition (RQ1 / Eq. 2c)
    # -------------------------------------------------------------------------
    print("\n1. [Suite A] Computing Multi-Scale Income, Continent, Bloc, SIDS/LDC & Subregion Matrices...")
    # Also compute exporter Chapter 16 (Prepared/Preserved) export share and exact Value-Weighted SCRI per country
    bilat_all = pd.read_parquet(
        os.path.join(DERIVED_DIR, "global_threatened_taxa_bilateral_panel.parquet"),
        columns=["Year", "Trade_Flow", "Reporter_ISO3", "Partner_ISO3", "HS6_Code", "Weight_KG", "Real_Trade_Value_USD", "Processing_Form", "Conservation_Category", "CITES_Status", "Threatened_Taxa_Flag", "Trophic_Level"]
    )
    bilat_all["HS6_Code"] = bilat_all["HS6_Code"].astype(str).str.zfill(6)
    bilat_all = bilat_all.merge(scri_audit[["HS6_Code", "SCRI", "Taxonomic_Pooling_Index_Tau"]], on="HS6_Code", how="left")
    bilat_all["SCRI"] = bilat_all["SCRI"].fillna(0.05)
    bilat_all["Is_Ch16"] = bilat_all["HS6_Code"].str.startswith("16").astype(int)
    bilat_all["Is_Pooled_Code"] = (bilat_all["SCRI"] < 1.0).astype(int)

    # Load full bilateral trade for sovereign Chapter 16 share and import TAR (for BTAR across all 210 countries)
    raw_full = pd.read_parquet(
        MASTER_PARQUET,
        columns=["Year", "Trade_Flow", "Reporter_ISO3", "Partner_ISO3", "HS6_Code", "Weight_KG", "Trade_Value_USD"]
    )
    raw_full = raw_full[(raw_full["Year"] <= 2024) & (raw_full["Weight_KG"] > 0) & (raw_full["Trade_Value_USD"] > 0)].copy()
    raw_full["HS6_Code"] = raw_full["HS6_Code"].astype(str).str.zfill(6)
    raw_full["Real_USD"] = raw_full["Trade_Value_USD"] * (BASE_CPI / raw_full["Year"].map(CPI_DEFLATOR))
    raw_full = raw_full.merge(scri_audit[["HS6_Code", "SCRI"]], on="HS6_Code", how="left")
    raw_full["SCRI"] = raw_full["SCRI"].fillna(0.05)
    raw_full["Is_Pooled"] = (raw_full["SCRI"] < 1.0).astype(int)
    raw_full["Is_Ch16"] = raw_full["HS6_Code"].str.startswith("16").astype(int)
    raw_full["Pooled_USD_Row"] = raw_full["Real_USD"] * raw_full["Is_Pooled"]
    raw_full["Ch16_USD_Row"] = raw_full["Real_USD"] * raw_full["Is_Ch16"]
    raw_full["SCRI_USD_Row"] = raw_full["Real_USD"] * raw_full["SCRI"]

    exp_full = raw_full[raw_full["Trade_Flow"] == "Export"].groupby("Reporter_ISO3", as_index=False).agg(
        Exp_Total_USD=("Real_USD", "sum"),
        Exp_Pooled_USD=("Pooled_USD_Row", "sum"),
        Exp_Ch16_USD=("Ch16_USD_Row", "sum"),
        Exp_SCRI_USD=("SCRI_USD_Row", "sum")
    ).rename(columns={"Reporter_ISO3": "ISO3"})
    exp_full["Exp_Weighted_SCRI"] = exp_full["Exp_SCRI_USD"] / np.maximum(exp_full["Exp_Total_USD"], 1.0)
    exp_full["Ch16_Export_Share"] = exp_full["Exp_Ch16_USD"] / np.maximum(exp_full["Exp_Total_USD"], 1.0)
    exp_full["Exp_TAR_Val_Pct"] = 100.0 * exp_full["Exp_Pooled_USD"] / np.maximum(exp_full["Exp_Total_USD"], 1.0)

    imp_full = raw_full[raw_full["Trade_Flow"] == "Import"].groupby("Reporter_ISO3", as_index=False).agg(
        Imp_Total_USD=("Real_USD", "sum"),
        Imp_Pooled_USD=("Pooled_USD_Row", "sum"),
        Imp_SCRI_USD=("SCRI_USD_Row", "sum")
    ).rename(columns={"Reporter_ISO3": "ISO3"})
    imp_full["Imp_Weighted_SCRI"] = imp_full["Imp_SCRI_USD"] / np.maximum(imp_full["Imp_Total_USD"], 1.0)
    imp_full["Imp_TAR_Val_Pct"] = 100.0 * imp_full["Imp_Pooled_USD"] / np.maximum(imp_full["Imp_Total_USD"], 1.0)

    # Merge sovereign TAR, TBT, and full-trade stats onto cs
    sov_master = (
        cs.merge(sov_tar.drop(columns=["Reporter_Country", "Reporter_Bloc"], errors="ignore"), left_on="ISO3", right_on="Reporter_ISO3", how="left")
        .merge(sov_tbt.drop(columns=["Reporter_Country", "Reporter_Bloc", "Total_Export_Real_USD_B"], errors="ignore"), left_on="ISO3", right_on="Reporter_ISO3", how="left", suffixes=("", "_tbt"))
        .merge(exp_full, on="ISO3", how="left")
        .merge(imp_full, on="ISO3", how="left")
    )
    sov_master["BTAR_pp"] = sov_master["TAR_Value_Percent"] - sov_master["Imp_TAR_Val_Pct"]
    sov_master["Export_to_Import_Price_Ratio"] = sov_master["Unit_Price_USD_per_KG_Export"] / sov_master["Unit_Price_USD_per_KG_Import"]

    # Build Multi-Scale Group Summary Table (Supplementary Table S18)
    def summarize_group(df_sub: pd.DataFrame, dim_name: str, grp_label: str) -> dict:
        exp_u = df_sub["Total_Real_USD_Export"].sum() / 1e9
        imp_u = df_sub["Total_Real_USD_Import"].sum() / 1e9
        exp_w = df_sub["Total_Weight_KG_Export"].sum() / 1e9
        imp_w = df_sub["Total_Weight_KG_Import"].sum() / 1e9
        tar_exp_u = df_sub["Total_Export_Real_USD"].sum()
        tar_agg_u = df_sub["Aggregated_Export_Real_USD"].sum()
        tar_exp_w = df_sub["Total_Export_Weight_KG"].sum()
        tar_agg_w = df_sub["Aggregated_Export_Weight_KG"].sum()
        tar_val_pct = 100.0 * tar_agg_u / max(tar_exp_u, 1.0)
        tar_vol_pct = 100.0 * tar_agg_w / max(tar_exp_w, 1.0)
        imp_tar_pct = 100.0 * df_sub["Imp_Pooled_USD"].sum() / max(df_sub["Imp_Total_USD"].sum(), 1.0)
        scri_val = float(np.sum(df_sub["Exp_Total_USD"].fillna(0) * df_sub["Exp_Weighted_SCRI"].fillna(0.5)) / max(df_sub["Exp_Total_USD"].sum(), 1.0))

        mtl_x = float(np.sum(df_sub["Total_Weight_KG_Export"].fillna(0) * df_sub["MTL_Export"].fillna(3.2)) / max(df_sub["Total_Weight_KG_Export"].sum(), 1.0))
        mtl_m = float(np.sum(df_sub["Total_Weight_KG_Import"].fillna(0) * df_sub["MTL_Import"].fillna(3.2)) / max(df_sub["Total_Weight_KG_Import"].sum(), 1.0))
        px = (df_sub["Total_Real_USD_Export"].sum() / max(df_sub["Total_Weight_KG_Export"].sum(), 1.0))
        pm = (df_sub["Total_Real_USD_Import"].sum() / max(df_sub["Total_Weight_KG_Import"].sum(), 1.0))

        return {
            "Classification_Dimension": dim_name,
            "Group_Name": grp_label,
            "Active_Reporters_N": int(df_sub["Total_Export_Real_USD"].notna().sum()),
            "Export_Value_2024USD_B": round(tar_exp_u / 1e9, 2),
            "Import_Value_2024USD_B": round(imp_u, 2),
            "Net_Value_Balance_USD_B": round((tar_exp_u / 1e9) - imp_u, 2),
            "Export_Volume_MMT": round(tar_exp_w / 1e9, 2),
            "Import_Volume_MMT": round(imp_w, 2),
            "Net_Volume_Balance_MMT": round((tar_exp_w / 1e9) - imp_w, 2),
            "Pooled_TAR_Value_Pct": round(tar_val_pct, 2),
            "Pooled_TAR_Volume_Pct": round(tar_vol_pct, 2),
            "Import_TAR_Value_Pct": round(imp_tar_pct, 2),
            "BTAR_Net_Masking_pp": round(tar_val_pct - imp_tar_pct, 2),
            "Value_Weighted_SCRI": round(scri_val, 4),
            "Export_Unit_Price_USD_kg": round(px, 2),
            "Import_Unit_Price_USD_kg": round(pm, 2),
            "Export_to_Import_Price_Ratio": round(px / max(pm, 1e-6), 2),
            "MTL_Export": round(mtl_x, 3),
            "MTL_Import": round(mtl_m, 3),
            "Trophic_Balance_TBT": round(mtl_x - mtl_m, 3),
            "Mean_Gov_Capacity_Index": round(float(df_sub["Gov_Capacity_Index"].mean()), 2),
            "Mean_PSMA_SDG14_6_1": round(float(df_sub["SDG14_6_1_IUU_PSMA_Score"].mean()), 2),
            "Mean_WTO_MFN_Tariff_Pct": round(float(df_sub["WTO_MFN_Seafood_Tariff_Pct"].mean()), 2),
            "Mean_FAO_Aqua_Share_Pct": round(float(df_sub["Mean_FAO_Aqua_Share"].mean() * 100.0), 1)
        }

    active_sov = sov_master[sov_master["Total_Export_Real_USD"].notna() | sov_master["Total_Real_USD_Import"].notna()].copy()
    matrix_rows = []

    # 1. World Bank Income Tiers
    for inc in ["HIC", "UMC", "LMC", "LIC"]:
        matrix_rows.append(summarize_group(active_sov[active_sov["WB_Income_Group"] == inc], "1. World Bank Income Tier", inc))
    # 2. Macro Development & Vulnerability Groupings
    matrix_rows.append(summarize_group(active_sov[active_sov["Is_OECD"] == 1], "2. Development & Vulnerability Tier", "OECD Members (38)"))
    matrix_rows.append(summarize_group(active_sov[active_sov["Is_Global_South"] == 1], "2. Development & Vulnerability Tier", "Global South (UMC + LMC + LIC)"))
    matrix_rows.append(summarize_group(active_sov[active_sov["Is_SIDS"] == 1], "2. Development & Vulnerability Tier", "Small Island Developing States (SIDS)"))
    matrix_rows.append(summarize_group(active_sov[active_sov["Is_LDC"] == 1], "2. Development & Vulnerability Tier", "Least Developed Countries (LDCs)"))
    matrix_rows.append(summarize_group(active_sov[active_sov["Is_BRICS_Plus"] == 1], "2. Development & Vulnerability Tier", "BRICS+ Economies"))
    # 3. UN Continents (5)
    for cont in ["Europe", "Americas", "Oceania", "Africa", "Asia"]:
        matrix_rows.append(summarize_group(active_sov[active_sov["UN_Continent"] == cont], "3. UN Continent", cont))
    # 4. Regional Trade & Customs Blocs
    for rta_label, col_flag in [
        ("EU-27 (TARIC / CFP)", "Is_EU27"),
        ("EFTA (Norway / Iceland / Switzerland)", "Is_EFTA"),
        ("USMCA (USA / Canada / Mexico)", "Is_USMCA"),
        ("ASEAN (AHTN Processing Hub)", "Is_ASEAN"),
        ("RCEP (15-Nation Asia-Pacific Bloc)", "Is_RCEP"),
        ("CPTPP (Trans-Pacific Partnership)", "Is_CPTPP"),
        ("Mercosur (South American Customs Union)", "Is_Mercosur"),
        ("AfCFTA (African Continental FTA)", "Is_AfCFTA"),
        ("Track 1 Immediate 2028 Customs Schedules", "Is_Track1_2028_Customs")
    ]:
        matrix_rows.append(summarize_group(active_sov[active_sov[col_flag] == 1], "4. Regional Trade & Customs Bloc", rta_label))
    # 5. All 22 UN M49 Subregions
    for subr in sorted(active_sov["UN_M49_Subregion"].dropna().unique()):
        if subr == "Other":
            continue
        sub_df = active_sov[active_sov["UN_M49_Subregion"] == subr]
        if len(sub_df) >= 1:
            matrix_rows.append(summarize_group(sub_df, "5. UN M49 Subregion", subr))

    mat_df = pd.DataFrame(matrix_rows)
    mat_out = os.path.join(DERIVED_DIR, "multi_scale_bloc_income_trophic_customs_matrix.csv")
    mat_df.to_csv(mat_out, index=False)
    print(f"   -> Saved Multi-Scale Sovereign/Bloc Matrix: {mat_out} ({len(mat_df)} strata)")

    # Suite A Econometric Decomposition of Sovereign Export Pooling (Eq. 2c)
    reg_a = sov_master[sov_master["Total_Export_Real_USD"].fillna(0) > 1e5].copy()
    reg_a["TAR_Prop"] = np.clip(reg_a["TAR_Value_Percent"] / 100.0, 1e-4, 1.0 - 1e-4)
    reg_a["SCRI_Prop"] = np.clip(reg_a["Exp_Weighted_SCRI"].fillna(1.0 - reg_a["TAR_Prop"]), 1e-4, 1.0 - 1e-4)
    reg_a["ln_Biodiv_Richness"] = np.log(np.maximum(reg_a["Marine_Threatened_Biodiversity_Richness"], 1.0))
    reg_a["Abs_Lat_10deg"] = reg_a["Abs_Latitude"] / 10.0
    reg_a["ln_GDPpc"] = np.log(np.maximum(reg_a["GDP_Per_Capita_Real_USD"], 100.0))
    reg_a["Ch16_Share_Pct"] = reg_a["Ch16_Export_Share"].fillna(0.0) * 100.0
    reg_a["Aqua_Share_Pct"] = reg_a["Mean_FAO_Aqua_Share"].fillna(0.0) * 100.0
    reg_a["ln_Export_Weight"] = np.log(np.maximum(reg_a["Total_Export_Real_USD"], 1.0))

    # Model A1: Unweighted OLS (HC3) on TAR_Value_Percent
    m_a1 = smf.ols(
        "TAR_Value_Percent ~ ln_Biodiv_Richness + Abs_Lat_10deg + Gov_Capacity_Index + Ch16_Share_Pct + Aqua_Share_Pct",
        data=reg_a
    ).fit(cov_type="HC3")
    # Model A2: Trade-Weighted WLS on TAR_Value_Percent
    m_a2 = smf.wls(
        "TAR_Value_Percent ~ ln_Biodiv_Richness + Abs_Lat_10deg + Gov_Capacity_Index + Ch16_Share_Pct + Aqua_Share_Pct",
        data=reg_a, weights=reg_a["ln_Export_Weight"]
    ).fit(cov_type="HC3")
    # Model A3: Papke-Wooldridge Fractional Logit QMLE on TAR_Prop
    m_a3 = smf.glm(
        "TAR_Prop ~ ln_Biodiv_Richness + Abs_Lat_10deg + Gov_Capacity_Index + Ch16_Share_Pct + Aqua_Share_Pct",
        data=reg_a, family=sm.families.Binomial()
    ).fit(cov_type="HC3")

    shapley_res = compute_shapley_r2(
        reg_a, "TAR_Value_Percent",
        {
            "Biogeographic_Temperate_vs_Tropical_Endowment": ["ln_Biodiv_Richness", "Abs_Lat_10deg"],
            "Sovereign_Governance_and_Customs_Capacity": ["Gov_Capacity_Index"],
            "Processing_Hub_and_Aquaculture_Structure": ["Ch16_Share_Pct", "Aqua_Share_Pct"]
        },
        weights_col="ln_Export_Weight"
    )
    print("   -> Suite A Shapley R^2 Decomposition (Total R^2 = {:.3f}):".format(shapley_res["Total_R2"]))
    for k_sh, v_sh in shapley_res["Shapley_Share_Pct"].items():
        print(f"      * {k_sh}: {v_sh:.1f}% of explained variance (R^2 = {shapley_res['Shapley_R2'][k_sh]:.3f})")

    suite_a_rows = []
    for m_label, m_obj in [
        ("OLS_HC3_TAR_Pct", m_a1),
        ("WLS_LogTradeWeighted_TAR_Pct", m_a2),
        ("Fractional_Logit_QMLE_TAR_Prop", m_a3)
    ]:
        for param in m_obj.params.index:
            suite_a_rows.append({
                "Model": m_label,
                "Parameter": param,
                "Estimate": round(float(m_obj.params[param]), 4),
                "Std_Error_HC3": round(float(m_obj.bse[param]), 4),
                "z_or_t_stat": round(float(m_obj.tvalues[param]), 3),
                "p_value": float(m_obj.pvalues[param]),
                "N_Countries": int(m_obj.nobs),
                "Total_WLS_R2": round(shapley_res["Total_R2"], 4),
                "Shapley_Biogeographic_Pct": round(shapley_res["Shapley_Share_Pct"]["Biogeographic_Temperate_vs_Tropical_Endowment"], 2),
                "Shapley_Governance_Capacity_Pct": round(shapley_res["Shapley_Share_Pct"]["Sovereign_Governance_and_Customs_Capacity"], 2),
                "Shapley_Processing_Aquaculture_Pct": round(shapley_res["Shapley_Share_Pct"]["Processing_Hub_and_Aquaculture_Structure"], 2)
            })
    pd.DataFrame(suite_a_rows).to_csv(os.path.join(DERIVED_DIR, "sovereign_customs_consolidation_decomposition.csv"), index=False)

    # -------------------------------------------------------------------------
    # SUITE B: Income-Stratified Luxury Demand & Clade-Specific Aquaculture Moderation (RQ2 / Eq. 4c & S4d)
    # -------------------------------------------------------------------------
    print("\n2. [Suite B] Estimating Dual-IV Cross-Partial Moderation Models (Eq. 4c & ODE Eq. S4d)...")
    conc = pd.read_csv(os.path.join(DERIVED_DIR, "global_hs6_conservation_concordance.csv"))
    conc["HS6_Code"] = conc["HS6_Code"].astype(str).str.zfill(6)
    df_m = raw_full.merge(conc[["HS6_Code", "Conservation_Category"]].drop_duplicates("HS6_Code"), on="HS6_Code", how="inner")
    df_m = df_m[(df_m["Weight_KG"] > 10) & (df_m["Trade_Value_USD"] > 10) & (df_m["Reporter_ISO3"] != df_m["Partner_ISO3"])].copy()

    m_exp = df_m[df_m["Trade_Flow"] == "Export"].groupby(
        ["Year", "HS6_Code", "Conservation_Category", "Reporter_ISO3", "Partner_ISO3"], as_index=False
    ).agg(Exp_KG=("Weight_KG", "sum"), Exp_USD=("Real_USD", "sum"))
    tot_exp = m_exp.groupby(["Year", "HS6_Code", "Reporter_ISO3"])["Exp_KG"].transform("sum")
    m_exp["LOO_Supply_KG"] = tot_exp - m_exp["Exp_KG"]

    m_imp = df_m[df_m["Trade_Flow"] == "Import"].groupby(
        ["Year", "HS6_Code", "Partner_ISO3", "Reporter_ISO3"], as_index=False
    ).agg(Imp_KG=("Weight_KG", "sum"), Imp_USD=("Real_USD", "sum")).rename(
        columns={"Partner_ISO3": "Reporter_ISO3", "Reporter_ISO3": "Partner_ISO3"}
    )
    tot_imp = m_imp.groupby(["Year", "HS6_Code", "Partner_ISO3"])["Imp_USD"].transform("sum")
    m_imp["LOO_Demand_USD"] = tot_imp - m_imp["Imp_USD"]

    mir = pd.merge(m_exp, m_imp, on=["Year", "HS6_Code", "Reporter_ISO3", "Partner_ISO3"], how="inner")
    mir["Exp_UV"] = mir["Exp_USD"] / mir["Exp_KG"]
    mir = mir[(mir["Exp_UV"] >= 0.10) & (mir["Exp_UV"] <= 5000.0) & (~mir["HS6_Code"].isin(["160417", "030812"]))].copy()
    mir["route_id"] = mir["Reporter_ISO3"] + "_" + mir["Partner_ISO3"] + "_" + mir["HS6_Code"]
    cnts = mir.groupby("route_id")["Year"].transform("count")
    mir = mir[cnts >= 3].copy()

    # Add 30-year BACI HS92 routes for Sturgeons & Caviar (160430) and Chondrichthyans (030265, 030375)
    baci_raw_b = pd.read_parquet(os.path.join(PROJ_DIR, "data", "raw", "baci_hs92_seafood_1995_2024.parquet"))
    baci_raw_b["HS92_Code"] = baci_raw_b["HS92_Code"].astype(str).str.zfill(6)
    baci_raw_b = baci_raw_b[
        (baci_raw_b["Year"].between(1995, 2024))
        & (baci_raw_b["Exporter_ISO3"] != baci_raw_b["Importer_ISO3"])
        & (baci_raw_b["HS92_Code"].isin(["160430", "030265", "030375"]))
        & (baci_raw_b["Quantity_t"] > 0.01)
    ].copy()
    baci_raw_b["Conservation_Category"] = np.where(
        baci_raw_b["HS92_Code"] == "160430",
        "Sturgeons & Caviar (Acipenseriformes) [BACI 30yr]",
        "Chondrichthyans (Sharks & Rays) [BACI 30yr]"
    )
    baci_raw_b["Clade_Match"] = np.where(
        baci_raw_b["HS92_Code"] == "160430",
        "Sturgeons & Caviar (Acipenseriformes)",
        "Chondrichthyans (Sharks & Rays)"
    )
    baci_raw_b["HS6_Code"] = baci_raw_b["HS92_Code"]
    baci_raw_b["Reporter_ISO3"] = baci_raw_b["Exporter_ISO3"]
    baci_raw_b["Partner_ISO3"] = baci_raw_b["Importer_ISO3"]
    baci_raw_b["Exp_KG"] = baci_raw_b["Quantity_t"].astype(float) * 1000.0
    baci_raw_b["Imp_KG"] = baci_raw_b["Exp_KG"]
    baci_raw_b["Real_Factor"] = BASE_CPI / baci_raw_b["Year"].map(CPI_DEFLATOR).fillna(BASE_CPI)
    baci_raw_b["Exp_USD"] = baci_raw_b["Value_USD"].astype(float) * baci_raw_b["Real_Factor"]
    baci_raw_b["Exp_UV"] = baci_raw_b["Exp_USD"] / baci_raw_b["Exp_KG"]
    baci_raw_b = baci_raw_b[(baci_raw_b["Exp_UV"] >= 0.10) & (baci_raw_b["Exp_UV"] <= 5000.0)].copy()
    tot_b_exp = baci_raw_b.groupby(["Year", "HS6_Code", "Reporter_ISO3"])["Exp_KG"].transform("sum")
    baci_raw_b["LOO_Supply_KG"] = np.maximum(tot_b_exp - baci_raw_b["Exp_KG"], 1.0)
    tot_b_imp = baci_raw_b.groupby(["Year", "HS6_Code", "Partner_ISO3"])["Exp_USD"].transform("sum")
    baci_raw_b["LOO_Demand_USD"] = np.maximum(tot_b_imp - baci_raw_b["Exp_USD"], 1.0)
    baci_raw_b["route_id"] = "BACI_" + baci_raw_b["Reporter_ISO3"] + "_" + baci_raw_b["Partner_ISO3"] + "_" + baci_raw_b["HS6_Code"]
    b_cnts = baci_raw_b.groupby("route_id")["Year"].transform("count")
    baci_raw_b = baci_raw_b[b_cnts >= 3].copy()

    mir["Clade_Match"] = mir["Conservation_Category"].replace({
        "Demersal Commercial Taxonomic Groups": "Demersal Commercial Controls"
    })

    # Load exporter clade-specific FAO Aquaculture Share (and national fallback) + importer GDPpc
    clade_aqua = pd.read_csv(os.path.join(DERIVED_DIR, "fao_country_clade_aquaculture_share_panel.csv")).rename(
        columns={"ISO3": "Reporter_ISO3", "Clade": "Clade_Match"}
    )
    clade_aqua_mean = clade_aqua.groupby(["Reporter_ISO3", "Clade_Match"], as_index=False)["Clade_Aqua_Share"].mean().rename(
        columns={"Clade_Aqua_Share": "Clade_Aqua_Share_Mean"}
    )
    exp_aqua = panel_gov[["ISO3", "Year", "FAO_Aqua_Share", "WB_Income_Group", "Gov_Capacity_Index"]].rename(
        columns={"ISO3": "Reporter_ISO3", "FAO_Aqua_Share": "Nat_Aqua_Share", "WB_Income_Group": "Exp_Income_Group", "Gov_Capacity_Index": "Exp_Gov"}
    )
    imp_macro = cs[["ISO3", "WB_Income_Group", "GDP_Per_Capita_Real_USD", "Gov_Capacity_Index"]].rename(
        columns={"ISO3": "Partner_ISO3", "WB_Income_Group": "Imp_Income_Group", "GDP_Per_Capita_Real_USD": "Imp_GDPpc", "Gov_Capacity_Index": "Imp_Gov"}
    )

    common_b_cols = ["Year", "HS6_Code", "Conservation_Category", "Clade_Match", "Reporter_ISO3", "Partner_ISO3", "route_id", "Exp_KG", "Imp_KG", "Exp_UV", "LOO_Supply_KG", "LOO_Demand_USD"]
    mir_combined = pd.concat([mir[common_b_cols], baci_raw_b[common_b_cols]], ignore_index=True)
    mir_combined = (
        mir_combined.merge(exp_aqua, on=["Reporter_ISO3", "Year"], how="left")
        .merge(clade_aqua[["Reporter_ISO3", "Clade_Match", "Year", "Clade_Aqua_Share"]], on=["Reporter_ISO3", "Clade_Match", "Year"], how="left")
        .merge(clade_aqua_mean, on=["Reporter_ISO3", "Clade_Match"], how="left")
        .merge(imp_macro, on="Partner_ISO3", how="left")
    )
    mir_combined["Clade_Aqua_Share_Filled"] = mir_combined["Clade_Aqua_Share"].fillna(mir_combined["Clade_Aqua_Share_Mean"]).fillna(0.0)
    mir_combined["Exp_Aqua_Share"] = mir_combined["Clade_Aqua_Share_Filled"]
    mir_combined["Nat_Aqua_Share"] = mir_combined["Nat_Aqua_Share"].fillna(0.0)
    mir_combined["ln_Imp_GDPpc_c"] = np.log(np.maximum(mir_combined["Imp_GDPpc"].fillna(15000.0), 100.0)) - np.log(15000.0)
    mir_combined["ln_p"] = np.log(mir_combined["Exp_UV"])
    mir_combined["ln_q_exp"] = np.log(mir_combined["Exp_KG"])
    mir_combined["ln_q_imp"] = np.where(
        mir_combined["route_id"].str.startswith("BACI_"),
        np.log(np.maximum(mir_combined["LOO_Supply_KG"], 1.0)),
        np.log(mir_combined["Imp_KG"])
    )
    mir_combined["ln_loo_m"] = np.log(np.maximum(mir_combined["LOO_Demand_USD"], 1.0))

    suite_b_rows = []
    luxury_clades = [
        "Sea Cucumbers (Holothuroidea)",
        "High-Trophic Tunas (Thunnus spp.)",
        "Diadromous Eels (Anguilla spp.)"
    ]
    b_scopes = [
        ("Pooled Luxury Threatened Clades (Holothuroidea, Thunnus, Anguilla)", mir_combined[mir_combined["Conservation_Category"].isin(luxury_clades)].copy(), "Nat_Aqua_Share"),
        ("Sea Cucumbers (Holothuroidea) [Nat. Aqua Share Robustness]", mir_combined[mir_combined["Conservation_Category"] == "Sea Cucumbers (Holothuroidea)"].copy(), "Nat_Aqua_Share"),
        ("Sea Cucumbers (Holothuroidea)", mir_combined[mir_combined["Conservation_Category"] == "Sea Cucumbers (Holothuroidea)"].copy(), "Exp_Aqua_Share"),
        ("High-Trophic Tunas (Thunnus spp.)", mir_combined[mir_combined["Conservation_Category"] == "High-Trophic Tunas (Thunnus spp.)"].copy(), "Exp_Aqua_Share"),
        ("Diadromous Eels (Anguilla spp.)", mir_combined[mir_combined["Conservation_Category"] == "Diadromous Eels (Anguilla spp.)"].copy(), "Exp_Aqua_Share"),
        ("Sturgeons & Caviar (Acipenseriformes) [BACI 30yr LOO-IV]", mir_combined[mir_combined["Conservation_Category"] == "Sturgeons & Caviar (Acipenseriformes) [BACI 30yr]"].copy(), "Exp_Aqua_Share"),
        ("Chondrichthyans (Sharks & Rays) [BACI 30yr LOO-IV]", mir_combined[mir_combined["Conservation_Category"] == "Chondrichthyans (Sharks & Rays) [BACI 30yr]"].copy(), "Exp_Aqua_Share"),
        ("Bulk Demersal Control Clades", mir_combined[mir_combined["Conservation_Category"] == "Demersal Commercial Taxonomic Groups"].copy(), "Exp_Aqua_Share"),
    ]
    for scope_label, sub_b, aqua_col in b_scopes:
        if len(sub_b) < 100:
            continue
        # Remove singletons in route or Year
        while True:
            n0 = len(sub_b)
            sub_b = sub_b[sub_b.groupby("route_id")["Year"].transform("count") >= 2]
            sub_b = sub_b[sub_b.groupby("Year")["route_id"].transform("count") >= 2]
            if len(sub_b) == n0 or len(sub_b) < 50:
                break
        if len(sub_b) < 100:
            continue

        has_aqua_var = float(sub_b[aqua_col].std()) > 1e-6
        sub_b["ln_q_x_Aqua"] = sub_b["ln_q_exp"] * sub_b[aqua_col]
        sub_b["z_q_x_Aqua"] = sub_b["ln_q_imp"] * sub_b[aqua_col]
        sub_b["ln_q_x_GDPpc"] = sub_b["ln_q_exp"] * sub_b["ln_Imp_GDPpc_c"]
        sub_b["z_q_x_GDPpc"] = sub_b["ln_q_imp"] * sub_b["ln_Imp_GDPpc_c"]

        # Exact converged 2WFE alternating-projection demeaning (tol = 1e-11, max_iter = 200)
        dm_cols = ["ln_p", "ln_q_exp", "ln_q_imp", "ln_loo_m", "ln_q_x_GDPpc", "z_q_x_GDPpc"]
        if has_aqua_var:
            dm_cols += ["ln_q_x_Aqua", "z_q_x_Aqua"]
        for col in dm_cols:
            vals = sub_b[col].values.copy()
            for _ in range(200):
                prev = vals.copy()
                sub_b[col] = vals - sub_b.groupby("route_id")[col].transform("mean").values
                sub_b[col] = sub_b[col].values - sub_b.groupby("Year")[col].transform("mean").values
                vals = sub_b[col].values
                if np.max(np.abs(vals - prev)) < 1e-11:
                    break

        Y = sub_b["ln_p"].values
        if has_aqua_var:
            X_endog = sub_b[["ln_q_exp", "ln_q_x_Aqua", "ln_q_x_GDPpc", "ln_loo_m"]].values
            Z_inst = sub_b[["ln_q_imp", "z_q_x_Aqua", "z_q_x_GDPpc", "ln_loo_m"]].values
        else:
            X_endog = sub_b[["ln_q_exp", "ln_q_x_GDPpc", "ln_loo_m"]].values
            Z_inst = sub_b[["ln_q_imp", "z_q_x_GDPpc", "ln_loo_m"]].values

        PZ_X = Z_inst @ np.linalg.lstsq(Z_inst, X_endog, rcond=None)[0]
        beta_iv = np.linalg.lstsq(PZ_X, Y, rcond=None)[0]
        resid = Y - X_endog @ beta_iv

        routes = sub_b["route_id"].values
        uniq_r, r_idx = np.unique(routes, return_inverse=True)
        G = len(uniq_r)
        N, K = X_endog.shape
        bread = np.linalg.pinv(PZ_X.T @ PZ_X)
        scores = np.zeros((G, K))
        for k_i in range(K):
            scores[:, k_i] = np.bincount(r_idx, weights=PZ_X[:, k_i] * resid)
        meat = scores.T @ scores * (G / max(G - 1, 1)) * ((N - 1) / max(N - K, 1))
        vcov = bread @ meat @ bread
        se_iv = np.sqrt(np.maximum(np.diag(vcov), 1e-12))
        t_iv = beta_iv / se_iv
        p_iv = 2.0 * (1.0 - stats.t.cdf(np.abs(t_iv), df=max(G - 1, 1)))

        suite_b_rows.append({
            "Clade_Scope": scope_label,
            "Aqua_Measure": aqua_col if has_aqua_var else "Zero_Wild_Only",
            "N_Obs": int(N),
            "N_Routes": int(G),
            "Beta_ln_Q_Base": round(float(beta_iv[0]), 4),
            "SE_ln_Q_Base": round(float(se_iv[0]), 4),
            "p_ln_Q_Base": round(float(p_iv[0]), 4),
            "Theta2_ln_Q_x_ExporterAquaShare": round(float(beta_iv[1]), 4) if has_aqua_var else np.nan,
            "SE_Theta2_Aqua": round(float(se_iv[1]), 4) if has_aqua_var else np.nan,
            "p_Theta2_Aqua": round(float(p_iv[1]), 4) if has_aqua_var else np.nan,
            "Theta1_ln_Q_x_ImporterLnGDPpc": round(float(beta_iv[2 if has_aqua_var else 1]), 4),
            "SE_Theta1_GDPpc": round(float(se_iv[2 if has_aqua_var else 1]), 4),
            "p_Theta1_GDPpc": round(float(p_iv[2 if has_aqua_var else 1]), 4),
            "Gamma_LOO_Demand": round(float(beta_iv[-1]), 4)
        })
    df_suite_b = pd.DataFrame(suite_b_rows)
    df_suite_b.to_csv(os.path.join(DERIVED_DIR, "sovereign_allee_income_aquaculture_moderation.csv"), index=False)
    print("   -> Suite B Dual-IV Cross-Partial Results:")
    print(df_suite_b[["Clade_Scope", "N_Obs", "Beta_ln_Q_Base", "Theta2_ln_Q_x_ExporterAquaShare", "p_Theta2_Aqua", "Theta1_ln_Q_x_ImporterLnGDPpc", "p_Theta1_GDPpc"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # SUITE C: Governance-Moderated Gravity PPML & WTO Tariff-vs.-CITES Mirror Evasion (RQ3 / Eq. 6d, 9d & S3d)
    # -------------------------------------------------------------------------
    print("\n3. [Suite C] Estimating Governance-Moderated Gravity PPML & Tariff-vs.-CITES Evasion Models...")
    # Model C1: Bilateral Zero-Retaining Gravity PPML with Governance & PSMA Moderation
    baci_path = os.path.join(PROJ_DIR, "data", "raw", "baci_hs92_seafood_1995_2024.parquet")
    baci = pd.read_parquet(baci_path)
    baci["HS92_Code"] = baci["HS92_Code"].astype(str).str.zfill(6)
    baci = baci[(baci["Year"] >= 1995) & (baci["Year"] <= 2024) & (baci["Exporter_ISO3"] != baci["Importer_ISO3"])].copy()
    baci["Quantity_MT"] = baci["Quantity_t"].astype(float)
    baci["route_id"] = baci["Exporter_ISO3"] + "_" + baci["Importer_ISO3"] + "_" + baci["HS92_Code"]

    th_exp = bilat_all[
        (bilat_all["Trade_Flow"] == "Export") &
        (bilat_all["Year"] <= 2024) &
        (bilat_all["Reporter_ISO3"] != bilat_all["Partner_ISO3"]) &
        (~bilat_all["HS6_Code"].isin(["160417", "030812"]))
    ].copy()
    th_exp["Quantity_MT"] = th_exp["Weight_KG"] / 1000.0
    th_exp["route_id"] = th_exp["Reporter_ISO3"] + "_" + th_exp["Partner_ISO3"] + "_" + th_exp["HS6_Code"]
    th_exp["Exporter_ISO3"] = th_exp["Reporter_ISO3"]
    th_exp["Importer_ISO3"] = th_exp["Partner_ISO3"]

    gov_lookup = dict(zip(cs["ISO3"], cs["Gov_Capacity_Index"]))
    psma_lookup = dict(zip(cs["ISO3"], cs["SDG14_6_1_IUU_PSMA_Score"]))

    ppml_gov_rows = []
    for reg_name, int_yr, src_type in [
        ("High-Trophic Tunas (Thunnus spp.)", 2008, "comtrade"),
        ("Diadromous Eels (Anguilla spp.)", 2009, "comtrade"),
        ("Chondrichthyans (Sharks & Rays)", 2014, "baci"),
        ("Sturgeons & Caviar (Acipenseriformes)", 2006, "baci"),
        ("Sea Cucumbers (Holothuroidea)", 2020, "comtrade")
    ]:
        if src_type == "baci":
            t_codes = ["030265", "030375"] if "Chondrichthyans" in reg_name else ["160430"]
            c_codes = ["030250", "030262", "030263", "030360", "030372", "030373", "030378"]
            sub_g = baci[baci["HS92_Code"].isin(t_codes + c_codes)][["Year", "route_id", "Exporter_ISO3", "Importer_ISO3", "HS92_Code", "Quantity_MT"]].copy()
            sub_g["Treated"] = sub_g["HS92_Code"].isin(t_codes).astype(int)
        else:
            min_cat_yr = int(th_exp[th_exp["Conservation_Category"] == reg_name]["Year"].min())
            sub_g = th_exp[
                (th_exp["Conservation_Category"].isin([reg_name, "Demersal Commercial Taxonomic Groups"])) &
                (th_exp["Year"] >= min_cat_yr)
            ][["Year", "route_id", "Exporter_ISO3", "Importer_ISO3", "Conservation_Category", "Quantity_MT"]].copy()
            sub_g["Treated"] = (sub_g["Conservation_Category"] == reg_name).astype(int)

        grp = sub_g.groupby(["route_id", "Exporter_ISO3", "Importer_ISO3", "Treated", "Year"], as_index=False)["Quantity_MT"].sum()
        cnt = grp.groupby("route_id")["Year"].nunique()
        grp = grp[grp["route_id"].isin(cnt[cnt >= 4].index)].copy()
        t_routes = grp[grp["Treated"] == 1]["route_id"].unique()
        c_routes = grp[grp["Treated"] == 0].groupby("route_id")["Quantity_MT"].sum().sort_values(ascending=False).head(1500).index
        grp = grp[grp["route_id"].isin(np.concatenate([t_routes, c_routes]))].copy()

        yrs = sorted(grp["Year"].unique())
        rmeta = grp[["route_id", "Exporter_ISO3", "Importer_ISO3", "Treated"]].drop_duplicates("route_id")
        bal = rmeta.assign(_k=1).merge(pd.DataFrame({"Year": yrs, "_k": 1}), on="_k").drop(columns="_k")
        bal = bal.merge(grp[["route_id", "Year", "Quantity_MT"]], on=["route_id", "Year"], how="left")
        bal["Quantity_MT"] = bal["Quantity_MT"].fillna(0.0)

        bal["Exp_Gov"] = bal["Exporter_ISO3"].map(gov_lookup).fillna(0.0)
        bal["Imp_Gov"] = bal["Importer_ISO3"].map(gov_lookup).fillna(0.0)
        bal["Corridor_Gov"] = 0.5 * (bal["Exp_Gov"] + bal["Imp_Gov"])
        # Center Corridor_Gov across treated routes so Beta_Treat_Post is evaluated at mean corridor governance
        mean_cgov = float(bal.loc[bal["Treated"] == 1, "Corridor_Gov"].mean())
        bal["Corridor_Gov_c"] = bal["Corridor_Gov"] - mean_cgov
        bal["Post"] = (bal["Year"] >= int_yr).astype(int)
        bal["Treat_Post"] = bal["Treated"] * bal["Post"]
        bal["Treat_Post_x_Gov"] = bal["Treat_Post"] * bal["Corridor_Gov_c"]

        fit_g = pf.fepois(
            "Quantity_MT ~ Treat_Post + Treat_Post_x_Gov | route_id + Year",
            data=bal, vcov={"CRV1": "route_id"}
        )
        b0 = float(fit_g.coef()["Treat_Post"])
        se0 = float(fit_g.se()["Treat_Post"])
        p0 = float(fit_g.pvalue()["Treat_Post"])
        bg = float(fit_g.coef()["Treat_Post_x_Gov"])
        seg = float(fit_g.se()["Treat_Post_x_Gov"])
        pg = float(fit_g.pvalue()["Treat_Post_x_Gov"])

        ppml_gov_rows.append({
            "Regime": reg_name,
            "Intervention_Year": int_yr,
            "N_Cells": int(fit_g._N),
            "Beta_Treat_Post_AtMeanGov": round(b0, 4),
            "SE_Treat_Post": round(se0, 4),
            "p_Treat_Post": round(p0, 4),
            "CrossPartial_TreatPost_x_CorridorGov": round(bg, 4),
            "SE_CrossPartial_Gov": round(seg, 4),
            "p_CrossPartial_Gov": round(pg, 4),
            "Implied_Pct_Change_HighGov_Plus1SD": round(100.0 * (np.exp(b0 + 1.0 * bg) - 1.0), 2),
            "Implied_Pct_Change_LowGov_Minus1SD": round(100.0 * (np.exp(b0 - 1.0 * bg) - 1.0), 2)
        })
    df_ppml_gov = pd.DataFrame(ppml_gov_rows)
    print("   -> Model C1 Governance-Moderated Gravity PPML Results:")
    print(df_ppml_gov[["Regime", "Beta_Treat_Post_AtMeanGov", "CrossPartial_TreatPost_x_CorridorGov", "p_CrossPartial_Gov", "Implied_Pct_Change_HighGov_Plus1SD", "Implied_Pct_Change_LowGov_Minus1SD"]].to_string(index=False))

    # Model C2: Fisman-Wei WTO Tariff Evasion vs. CITES Treaty Evasion & Corridor Governance (Eq. 9d & S3d)
    # Construct matched + unmatched mirror cells across Exporter x Importer x HS6 x Year with time-varying CITES & HS6-level WTO tariffs
    raw_recent = raw_full[raw_full["Year"] >= 2012].copy()
    raw_recent = raw_recent.merge(
        conc[["HS6_Code", "Conservation_Category", "CITES_Status", "Threatened_Taxa_Flag"]].drop_duplicates("HS6_Code"),
        on="HS6_Code", how="left"
    )
    cites_yr_active = (
        ((raw_recent["Conservation_Category"] == "Sturgeons & Caviar (Acipenseriformes)") & (raw_recent["Year"] >= 1998)) |
        ((raw_recent["Conservation_Category"] == "Diadromous Eels (Anguilla spp.)") & (raw_recent["Year"] >= 2009)) |
        ((raw_recent["Conservation_Category"] == "Chondrichthyans (Sharks & Rays)") & (raw_recent["Year"] >= 2014)) |
        ((raw_recent["Conservation_Category"] == "Sea Cucumbers (Holothuroidea)") & (raw_recent["Year"] >= 2020))
    )
    raw_recent["Is_CITES"] = cites_yr_active.astype(int)
    raw_recent["Pooling_Tau"] = 1.0 - raw_recent["SCRI"]

    exp_c = raw_recent[(raw_recent["Trade_Flow"] == "Export") & (raw_recent["Reporter_ISO3"] != raw_recent["Partner_ISO3"])].groupby(
        ["Year", "HS6_Code", "Reporter_ISO3", "Partner_ISO3", "Is_Pooled", "Pooling_Tau", "Is_CITES", "Is_Ch16"], as_index=False
    ).agg(Exp_KG=("Weight_KG", "sum"), Exp_USD=("Real_USD", "sum"))

    imp_c = raw_recent[(raw_recent["Trade_Flow"] == "Import") & (raw_recent["Reporter_ISO3"] != raw_recent["Partner_ISO3"])].groupby(
        ["Year", "HS6_Code", "Partner_ISO3", "Reporter_ISO3", "Is_Pooled", "Pooling_Tau", "Is_CITES", "Is_Ch16"], as_index=False
    ).agg(Imp_KG=("Weight_KG", "sum"), Imp_USD=("Real_USD", "sum")).rename(
        columns={"Partner_ISO3": "Reporter_ISO3", "Reporter_ISO3": "Partner_ISO3"}
    )

    mir_c = pd.merge(
        exp_c, imp_c,
        on=["Year", "HS6_Code", "Reporter_ISO3", "Partner_ISO3", "Is_Pooled", "Pooling_Tau", "Is_CITES", "Is_Ch16"],
        how="outer"
    )
    mir_c["Exp_KG"] = mir_c["Exp_KG"].fillna(0.0)
    mir_c["Imp_KG"] = mir_c["Imp_KG"].fillna(0.0)
    mir_c = mir_c[(mir_c["Exp_KG"] + mir_c["Imp_KG"]) > 100.0].copy()
    mir_c["Symmetric_MDI"] = np.abs(mir_c["Exp_KG"] - mir_c["Imp_KG"]) / (mir_c["Exp_KG"] + mir_c["Imp_KG"])
    mir_c["Is_Orphan"] = ((mir_c["Exp_KG"] == 0.0) | (mir_c["Imp_KG"] == 0.0)).astype(float)

    # Merge exact HS6-by-importer-by-year WTO MFN duties (and bilateral preferential margins), falling back to importer mean
    wto_mfn_raw = pd.read_csv(os.path.join(DB_DIR, "WTO_Workspace", "data", "wto_seafood_mfn_tariffs_multicountry_2015_2024.csv"))
    wto_mfn_raw["HS6_Code"] = wto_mfn_raw["HS_Code"].astype(str).str.strip().str.zfill(6)
    name_to_iso = dict(zip(cs["Country_Name"].str.lower(), cs["ISO3"]))
    name_to_iso.update({
        "european union": "EU27", "united states of america": "USA", "united states": "USA",
        "china": "CHN", "japan": "JPN", "korea, republic of": "KOR", "republic of korea": "KOR",
        "viet nam": "VNM", "russian federation": "RUS", "united kingdom": "GBR",
        "taipei, chinese": "TWN", "hong kong, china": "HKG", "turkiye": "TUR", "türkiye": "TUR",
        "saudi arabia, kingdom of": "SAU"
    })
    wto_mfn_raw["ISO3"] = wto_mfn_raw["Reporter_Name"].astype(str).str.strip().str.lower().map(name_to_iso)
    wto_hs6 = wto_mfn_raw[(wto_mfn_raw["HS6_Code"].str.startswith(("03", "16"))) & (wto_mfn_raw["ISO3"].notna())][
        ["ISO3", "HS6_Code", "Year", "MFN_Applied_Duty_Pct"]
    ].copy()
    eu_members = sorted(list(cs.loc[cs["Is_EU27"] == 1, "ISO3"].unique()))
    eu_rows = wto_hs6[wto_hs6["ISO3"] == "EU27"]
    if not eu_rows.empty and eu_members:
        eu_exp = eu_rows.drop(columns=["ISO3"]).assign(_k=1).merge(pd.DataFrame({"ISO3": eu_members, "_k": 1}), on="_k").drop(columns="_k")
        wto_hs6 = pd.concat([wto_hs6[wto_hs6["ISO3"] != "EU27"], eu_exp], ignore_index=True)

    wto_hs6_yr = wto_hs6.groupby(["ISO3", "HS6_Code", "Year"], as_index=False)["MFN_Applied_Duty_Pct"].mean().rename(
        columns={"ISO3": "Partner_ISO3", "MFN_Applied_Duty_Pct": "Duty_HS6_Yr"}
    )
    wto_hs6_imp = wto_hs6.groupby(["ISO3", "HS6_Code"], as_index=False)["MFN_Applied_Duty_Pct"].mean().rename(
        columns={"ISO3": "Partner_ISO3", "MFN_Applied_Duty_Pct": "Duty_HS6_Mean"}
    )

    mfn_map = dict(zip(cs["ISO3"], cs["WTO_MFN_Seafood_Tariff_Pct"]))
    inc_grp_map = dict(zip(cs["ISO3"], cs["WB_Income_Group"]))
    mir_c = mir_c.merge(wto_hs6_yr, on=["Partner_ISO3", "HS6_Code", "Year"], how="left").merge(
        wto_hs6_imp, on=["Partner_ISO3", "HS6_Code"], how="left"
    )
    mir_c["Imp_Tariff_Pct"] = mir_c["Duty_HS6_Yr"].fillna(mir_c["Duty_HS6_Mean"]).fillna(mir_c["Partner_ISO3"].map(mfn_map)).fillna(10.5)
    mir_c["Imp_MFN_Tariff_Std"] = (mir_c["Imp_Tariff_Pct"] - 10.5) / 8.0
    mir_c["Exp_Gov"] = mir_c["Reporter_ISO3"].map(gov_lookup).fillna(0.0)
    mir_c["Imp_Gov"] = mir_c["Partner_ISO3"].map(gov_lookup).fillna(0.0)
    mir_c["Min_Corridor_Gov"] = np.minimum(mir_c["Exp_Gov"], mir_c["Imp_Gov"])
    mir_c["Exp_South"] = (~mir_c["Reporter_ISO3"].map(inc_grp_map).eq("HIC")).astype(int)
    mir_c["Imp_North"] = (mir_c["Partner_ISO3"].map(inc_grp_map).eq("HIC")).astype(int)
    mir_c["Corridor_Type"] = np.select(
        [
            (mir_c["Exp_South"] == 1) & (mir_c["Imp_North"] == 1),
            (mir_c["Exp_South"] == 1) & (mir_c["Imp_North"] == 0),
            (mir_c["Exp_South"] == 0) & (mir_c["Imp_North"] == 1)
        ],
        ["South_to_North", "South_to_South", "North_to_North"],
        default="North_to_South"
    )
    mir_c["dyad_id"] = mir_c["Reporter_ISO3"] + "_" + mir_c["Partner_ISO3"]
    mir_c["Pooled_x_Tariff"] = mir_c["Is_Pooled"] * mir_c["Imp_MFN_Tariff_Std"]
    mir_c["Pooled_x_CITES"] = mir_c["Is_Pooled"] * mir_c["Is_CITES"]
    mir_c["CITES_x_MinGov"] = mir_c["Is_CITES"] * mir_c["Min_Corridor_Gov"]

    # Also merge GFW transshipment / dark-fleet exposure & FAO bilateral border rejections for auxiliary validation
    gfw_exp_df = pd.read_csv(os.path.join(DERIVED_DIR, "gfw_port_transshipment_exposure.csv"))
    rej_dyad = pd.read_csv(os.path.join(DERIVED_DIR, "fao_bilateral_border_rejections_dyad.csv")).rename(
        columns={"Exporter_ISO3": "Reporter_ISO3", "Importer_ISO3": "Partner_ISO3"}
    )
    gfw_trans_map = dict(zip(gfw_exp_df["ISO3"], np.log1p(gfw_exp_df["GFW_Transshipment_MT"])))
    mir_c["Exp_ln_GFW_Transship"] = mir_c["Reporter_ISO3"].map(gfw_trans_map).fillna(0.0)
    mir_c = mir_c.merge(rej_dyad[["Reporter_ISO3", "Partner_ISO3", "Dyad_Doc_Fraud_Rejections"]], on=["Reporter_ISO3", "Partner_ISO3"], how="left")
    mir_c["Dyad_Has_Doc_Rejection"] = (mir_c["Dyad_Doc_Fraud_Rejections"].fillna(0.0) > 0).astype(float)

    # Estimate high-dimensional FE models via pyfixest (absorbing Exporter x Importer dyad + Year FE)
    mir_c["CITES_x_GFW_Transship"] = mir_c["Is_CITES"] * mir_c["Exp_ln_GFW_Transship"]
    mir_c["CITES_x_DocRej"] = mir_c["Is_CITES"] * mir_c["Dyad_Has_Doc_Rejection"]

    fit_mdi_all = pf.feols(
        "Symmetric_MDI ~ Is_Pooled + Is_CITES + Imp_MFN_Tariff_Std + Pooled_x_Tariff + Pooled_x_CITES + CITES_x_MinGov + Is_Ch16 | dyad_id + Year",
        data=mir_c, vcov={"CRV1": "dyad_id"}
    )
    fit_orph_all = pf.feols(
        "Is_Orphan ~ Is_Pooled + Is_CITES + Imp_MFN_Tariff_Std + Pooled_x_Tariff + Pooled_x_CITES + CITES_x_MinGov + Is_Ch16 | dyad_id + Year",
        data=mir_c, vcov={"CRV1": "dyad_id"}
    )
    fit_mdi_gfw_rej = pf.feols(
        "Symmetric_MDI ~ Is_Pooled + Is_CITES + Imp_MFN_Tariff_Std + Pooled_x_Tariff + Pooled_x_CITES + CITES_x_MinGov + CITES_x_GFW_Transship + CITES_x_DocRej + Is_Ch16 | dyad_id + Year",
        data=mir_c, vcov={"CRV1": "dyad_id"}
    )

    evasion_rows = []
    for m_tag, f_obj in [
        ("Symmetric_MDI_Intensive_And_Extensive", fit_mdi_all),
        ("Unilateral_Orphan_Extensive_LPM", fit_orph_all),
        ("Symmetric_MDI_GFW_Transshipment_and_Rejections", fit_mdi_gfw_rej),
    ]:
        for p_name in f_obj.coef().index:
            evasion_rows.append({
                "Model": m_tag,
                "Corridor_Stratum": "All_Global_Corridors",
                "Parameter": p_name,
                "Estimate": round(float(f_obj.coef()[p_name]), 5),
                "Std_Error": round(float(f_obj.se()[p_name]), 5),
                "t_stat": round(float(f_obj.tstat()[p_name]), 3),
                "p_value": float(f_obj.pvalue()[p_name]),
                "N_Obs": int(f_obj._N)
            })

    # Stratified by Corridor Type (South->North, South->South, North->North)
    for c_type in ["South_to_North", "South_to_South", "North_to_North"]:
        sub_ct = mir_c[mir_c["Corridor_Type"] == c_type]
        if len(sub_ct) > 1000:
            f_ct = pf.feols(
                "Symmetric_MDI ~ Is_Pooled + Is_CITES + Pooled_x_Tariff + Pooled_x_CITES + CITES_x_MinGov + Is_Ch16 | dyad_id + Year",
                data=sub_ct, vcov={"CRV1": "dyad_id"}
            )
            for p_name in f_ct.coef().index:
                evasion_rows.append({
                    "Model": "Symmetric_MDI_By_Corridor",
                    "Corridor_Stratum": c_type,
                    "Parameter": p_name,
                    "Estimate": round(float(f_ct.coef()[p_name]), 5),
                    "Std_Error": round(float(f_ct.se()[p_name]), 5),
                    "t_stat": round(float(f_ct.tstat()[p_name]), 3),
                    "p_value": float(f_ct.pvalue()[p_name]),
                    "N_Obs": int(f_ct._N)
                })

    df_evasion = pd.DataFrame(evasion_rows)
    df_evasion.to_csv(os.path.join(DERIVED_DIR, "governance_moderated_ppml_and_evasion_results.csv"), index=False)
    df_ppml_gov.to_csv(os.path.join(DERIVED_DIR, "governance_moderated_gravity_ppml_summary.csv"), index=False)
    print("   -> Model C2 Tariff-vs.-CITES Mirror Evasion Cross-Partials (Global):")
    print(df_evasion[df_evasion["Corridor_Stratum"] == "All_Global_Corridors"][["Model", "Parameter", "Estimate", "Std_Error", "t_stat", "p_value"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # SUITE D: Bilateral 4x4 Income & Bloc Matrices, CWON + FAO Capture Depletion (Eq. 11c/d) & SRECI CBDR Index (Eq. 13b)
    # -------------------------------------------------------------------------
    print("\n4. [Suite D] Computing 4x4 Income-Tier Exchange Matrix, Natural Capital / Wild Threatened Capture Models & CBDR SRECI Index...")
    # 1. Bilateral 4x4 World Bank Income-Tier Matrix from bilat_all Exports
    exp_flows = bilat_all[(bilat_all["Trade_Flow"] == "Export") & (bilat_all["Reporter_ISO3"] != bilat_all["Partner_ISO3"])].copy()
    exp_flows["Exp_Income"] = exp_flows["Reporter_ISO3"].map(inc_grp_map).fillna("UMC")
    exp_flows["Imp_Income"] = exp_flows["Partner_ISO3"].map(inc_grp_map).fillna("HIC")
    exp_flows["Is_Predator_TL38"] = (exp_flows["Trophic_Level"] >= 3.8).astype(int)
    exp_flows["TL_Weight_Row"] = exp_flows["Weight_KG"] * exp_flows["Trophic_Level"]
    exp_flows["Pred_Weight_Row"] = exp_flows["Weight_KG"] * exp_flows["Is_Predator_TL38"]
    exp_flows["Threat_USD_Row"] = exp_flows["Real_Trade_Value_USD"] * exp_flows["Threatened_Taxa_Flag"].astype(int)

    inc_bilat = exp_flows.groupby(["Exp_Income", "Imp_Income"], as_index=False).agg(
        Total_Real_USD_B=("Real_Trade_Value_USD", lambda s: s.sum() / 1e9),
        Total_Weight_MMT=("Weight_KG", lambda s: s.sum() / 1e9),
        TL_Weight_Sum=("TL_Weight_Row", "sum"),
        Pred_Weight_Sum=("Pred_Weight_Row", "sum"),
        Threat_USD_Sum=("Threat_USD_Row", "sum")
    )
    inc_bilat["Mean_MTL"] = inc_bilat["TL_Weight_Sum"] / np.maximum(inc_bilat["Total_Weight_MMT"] * 1e9, 1.0)
    inc_bilat["Predator_Mass_Share_Pct"] = 100.0 * inc_bilat["Pred_Weight_Sum"] / np.maximum(inc_bilat["Total_Weight_MMT"] * 1e9, 1.0)
    inc_bilat["Threatened_Value_Share_Pct"] = 100.0 * inc_bilat["Threat_USD_Sum"] / np.maximum(inc_bilat["Total_Real_USD_B"] * 1e9, 1.0)
    inc_bilat = inc_bilat.drop(columns=["TL_Weight_Sum", "Pred_Weight_Sum", "Threat_USD_Sum"])
    inc_bilat["Unit_Price_USD_kg"] = inc_bilat["Total_Real_USD_B"] / np.maximum(inc_bilat["Total_Weight_MMT"], 1e-6)
    inc_bilat.to_csv(os.path.join(DERIVED_DIR, "bilateral_income_and_bloc_transfer_matrix.csv"), index=False)

    # 2a. World Bank CWON Fisheries Natural Capital Subsample (14 reporting economies, Eq. 11c)
    wb_cwon = pd.read_csv(os.path.join(DB_DIR, "World_Bank_Workspace", "data", "WorldBank_Fisheries_Natural_Capital_Wealth.csv"))
    cwon_pc = wb_cwon[wb_cwon["indicator_id"] == "NW.NCA.FISH.PC"].copy()
    cwon_pc["ISO3"] = cwon_pc["country_iso3"].str.strip().str.upper()
    cwon_pc["Year"] = pd.to_numeric(cwon_pc["year"], errors="coerce")
    cwon_pc = cwon_pc.sort_values(["ISO3", "Year"])
    cwon_pc["ln_CWON_Fish_PC"] = np.log(np.maximum(cwon_pc["value"], 0.01))
    cwon_pc["d_ln_CWON_Fish_PC"] = cwon_pc.groupby("ISO3")["ln_CWON_Fish_PC"].diff()

    cwon_panel = cwon_pc.merge(
        sov_master[["ISO3", "TBT", "MTL_Export", "Gov_Capacity_Index", "SDG14_6_1_IUU_PSMA_Score", "WB_Income_Group", "OECD_MCS_to_Total_Support_Share", "GFW_Foreign_DWF_Share"]],
        on="ISO3", how="left"
    ).dropna(subset=["d_ln_CWON_Fish_PC", "TBT", "Gov_Capacity_Index"])
    cwon_panel["TBT_x_Gov"] = cwon_panel["TBT"] * cwon_panel["Gov_Capacity_Index"]
    m_cwon = smf.ols("d_ln_CWON_Fish_PC ~ TBT + Gov_Capacity_Index + TBT_x_Gov", data=cwon_panel).fit(cov_type="HC3")

    # 2b. Global FAO Wild Threatened Capture Stability Panel (170 countries, 1989-2024)
    fao_thr_cy = pd.read_csv(os.path.join(DERIVED_DIR, "fao_country_year_threatened_capture_panel.csv"))
    thr_panel = fao_thr_cy.merge(
        sov_master[["ISO3", "TBT", "Gov_Capacity_Index", "GFW_Foreign_DWF_Share", "OECD_MCS_to_Total_Support_Share", "OECD_CapEnh_to_Total_Support_Share"]],
        on="ISO3", how="inner"
    ).dropna(subset=["dln_Threatened_Capture", "TBT", "Gov_Capacity_Index"]).copy()
    thr_panel["TBT_x_Gov"] = thr_panel["TBT"] * thr_panel["Gov_Capacity_Index"]
    m_thr_glob = smf.ols(
        "dln_Threatened_Capture ~ TBT + Gov_Capacity_Index + TBT_x_Gov + GFW_Foreign_DWF_Share",
        data=thr_panel
    ).fit(cov_type="cluster", cov_kwds={"groups": thr_panel["ISO3"]})

    # 2c. OECD Fisheries Support Estimate (FSE) MCS vs. Capacity-Enhancing Subsidies Subsample (43 economies)
    oecd_sub = thr_panel.dropna(subset=["OECD_MCS_to_Total_Support_Share", "OECD_CapEnh_to_Total_Support_Share"]).copy()
    m_oecd_mcs = smf.ols(
        "dln_Threatened_Capture ~ OECD_MCS_to_Total_Support_Share + OECD_CapEnh_to_Total_Support_Share + Gov_Capacity_Index",
        data=oecd_sub
    ).fit(cov_type="cluster", cov_kwds={"groups": oecd_sub["ISO3"]}) if len(oecd_sub) > 50 else None

    # 2d. Cross-sectional Sovereign Trophic Drain Determinants across all active exporters
    sov_drain = active_sov.dropna(subset=["TBT", "Export_to_Import_Price_Ratio", "Gov_Capacity_Index"]).copy()
    sov_drain["TBT_x_Gov"] = sov_drain["TBT"] * sov_drain["Gov_Capacity_Index"]
    sov_drain["ln_Price_Ratio"] = np.log(np.clip(sov_drain["Export_to_Import_Price_Ratio"], 0.05, 50.0))
    m_drain = smf.ols("TBT ~ Gov_Capacity_Index + Is_SIDS + Is_LDC + GFW_Foreign_DWF_Share + Mean_FAO_Aqua_Share", data=sov_drain).fit(cov_type="HC3")

    cwon_rows = []
    d_models = [
        ("CWON_Fisheries_Natural_Capital_14Countries_Eq11c", m_cwon),
        ("Global_FAO_Threatened_Capture_Panel_170Countries_Eq11d", m_thr_glob),
        ("Sovereign_Trophic_Drain_Determinants", m_drain),
    ]
    if m_oecd_mcs is not None:
        d_models.append(("OECD_MCS_vs_CapEnh_Subsidy_Capture_Subsample", m_oecd_mcs))
    for m_lbl, m_o in d_models:
        for p in m_o.params.index:
            cwon_rows.append({
                "Model": m_lbl,
                "Parameter": p,
                "Estimate": round(float(m_o.params[p]), 4),
                "Std_Error_HC3": round(float(m_o.bse[p]), 4),
                "t_stat": round(float(m_o.tvalues[p]), 3),
                "p_value": float(m_o.pvalues[p]),
                "N_Obs": int(m_o.nobs),
                "R2": round(float(m_o.rsquared), 4)
            })
    pd.DataFrame(cwon_rows).to_csv(os.path.join(DERIVED_DIR, "brander_taylor_cwon_natural_capital_results.csv"), index=False)

    # 3. Sovereign Reform Exposure & Capacity Index (SRECI_r, Eq. 13b) across all active reporters
    # Compute each country's pooled threatened + CITES trade exposure
    bilat_all["Is_Thr_or_CITES"] = (bilat_all["Threatened_Taxa_Flag"] == True) | (bilat_all["CITES_Status"].astype(str).str.contains("Appendix", na=False))
    thr_all = bilat_all[bilat_all["Is_Thr_or_CITES"]].copy()
    thr_all["Pooled_Thr_USD_Row"] = thr_all["Real_Trade_Value_USD"] * thr_all["Is_Pooled_Code"]

    thr_by_r = thr_all[thr_all["Trade_Flow"] == "Export"].groupby("Reporter_ISO3", as_index=False).agg(
        Threatened_CITES_Export_USD=("Real_Trade_Value_USD", "sum"),
        Pooled_Threatened_CITES_Export_USD=("Pooled_Thr_USD_Row", "sum")
    ).rename(columns={"Reporter_ISO3": "ISO3"})

    thr_by_m = thr_all[thr_all["Trade_Flow"] == "Import"].groupby("Reporter_ISO3", as_index=False).agg(
        Threatened_CITES_Import_USD=("Real_Trade_Value_USD", "sum"),
        Pooled_Threatened_CITES_Import_USD=("Pooled_Thr_USD_Row", "sum")
    ).rename(columns={"Reporter_ISO3": "ISO3"})

    sreci_df = sov_master.merge(thr_by_r, on="ISO3", how="left").merge(thr_by_m, on="ISO3", how="left")
    sreci_df["Threatened_CITES_Export_USD"] = sreci_df["Threatened_CITES_Export_USD"].fillna(0.0)
    sreci_df["Pooled_Threatened_CITES_Export_USD"] = sreci_df["Pooled_Threatened_CITES_Export_USD"].fillna(0.0)
    sreci_df["Threatened_CITES_Import_USD"] = sreci_df["Threatened_CITES_Import_USD"].fillna(0.0)
    sreci_df["Pooled_Threatened_CITES_Import_USD"] = sreci_df["Pooled_Threatened_CITES_Import_USD"].fillna(0.0)

    sreci_df["Reform_Exposure_Share"] = np.clip(
        sreci_df["Pooled_Threatened_CITES_Export_USD"] / np.maximum(sreci_df["Total_Export_Real_USD"].fillna(0.0), 1.0),
        0.0, 1.0
    )
    sreci_df["Capacity_Deficit_0_1"] = 1.0 - sreci_df["Customs_Admin_Capacity_0_1"]
    sreci_df["SRECI_Score"] = sreci_df["Reform_Exposure_Share"] * sreci_df["Capacity_Deficit_0_1"]

    # Assign CBDR Implementation Tier
    sreci_df["CBDR_Implementation_Track"] = np.select(
        [
            sreci_df["Is_Track1_2028_Customs"] == 1,
            sreci_df["Is_ASEAN"] == 1,
            (sreci_df["Is_SIDS"] == 1) | (sreci_df["Is_LDC"] == 1) | (sreci_df["WB_Income_Group"].isin(["LIC", "LMC"]))
        ],
        [
            "Track 1A: Immediate Jan 2028 National 8-10 Digit Adopters (EU TARIC, US HTS, JPN, CHN, GBR, EFTA, CAN, AUS, KOR)",
            "Track 1B: Regional Processing Hub Harmonized Schedule (ASEAN AHTN 8-Digit Co-Adoption)",
            "Track 2A: Priority CBDR eCITES & Customs Co-Financing States (SIDS, LDCs, LIC & LMC Exporters)"
        ],
        default="Track 2B: Phased WCO 2033 Standard Six-Digit Transition (Other UMC Exporters)"
    )

    tot_thr_imp = sreci_df["Pooled_Threatened_CITES_Import_USD"].sum()
    tot_thr_exp = sreci_df["Pooled_Threatened_CITES_Export_USD"].sum()
    track1a_imp_pct = 100.0 * sreci_df.loc[sreci_df["Is_Track1_2028_Customs"] == 1, "Pooled_Threatened_CITES_Import_USD"].sum() / max(tot_thr_imp, 1.0)
    track1ab_imp_pct = 100.0 * sreci_df.loc[(sreci_df["Is_Track1_2028_Customs"] == 1) | (sreci_df["Is_ASEAN"] == 1), "Pooled_Threatened_CITES_Import_USD"].sum() / max(tot_thr_imp, 1.0)
    track1ab_exp_pct = 100.0 * sreci_df.loc[(sreci_df["Is_Track1_2028_Customs"] == 1) | (sreci_df["Is_ASEAN"] == 1), "Pooled_Threatened_CITES_Export_USD"].sum() / max(tot_thr_exp, 1.0)

    print(f"   -> Track 1A (High-Capacity 8-10 Digit Schedules) covers {track1a_imp_pct:.2f}% of global pooled threatened/CITES imports!")
    print(f"   -> Track 1A + 1B (+ ASEAN AHTN) covers {track1ab_imp_pct:.2f}% of global imports and {track1ab_exp_pct:.2f}% of exports!")

    sreci_out = os.path.join(DERIVED_DIR, "sovereign_cbdr_reform_exposure_sreci.csv")
    sreci_df.to_csv(sreci_out, index=False)
    print(f"   -> Saved Sovereign CBDR Reform Exposure & Capacity Schedule: {sreci_out}")


if __name__ == "__main__":
    main()
