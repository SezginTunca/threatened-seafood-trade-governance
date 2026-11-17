#!/usr/bin/env python3
"""
Module 08: Scarcity Pricing, Dual-IV Regressions, and Processing Premiums (Eqs. 4-5)
====================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Estimates aggregate Newey-West HAC price-to-volume elasticities across UN Comtrade
(1988-2024) and CEPII BACI HS92 (1995-2024), converged route-by-year two-way
fixed-effects Mirror-IV and Leave-One-Out Supply-IV 2SLS regressions, and
full-sample wild-versus-farmed Mann-Whitney U and rank-biserial price tests.

Outputs:
  - data/derived/annual_taxa_allee_time_series.csv
  - data/derived/global_allee_effect_econometric_results.csv
  - data/derived/baci_vs_comtrade_robustness_validation.csv
  - data/derived/wild_vs_farmed_price_premium_tests.csv
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import pyfixest as pf

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"

MASTER_PARQUET = DATA_RAW / "master_global_bilateral_panel.parquet"
BACI_PARQUET = DATA_RAW / "baci_hs92_seafood_1995_2024.parquet"
THREATENED_PARQUET = DATA_DERIVED / "global_threatened_taxa_bilateral_panel.parquet"

# Canonical US Bureau of Labor Statistics CPI-U Index (base 2024 = 313.0)
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
BASE_CPI = 313.0


def run_2wfe_iv_diagnostics(df_sub, endog_col, instr_cols, exog_col="ln_loo_m"):
    """
    Estimate converged 2WFE (route_id + Year) 2SLS with route-clustered CRV1 SEs,
    first-stage F-statistic, weak-IV-robust Anderson-Rubin reduced-form p-value,
    and (when overidentified, len(instr_cols) >= 2) Sargan-Hansen J p-value.
    """
    out = {
        "beta": np.nan, "se": np.nan, "pval": np.nan,
        "ci_low": np.nan, "ci_high": np.nan,
        "f_first": np.nan, "ar_pval": np.nan, "j_pval": np.nan, "n_obs": 0
    }
    if len(df_sub) < 50:
        return out

    cnts = df_sub["route_id"].value_counts()
    sub = df_sub[df_sub["route_id"].isin(cnts[cnts >= 3].index)].copy()
    if len(sub) < 50 or sub["route_id"].nunique() < 15 or sub["Year"].nunique() < 3:
        return out

    rhs_instr = " + ".join(instr_cols)
    exog_part = f"{exog_col}" if exog_col in sub.columns else "1"

    try:
        if len(instr_cols) == 1:
            fml = f"ln_p ~ {exog_part} | route_id + Year | {endog_col} ~ {rhs_instr}"
            fit = pf.feols(fml, data=sub, vcov={"CRV1": "route_id"})
            b = float(fit.coef()[endog_col])
            se = float(fit.se()[endog_col])
            pv = float(fit.pvalue()[endog_col])
            f1 = float(getattr(fit, "_f_stat_1st_stage", np.nan))
            n_used = int(fit._N)

            # Exact weak-IV-robust Anderson-Rubin (1949) test of H0: beta = 0
            # via 2WFE reduced-form regression of ln_p on instrument(s) with CRV1 clustering
            fml_rf = f"ln_p ~ {rhs_instr} + {exog_part} | route_id + Year"
            fit_rf = pf.feols(fml_rf, data=sub, vcov={"CRV1": "route_id"})
            ar_pv = float(fit_rf.pvalue()[instr_cols[0]])

            out.update({
                "beta": b, "se": se, "pval": pv,
                "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
                "f_first": f1, "ar_pval": ar_pv, "n_obs": n_used
            })
        else:
            # Overidentified 2SLS: exact converged alternating-projection 2WFE demeaning + 2SLS + Hansen-Sargan J
            cols_dm = ["ln_p", endog_col] + instr_cols + ([exog_col] if exog_col in sub.columns else [])
            dm = sub[["route_id", "Year"] + cols_dm].dropna().copy()
            # Drop singletons iteratively
            for _ in range(5):
                rc = dm["route_id"].value_counts()
                yc = dm["Year"].value_counts()
                dm = dm[dm["route_id"].isin(rc[rc >= 2].index) & dm["Year"].isin(yc[yc >= 2].index)]
            if len(dm) < 60:
                return out
            # Converge 2WFE demeaning to 1e-11 tolerance
            vals = {c: dm[c].values.astype(float).copy() for c in cols_dm}
            r_codes = pd.Categorical(dm["route_id"]).codes
            y_codes = pd.Categorical(dm["Year"]).codes
            n_r = r_codes.max() + 1
            n_y = y_codes.max() + 1
            r_counts = np.bincount(r_codes, minlength=n_r)
            y_counts = np.bincount(y_codes, minlength=n_y)
            for c in cols_dm:
                v = vals[c]
                for _ in range(200):
                    v_old = v.copy()
                    v = v - (np.bincount(r_codes, weights=v, minlength=n_r) / np.maximum(r_counts, 1))[r_codes]
                    v = v - (np.bincount(y_codes, weights=v, minlength=n_y) / np.maximum(y_counts, 1))[y_codes]
                    if np.max(np.abs(v - v_old)) < 1e-11:
                        break
                vals[c] = v

            from linearmodels.iv import IV2SLS
            df_dm = pd.DataFrame(vals, index=dm.index)
            exog_df = df_dm[[exog_col]] if exog_col in df_dm.columns else None
            iv_mod = IV2SLS(
                dependent=df_dm["ln_p"],
                exog=exog_df,
                endog=df_dm[[endog_col]],
                instruments=df_dm[instr_cols]
            ).fit(cov_type="clustered", clusters=dm["route_id"])

            b = float(iv_mod.params[endog_col])
            se = float(iv_mod.std_errors[endog_col])
            pv = float(iv_mod.pvalues[endog_col])
            f1 = float(iv_mod.first_stage.diagnostics["f.stat"].iloc[0])
            j_pv = float(iv_mod.wooldridge_overid.pval)
            out.update({
                "beta": b, "se": se, "pval": pv,
                "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
                "f_first": f1, "j_pval": j_pv, "n_obs": int(len(dm))
            })
    except Exception as exc:
        print(f"      [Warn] 2WFE IV failed: {exc}")
    return out


def main():
    print("=" * 80)
    print("MODULE 08: GLOBAL BIOECONOMIC SCARCITY ELASTICITIES (DUAL-IV) & PRICE PREMIUMS")
    print("=" * 80)

    if not THREATENED_PARQUET.exists() or not BACI_PARQUET.exists():
        print("Error: Required parquet inputs missing.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Load Native-HS UN Comtrade Threatened Panel (1988-2024)
    # -------------------------------------------------------------------------
    print(f"1. Loading UN Comtrade threatened taxa panel: {THREATENED_PARQUET}...")
    df = pd.read_parquet(THREATENED_PARQUET)
    df_exp = df[(df["Trade_Flow"] == "Export") & (df["Year"] <= 2024)].copy()
    df_exp = df_exp[(df_exp["Weight_KG"] > 10) & (df_exp["Real_Trade_Value_USD"] > 10)].copy()
    df_exp["HS6_Code"] = df_exp["HS6_Code"].astype(str).str.zfill(6)
    df_exp["Unit_Price"] = df_exp["Real_Trade_Value_USD"] / df_exp["Weight_KG"]
    df_exp = df_exp[(df_exp["Unit_Price"] >= 0.1) & (df_exp["Unit_Price"] <= 5000)].copy()

    # Maintain longitudinal HS basket consistency across revisions by excluding mid-series
    # processing-form split additions (160417 prepared eel added in 2012; 030812 frozen sea cucumber added in 2017)
    df_com_long = df_exp[~df_exp["HS6_Code"].isin(["160417", "030812"])].copy()

    # Native Comtrade categories with >= 13 years of dedicated codes:
    # Diadromous Eels (1988-2024, 37 yrs), High-Trophic Tunas (2002-2024, 23 yrs),
    # Sea Cucumbers (2012-2024, 13 yrs), Demersal Commercial (1988-2024, 37 yrs)
    com_cats_keep = [
        "Diadromous Eels (Anguilla spp.)",
        "High-Trophic Tunas (Thunnus spp.)",
        "Sea Cucumbers (Holothuroidea)",
        "Demersal Commercial Taxonomic Groups",
    ]
    taxa_com_ann = df_com_long[df_com_long["Conservation_Category"].isin(com_cats_keep)].groupby(
        ["Year", "Conservation_Category"], as_index=False
    ).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum"),
        Transaction_Count=("Real_Trade_Value_USD", "count")
    )
    taxa_com_ann["Data_Source"] = "UN Comtrade Native HS (1988-2024)"

    # -------------------------------------------------------------------------
    # 2. Load Harmonized CEPII BACI HS92 Panel (1995-2024, 30 consecutive years)
    #    Unlocks full 30-year series for Chondrichthyans (030265, 030375) and
    #    Sturgeons & Caviar (160430), plus cross-database validation across clades
    # -------------------------------------------------------------------------
    print(f"2. Loading harmonized CEPII BACI HS92 panel (1995-2024): {BACI_PARQUET}...")
    baci = pd.read_parquet(BACI_PARQUET)
    baci["HS92_Code"] = baci["HS92_Code"].astype(str).str.zfill(6)
    baci = baci[(baci["Year"] >= 1995) & (baci["Year"] <= 2024)].copy()
    baci["Weight_KG"] = baci["Quantity_t"] * 1000.0
    baci["Real_Trade_Value_USD"] = baci["Value_USD"] * (BASE_CPI / baci["Year"].map(CPI_DEFLATOR))
    baci = baci[
        (baci["Weight_KG"] > 10) &
        (baci["Real_Trade_Value_USD"] > 10) &
        (baci["Exporter_ISO3"] != baci["Importer_ISO3"])
    ].copy()
    baci["Unit_Price"] = baci["Real_Trade_Value_USD"] / baci["Weight_KG"]
    baci = baci[(baci["Unit_Price"] >= 0.1) & (baci["Unit_Price"] <= 5000)].copy()

    baci_clade_map = {
        "030265": "Chondrichthyans (Sharks & Rays)",
        "030375": "Chondrichthyans (Sharks & Rays)",
        "160430": "Sturgeons & Caviar (Acipenseriformes)",
        "030192": "Diadromous Eels (Anguilla spp.)",
        "030266": "Diadromous Eels (Anguilla spp.)",
        "030376": "Diadromous Eels (Anguilla spp.)",
        "030231": "High-Trophic Tunas (Thunnus spp.)",
        "030232": "High-Trophic Tunas (Thunnus spp.)",
        "030233": "High-Trophic Tunas (Thunnus spp.)",
        "030239": "High-Trophic Tunas (Thunnus spp.)",
        "030341": "High-Trophic Tunas (Thunnus spp.)",
        "030342": "High-Trophic Tunas (Thunnus spp.)",
        "030343": "High-Trophic Tunas (Thunnus spp.)",
        "030349": "High-Trophic Tunas (Thunnus spp.)",
        "030250": "Demersal Commercial Taxonomic Groups",
        "030262": "Demersal Commercial Taxonomic Groups",
        "030263": "Demersal Commercial Taxonomic Groups",
        "030360": "Demersal Commercial Taxonomic Groups",
        "030372": "Demersal Commercial Taxonomic Groups",
        "030373": "Demersal Commercial Taxonomic Groups",
        "030378": "Demersal Commercial Taxonomic Groups",
    }
    baci["Conservation_Category"] = baci["HS92_Code"].map(baci_clade_map)
    baci_mapped = baci.dropna(subset=["Conservation_Category"]).copy()

    # Pre-compute BACI LOO exporter supply & LOO importer demand for 30-year 2WFE IV
    tot_baci_exp = baci_mapped.groupby(["Year", "HS92_Code", "Exporter_ISO3"])["Weight_KG"].transform("sum")
    baci_mapped["LOO_Supply_KG"] = tot_baci_exp - baci_mapped["Weight_KG"]
    tot_baci_imp = baci_mapped.groupby(["Year", "HS92_Code", "Importer_ISO3"])["Real_Trade_Value_USD"].transform("sum")
    baci_mapped["LOO_Demand_USD"] = tot_baci_imp - baci_mapped["Real_Trade_Value_USD"]
    baci_mapped["route_id"] = (
        baci_mapped["Exporter_ISO3"] + "_" + baci_mapped["Importer_ISO3"] + "_" + baci_mapped["HS92_Code"]
    )
    baci_mapped["ln_p"] = np.log(baci_mapped["Unit_Price"])
    baci_mapped["ln_q_exp"] = np.log(baci_mapped["Weight_KG"])
    baci_mapped["ln_loo_q"] = np.log(np.maximum(baci_mapped["LOO_Supply_KG"], 1.0))
    baci_mapped["ln_loo_m"] = np.log(np.maximum(baci_mapped["LOO_Demand_USD"], 1.0))

    taxa_baci_sharks_sturg = baci_mapped[
        baci_mapped["Conservation_Category"].isin([
            "Chondrichthyans (Sharks & Rays)",
            "Sturgeons & Caviar (Acipenseriformes)"
        ])
    ].groupby(["Year", "Conservation_Category"], as_index=False).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum"),
        Transaction_Count=("Real_Trade_Value_USD", "count")
    )
    taxa_baci_sharks_sturg["Data_Source"] = "CEPII BACI HS92 Harmonized (1995-2024)"

    # Combine into complete 6-category longitudinal time series (13 to 37 years for EVERY clade)
    taxa_annual = pd.concat([taxa_com_ann, taxa_baci_sharks_sturg], ignore_index=True)
    taxa_annual = taxa_annual.sort_values(["Conservation_Category", "Year"]).reset_index(drop=True)
    taxa_annual["Total_Weight_MT"] = taxa_annual["Total_Weight_KG"] / 1e3
    taxa_annual["Total_Real_USD_M"] = taxa_annual["Total_Real_USD"] / 1e6
    taxa_annual["Weighted_Unit_Price_USD_per_KG"] = taxa_annual["Total_Real_USD"] / taxa_annual["Total_Weight_KG"]

    out_ts = DATA_DERIVED / "annual_taxa_allee_time_series.csv"
    taxa_annual.to_csv(out_ts, index=False)
    print(f"   -> Saved complete 6-category longitudinal series: {out_ts} ({len(taxa_annual)} clade-years)")

    # -------------------------------------------------------------------------
    # 3. Prepare UN Comtrade Matched Mirror + LOO Supply Panel for Dual-IV 2SLS
    # -------------------------------------------------------------------------
    print("\n3. Constructing UN Comtrade bilateral mirror + LOO supply panel for converged 2WFE 2SLS...")
    raw_cols = [
        "Year", "Trade_Flow", "Reporter_ISO3", "Partner_ISO3",
        "HS6_Code", "Weight_KG", "Trade_Value_USD", "UV_Outlier_Flag"
    ]
    df_raw = pd.read_parquet(MASTER_PARQUET, columns=raw_cols)
    df_raw = df_raw[
        (df_raw["Year"] <= 2024) &
        (df_raw["Weight_KG"] > 10) &
        (df_raw["Trade_Value_USD"] > 10) &
        (~df_raw["UV_Outlier_Flag"])
    ].copy()
    df_raw["HS6_Code"] = df_raw["HS6_Code"].astype(str).str.zfill(6)
    conc = pd.read_csv(DATA_DERIVED / "global_hs6_conservation_concordance.csv")
    conc["HS6_Code"] = conc["HS6_Code"].astype(str).str.zfill(6)
    df_raw = df_raw.merge(conc[["HS6_Code", "Conservation_Category"]].drop_duplicates("HS6_Code"), on="HS6_Code", how="left")
    df_raw["Real_USD"] = df_raw["Trade_Value_USD"] * (BASE_CPI / df_raw["Year"].map(CPI_DEFLATOR))

    m_exp = df_raw[(df_raw["Trade_Flow"] == "Export") & (df_raw["Reporter_ISO3"] != df_raw["Partner_ISO3"])].groupby(
        ["Year", "HS6_Code", "Conservation_Category", "Reporter_ISO3", "Partner_ISO3"], as_index=False
    ).agg(Exp_KG=("Weight_KG", "sum"), Exp_USD=("Real_USD", "sum"))

    tot_com_exp = m_exp.groupby(["Year", "HS6_Code", "Reporter_ISO3"])["Exp_KG"].transform("sum")
    m_exp["LOO_Supply_KG"] = tot_com_exp - m_exp["Exp_KG"]

    m_imp = df_raw[(df_raw["Trade_Flow"] == "Import") & (df_raw["Reporter_ISO3"] != df_raw["Partner_ISO3"])].groupby(
        ["Year", "HS6_Code", "Conservation_Category", "Partner_ISO3", "Reporter_ISO3"], as_index=False
    ).agg(Imp_KG=("Weight_KG", "sum"), Imp_USD=("Real_USD", "sum")).rename(
        columns={"Partner_ISO3": "Reporter_ISO3", "Reporter_ISO3": "Partner_ISO3"}
    )
    tot_com_imp = m_imp.groupby(["Year", "HS6_Code", "Partner_ISO3"])["Imp_USD"].transform("sum")
    m_imp["LOO_Demand_USD"] = tot_com_imp - m_imp["Imp_USD"]
    tot_imp_all_clades = m_imp.groupby(["Year", "Partner_ISO3"])["Imp_USD"].transform("sum")
    tot_imp_this_clade = m_imp.groupby(["Year", "Conservation_Category", "Partner_ISO3"])["Imp_USD"].transform("sum")
    m_imp["LOO_Clade_Demand_USD"] = np.maximum(tot_imp_all_clades - tot_imp_this_clade, 1.0)

    mir = pd.merge(m_exp, m_imp.drop(columns=["Conservation_Category"], errors="ignore"), on=["Year", "HS6_Code", "Reporter_ISO3", "Partner_ISO3"], how="inner")
    mir["Exp_UV"] = mir["Exp_USD"] / mir["Exp_KG"]
    mir = mir[(mir["Exp_UV"] >= 0.1) & (mir["Exp_UV"] <= 5000) & (~mir["HS6_Code"].isin(["160417", "030812"]))].copy()
    mir["route_id"] = mir["Reporter_ISO3"] + "_" + mir["Partner_ISO3"] + "_" + mir["HS6_Code"]
    mir["ln_p"] = np.log(mir["Exp_UV"])
    mir["ln_q_exp"] = np.log(mir["Exp_KG"])
    mir["ln_q_imp"] = np.log(mir["Imp_KG"])
    mir["ln_loo_q"] = np.log(np.maximum(mir["LOO_Supply_KG"], 1.0))
    mir["ln_loo_m"] = np.log(np.maximum(mir["LOO_Demand_USD"], 1.0))
    mir["ln_loo_m_clade"] = np.log(np.maximum(mir["LOO_Clade_Demand_USD"], 1.0))

    # -------------------------------------------------------------------------
    # 4. Estimate Aggregate OLS (HAC) + Converged 2WFE Dual-IV Across All 6 Clades
    # -------------------------------------------------------------------------
    print("\n4. Estimating Aggregate HAC OLS + Route-Level 2WFE Mirror-IV & LOO-Supply-IV...")
    allee_results = []
    for cat in sorted(taxa_annual["Conservation_Category"].unique()):
        sub_reg = taxa_annual[(taxa_annual["Conservation_Category"] == cat) & (taxa_annual["Year"] <= 2024)].sort_values("Year").copy()
        q = sub_reg["Total_Weight_MT"].values
        p = sub_reg["Weighted_Unit_Price_USD_per_KG"].values

        ln_q = np.log(q)
        ln_p = np.log(p)
        X_ols = sm.add_constant(ln_q)
        ols_res = sm.OLS(ln_p, X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        beta_k = float(ols_res.params[1])
        se_beta_k = float(ols_res.bse[1])
        p_ols = float(ols_res.pvalues[1])
        rho, p_val = stats.spearmanr(q, p)

        init_vol, final_vol = q[:3].mean(), q[-3:].mean()
        vol_change_pct = ((final_vol - init_vol) / init_vol) * 100.0
        init_p, final_p = p[:3].mean(), p[-3:].mean()
        price_change_pct = ((final_p - init_p) / init_p) * 100.0
        aae_flag = (beta_k < -0.15 and p_ols < 0.05) and (rho < -0.35 and p_val < 0.10)

        # Spec A: Comtrade Mirror-Quantity IV (purges unit-value division bias)
        sub_m = mir[mir["Conservation_Category"] == cat].copy()
        res_mir = run_2wfe_iv_diagnostics(sub_m, endog_col="ln_q_exp", instr_cols=["ln_q_imp"])
        iv_r = np.nan
        if res_mir["n_obs"] >= 50:
            iv_r = float(np.corrcoef(sub_m["ln_q_exp"], sub_m["ln_q_imp"])[0, 1])

        # Spec B: Exporter Leave-One-Destination-Out Supply-Shifter IV
        # Use 30-year BACI panel for Chondrichthyans & Sturgeons (and where BACI exists),
        # and Comtrade LOO panel for Sea Cucumbers (not separated in HS92)
        if cat in ["Chondrichthyans (Sharks & Rays)", "Sturgeons & Caviar (Acipenseriformes)"]:
            sub_baci = baci_mapped[(baci_mapped["Conservation_Category"] == cat) & (baci_mapped["LOO_Supply_KG"] > 10)].copy()
            res_loo = run_2wfe_iv_diagnostics(sub_baci, endog_col="ln_q_exp", instr_cols=["ln_loo_q"])
            loo_source = "BACI_30yr"
        else:
            sub_loo_c = sub_m[sub_m["LOO_Supply_KG"] > 10].copy()
            res_loo = run_2wfe_iv_diagnostics(sub_loo_c, endog_col="ln_q_exp", instr_cols=["ln_loo_q"])
            loo_source = "Comtrade"

        # Spec C: Overidentified Dual-IV (Mirror + LOO Supply) on Comtrade matched panel
        sub_dual = sub_m[sub_m["LOO_Supply_KG"] > 10].copy()
        res_dual = run_2wfe_iv_diagnostics(sub_dual, endog_col="ln_q_exp", instr_cols=["ln_q_imp", "ln_loo_q"])

        # Spec D: Leave-one-clade-out destination demand control (ln M_{j,-k,t}) sensitivity check
        res_mir_clade = run_2wfe_iv_diagnostics(sub_m, endog_col="ln_q_exp", instr_cols=["ln_q_imp"], exog_col="ln_loo_m_clade")

        allee_results.append({
            "Conservation_Category": cat,
            "Longitudinal_Source": sub_reg["Data_Source"].iloc[0],
            "Year_Span": f"{int(sub_reg['Year'].min())}-{int(sub_reg['Year'].max())}",
            "Observations_Years": int(len(sub_reg)),
            "OLS_Slope_Beta": np.round(beta_k, 4),
            "OLS_SE_HAC": np.round(se_beta_k, 4),
            "OLS_P_Value": np.round(p_ols, 5),
            "Spearman_Rho": np.round(rho, 4),
            "P_Value": np.round(p_val, 5),
            "MirrorIV_2WFE_Beta": np.round(res_mir["beta"], 4) if not np.isnan(res_mir["beta"]) else np.nan,
            "MirrorIV_2WFE_SE": np.round(res_mir["se"], 4) if not np.isnan(res_mir["se"]) else np.nan,
            "MirrorIV_2WFE_Pval": np.round(res_mir["pval"], 5) if not np.isnan(res_mir["pval"]) else np.nan,
            "MirrorIV_FirstStage_F": np.round(res_mir["f_first"], 1) if not np.isnan(res_mir["f_first"]) else np.nan,
            "MirrorIV_AR_WeakIV_Pval": np.round(res_mir["ar_pval"], 5) if not np.isnan(res_mir["ar_pval"]) else np.nan,
            "MirrorIV_1st_Stage_r": np.round(iv_r, 3) if not np.isnan(iv_r) else np.nan,
            "MirrorIV_CladeLOO_Beta": np.round(res_mir_clade["beta"], 4) if not np.isnan(res_mir_clade["beta"]) else np.nan,
            "MirrorIV_CladeLOO_Pval": np.round(res_mir_clade["pval"], 5) if not np.isnan(res_mir_clade["pval"]) else np.nan,
            "Dyad_Year_Observations": int(res_mir["n_obs"]),
            "LOOSupplyIV_Source": loo_source,
            "LOOSupplyIV_2WFE_Beta": np.round(res_loo["beta"], 4) if not np.isnan(res_loo["beta"]) else np.nan,
            "LOOSupplyIV_2WFE_SE": np.round(res_loo["se"], 4) if not np.isnan(res_loo["se"]) else np.nan,
            "LOOSupplyIV_2WFE_Pval": np.round(res_loo["pval"], 5) if not np.isnan(res_loo["pval"]) else np.nan,
            "LOOSupplyIV_FirstStage_F": np.round(res_loo["f_first"], 1) if not np.isnan(res_loo["f_first"]) else np.nan,
            "LOOSupplyIV_N": int(res_loo["n_obs"]),
            "DualIV_2WFE_Beta": np.round(res_dual["beta"], 4) if not np.isnan(res_dual["beta"]) else np.nan,
            "DualIV_2WFE_SE": np.round(res_dual["se"], 4) if not np.isnan(res_dual["se"]) else np.nan,
            "DualIV_2WFE_Pval": np.round(res_dual["pval"], 5) if not np.isnan(res_dual["pval"]) else np.nan,
            "DualIV_FirstStage_F": np.round(res_dual["f_first"], 1) if not np.isnan(res_dual["f_first"]) else np.nan,
            "DualIV_Hansen_J_Pval": np.round(res_dual["j_pval"], 4) if not np.isnan(res_dual["j_pval"]) else np.nan,
            "Volume_Change_Percent": np.round(vol_change_pct, 1),
            "Real_Price_Change_Percent": np.round(price_change_pct, 1),
            "Baseline_Real_Price_USD_KG": np.round(init_p, 2),
            "Modern_Real_Price_USD_KG": np.round(final_p, 2),
            "Peak_Real_Price_USD_KG": np.round(p.max(), 2),
            "Peak_Price_Year": int(sub_reg.iloc[int(np.argmax(p))]["Year"]),
            "Empirical_AAE_Confirmed": bool(aae_flag),
            "Wild_Scarcity_MirrorIV_Confirmed": bool(
                not np.isnan(res_mir["beta"]) and res_mir["beta"] < -0.08 and res_mir["pval"] < 0.05
            )
        })

    df_allee = pd.DataFrame(allee_results)
    out_allee = DATA_DERIVED / "global_allee_effect_econometric_results.csv"
    df_allee.to_csv(out_allee, index=False)
    print(f"   -> Saved upgraded Dual-IV econometric results: {out_allee}")
    print("\n--- Upgraded 6-Clade AAE & Converged 2WFE Dual-IV Summary ---")
    print(df_allee[[
        "Conservation_Category", "Year_Span", "Observations_Years",
        "OLS_Slope_Beta", "OLS_P_Value", "MirrorIV_2WFE_Beta", "MirrorIV_2WFE_Pval",
        "MirrorIV_FirstStage_F", "LOOSupplyIV_2WFE_Beta", "LOOSupplyIV_2WFE_Pval", "LOOSupplyIV_FirstStage_F"
    ]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Cross-Database Validation (BACI 1995-2024 vs Comtrade) & Aquaculture Regime Split
    # -------------------------------------------------------------------------
    print("\n5. Running BACI vs Comtrade cross-validation & Sturgeon wild-to-aquaculture regime split...")
    val_rows = []
    for cat_v in sorted(baci_mapped["Conservation_Category"].unique()):
        sb = baci_mapped[(baci_mapped["Conservation_Category"] == cat_v) & (baci_mapped["LOO_Supply_KG"] > 10)].copy()
        ann_b = sb.groupby("Year", as_index=False).agg(V=("Real_Trade_Value_USD", "sum"), W=("Weight_KG", "sum"))
        ols_b = sm.OLS(np.log(ann_b["V"] / ann_b["W"]), sm.add_constant(np.log(ann_b["W"] / 1e3))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 2}
        )
        iv_b = run_2wfe_iv_diagnostics(sb, endog_col="ln_q_exp", instr_cols=["ln_loo_q"])
        val_rows.append({
            "Clade_Regime": f"{cat_v} (BACI 1995-2024 Full 30yr)",
            "Years": int(len(ann_b)),
            "Route_Year_N": int(iv_b["n_obs"]),
            "Macro_OLS_Beta": np.round(float(ols_b.params.iloc[1]), 4),
            "Macro_OLS_Pval": np.round(float(ols_b.pvalues.iloc[1]), 5),
            "Route_LOO_IV_Beta": np.round(iv_b["beta"], 4),
            "Route_LOO_IV_SE": np.round(iv_b["se"], 4),
            "Route_LOO_IV_Pval": np.round(iv_b["pval"], 5),
            "First_Stage_F": np.round(iv_b["f_first"], 1),
        })

    # Test H2a Aquaculture Decoupling directly on Sturgeons & Caviar (160430):
    # Pre-2008 Wild Caspian Collapse Era (1995-2007) vs Post-2008 Farmed Caviar Expansion Era (2008-2024)
    for era_label, y0, y1 in [
        ("Sturgeons & Caviar: Wild Collapse Era (1995-2007)", 1995, 2007),
        ("Sturgeons & Caviar: Aquaculture Expansion Era (2008-2024)", 2008, 2024),
    ]:
        sb_e = baci_mapped[
            (baci_mapped["Conservation_Category"] == "Sturgeons & Caviar (Acipenseriformes)") &
            (baci_mapped["Year"] >= y0) & (baci_mapped["Year"] <= y1) &
            (baci_mapped["LOO_Supply_KG"] > 10)
        ].copy()
        ann_e = sb_e.groupby("Year", as_index=False).agg(V=("Real_Trade_Value_USD", "sum"), W=("Weight_KG", "sum"))
        ols_e = sm.OLS(np.log(ann_e["V"] / ann_e["W"]), sm.add_constant(np.log(ann_e["W"] / 1e3))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1}
        )
        iv_e = run_2wfe_iv_diagnostics(sb_e, endog_col="ln_q_exp", instr_cols=["ln_loo_q"])
        val_rows.append({
            "Clade_Regime": era_label,
            "Years": int(len(ann_e)),
            "Route_Year_N": int(iv_e["n_obs"]),
            "Macro_OLS_Beta": np.round(float(ols_e.params.iloc[1]), 4),
            "Macro_OLS_Pval": np.round(float(ols_e.pvalues.iloc[1]), 5),
            "Route_LOO_IV_Beta": np.round(iv_e["beta"], 4),
            "Route_LOO_IV_SE": np.round(iv_e["se"], 4),
            "Route_LOO_IV_Pval": np.round(iv_e["pval"], 5),
            "First_Stage_F": np.round(iv_e["f_first"], 1),
        })

    df_val = pd.DataFrame(val_rows)
    out_val = DATA_DERIVED / "baci_vs_comtrade_robustness_validation.csv"
    df_val.to_csv(out_val, index=False)
    print(f"   -> Saved BACI cross-validation & regime-split table: {out_val}")
    print(df_val.to_string(index=False))

    # -------------------------------------------------------------------------
    # 6. Wild-versus-Farmed Real Export Price Premium Tests (H2b, Exact Mann-Whitney U)
    # -------------------------------------------------------------------------
    print("\n6. Evaluating Wild-versus-Farmed Real Export Price Disparities Across Processing Forms...")
    cols_typ = [
        "Year", "Trade_Flow", "Processing_Form", "Production_Typology",
        "Weight_KG", "Trade_Value_USD", "Species_Group", "UV_Outlier_Flag"
    ]
    df_all = pd.read_parquet(MASTER_PARQUET, columns=cols_typ)
    df_all = df_all[
        (df_all["Year"] <= 2024) &
        (df_all["Trade_Flow"] == "Export") &
        (df_all["Weight_KG"] > 10) &
        (df_all["Trade_Value_USD"] > 10) &
        (~df_all["UV_Outlier_Flag"])
    ].copy()
    df_all["CPI"] = df_all["Year"].map(CPI_DEFLATOR).fillna(BASE_CPI)
    df_all["Real_Trade_Value_USD"] = df_all["Trade_Value_USD"] * (BASE_CPI / df_all["CPI"])
    df_all["Unit_Price"] = df_all["Real_Trade_Value_USD"] / df_all["Weight_KG"]
    df_all = df_all[(df_all["Unit_Price"] >= 0.1) & (df_all["Unit_Price"] <= 1000)].copy()

    premium_results = []
    for form in sorted(df_all["Processing_Form"].unique()):
        sub_form = df_all[df_all["Processing_Form"] == form]
        wild_prices = sub_form[sub_form["Production_Typology"].str.contains("Capture", case=False, na=False)]["Unit_Price"].values
        farm_prices = sub_form[sub_form["Production_Typology"].str.contains("Aquaculture", case=False, na=False)]["Unit_Price"].values

        if len(wild_prices) >= 100 and len(farm_prices) >= 100:
            u_stat, p_val = stats.mannwhitneyu(wild_prices, farm_prices, alternative="two-sided")
            n1, n2 = float(len(wild_prices)), float(len(farm_prices))
            r_biserial = (2.0 * u_stat) / (n1 * n2) - 1.0
            med_wild = float(np.median(wild_prices))
            med_farm = float(np.median(farm_prices))
            ratio = med_wild / med_farm if med_farm > 0 else np.nan

            premium_results.append({
                "Processing_Form": form,
                "N_Wild_Total": int(len(wild_prices)),
                "N_Farmed_Total": int(len(farm_prices)),
                "Median_Wild_USD_KG": np.round(med_wild, 2),
                "Median_Farmed_USD_KG": np.round(med_farm, 2),
                "Price_Ratio_Wild_to_Farm": np.round(ratio, 2),
                "Mann_Whitney_U": np.round(float(u_stat), 1),
                "P_Value": float(p_val),
                "Rank_Biserial_Correlation_r": np.round(r_biserial, 4),
                "Wild_Premium_Stat_Sig": bool(p_val < 0.001 and ratio > 1.0)
            })

    wild_luxury = df_all[df_all["Species_Group"].isin([
        "Bluefin Tuna & Major Tunas", "Crabs, Lobsters & Other Crustaceans", "Flatfish & Turbot"
    ])]["Unit_Price"].values
    farm_core = df_all[df_all["Species_Group"].isin([
        "Atlantic & Pacific Salmon", "Rainbow & Freshwater Trout", "Pangasius, Tilapia & Freshwater Catfish"
    ])]["Unit_Price"].values

    u_lux, p_lux = stats.mannwhitneyu(wild_luxury, farm_core, alternative="two-sided")
    n1_lux, n2_lux = float(len(wild_luxury)), float(len(farm_core))
    r_lux = (2.0 * u_lux) / (n1_lux * n2_lux) - 1.0
    med_w_lux = float(np.median(wild_luxury))
    med_f_core = float(np.median(farm_core))

    premium_results.append({
        "Processing_Form": "Luxury Marine Capture vs Industrial Aquaculture",
        "N_Wild_Total": int(len(wild_luxury)),
        "N_Farmed_Total": int(len(farm_core)),
        "Median_Wild_USD_KG": np.round(med_w_lux, 2),
        "Median_Farmed_USD_KG": np.round(med_f_core, 2),
        "Price_Ratio_Wild_to_Farm": np.round(med_w_lux / med_f_core, 2),
        "Mann_Whitney_U": np.round(float(u_lux), 1),
        "P_Value": float(p_lux),
        "Rank_Biserial_Correlation_r": np.round(r_lux, 4),
        "Wild_Premium_Stat_Sig": bool(p_lux < 0.001 and (med_w_lux > med_f_core))
    })

    df_prem = pd.DataFrame(premium_results)
    out_prem = DATA_DERIVED / "wild_vs_farmed_price_premium_tests.csv"
    df_prem.to_csv(out_prem, index=False)
    print(f"   -> Saved wild-vs-farmed processing inversion tests: {out_prem}")
    print("=" * 80)
    print("MODULE 08 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
