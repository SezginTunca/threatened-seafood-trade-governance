#!/usr/bin/env python3
"""
Module 12: Bilateral Mirror Discrepancies and Strategic Evasion Models (Eqs. 7-10)
==================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Computes CIF/FOB-adjusted value and weight Mirror Discrepancy Indices (MDI),
Bilateral Taxonomic Aggregation Asymmetry (BTAR), and Entrepot Masking Ratios,
and estimates high-dimensional exporter-importer-year fixed-effects models for
symmetric evasion and one-sided unmatched border declarations.

Outputs:
  - data/derived/global_mirror_discrepancy_by_taxa.csv
  - data/derived/global_mirror_discrepancy_by_bloc.csv
  - data/derived/global_mirror_discrepancy_annual_timeline.csv
  - data/derived/global_mirror_discrepancy_granularity_comparison.csv
  - data/derived/global_mirror_discrepancy_theta_sensitivity.csv
  - data/derived/global_mirror_discrepancy_theta_sensitivity_by_taxa.csv
  - data/derived/global_entrepot_masking_btar_ecr.csv
  - data/derived/global_mirror_fe_regression_results.csv
  - data/derived/global_mirror_strategic_evasion_fe_models.csv
  - data/derived/global_mirror_extensive_margin_orphans.csv
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyfixest as pf

warnings.filterwarnings("ignore", category=UserWarning)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"

MASTER_PARQUET = DATA_RAW / "master_global_bilateral_panel.parquet"
CONCORDANCE_CSV = DATA_DERIVED / "global_hs6_conservation_concordance.csv"
SCRI_AUDIT_CSV = DATA_DERIVED / "global_hs6_customs_scri_audit.csv"

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
THETA = 0.11


def main():
    print("=" * 80)
    print("MODULE 12: GLOBAL BILATERAL MIRROR DISCREPANCY & STRATEGIC EVASION FORENSICS")
    print("=" * 80)

    if not MASTER_PARQUET.exists():
        print(f"Error: {MASTER_PARQUET} not found.")
        sys.exit(1)

    print("1. Loading concordance taxonomy and empirical SCRI audit...")
    df_conc = pd.read_csv(CONCORDANCE_CSV)
    df_conc["HS6_Code"] = df_conc["HS6_Code"].astype(str).str.zfill(6)
    conc_cols = [
        "HS6_Code", "Customs_Granularity", "Conservation_Category",
        "Threatened_Taxa_Flag", "IUCN_Threat_Tier"
    ]
    df_conc_clean = df_conc[conc_cols].drop_duplicates(subset=["HS6_Code"])

    if SCRI_AUDIT_CSV.exists():
        df_scri = pd.read_csv(SCRI_AUDIT_CSV)[["HS6_Code", "Subheading_SCRI", "CITES_Status"]].copy()
        df_scri["HS6_Code"] = df_scri["HS6_Code"].astype(str).str.zfill(6)
        df_conc_clean = df_conc_clean.merge(df_scri.drop_duplicates("HS6_Code"), on="HS6_Code", how="left")
    else:
        df_conc_clean["Subheading_SCRI"] = np.where(df_conc_clean["Customs_Granularity"] == "Taxon-Specific", 1.0, 0.45)
        df_conc_clean["CITES_Status"] = "Not Listed"

    print("2. Loading master bilateral panel (core columns)...")
    cols_load = [
        "Year", "Trade_Flow", "Reporter_ISO3", "Reporter_Country", "Partner_ISO3",
        "Reporter_Bloc", "Partner_Bloc", "HS6_Code", "Weight_KG",
        "Trade_Value_USD", "Species_Group", "UV_Outlier_Flag"
    ]
    df = pd.read_parquet(MASTER_PARQUET, columns=cols_load)
    df["HS6_Code"] = df["HS6_Code"].astype(str).str.zfill(6)

    df = df[
        (df["Year"] <= 2024) &
        (df["Weight_KG"] > 0) &
        (df["Trade_Value_USD"] > 0) &
        (df["Reporter_ISO3"].notna()) &
        (df["Partner_ISO3"].notna()) &
        (df["Reporter_ISO3"] != df["Partner_ISO3"]) &
        (~df["UV_Outlier_Flag"])
    ].copy()
    print(f"   Retained {len(df):,} clean non-zero records with valid bilateral partners (1988-2024).")

    df["CPI"] = df["Year"].map(CPI_DEFLATOR).fillna(BASE_CPI)
    df["Real_Trade_Value_USD"] = df["Trade_Value_USD"] * (BASE_CPI / df["CPI"])

    df = df.merge(df_conc_clean, on="HS6_Code", how="left")
    df["Customs_Granularity"] = df["Customs_Granularity"].fillna("Aggregated Multi-Taxa")
    df["Conservation_Category"] = df["Conservation_Category"].fillna("General Commercial")
    df["Threatened_Taxa_Flag"] = df["Threatened_Taxa_Flag"].fillna(False)
    df["Subheading_SCRI"] = df["Subheading_SCRI"].fillna(0.40)

    print("3. Constructing bilateral mirror pairs...")
    exp = df[df["Trade_Flow"] == "Export"].rename(columns={
        "Reporter_ISO3": "Origin_ISO3",
        "Partner_ISO3": "Dest_ISO3",
        "Reporter_Bloc": "Origin_Bloc",
        "Partner_Bloc": "Dest_Bloc",
        "Weight_KG": "Exp_Weight_KG",
        "Real_Trade_Value_USD": "Exp_Real_USD"
    })
    imp = df[df["Trade_Flow"] == "Import"].rename(columns={
        "Reporter_ISO3": "Dest_ISO3",
        "Partner_ISO3": "Origin_ISO3",
        "Reporter_Bloc": "Dest_Bloc",
        "Partner_Bloc": "Origin_Bloc",
        "Weight_KG": "Imp_Weight_KG",
        "Real_Trade_Value_USD": "Imp_Real_USD"
    })

    merge_keys = ["Year", "HS6_Code", "Origin_ISO3", "Dest_ISO3"]
    exp_grp = exp.groupby(merge_keys, as_index=False).agg({
        "Origin_Bloc": "first",
        "Dest_Bloc": "first",
        "Species_Group": "first",
        "Customs_Granularity": "first",
        "Conservation_Category": "first",
        "Threatened_Taxa_Flag": "first",
        "IUCN_Threat_Tier": "first",
        "Subheading_SCRI": "first",
        "Exp_Weight_KG": "sum",
        "Exp_Real_USD": "sum"
    })
    imp_grp = imp.groupby(merge_keys, as_index=False).agg({
        "Imp_Weight_KG": "sum",
        "Imp_Real_USD": "sum"
    })

    mirror = pd.merge(exp_grp, imp_grp, on=merge_keys, how="inner")
    print(f"   Matched {len(mirror):,} exact bilateral mirror transaction pairs across 1988-2024.")

    print("4. Calculating CIF/FOB transport-adjusted Mirror Discrepancy Indices...")
    mirror["Adj_Exp_Real_USD"] = mirror["Exp_Real_USD"] * (1.0 + THETA)
    max_val = np.maximum(mirror["Imp_Real_USD"], mirror["Adj_Exp_Real_USD"])
    mirror["MDI_Val"] = (mirror["Imp_Real_USD"] - mirror["Adj_Exp_Real_USD"]) / max_val
    mirror["Abs_MDI_Val"] = np.abs(mirror["MDI_Val"])
    mirror["Val_Gap_USD"] = mirror["Imp_Real_USD"] - mirror["Adj_Exp_Real_USD"]
    mirror["Abs_Val_Gap_USD"] = np.abs(mirror["Val_Gap_USD"])

    max_vol = np.maximum(mirror["Imp_Weight_KG"], mirror["Exp_Weight_KG"])
    mirror["MDI_Vol"] = (mirror["Imp_Weight_KG"] - mirror["Exp_Weight_KG"]) / max_vol
    mirror["Abs_MDI_Vol"] = np.abs(mirror["MDI_Vol"])
    mirror["Vol_Gap_KG"] = mirror["Imp_Weight_KG"] - mirror["Exp_Weight_KG"]

    # Sensitivity across theta in {0.08, 0.11, 0.15}
    sens_rows = []
    for theta_test, label_theta in [
        (0.08, "Low Freight Margin (theta = 0.08)"),
        (0.11, "Baseline Freight Margin (theta = 0.11)"),
        (0.15, "High Freight Margin (theta = 0.15)")
    ]:
        adj_x = mirror["Exp_Real_USD"] * (1.0 + theta_test)
        mx_v = np.maximum(mirror["Imp_Real_USD"], adj_x)
        mdi_v = (mirror["Imp_Real_USD"] - adj_x) / mx_v
        abs_mdi_v = np.abs(mdi_v)
        net_gap_b = (mirror["Imp_Real_USD"] - adj_x).sum() / 1e9
        abs_gap_b = np.abs(mirror["Imp_Real_USD"] - adj_x).sum() / 1e9

        mask_agg = mirror["Customs_Granularity"] == "Aggregated Multi-Taxa"
        mask_spec = mirror["Customs_Granularity"] == "Taxon-Specific"

        sens_rows.append({
            "Theta_Specification": label_theta,
            "Theta_Value": theta_test,
            "Matched_Pairs": int(len(mirror)),
            "Mean_MDI_Val": np.round(float(mdi_v.mean()), 4),
            "Mean_Abs_MDI_Val": np.round(float(abs_mdi_v.mean()), 4),
            "Severe_Mismatch_Rate_All_Pct": np.round(float((abs_mdi_v > 0.50).mean() * 100), 2),
            "Severe_Mismatch_Aggregated_Pct": np.round(float((abs_mdi_v[mask_agg] > 0.50).mean() * 100), 2),
            "Severe_Mismatch_Specific_Pct": np.round(float((abs_mdi_v[mask_spec] > 0.50).mean() * 100), 2),
            "Net_Val_Gap_USD_B": np.round(float(net_gap_b), 2),
            "Cumulative_Abs_Val_Gap_USD_B": np.round(float(abs_gap_b), 2)
        })
    pd.DataFrame(sens_rows).to_csv(DATA_DERIVED / "global_mirror_discrepancy_theta_sensitivity.csv", index=False)

    taxa_sens_rows = []
    for cat in [
        "Sea Cucumbers (Holothuroidea)",
        "Demersal Commercial Taxonomic Groups",
        "General Commercial",
        "High-Trophic Tunas (Thunnus spp.)",
        "Chondrichthyans (Sharks & Rays)",
        "Diadromous Eels (Anguilla spp.)",
        "Sturgeons & Caviar (Acipenseriformes)",
    ]:
        sub_c = mirror[mirror["Conservation_Category"] == cat]
        row_c = {"Category": cat, "Matched_Pairs": int(len(sub_c))}
        for th_val, th_col in [(0.08, "Severe_Mismatch_008_Pct"), (0.11, "Severe_Mismatch_011_Pct"), (0.15, "Severe_Mismatch_015_Pct")]:
            adj_c = sub_c["Exp_Real_USD"] * (1.0 + th_val)
            mdi_c = np.abs((sub_c["Imp_Real_USD"] - adj_c) / np.maximum(sub_c["Imp_Real_USD"], adj_c))
            row_c[th_col] = np.round(float((mdi_c > 0.50).mean() * 100), 2)
            if th_val == 0.11:
                row_c["Net_Val_Gap_USD_B"] = np.round(float((sub_c["Imp_Real_USD"] - adj_c).sum() / 1e9), 2)
                row_c["Abs_Val_Gap_USD_B"] = np.round(float(np.abs(sub_c["Imp_Real_USD"] - adj_c).sum() / 1e9), 2)
        taxa_sens_rows.append(row_c)
    pd.DataFrame(taxa_sens_rows).to_csv(DATA_DERIVED / "global_mirror_discrepancy_theta_sensitivity_by_taxa.csv", index=False)

    print("5. Computing Bilateral Taxonomic Aggregation Asymmetry (BTAR) & Entrepot Masking Ratio (ECR)...")
    df["Is_Aggregated_Weight_KG"] = np.where(df["Customs_Granularity"] == "Aggregated Multi-Taxa", df["Weight_KG"], 0.0)
    df["Is_Aggregated_Real_USD"] = np.where(df["Customs_Granularity"] == "Aggregated Multi-Taxa", df["Real_Trade_Value_USD"], 0.0)

    sov_flow = df.groupby(["Reporter_ISO3", "Reporter_Country", "Reporter_Bloc", "Trade_Flow"]).agg(
        Total_Weight_KG=("Weight_KG", "sum"),
        Agg_Weight_KG=("Is_Aggregated_Weight_KG", "sum"),
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Agg_Real_USD=("Is_Aggregated_Real_USD", "sum")
    ).reset_index()
    sov_flow["TAR_Vol_Pct"] = (sov_flow["Agg_Weight_KG"] / sov_flow["Total_Weight_KG"]) * 100.0
    sov_flow["TAR_Val_Pct"] = (sov_flow["Agg_Real_USD"] / sov_flow["Total_Real_USD"]) * 100.0

    piv_btar = sov_flow.pivot(
        index=["Reporter_ISO3", "Reporter_Country", "Reporter_Bloc"],
        columns="Trade_Flow",
        values=["Total_Weight_KG", "Agg_Weight_KG", "Total_Real_USD", "TAR_Vol_Pct", "TAR_Val_Pct"]
    ).reset_index()
    piv_btar.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in piv_btar.columns]
    piv_btar = piv_btar.dropna(subset=["Total_Weight_KG_Export", "Total_Weight_KG_Import"]).copy()
    piv_btar["BTAR_Volume_pp"] = np.round(piv_btar["TAR_Vol_Pct_Export"] - piv_btar["TAR_Vol_Pct_Import"], 2)
    piv_btar["BTAR_Value_pp"] = np.round(piv_btar["TAR_Val_Pct_Export"] - piv_btar["TAR_Val_Pct_Import"], 2)
    piv_btar["ECR_Ratio"] = np.round(piv_btar["Agg_Weight_KG_Export"] / piv_btar["Total_Weight_KG_Import"], 4)
    piv_btar["Total_Trade_USD_B"] = np.round((piv_btar["Total_Real_USD_Export"] + piv_btar["Total_Real_USD_Import"]) / 1e9, 2)
    piv_btar = piv_btar.sort_values("Total_Trade_USD_B", ascending=False)
    piv_btar.to_csv(DATA_DERIVED / "global_entrepot_masking_btar_ecr.csv", index=False)

    print("6. Generating Analytical Summaries & High-Dimensional Strategic Evasion FE Models (H3b)...")
    taxa_sum = mirror.groupby("Conservation_Category").agg(
        Pairs=("Year", "count"),
        Total_Adj_Exp_Real_USD_B=("Adj_Exp_Real_USD", lambda x: x.sum() / 1e9),
        Total_Imp_Real_USD_B=("Imp_Real_USD", lambda x: x.sum() / 1e9),
        Net_Val_Gap_USD_B=("Val_Gap_USD", lambda x: x.sum() / 1e9),
        Abs_Val_Gap_USD_B=("Abs_Val_Gap_USD", lambda x: x.sum() / 1e9),
        Mean_MDI_Val=("MDI_Val", "mean"),
        Median_MDI_Val=("MDI_Val", "median"),
        Mean_Abs_MDI_Val=("Abs_MDI_Val", "mean"),
        Mean_MDI_Vol=("MDI_Vol", "mean"),
        Mean_Abs_MDI_Vol=("Abs_MDI_Vol", "mean"),
        Severe_Mismatch_Rate=("Abs_MDI_Val", lambda x: (x > 0.50).mean() * 100)
    ).reset_index()
    taxa_sum.to_csv(DATA_DERIVED / "global_mirror_discrepancy_by_taxa.csv", index=False)

    gran_sum = mirror.groupby("Customs_Granularity").agg(
        Pairs=("Year", "count"),
        Total_Adj_Exp_Real_USD_B=("Adj_Exp_Real_USD", lambda x: x.sum() / 1e9),
        Total_Imp_Real_USD_B=("Imp_Real_USD", lambda x: x.sum() / 1e9),
        Net_Val_Gap_USD_B=("Val_Gap_USD", lambda x: x.sum() / 1e9),
        Abs_Val_Gap_USD_B=("Abs_Val_Gap_USD", lambda x: x.sum() / 1e9),
        Mean_MDI_Val=("MDI_Val", "mean"),
        Median_MDI_Val=("MDI_Val", "median"),
        Mean_Abs_MDI_Val=("Abs_MDI_Val", "mean"),
        Mean_Abs_MDI_Vol=("Abs_MDI_Vol", "mean"),
        Severe_Mismatch_Rate=("Abs_MDI_Val", lambda x: (x > 0.50).mean() * 100)
    ).reset_index()
    gran_sum.to_csv(DATA_DERIVED / "global_mirror_discrepancy_granularity_comparison.csv", index=False)

    # Construct covariates for Exporter x Importer x Year Fixed Effects Strategic Evasion Models
    mirror["Severe_50"] = (mirror["Abs_MDI_Val"] > 0.50).astype(float)
    mirror["Severe_30"] = (mirror["Abs_MDI_Val"] > 0.30).astype(float)
    mirror["Severe_70"] = (mirror["Abs_MDI_Val"] > 0.70).astype(float)
    mirror["Severe_Vol_50"] = (mirror["Abs_MDI_Vol"] > 0.50).astype(float)
    mirror["Is_Agg"] = (mirror["Customs_Granularity"] == "Aggregated Multi-Taxa").astype(float)
    mirror["Pooling_Index"] = 1.0 - mirror["Subheading_SCRI"].clip(0.0, 1.0)
    mirror["dyad_yr"] = mirror["Origin_ISO3"] + "_" + mirror["Dest_ISO3"] + "_" + mirror["Year"].astype(str)

    # Time-varying CITES Appendix I/II regulation active on subheading in year t
    cites_active = (
        ((mirror["Conservation_Category"] == "Sturgeons & Caviar (Acipenseriformes)") & (mirror["Year"] >= 1998)) |
        ((mirror["Conservation_Category"] == "Diadromous Eels (Anguilla spp.)") & (mirror["Year"] >= 2009)) |
        ((mirror["Conservation_Category"] == "Chondrichthyans (Sharks & Rays)") & (mirror["Year"] >= 2014)) |
        ((mirror["Conservation_Category"] == "Sea Cucumbers (Holothuroidea)") & (mirror["Year"] >= 2020))
    )
    mirror["CITES_Active"] = cites_active.astype(float)

    # Subheading baseline log median unit price (predetermined product value intensity, centered at global median)
    hs6_med_uv = mirror.assign(uv=mirror["Exp_Real_USD"] / mirror["Exp_Weight_KG"]).groupby("HS6_Code")["uv"].median()
    mirror["ln_HS6_UV"] = np.log(mirror["HS6_Code"].map(hs6_med_uv).clip(lower=0.1))
    mirror["ln_HS6_UV_c"] = mirror["ln_HS6_UV"] - float(mirror["ln_HS6_UV"].median())

    mirror["Pool_x_CITES"] = mirror["Pooling_Index"] * mirror["CITES_Active"]
    mirror["Pool_x_lnUV"] = mirror["Pooling_Index"] * mirror["ln_HS6_UV_c"]
    mirror["Agg_x_CITES"] = mirror["Is_Agg"] * mirror["CITES_Active"]
    mirror["Agg_x_lnUV"] = mirror["Is_Agg"] * mirror["ln_HS6_UV_c"]

    # Baseline exact backward-compatible FE regression
    dy_var = mirror.groupby("dyad_yr")["Is_Agg"].transform("std") > 0
    both_fe = mirror[dy_var].copy()
    both_fe["Sev_dm"] = both_fe["Severe_50"] - both_fe.groupby("dyad_yr")["Severe_50"].transform("mean")
    both_fe["Agg_dm"] = both_fe["Is_Agg"] - both_fe.groupby("dyad_yr")["Is_Agg"].transform("mean")
    fe_mod = sm.OLS(both_fe["Sev_dm"], both_fe[["Agg_dm"]]).fit(cov_type="cluster", cov_kwds={"groups": both_fe["HS6_Code"]})
    fe_beta_pp = float(fe_mod.params["Agg_dm"] * 100.0)
    fe_se_pp = float(fe_mod.bse["Agg_dm"] * 100.0)
    fe_pval = float(fe_mod.pvalues["Agg_dm"])
    fe_tval = float(fe_mod.tvalues["Agg_dm"])

    df_fe_res = pd.DataFrame([{
        "Model": "Within-Dyad-Year Fixed Effects (Eq. 9 Causal Proof)",
        "Observations": int(len(both_fe)),
        "Dyad_Year_Clusters": int(both_fe["dyad_yr"].nunique()),
        "HS6_Clusters": int(both_fe["HS6_Code"].nunique()),
        "Aggregated_Coefficient_pp": np.round(fe_beta_pp, 2),
        "Clustered_SE_pp": np.round(fe_se_pp, 2),
        "t_Statistic": np.round(fe_tval, 2),
        "P_Value": np.round(fe_pval, 6),
        "R_Squared_Within": np.round(float(fe_mod.rsquared), 4)
    }])
    df_fe_res.to_csv(DATA_DERIVED / "global_mirror_fe_regression_results.csv", index=False)

    # Extended Strategic Evasion Suite using pyfixest (Exporter x Importer x Year FEs, HS6-clustered SEs)
    evasion_specs = [
        ("M1: Binary Pooling (|MDI_Val| > 0.50)", "Severe_50 ~ Is_Agg | dyad_yr"),
        ("M2: Continuous Pooling (1 - SCRI_k, |MDI_Val| > 0.50)", "Severe_50 ~ Pooling_Index | dyad_yr"),
        ("M3: Strategic CITES & Luxury Value Interactions (|MDI_Val| > 0.50)",
         "Severe_50 ~ Pooling_Index + CITES_Active + Pool_x_CITES + ln_HS6_UV_c + Pool_x_lnUV | dyad_yr"),
        ("M4: Moderate Discordance Threshold (|MDI_Val| > 0.30)",
         "Severe_30 ~ Pooling_Index + CITES_Active + Pool_x_CITES + ln_HS6_UV_c + Pool_x_lnUV | dyad_yr"),
        ("M5: Extreme Discordance Threshold (|MDI_Val| > 0.70)",
         "Severe_70 ~ Pooling_Index + CITES_Active + Pool_x_CITES + ln_HS6_UV_c + Pool_x_lnUV | dyad_yr"),
        ("M6: Physical Weight Discordance (|MDI_Vol| > 0.50)",
         "Severe_Vol_50 ~ Pooling_Index + CITES_Active + Pool_x_CITES + ln_HS6_UV_c + Pool_x_lnUV | dyad_yr"),
        ("M7: Continuous |MDI_Val| Intensity (0 to 100 pp)",
         "Abs_MDI_Val ~ Pooling_Index + CITES_Active + Pool_x_CITES + ln_HS6_UV_c + Pool_x_lnUV | dyad_yr"),
    ]
    ev_rows = []
    for spec_name, fml in evasion_specs:
        fit_ev = pf.feols(fml, data=both_fe, vcov={"CRV1": "HS6_Code"})
        coefs = fit_ev.coef()
        ses = fit_ev.se()
        pvs = fit_ev.pvalue()
        for term in coefs.index:
            ev_rows.append({
                "Specification": spec_name,
                "Regressor": str(term),
                "Effect_pp": np.round(float(coefs[term]) * 100.0, 3),
                "HS6_Clustered_SE_pp": np.round(float(ses[term]) * 100.0, 3),
                "P_Value": np.round(float(pvs[term]), 6),
                "Observations": int(fit_ev._N),
            })
    df_ev = pd.DataFrame(ev_rows)
    out_ev = DATA_DERIVED / "global_mirror_strategic_evasion_fe_models.csv"
    df_ev.to_csv(out_ev, index=False)
    print(f"   -> Saved Strategic Evasion FE suite: {out_ev}")
    print("\n--- Within-Dyad-Year Strategic Evasion FE Highlights ---")
    print(df_ev[df_ev["Specification"].str.startswith(("M1", "M2", "M3", "M6"))].to_string(index=False))

    # Extensive Margin Orphan Flow Analysis (also broken out by CITES-regulated vs Unlisted)
    exp_all_flows = exp.groupby(["Year", "HS6_Code", "Customs_Granularity", "Conservation_Category", "Origin_ISO3", "Dest_ISO3"], as_index=False).agg(Exp_USD=("Exp_Real_USD", "sum"))
    imp_all_flows = imp.groupby(["Year", "HS6_Code", "Customs_Granularity", "Conservation_Category", "Origin_ISO3", "Dest_ISO3"], as_index=False).agg(Imp_USD=("Imp_Real_USD", "sum"))
    mir_full = pd.merge(
        exp_all_flows, imp_all_flows,
        on=["Year", "HS6_Code", "Customs_Granularity", "Conservation_Category", "Origin_ISO3", "Dest_ISO3"],
        how="outer", indicator=True
    )

    orphan_rows = []
    for gran in ["Aggregated Multi-Taxa", "Taxon-Specific"]:
        sub_g = mir_full[mir_full["Customs_Granularity"] == gran]
        tot_g = len(sub_g)
        matched_g = int((sub_g["_merge"] == "both").sum())
        exp_only_g = int((sub_g["_merge"] == "left_only").sum())
        imp_only_g = int((sub_g["_merge"] == "right_only").sum())
        orphans_g = exp_only_g + imp_only_g
        orphan_rows.append({
            "Customs_Granularity": gran,
            "Total_Unique_Transactions": tot_g,
            "Bilateral_Matched_Count": matched_g,
            "Bilateral_Matched_Pct": np.round(matched_g / tot_g * 100, 2),
            "Exporter_Only_Orphans": exp_only_g,
            "Importer_Only_Orphans": imp_only_g,
            "Total_Orphan_Transactions": orphans_g,
            "Orphan_Rate_Pct": np.round(orphans_g / tot_g * 100, 2)
        })
    pd.DataFrame(orphan_rows).to_csv(DATA_DERIVED / "global_mirror_extensive_margin_orphans.csv", index=False)

    # CITES x Granularity Extensive-Margin Orphan Forensics (Subheading Code-Switching Test, H3b)
    mir_full["Orphan"] = (mir_full["_merge"] != "both").astype(float)
    mir_full["Is_Agg"] = (mir_full["Customs_Granularity"] == "Aggregated Multi-Taxa").astype(float)
    mir_full["CITES_Active"] = (
        ((mir_full["Conservation_Category"] == "Sturgeons & Caviar (Acipenseriformes)") & (mir_full["Year"] >= 1998)) |
        ((mir_full["Conservation_Category"] == "Diadromous Eels (Anguilla spp.)") & (mir_full["Year"] >= 2009)) |
        ((mir_full["Conservation_Category"] == "Chondrichthyans (Sharks & Rays)") & (mir_full["Year"] >= 2014)) |
        ((mir_full["Conservation_Category"] == "Sea Cucumbers (Holothuroidea)") & (mir_full["Year"] >= 2020))
    ).astype(float)
    cites_orph = mir_full.groupby(["CITES_Active", "Customs_Granularity"], as_index=False).agg(
        Total_Declarations=("Orphan", "count"),
        Orphan_Declarations=("Orphan", "sum"),
        Orphan_Rate_Pct=("Orphan", lambda x: np.round(x.mean() * 100.0, 2)),
        Matched_Rate_Pct=("Orphan", lambda x: np.round((1.0 - x.mean()) * 100.0, 2)),
    )
    mir_full["dyad_yr"] = mir_full["Origin_ISO3"] + "_" + mir_full["Dest_ISO3"] + "_" + mir_full["Year"].astype(str)
    mir_full["Agg_x_CITES"] = mir_full["Is_Agg"] * mir_full["CITES_Active"]
    fit_co = pf.feols("Orphan ~ Is_Agg + CITES_Active + Agg_x_CITES | dyad_yr", data=mir_full, vcov={"CRV1": "HS6_Code"})
    cites_orph["WithinDyadYr_IsAgg_pp"] = np.round(float(fit_co.coef()["Is_Agg"]) * 100.0, 2)
    cites_orph["WithinDyadYr_CITES_pp"] = np.round(float(fit_co.coef()["CITES_Active"]) * 100.0, 2)
    cites_orph["WithinDyadYr_AggxCITES_pp"] = np.round(float(fit_co.coef()["Agg_x_CITES"]) * 100.0, 2)
    cites_orph["WithinDyadYr_N"] = int(fit_co._N)
    cites_orph.to_csv(DATA_DERIVED / "global_mirror_cites_orphan_forensics.csv", index=False)
    print("   -> Saved CITES x Granularity extensive-margin orphan forensics:")
    print(cites_orph.to_string(index=False))

    bloc_sum = mirror.groupby(["Origin_Bloc", "Dest_Bloc"]).agg(
        Pairs=("Year", "count"),
        Total_Adj_Exp_USD_B=("Adj_Exp_Real_USD", lambda x: x.sum() / 1e9),
        Total_Imp_USD_B=("Imp_Real_USD", lambda x: x.sum() / 1e9),
        Net_Val_Gap_USD_B=("Val_Gap_USD", lambda x: x.sum() / 1e9),
        Abs_Val_Gap_USD_B=("Abs_Val_Gap_USD", lambda x: x.sum() / 1e9),
        Mean_MDI_Val=("MDI_Val", "mean"),
        Mean_Abs_MDI_Val=("Abs_MDI_Val", "mean"),
        Severe_Mismatch_Rate=("Abs_MDI_Val", lambda x: (x > 0.50).mean() * 100)
    ).reset_index()
    bloc_sum.to_csv(DATA_DERIVED / "global_mirror_discrepancy_by_bloc.csv", index=False)

    timeline_sum = mirror.groupby("Year").agg(
        Pairs=("HS6_Code", "count"),
        Total_Adj_Exp_USD_B=("Adj_Exp_Real_USD", lambda x: x.sum() / 1e9),
        Total_Imp_USD_B=("Imp_Real_USD", lambda x: x.sum() / 1e9),
        Net_Val_Gap_USD_B=("Val_Gap_USD", lambda x: x.sum() / 1e9),
        Abs_Val_Gap_USD_B=("Abs_Val_Gap_USD", lambda x: x.sum() / 1e9),
        Mean_MDI_Val=("MDI_Val", "mean"),
        Mean_Abs_MDI_Val=("Abs_MDI_Val", "mean"),
        Mean_Abs_MDI_Vol=("Abs_MDI_Vol", "mean"),
        Severe_Mismatch_Rate=("Abs_MDI_Val", lambda x: (x > 0.50).mean() * 100)
    ).reset_index()
    timeline_sum.to_csv(DATA_DERIVED / "global_mirror_discrepancy_annual_timeline.csv", index=False)

    print("=" * 80)
    print("MODULE 12 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
