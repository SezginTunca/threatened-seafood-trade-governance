#!/usr/bin/env python3
"""
Module 11: Marine Trophic Transfers and Live-Weight-Equivalent Balances (Eqs. 11-11b)
=====================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Computes export and import Mean Trophic Levels (MTL_X, MTL_M), net Trophic
Balance of Trade (TBT = MTL_X - MTL_M), and FAO Live-Weight Equivalent (LWE)
conversions across 210 reporting jurisdictions and eight regional trade blocs
(Supplementary Table S8).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"

MASTER_PARQUET = DATA_RAW / "master_global_bilateral_panel.parquet"

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


def main():
    print("=" * 80)
    print("MODULE 11: GLOBAL TROPHIC LEVEL TRANSFERS & BIOMASS REDISTRIBUTION")
    print("=" * 80)

    if not MASTER_PARQUET.exists():
        print(f"Error: {MASTER_PARQUET} not found.")
        sys.exit(1)

    print("1. Loading master panel with trophic and geographic dimensions...")
    cols_load = [
        "Year", "Trade_Flow", "Reporter_ISO3", "Partner_ISO3",
        "Reporter_Bloc", "Partner_Bloc", "Weight_KG",
        "Trade_Value_USD", "Processing_Form", "Species_Group", "Trophic_Level", "UV_Outlier_Flag"
    ]
    df = pd.read_parquet(MASTER_PARQUET, columns=cols_load)
    print(f"   Loaded {len(df):,} records.")

    df = df[
        (df["Year"] <= 2024) &
        (df["Weight_KG"] > 0) &
        (df["Trade_Value_USD"] > 0) &
        (df["Trophic_Level"].notna()) &
        (df["Reporter_Bloc"].notna()) &
        (~df["UV_Outlier_Flag"])
    ].copy()

    df["CPI"] = df["Year"].map(CPI_DEFLATOR).fillna(BASE_CPI)
    df["Real_Trade_Value_USD"] = df["Trade_Value_USD"] * (BASE_CPI / df["CPI"])
    df["Weight_MT"] = df["Weight_KG"] / 1000.0

    conditions = [
        df["Trophic_Level"] <= 2.8,
        (df["Trophic_Level"] > 2.8) & (df["Trophic_Level"] < 3.8),
        df["Trophic_Level"] >= 3.8
    ]
    labels = ["Low Trophic (TL <= 2.8)", "Mid Trophic (2.8 < TL < 3.8)", "Apex / High Trophic (TL >= 3.8)"]
    df["Trophic_Tier"] = np.select(conditions, labels, default="Mid Trophic (2.8 < TL < 3.8)")

    df["Trophic_Biomass_KG"] = df["Weight_KG"] * df["Trophic_Level"]

    print("2. Calculating Annual Mean Trophic Level (MTL) and TBT by Regional Bloc...")
    ann_bloc = df.groupby(["Year", "Reporter_Bloc", "Trade_Flow"]).agg(
        Total_Weight_MT=("Weight_MT", "sum"),
        Total_Real_USD_B=("Real_Trade_Value_USD", lambda x: x.sum() / 1e9),
        Total_Trophic_Biomass_KG=("Trophic_Biomass_KG", "sum"),
        Total_Weight_KG=("Weight_KG", "sum")
    ).reset_index()

    ann_bloc["MTL"] = ann_bloc["Total_Trophic_Biomass_KG"] / ann_bloc["Total_Weight_KG"]
    ann_bloc["Real_Unit_Value_USD_per_KG"] = (ann_bloc["Total_Real_USD_B"] * 1e9) / ann_bloc["Total_Weight_KG"]

    pivot_mtl = ann_bloc.pivot(
        index=["Year", "Reporter_Bloc"],
        columns="Trade_Flow",
        values=["MTL", "Total_Weight_MT", "Total_Real_USD_B", "Real_Unit_Value_USD_per_KG", "Total_Trophic_Biomass_KG", "Total_Weight_KG"]
    ).reset_index()
    pivot_mtl.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_mtl.columns]

    pivot_mtl["TBT"] = pivot_mtl["MTL_Export"] - pivot_mtl["MTL_Import"]
    pivot_mtl["Net_Biomass_Export_MT"] = pivot_mtl["Total_Weight_MT_Export"] - pivot_mtl["Total_Weight_MT_Import"]
    pivot_mtl["Net_Trade_Balance_USD_B"] = pivot_mtl["Total_Real_USD_B_Export"] - pivot_mtl["Total_Real_USD_B_Import"]
    pivot_mtl["Unit_Value_Premium_Exp_over_Imp"] = pivot_mtl["Real_Unit_Value_USD_per_KG_Export"] - pivot_mtl["Real_Unit_Value_USD_per_KG_Import"]

    out_ann = DATA_DERIVED / "annual_trophic_level_by_bloc.csv"
    pivot_mtl.to_csv(out_ann, index=False)
    print(f"   Exported annual bloc trophic metrics: {out_ann}")

    # Live-Weight Equivalent (LWE) conversion check for TBT invariance (FAO standard conversion factors)
    lwe_map = {
        "Live": 1.00,
        "Live / Fresh / Chilled Shellfish": 1.00,
        "Fresh / Chilled Whole Round": 1.10,
        "Frozen Whole Round": 1.12,
        "Fresh / Chilled Fillet & Meat": 2.60,
        "Frozen Fillet & Meat": 2.70,
        "Prepared / Preserved / Canned": 1.85,
        "Smoked / Cured / Dried / Salted": 3.20,
        "Meals / Flours / Pellets": 4.50,
        "Other Form": 1.50,
    }
    df["CF"] = df["Processing_Form"].map(lwe_map).fillna(1.50)
    df["LWE_KG"] = df["Weight_KG"] * df["CF"]
    df["TL_LWE"] = df["LWE_KG"] * df["Trophic_Level"]

    g_lwe = df.groupby(["Reporter_Bloc", "Trade_Flow"]).agg(
        LWE_KG=("LWE_KG", "sum"),
        TL_LWE=("TL_LWE", "sum")
    ).reset_index()
    g_lwe["MTL_LWE"] = g_lwe["TL_LWE"] / g_lwe["LWE_KG"]
    p_lwe = g_lwe.pivot(index="Reporter_Bloc", columns="Trade_Flow", values="MTL_LWE").reset_index()
    p_lwe["Mean_TBT_LWE"] = np.round(p_lwe["Export"] - p_lwe["Import"], 3)
    p_lwe_dict = dict(zip(p_lwe["Reporter_Bloc"], p_lwe["Mean_TBT_LWE"]))

    # Overall 37-year Bloc Summary (Volume-Weighted Cumulative MTL and TBT so MTL_Exp - MTL_Imp == TBT holds identically)
    bloc_cum = ann_bloc.groupby(["Reporter_Bloc", "Trade_Flow"]).agg(
        Cum_Weight_MT=("Total_Weight_MT", "sum"),
        Cum_Weight_KG=("Total_Weight_KG", "sum"),
        Cum_Trophic_Biomass_KG=("Total_Trophic_Biomass_KG", "sum"),
        Cum_Real_USD_B=("Total_Real_USD_B", "sum")
    ).reset_index()
    bloc_cum["Cum_MTL"] = bloc_cum["Cum_Trophic_Biomass_KG"] / bloc_cum["Cum_Weight_KG"]
    bloc_cum["Cum_Unit_Price"] = (bloc_cum["Cum_Real_USD_B"] * 1e9) / bloc_cum["Cum_Weight_KG"]

    piv_cum = bloc_cum.pivot(
        index="Reporter_Bloc",
        columns="Trade_Flow",
        values=["Cum_MTL", "Cum_Weight_MT", "Cum_Real_USD_B", "Cum_Unit_Price"]
    ).reset_index()
    piv_cum.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in piv_cum.columns]

    mtl_exp_r = np.round(piv_cum["Cum_MTL_Export"], 3)
    mtl_imp_r = np.round(piv_cum["Cum_MTL_Import"], 3)
    bloc_overall = pd.DataFrame({
        "Reporter_Bloc": piv_cum["Reporter_Bloc"],
        "Mean_MTL_Export": mtl_exp_r,
        "Mean_MTL_Import": mtl_imp_r,
        "Mean_TBT": np.round(mtl_exp_r - mtl_imp_r, 3),
        "Mean_TBT_LWE": piv_cum["Reporter_Bloc"].map(p_lwe_dict),
        "Cumulative_Export_MT": np.round(piv_cum["Cum_Weight_MT_Export"], 2),
        "Cumulative_Import_MT": np.round(piv_cum["Cum_Weight_MT_Import"], 2),
        "Cumulative_Export_USD_B": np.round(piv_cum["Cum_Real_USD_B_Export"], 2),
        "Cumulative_Import_USD_B": np.round(piv_cum["Cum_Real_USD_B_Import"], 2),
        "Cumulative_Net_Biomass_MT": np.round(piv_cum["Cum_Weight_MT_Export"] - piv_cum["Cum_Weight_MT_Import"], 2),
        "Cumulative_Net_Value_USD_B": np.round(piv_cum["Cum_Real_USD_B_Export"] - piv_cum["Cum_Real_USD_B_Import"], 2),
        "Mean_Unit_Price_Export": np.round(piv_cum["Cum_Unit_Price_Export"], 2),
        "Mean_Unit_Price_Import": np.round(piv_cum["Cum_Unit_Price_Import"], 2)
    }).sort_values("Mean_TBT", ascending=False)

    out_bloc_tbt = DATA_DERIVED / "bloc_trophic_balance_of_trade.csv"
    bloc_overall.to_csv(out_bloc_tbt, index=False)
    print(f"   Exported 37-year cumulative bloc TBT: {out_bloc_tbt}")

    # Export dedicated LWE sensitivity comparison table
    df_lwe_comp = bloc_overall[["Reporter_Bloc", "Mean_MTL_Export", "Mean_MTL_Import", "Mean_TBT", "Mean_TBT_LWE"]].copy()
    df_lwe_comp["Sign_Preserved"] = (np.sign(df_lwe_comp["Mean_TBT"]) == np.sign(df_lwe_comp["Mean_TBT_LWE"]))
    out_lwe_comp = DATA_DERIVED / "bloc_trophic_balance_lwe_comparison.csv"
    df_lwe_comp.to_csv(out_lwe_comp, index=False)
    print(f"   Exported LWE TBT sensitivity comparison: {out_lwe_comp}")

    print("3. Analyzing Global Bilateral Origin-Destination Trophic Flows...")
    df_exp = df[df["Trade_Flow"] == "Export"].copy()
    bilat_trophic = df_exp.groupby(["Reporter_Bloc", "Partner_Bloc"]).agg(
        Total_Weight_MT=("Weight_MT", "sum"),
        Total_Real_USD_B=("Real_Trade_Value_USD", lambda x: x.sum() / 1e9),
        Total_Trophic_Biomass_KG=("Trophic_Biomass_KG", "sum"),
        Total_Weight_KG=("Weight_KG", "sum")
    ).reset_index()

    bilat_trophic["Mean_Trophic_Level"] = bilat_trophic["Total_Trophic_Biomass_KG"] / bilat_trophic["Total_Weight_KG"]
    bilat_trophic["Real_Unit_Value_USD_per_KG"] = (bilat_trophic["Total_Real_USD_B"] * 1e9) / bilat_trophic["Total_Weight_KG"]
    bilat_trophic = bilat_trophic.drop(columns=["Total_Trophic_Biomass_KG", "Total_Weight_KG"])

    out_bilat = DATA_DERIVED / "global_bilateral_trophic_flows.csv"
    bilat_trophic.to_csv(out_bilat, index=False)
    print(f"   Exported bilateral origin-destination trophic flows: {out_bilat}")

    print("4. Quantifying Economic Summary across Trophic Tiers...")
    trophic_tier_sum = df.groupby(["Trophic_Tier", "Trade_Flow"]).agg(
        Total_Weight_MT=("Weight_MT", "sum"),
        Total_Real_USD_B=("Real_Trade_Value_USD", lambda x: x.sum() / 1e9),
        Total_Trophic_Biomass_KG=("Trophic_Biomass_KG", "sum"),
        Total_Weight_KG=("Weight_KG", "sum"),
        Total_Real_USD=("Real_Trade_Value_USD", "sum")
    ).reset_index()

    trophic_tier_sum["Mean_Trophic_Level"] = trophic_tier_sum["Total_Trophic_Biomass_KG"] / trophic_tier_sum["Total_Weight_KG"]
    trophic_tier_sum["Real_Unit_Value_USD_per_KG"] = trophic_tier_sum["Total_Real_USD"] / trophic_tier_sum["Total_Weight_KG"]
    trophic_tier_sum = trophic_tier_sum.drop(columns=["Total_Trophic_Biomass_KG", "Total_Weight_KG", "Total_Real_USD"])

    out_tier = DATA_DERIVED / "trophic_tier_economic_summary.csv"
    trophic_tier_sum.to_csv(out_tier, index=False)
    print(f"   Exported trophic tier economic summary: {out_tier}")

    print("\n--- Key Empirical Highlights: 37-Year Trophic Balance of Trade (TBT) by Bloc ---")
    cols_disp = [
        "Reporter_Bloc", "Mean_MTL_Export", "Mean_MTL_Import", "Mean_TBT",
        "Mean_Unit_Price_Export", "Mean_Unit_Price_Import", "Cumulative_Net_Biomass_MT"
    ]
    print(bloc_overall[cols_disp].to_string(index=False))

    print("\n--- Key Empirical Highlights: Trophic Tier Distribution ---")
    print(trophic_tier_sum.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
