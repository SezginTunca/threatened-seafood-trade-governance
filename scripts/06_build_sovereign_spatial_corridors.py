#!/usr/bin/env python3
"""
Module 06: Sovereign Spatial Indicators and Bilateral Trade Corridors
=====================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Computes national Taxonomic Aggregation Ratios, sovereign Trophic Balance of
Trade indicators, and top bilateral geodesic corridors for threatened and
CITES-regulated marine taxa.

Outputs:
  - data/derived/sovereign_customs_aggregation_tar.csv
  - data/derived/sovereign_trophic_balance_tbt.csv
  - data/derived/top_bilateral_threatened_taxa_corridors.csv
  - data/derived/top_bilateral_trophic_flow_corridors.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"

MASTER_PARQUET = DATA_RAW / "master_global_bilateral_panel.parquet"
CONCORDANCE_CSV = DATA_DERIVED / "global_hs6_conservation_concordance.csv"

# CPI Base (2024 = 313.0)
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
    print("MODULE 06: BUILDING SOVEREIGN SPATIAL DATASETS FOR CARTOGRAPHY")
    print("=" * 80)

    print("1. Loading concordance taxonomy...")
    df_conc = pd.read_csv(CONCORDANCE_CSV)
    df_conc["HS6_Code"] = df_conc["HS6_Code"].astype(str).str.zfill(6)
    conc_cols = [
        "HS6_Code", "Customs_Granularity", "Conservation_Category",
        "Threatened_Taxa_Flag", "IUCN_Threat_Tier"
    ]
    df_conc_clean = df_conc[conc_cols].drop_duplicates(subset=["HS6_Code"])

    print("2. Loading master bilateral panel...")
    cols_load = [
        "Year", "Trade_Flow", "Reporter_ISO3", "Reporter_Country", "Reporter_Bloc",
        "Partner_ISO3", "Partner_Country", "Partner_Bloc", "HS6_Code",
        "Weight_KG", "Trade_Value_USD", "Trophic_Level", "UV_Outlier_Flag"
    ]
    df = pd.read_parquet(MASTER_PARQUET, columns=cols_load)
    df = df[
        (df["Weight_KG"] > 0) &
        (df["Trade_Value_USD"] > 0) &
        (df["Reporter_ISO3"].notna()) &
        (df["Partner_ISO3"].notna()) &
        (~df["UV_Outlier_Flag"])
    ].copy()

    df["HS6_Code"] = df["HS6_Code"].astype(str).str.zfill(6)
    df["CPI"] = df["Year"].map(CPI_DEFLATOR).fillna(BASE_CPI)
    df["Real_Trade_Value_USD"] = df["Trade_Value_USD"] * (BASE_CPI / df["CPI"])

    # Merge concordance
    df = df.merge(df_conc_clean, on="HS6_Code", how="left")
    df["Customs_Granularity"] = df["Customs_Granularity"].fillna("Aggregated Multi-Taxa")
    df["Conservation_Category"] = df["Conservation_Category"].fillna("General Commercial")
    df["Threatened_Taxa_Flag"] = df["Threatened_Taxa_Flag"].fillna(False)
    df["Trophic_Biomass_KG"] = df["Weight_KG"] * df["Trophic_Level"]

    print("3. Computing Sovereign Customs Aggregation Index (TAR)...")
    exp = df[df["Trade_Flow"] == "Export"].copy()
    sovereign_tar = exp.groupby(["Reporter_ISO3", "Reporter_Country", "Reporter_Bloc"]).agg(
        Total_Export_Real_USD=("Real_Trade_Value_USD", "sum"),
        Aggregated_Export_Real_USD=("Real_Trade_Value_USD", lambda x: x[exp.loc[x.index, "Customs_Granularity"] != "Taxon-Specific"].sum()),
        Specific_Export_Real_USD=("Real_Trade_Value_USD", lambda x: x[exp.loc[x.index, "Customs_Granularity"] == "Taxon-Specific"].sum()),
        Total_Export_Weight_KG=("Weight_KG", "sum"),
        Aggregated_Export_Weight_KG=("Weight_KG", lambda x: x[exp.loc[x.index, "Customs_Granularity"] != "Taxon-Specific"].sum()),
        Specific_Export_Weight_KG=("Weight_KG", lambda x: x[exp.loc[x.index, "Customs_Granularity"] == "Taxon-Specific"].sum())
    ).reset_index()

    sovereign_tar["TAR_Value_Percent"] = (sovereign_tar["Aggregated_Export_Real_USD"] / sovereign_tar["Total_Export_Real_USD"]) * 100
    sovereign_tar["TAR_Volume_Percent"] = (sovereign_tar["Aggregated_Export_Weight_KG"] / sovereign_tar["Total_Export_Weight_KG"]) * 100
    sovereign_tar["Total_Export_Real_USD_B"] = sovereign_tar["Total_Export_Real_USD"] / 1e9
    sovereign_tar["Total_Export_Weight_MMT"] = sovereign_tar["Total_Export_Weight_KG"] / 1e9

    out_tar = DATA_DERIVED / "sovereign_customs_aggregation_tar.csv"
    sovereign_tar.to_csv(out_tar, index=False)
    print(f"   Exported: {out_tar}")

    print("4. Computing Sovereign Trophic Balance of Trade (TBT)...")
    df["Trophic_Biomass_KG"] = df["Weight_KG"] * df["Trophic_Level"]

    sovereign_trophic = df.groupby(["Reporter_ISO3", "Reporter_Country", "Reporter_Bloc", "Trade_Flow"]).agg(
        Total_Weight_KG=("Weight_KG", "sum"),
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Trophic_Biomass_KG=("Trophic_Biomass_KG", "sum")
    ).reset_index()

    sovereign_trophic["MTL"] = sovereign_trophic["Total_Trophic_Biomass_KG"] / sovereign_trophic["Total_Weight_KG"]
    sovereign_trophic["Unit_Price_USD_per_KG"] = sovereign_trophic["Total_Real_USD"] / sovereign_trophic["Total_Weight_KG"]

    pivot_tbt = sovereign_trophic.pivot(
        index=["Reporter_ISO3", "Reporter_Country", "Reporter_Bloc"],
        columns="Trade_Flow",
        values=["MTL", "Total_Weight_KG", "Total_Real_USD", "Unit_Price_USD_per_KG"]
    ).reset_index()

    pivot_tbt.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_tbt.columns]

    pivot_tbt["TBT"] = pivot_tbt["MTL_Export"] - pivot_tbt["MTL_Import"]
    pivot_tbt["Net_Biomass_Export_MT"] = (pivot_tbt["Total_Weight_KG_Export"] - pivot_tbt["Total_Weight_KG_Import"]) / 1e3
    pivot_tbt["Net_Value_USD_B"] = (pivot_tbt["Total_Real_USD_Export"] - pivot_tbt["Total_Real_USD_Import"]) / 1e9
    pivot_tbt["Total_Export_Real_USD_B"] = pivot_tbt["Total_Real_USD_Export"] / 1e9

    out_tbt = DATA_DERIVED / "sovereign_trophic_balance_tbt.csv"
    pivot_tbt.to_csv(out_tbt, index=False)
    print(f"   Exported: {out_tbt}")

    print("5. Extracting Bilateral Threatened Taxa Corridors...")
    threat_exp = exp[exp["Threatened_Taxa_Flag"]].copy()
    threat_corridors = threat_exp.groupby([
        "Reporter_ISO3", "Reporter_Country", "Partner_ISO3", "Partner_Country", "Conservation_Category"
    ]).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_MT=("Weight_KG", lambda x: x.sum() / 1e3),
        Flow_Count=("Year", "count")
    ).reset_index().sort_values("Total_Real_USD", ascending=False)

    out_threat_corridors = DATA_DERIVED / "top_bilateral_threatened_taxa_corridors.csv"
    threat_corridors.to_csv(out_threat_corridors, index=False)
    print(f"   Exported: {out_threat_corridors}")

    print("6. Extracting Top Global Biomass Corridors with Trophic Tags...")
    biomass_corridors = exp.groupby([
        "Reporter_ISO3", "Reporter_Country", "Partner_ISO3", "Partner_Country"
    ]).agg(
        Total_Real_USD_M=("Real_Trade_Value_USD", lambda x: x.sum() / 1e6),
        Total_Weight_MMT=("Weight_KG", lambda x: x.sum() / 1e9),
        Total_Weight_KG=("Weight_KG", "sum"),
        Total_Trophic_Biomass_KG=("Trophic_Biomass_KG", "sum")
    ).reset_index()

    biomass_corridors["Mean_Trophic_Level"] = biomass_corridors["Total_Trophic_Biomass_KG"] / biomass_corridors["Total_Weight_KG"]
    biomass_corridors = biomass_corridors.sort_values("Total_Weight_MMT", ascending=False)

    out_biomass = DATA_DERIVED / "top_bilateral_trophic_flow_corridors.csv"
    biomass_corridors.to_csv(out_biomass, index=False)
    print(f"   Exported: {out_biomass}")

    print("=" * 80)
    print("SOVEREIGN SPATIAL DATASETS READY FOR HIGH-RESOLUTION CARTOGRAPHY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
