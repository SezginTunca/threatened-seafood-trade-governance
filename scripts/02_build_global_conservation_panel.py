#!/usr/bin/env python3
"""
Module 02: Bilateral Trade Panel Construction and Real Price Deflation (Eq. 3)
==============================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Processes the harmonized bilateral UN Comtrade panel (1988-2024), attaches
conservation and trophic concordances, deflates trade values to constant 2024
US dollars using the US BLS CPI-U series (2024 = 313.0), filters unit-value
outliers, and exports derived panel summaries.

Outputs (written locally when bilateral data are present):
  - data/derived/annual_global_conservation_trade_summary.csv
  - data/derived/annual_taxa_threat_tier_summary.csv
  - data/derived/bilateral_conservation_flows_summary.csv
  - data/derived/global_threatened_taxa_bilateral_panel.parquet
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"

MASTER_PARQUET = DATA_RAW / "master_global_bilateral_panel.parquet"
CONCORDANCE_CSV = DATA_DERIVED / "global_hs6_conservation_concordance.csv"

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
    print("MODULE 02: BUILDING GLOBAL CONSERVATION AND THREATENED TAXA PANEL")
    print("=" * 80)

    if not MASTER_PARQUET.exists():
        print(f"Error: Master parquet not found at {MASTER_PARQUET}")
        sys.exit(1)

    print(f"1. Loading conservation concordance: {CONCORDANCE_CSV}")
    df_concordance = pd.read_csv(CONCORDANCE_CSV)
    df_concordance["HS6_Code"] = df_concordance["HS6_Code"].astype(str).str.zfill(6)

    cols = [
        "Year", "Trade_Flow", "Reporter_ISO3", "Reporter_Country", "Reporter_Bloc",
        "Partner_ISO3", "Partner_Country", "Partner_Bloc",
        "HS6_Code", "Weight_KG", "Trade_Value_USD",
        "Species_Group", "Processing_Form", "Trophic_Level", "Production_Typology",
        "Unit_Price_USD_per_KG", "UV_Outlier_Flag"
    ]

    print(f"2. Reading master parquet panel from: {MASTER_PARQUET}...")
    df = pd.read_parquet(MASTER_PARQUET, columns=cols)
    print(f"   -> Loaded {len(df):,} total bilateral records across {df['Year'].nunique()} years ({df['Year'].min()}–{df['Year'].max()})")

    # Truncate incomplete 2025 reporting lag year to establish canonical 1988-2024 complete analytical panel
    # (In 2025, headings 0305-1605 have 0 reported records in Comtrade, creating an artificial composition cliff)
    df = df[
        (df["Year"] <= 2024) &
        (df["Weight_KG"] > 0) &
        (df["Trade_Value_USD"] > 0) &
        (~df["UV_Outlier_Flag"])
    ].copy()
    print(f"   -> Retained {len(df):,} clean validated records (1988–2024 complete window, 37 years) after outlier exclusion")

    df["HS6_Code"] = df["HS6_Code"].astype(str).str.zfill(6)

    print("3. Merging conservation, tier, and threat classifications...")
    df = df.merge(
        df_concordance[[
            "HS6_Code", "Customs_Granularity", "Taxonomic_Tier", "Conservation_Category",
            "CITES_Status", "IUCN_Threat_Tier", "Threatened_Taxa_Flag"
        ]],
        on="HS6_Code",
        how="left"
    )
    df["Customs_Granularity"] = df["Customs_Granularity"].fillna("Aggregated Multi-Taxa")
    df["Taxonomic_Tier"] = df["Taxonomic_Tier"].fillna("Tier 2: Clade-Pooled Multi-Species")
    df["Conservation_Category"] = df["Conservation_Category"].fillna("General Commercial")
    df["Threatened_Taxa_Flag"] = df["Threatened_Taxa_Flag"].fillna(False)

    print("4. Applying real price deflation (base year 2024 = 313.0 BLS CPI-U)...")
    df["CPI"] = df["Year"].map(CPI_DEFLATOR).fillna(BASE_CPI)
    df["Real_Trade_Value_USD"] = df["Trade_Value_USD"] * (BASE_CPI / df["CPI"])

    df["Real_Unit_Price_USD_per_KG"] = np.where(
        df["Weight_KG"] > 0,
        df["Real_Trade_Value_USD"] / df["Weight_KG"],
        np.nan
    )

    print("5. Generating annual conservation category summary...")
    ann_cat = df.groupby(["Year", "Conservation_Category", "Trade_Flow"]).agg(
        Total_Nominal_USD=("Trade_Value_USD", "sum"),
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum"),
        Active_Reporters=("Reporter_ISO3", "nunique"),
        Active_Partners=("Partner_ISO3", "nunique"),
        Bilateral_Pairs=("Trade_Value_USD", "count")
    ).reset_index()

    ann_cat["Total_Real_USD_M"] = ann_cat["Total_Real_USD"] / 1e6
    ann_cat["Total_Weight_MT"] = ann_cat["Total_Weight_KG"] / 1e3
    ann_cat["Real_Unit_Value_USD_per_KG"] = np.where(
        ann_cat["Total_Weight_KG"] > 0,
        ann_cat["Total_Real_USD"] / ann_cat["Total_Weight_KG"],
        np.nan
    )
    out_ann_cat = DATA_DERIVED / "annual_global_conservation_trade_summary.csv"
    ann_cat.to_csv(out_ann_cat, index=False)
    print(f"   -> Saved: {out_ann_cat}")

    print("6. Generating annual threat tier summary...")
    ann_threat = df.groupby(["Year", "IUCN_Threat_Tier", "Trade_Flow"]).agg(
        Total_Nominal_USD=("Trade_Value_USD", "sum"),
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum"),
        Bilateral_Pairs=("Trade_Value_USD", "count")
    ).reset_index()
    ann_threat["Total_Real_USD_M"] = ann_threat["Total_Real_USD"] / 1e6
    ann_threat["Total_Weight_MT"] = ann_threat["Total_Weight_KG"] / 1e3
    ann_threat["Real_Unit_Value_USD_per_KG"] = np.where(
        ann_threat["Total_Weight_KG"] > 0,
        ann_threat["Total_Real_USD"] / ann_threat["Total_Weight_KG"],
        np.nan
    )
    out_ann_threat = DATA_DERIVED / "annual_taxa_threat_tier_summary.csv"
    ann_threat.to_csv(out_ann_threat, index=False)
    print(f"   -> Saved: {out_ann_threat}")

    print("7. Generating inter-bloc conservation trade matrix...")
    bloc_flow = df[df["Trade_Flow"] == "Export"].groupby(
        ["Reporter_Bloc", "Partner_Bloc", "Conservation_Category"]
    ).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum"),
        Transactions=("Real_Trade_Value_USD", "count")
    ).reset_index()
    bloc_flow["Total_Real_USD_M"] = bloc_flow["Total_Real_USD"] / 1e6
    bloc_flow["Total_Weight_MT"] = bloc_flow["Total_Weight_KG"] / 1e3
    out_bloc = DATA_DERIVED / "bilateral_conservation_flows_summary.csv"
    bloc_flow.to_csv(out_bloc, index=False)
    print(f"   -> Saved: {out_bloc}")

    print("8. Exporting filtered micro-panel for threatened taxa...")
    df_threatened = df[df["Threatened_Taxa_Flag"] == True].copy()
    out_parquet = DATA_DERIVED / "global_threatened_taxa_bilateral_panel.parquet"
    df_threatened.to_parquet(out_parquet, index=False, compression="snappy")
    print(f"   -> Saved threatened micro-panel: {out_parquet} ({len(df_threatened):,} records)")

    print("=" * 80)
    print("GLOBAL CONSERVATION PANEL SUCCESSFULLY BUILT & EXPORTED")
    print("=" * 80)


if __name__ == "__main__":
    main()
