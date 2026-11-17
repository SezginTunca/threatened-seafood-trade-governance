#!/usr/bin/env python3
"""
Module 03: Customs Aggregation Ratios, Species Resolution (SCRI), and HS 2028 Audit
===================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Computes value- and volume-based Taxonomic Aggregation Ratios (TAR; Eqs. 1-2),
the cardinality-based Species Customs Resolution Index (SCRI; Eq. 2b / Eq. S1a),
processing-tier resolution compression, and a line-by-line legal audit of
adopted WCO HS 2028 amendments (Supplementary Table S13).
"""
import re
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"

MASTER_PARQUET = DATA_RAW / "master_global_bilateral_panel.parquet"
CONCORDANCE_CSV = DATA_DERIVED / "global_hs6_conservation_concordance.csv"
IUCN_CATALOG_CSV = DATA_DERIVED / "expanded_iucn_threatened_taxa_catalog.csv"

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
S_TOTAL = 767  # Total catalogued IUCN-assessed commercially traded marine species


def get_hs_era(year):
    if year < 1996:
        return "HS 1988 (1988-1995)"
    elif year < 2002:
        return "HS 1996 (1996-2001)"
    elif year < 2007:
        return "HS 2002 (2002-2006)"
    elif year < 2012:
        return "HS 2007 (2007-2011)"
    elif year < 2017:
        return "HS 2012 (2012-2016)"
    elif year < 2022:
        return "HS 2017 (2017-2021)"
    else:
        return "HS 2022 (2022-2024)"


# Empirical commercial species counts for major WCO genera/families/orders when "spp." or family is listed
GENUS_FAMILY_SPECIES_COUNTS = {
    "anguilla": 6,          # Commercial traded Anguilla spp. (anguilla, japonica, rostrata, bicolor, marmorata, australis)
    "thunnus": 8,           # All 8 Thunnus species when pooled
    "sarda": 4,
    "scomber": 3,
    "gadus": 3,
    "merluccius": 12,
    "urophycis": 4,
    "solea": 4,
    "hippoglossus": 2,
    "oncorhynchus": 7,
    "salmo": 2,
    "oreochromis": 8,
    "pangasius": 6,
    "cyprinus": 4,
    "crangon": 3,
    "pandalus": 6,
    "penaeus": 18,
    "homarus": 2,
    "palinurus": 6,
    "panulirus": 19,
    "jasus": 6,
    "mytilus": 4,
    "perna": 3,
    "ostrea": 8,
    "crassostrea": 8,
    "pecten": 15,
    "pectinidae": 25,
    "loligo": 12,
    "ommastrephidae": 18,
    "octopus": 18,
    "sepia": 15,
    "haliotis": 23,
    "holothuroidea": 66,    # Commercially exploited sea cucumbers (Purcell et al.)
    "strongylocentrotus": 8,
    "rajidae": 45,
    "batoidea": 117,        # Traded batoid rays/skates in IUCN catalog
    "sparidae": 24,
    "dicentrarchus": 2,
    "dissostichus": 2,
    "channa": 8,
    "clarias": 6,
    "silurus": 3,
    "ictalurus": 3,
    "scophthalmidae": 5,
    "pleuronectidae": 22,
    "bothidae": 16,
    "acipenseriformes": 25,
}


def compute_empirical_subheading_species_counts(df_conc, df_iucn):
    """
    Deterministically computes empirical species cardinality |S_k| >= 1 for each HS6 code:
    1. Parses parenthetical Latin nomenclature in WCO Commodity_Description:
       - Exact Latin binomials without 'spp.' or family '-idae'/'-oidea' -> exact binomial count (1 if mono-specific!)
       - Genus 'spp.' or family '-idae' -> sum of empirical genus/family commercial pools
    2. Cross-checks against empirical IUCN catalog species mapped to Primary_HS6_Code
    3. Assigns residual 'n.e.c.' / 'other' baskets to their empirical multi-order / family pool
    """
    iucn_counts = df_iucn.groupby("Primary_HS6_Code")["Scientific_Name"].nunique().to_dict()

    sk_map = {}
    for _, row in df_conc.iterrows():
        hs6 = str(row["HS6_Code"]).zfill(6)
        desc = str(row["Commodity_Description"])
        desc_low = desc.lower()
        tier = str(row["Taxonomic_Tier"])
        mapped_iucn = int(iucn_counts.get(hs6, 0))

        # Extract parenthetical blocks BEFORE any "excluding ..." clause
        main_clause = re.split(r"\bexcluding\b", desc, flags=re.IGNORECASE)[0]
        parens = re.findall(r"\(([^)]+)\)", main_clause)
        tax_parens = [
            p for p in parens
            if not any(w in p.lower() for w in [
                "whether", "other than", "minced", "fillets", "smoked",
                "cooked", "salted", "in shell", "heading", "item"
            ])
        ]

        estimated_sk = None
        if tax_parens:
            joined = ", ".join(tax_parens)
            tokens = [t.strip() for t in re.split(r"[,;]", joined) if t.strip()]
            has_spp_or_higher = any(
                ("spp" in t.lower()) or t.strip().lower().endswith(("idae", "oidea", "iformes", "ini"))
                for t in tokens
            )
            # Count explicit Latin binomials (Genus species)
            binomials = [
                t for t in tokens
                if re.match(r"^[A-Z][a-z]+\s+[a-z]{2,}", t.strip()) and "spp" not in t.lower()
            ]
            if binomials and not has_spp_or_higher:
                estimated_sk = len(set(binomials))
            else:
                pool = len(set(binomials))
                for t in tokens:
                    tl = t.lower()
                    matched_gf = False
                    for key, cnt in GENUS_FAMILY_SPECIES_COUNTS.items():
                        if key in tl:
                            pool += cnt
                            matched_gf = True
                            break
                    if not matched_gf and ("spp" in tl or tl.endswith(("idae", "oidea"))):
                        pool += 8
                estimated_sk = max(pool, mapped_iucn, 2)

        if estimated_sk is None:
            # No taxonomic parentheses in WCO legal text
            if "shark" in desc_low and ("dogfish" in desc_low or "fin" in desc_low):
                estimated_sk = max(mapped_iucn, 148)
            elif "caviar" in desc_low:
                estimated_sk = max(mapped_iucn, 25)
            elif "eel" in desc_low:
                estimated_sk = max(mapped_iucn, 6)
            elif "sea cucumber" in desc_low:
                estimated_sk = max(mapped_iucn, 66)
            elif "abalone" in desc_low:
                estimated_sk = max(mapped_iucn, 23)
            elif tier == "Tier 1: Taxon-Delimited":
                # Common-name delimited subheadings without Latin binomial (e.g., swordfish Xiphias gladius 030247/030357, toothfish, cobia)
                if any(mono in desc_low for mono in ["swordfish", "cobia", "nile perch", "milkfish", "alaska pollack", "alaska pollock", "norway lobster"]):
                    estimated_sk = 1
                else:
                    estimated_sk = max(mapped_iucn, 4)
            elif tier == "Tier 2: Clade-Pooled Multi-Species":
                estimated_sk = max(mapped_iucn, 28)
            else:
                # Tier 3: Residual Catch-All n.e.c.
                if any(broad in hs6 for broad in ["030289", "030389", "030489", "030499", "160419", "160420", "030559", "030569"]):
                    estimated_sk = max(mapped_iucn, 380)
                else:
                    estimated_sk = max(mapped_iucn, 95)

        # Enforce bounds [1, S_TOTAL]
        # Note: if a subheading is legally mono-specific (estimated_sk == 1), keep 1; otherwise ensure >= mapped_iucn if multi-species
        if estimated_sk > 1:
            estimated_sk = min(max(int(estimated_sk), mapped_iucn), S_TOTAL)
        else:
            estimated_sk = 1
        sk_map[hs6] = estimated_sk

    return sk_map


def export_wco_hs2028_legal_audit():
    """
    Exports verified Chapter 03 & Chapter 16 legal amendment audit from official
    WCO HS 2028 text (effective 1 January 2028) to test H1a.
    """
    rows = [
        {
            "WCO_HS2028_Amendment_Block": "Alaska Pollock vs. Cod Separation",
            "Chapter": "Chapter 03",
            "Affected_HS2028_Subheadings": "0302.51, 0302.55, 0303.63, 0303.67, 0304.71, 0304.75, 0304.94, 0305.33, 0305.51, 0305.53, 0305.55, 0305.62, 0305.65",
            "New_or_Restructured_Subheadings": 7,
            "Target_Taxa": "Gadus chalcogrammus vs. Gadus morhua/ogac/macrocephalus",
            "IUCN_Threat_Focus": "Least Concern / Industrial Whitefish",
            "CITES_Appendix_Species_Resolved_to_Genus_or_Species": 0,
            "Taxonomic_or_Form_Change": "Species-specific commercial disaggregation across fresh, frozen, fillet, dried & salted forms"
        },
        {
            "WCO_HS2028_Amendment_Block": "Crustaceans Heading 03.10 Restructuring (replacing 03.06)",
            "Chapter": "Chapter 03",
            "Affected_HS2028_Subheadings": "0310.11–0310.99",
            "New_or_Restructured_Subheadings": 38,
            "Target_Taxa": "Paralithodes camtschaticus, Chionoecetes spp., Homarus spp., Nephrops norvegicus, Pleoticus muelleri, Pandalus spp.",
            "IUCN_Threat_Focus": "Least Concern / Not Evaluated Industrial Crustaceans",
            "CITES_Appendix_Species_Resolved_to_Genus_or_Species": 0,
            "Taxonomic_or_Form_Change": "High-granularity commercial species & live/fresh/frozen disaggregation for king crab, snow crab & shrimp"
        },
        {
            "WCO_HS2028_Amendment_Block": "Bivalves & Sea Urchins Live/Fresh/Frozen Splits",
            "Chapter": "Chapter 03",
            "Affected_HS2028_Subheadings": "0307.13–19, 0307.23–29, 0307.33–39, 0308.23–29",
            "New_or_Restructured_Subheadings": 16,
            "Target_Taxa": "Oysters, Pectinidae scallops, Mytilus/Perna mussels, Strongylocentrotus/Paracentrotus sea urchins",
            "IUCN_Threat_Focus": "Least Concern / Aquaculture & Commercial Shellfish",
            "CITES_Appendix_Species_Resolved_to_Genus_or_Species": 0,
            "Taxonomic_or_Form_Change": "Physical preservation state splits (live vs. fresh/chilled vs. frozen)"
        },
        {
            "WCO_HS2028_Amendment_Block": "Catfish Genus Addition (Silurus spp.) & Edible Fish Flours (03.11)",
            "Chapter": "Chapter 03",
            "Affected_HS2028_Subheadings": "0302.72, 0303.24, 0304.32, 0304.62, 0311.10–90",
            "New_or_Restructured_Subheadings": 2,
            "Target_Taxa": "Silurus spp. added to aquaculture catfish headings; edible fish/invertebrate flours",
            "IUCN_Threat_Focus": "Least Concern / Aquaculture & Meal Commodities",
            "CITES_Appendix_Species_Resolved_to_Genus_or_Species": 0,
            "Taxonomic_or_Form_Change": "Nomenclature clarification & human-grade meal heading creation"
        },
        {
            "WCO_HS2028_Amendment_Block": "Batoid Rays & Skates Superorder Broadening & Fin Split",
            "Chapter": "Chapter 03",
            "Affected_HS2028_Subheadings": "0302.86, 0302.87, 0303.86, 0303.87, 0304.48, 0304.57, 0304.88, 0304.97, 0305.66, 0305.67",
            "New_or_Restructured_Subheadings": 6,
            "Target_Taxa": "Superorder Batoidea (~630 species pooled, replacing family Rajidae) — bodies vs. fins",
            "IUCN_Threat_Focus": "Superorder-wide pool (includes 19 CITES Mobulidae/Pristidae/Rhinopristiformes pooled with ~610 non-CITES batoids)",
            "CITES_Appendix_Species_Resolved_to_Genus_or_Species": 0,
            "Taxonomic_or_Form_Change": "Broadens Rajidae to Batoidea and separates batoid fins from meat, but creates ZERO family/genus codes for Mobulidae, Pristidae, Rhinidae, or Rhinobatidae"
        },
        {
            "WCO_HS2028_Amendment_Block": "Sea Cucumbers Physical State Split",
            "Chapter": "Chapter 03",
            "Affected_HS2028_Subheadings": "0308.13, 0308.14, 0308.15, 0308.19",
            "New_or_Restructured_Subheadings": 4,
            "Target_Taxa": "Class Holothuroidea (Stichopus japonicus, Holothuroidea)",
            "IUCN_Threat_Focus": "Class-wide pool (CITES Holothuria teatfish & Thelenota remain pooled with all other sea cucumbers)",
            "CITES_Appendix_Species_Resolved_to_Genus_or_Species": 0,
            "Taxonomic_or_Form_Change": "Splits live (0308.13) vs. fresh/chilled (0308.14) vs. frozen (0308.15) vs. dried/other (0308.19); ZERO genus/subgenus disaggregation"
        },
        {
            "WCO_HS2028_Amendment_Block": "Sharks, Anguillid Eels, Sturgeons & All Prepared/Preserved Seafood",
            "Chapter": "Chapters 03 & 16",
            "Affected_HS2028_Subheadings": "0302.81, 0303.81, 0305.71 (Sharks); 0301.92, 0302.74, 0303.26 (Eels); 1604.11–1605.69 (All Chapter 16)",
            "New_or_Restructured_Subheadings": 0,
            "Target_Taxa": "Carcharhinidae, Sphyrnidae, Lamnidae, Anguilla anguilla, Acipenseriformes, Chapter 16 prepared eels/caviar/sea cucumbers",
            "IUCN_Threat_Focus": "106+ CITES Appendix I/II & Critically Endangered/Endangered species",
            "CITES_Appendix_Species_Resolved_to_Genus_or_Species": 0,
            "Taxonomic_or_Form_Change": "ZERO amendments in the adopted HS 2028 schedule: sharks remain pooled as 'Dogfish and other sharks', eels as 'Anguilla spp.', and Chapter 16 has zero changes"
        }
    ]
    df_ng = pd.DataFrame(rows)
    out_ng = DATA_DERIVED / "wco_hs2028_legal_audit.csv"
    df_ng.to_csv(out_ng, index=False)
    return df_ng


def main():
    print("=" * 80)
    print("MODULE 03: CALCULATING GLOBAL CUSTOMS AGGREGATION & EMPIRICAL SCRI")
    print("=" * 80)

    print("1. Loading conservation concordance & IUCN threatened catalog...")
    df_conc = pd.read_csv(CONCORDANCE_CSV)
    df_conc["HS6_Code"] = df_conc["HS6_Code"].astype(str).str.zfill(6)
    df_iucn = pd.read_csv(IUCN_CATALOG_CSV)
    df_iucn["Primary_HS6_Code"] = df_iucn["Primary_HS6_Code"].astype(str).str.zfill(6)

    sk_map = compute_empirical_subheading_species_counts(df_conc, df_iucn)
    df_conc["Empirical_Species_Count_Sk"] = df_conc["HS6_Code"].map(sk_map).fillna(28).astype(int)
    # Exact Eq. 2b: SCRI_k = 1 - ln(|S_k|) / ln(|S_total|) -> mono-specific (|S_k|=1) gives SCRI_k = 1.0000!
    df_conc["Subheading_SCRI"] = np.round(1.0 - np.log(df_conc["Empirical_Species_Count_Sk"]) / np.log(S_TOTAL), 4)

    n_mono = int((df_conc["Empirical_Species_Count_Sk"] == 1).sum())
    print(f"   -> Computed empirical |S_k| across {len(df_conc)} HS6 codes ({n_mono} mono-specific codes with SCRI = 1.0000)")

    print("2. Reading master parquet panel (Year, HS6, Flows, Blocs, Values)...")
    cols = [
        "Year", "Trade_Flow", "HS6_Code", "Reporter_Bloc", "Processing_Form",
        "Weight_KG", "Trade_Value_USD", "UV_Outlier_Flag"
    ]
    df = pd.read_parquet(MASTER_PARQUET, columns=cols)
    df = df[
        (df["Year"] <= 2024) &
        (df["Weight_KG"] > 0) &
        (df["Trade_Value_USD"] > 0) &
        (~df["UV_Outlier_Flag"])
    ].copy()
    df["HS6_Code"] = df["HS6_Code"].astype(str).str.zfill(6)

    n_exp_clean = int((df["Trade_Flow"] == "Export").sum())
    n_imp_clean = int((df["Trade_Flow"] == "Import").sum())
    print(f"   -> Verified 1988–2024 clean bilateral declarations: {len(df):,} total ({n_exp_clean:,} Export + {n_imp_clean:,} Import)")

    df["CPI"] = df["Year"].map(CPI_DEFLATOR).fillna(BASE_CPI)
    df["Real_Trade_Value_USD"] = df["Trade_Value_USD"] * (BASE_CPI / df["CPI"])

    print("3. Merging customs granularity tiers and empirical Subheading_SCRI...")
    df = df.merge(
        df_conc[["HS6_Code", "Customs_Granularity", "Taxonomic_Tier", "Empirical_Species_Count_Sk", "Subheading_SCRI"]],
        on="HS6_Code",
        how="left"
    )
    df["Customs_Granularity"] = df["Customs_Granularity"].fillna("Aggregated Multi-Taxa")
    df["Taxonomic_Tier"] = df["Taxonomic_Tier"].fillna("Tier 2: Clade-Pooled Multi-Species")
    df["Empirical_Species_Count_Sk"] = df["Empirical_Species_Count_Sk"].fillna(28)
    df["Subheading_SCRI"] = df["Subheading_SCRI"].fillna(1.0 - np.log(28) / np.log(S_TOTAL))
    df["HS_Era"] = df["Year"].apply(get_hs_era)

    print("4. Computing annual global TAR & Empirical Species-Cardinal SCRI...")
    df_exp = df[df["Trade_Flow"] == "Export"].copy()
    df_exp["Weighted_SCRI"] = df_exp["Real_Trade_Value_USD"] * df_exp["Subheading_SCRI"]

    # Export HS6-level SCRI & trade audit (for H1a & H1b tests)
    hs6_trade = df_exp.groupby("HS6_Code", as_index=False).agg(
        Cum_Export_Real_USD_B=("Real_Trade_Value_USD", lambda x: x.sum() / 1e9),
        Cum_Export_Weight_MMT=("Weight_KG", lambda x: x.sum() / 1e9),
        Export_Records=("Real_Trade_Value_USD", "count")
    )
    iucn_wt_map = {"CR": 4, "EN": 3, "VU": 2, "NT": 1}
    df_iucn["Threat_Wt"] = df_iucn["Red_List_Category"].map(iucn_wt_map).fillna(0)
    hs6_iucn = df_iucn.groupby("Primary_HS6_Code", as_index=False).agg(
        Mapped_Threatened_Species=("Scientific_Name", "nunique"),
        Cumulative_IUCN_Threat_Score=("Threat_Wt", "sum")
    ).rename(columns={"Primary_HS6_Code": "HS6_Code"})

    hs6_audit = df_conc.merge(hs6_trade, on="HS6_Code", how="left").merge(hs6_iucn, on="HS6_Code", how="left")
    hs6_audit["Cum_Export_Real_USD_B"] = hs6_audit["Cum_Export_Real_USD_B"].fillna(0.0)
    hs6_audit["Cum_Export_Weight_MMT"] = hs6_audit["Cum_Export_Weight_MMT"].fillna(0.0)
    hs6_audit["Export_Records"] = hs6_audit["Export_Records"].fillna(0).astype(int)
    hs6_audit["Mapped_Threatened_Species"] = hs6_audit["Mapped_Threatened_Species"].fillna(0).astype(int)
    hs6_audit["Cumulative_IUCN_Threat_Score"] = hs6_audit["Cumulative_IUCN_Threat_Score"].fillna(0.0)
    out_hs6_scri = DATA_DERIVED / "global_hs6_customs_scri_audit.csv"
    hs6_audit.to_csv(out_hs6_scri, index=False)
    print(f"   -> Saved HS6 Empirical SCRI Audit: {out_hs6_scri}")

    # Test H1a: Correlation between SCRI and IUCN Threat Score vs Commercial Volume
    rho_threat, p_threat = stats.spearmanr(hs6_audit["Subheading_SCRI"], hs6_audit["Cumulative_IUCN_Threat_Score"])
    print(f"   -> H1a Test (SCRI vs IUCN Threat Score across {len(hs6_audit)} HS6 codes): Spearman rho = {rho_threat:.4f} (p = {p_threat:.4e})")

    ann_grp = df_exp.groupby(["Year", "Customs_Granularity"]).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum"),
        Transaction_Count=("Real_Trade_Value_USD", "count")
    ).unstack(fill_value=0)

    tier_grp = df_exp.groupby(["Year", "Taxonomic_Tier"]).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum")
    ).unstack(fill_value=0)

    scri_grp = df_exp.groupby("Year").agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weighted_SCRI=("Weighted_SCRI", "sum")
    )
    scri_series = scri_grp["Total_Weighted_SCRI"] / scri_grp["Total_Real_USD"]

    ann_summary = pd.DataFrame()
    ann_summary["Year"] = ann_grp.index
    ann_summary["HS_Era"] = ann_summary["Year"].apply(get_hs_era)

    val_agg = ann_grp["Total_Real_USD"]["Aggregated Multi-Taxa"].values
    val_spec = ann_grp["Total_Real_USD"]["Taxon-Specific"].values
    val_total = val_agg + val_spec

    wgt_agg = ann_grp["Total_Weight_KG"]["Aggregated Multi-Taxa"].values
    wgt_spec = ann_grp["Total_Weight_KG"]["Taxon-Specific"].values
    wgt_total = wgt_agg + wgt_spec

    val_clade = tier_grp["Total_Real_USD"]["Tier 2: Clade-Pooled Multi-Species"].values
    val_nec = tier_grp["Total_Real_USD"]["Tier 3: Residual Catch-All n.e.c."].values

    wgt_clade = tier_grp["Total_Weight_KG"]["Tier 2: Clade-Pooled Multi-Species"].values
    wgt_nec = tier_grp["Total_Weight_KG"]["Tier 3: Residual Catch-All n.e.c."].values

    ann_summary["Total_Export_Real_USD_B"] = val_total / 1e9
    ann_summary["Aggregated_Real_USD_B"] = val_agg / 1e9
    ann_summary["Specific_Real_USD_B"] = val_spec / 1e9
    ann_summary["Clade_Pooled_Real_USD_B"] = val_clade / 1e9
    ann_summary["Residual_NEC_Real_USD_B"] = val_nec / 1e9
    ann_summary["TAR_Value_Percent"] = np.round((val_agg / val_total) * 100, 2)
    ann_summary["TAR_Clade_Pooled_Percent"] = np.round((val_clade / val_total) * 100, 2)
    ann_summary["TAR_Residual_NEC_Percent"] = np.round((val_nec / val_total) * 100, 2)

    ann_summary["Total_Export_Weight_MMT"] = wgt_total / 1e9
    ann_summary["Aggregated_Weight_MMT"] = wgt_agg / 1e9
    ann_summary["Specific_Weight_MMT"] = wgt_spec / 1e9
    ann_summary["Clade_Pooled_Weight_MMT"] = wgt_clade / 1e9
    ann_summary["Residual_NEC_Weight_MMT"] = wgt_nec / 1e9
    ann_summary["TAR_Volume_Percent"] = np.round((wgt_agg / wgt_total) * 100, 2)
    ann_summary["TAR_Clade_Pooled_Volume_Percent"] = np.round((wgt_clade / wgt_total) * 100, 2)
    ann_summary["TAR_Residual_NEC_Volume_Percent"] = np.round((wgt_nec / wgt_total) * 100, 2)

    ann_summary["Species_Entropy_Customs_Resolution_Index"] = np.round(scri_series.values, 4)

    out_ann = DATA_DERIVED / "annual_global_customs_aggregation_tar.csv"
    ann_summary.to_csv(out_ann, index=False)
    print(f"   -> Saved Annual TAR & Empirical SCRI Summary: {out_ann}")
    print(f"   -> Empirical SCRI evolution: 1988 = {ann_summary.iloc[0]['Species_Entropy_Customs_Resolution_Index']:.4f} -> 2024 = {ann_summary.iloc[-1]['Species_Entropy_Customs_Resolution_Index']:.4f} (Mean = {ann_summary['Species_Entropy_Customs_Resolution_Index'].mean():.4f})")

    cum_total_val = val_total.sum() / 1e9
    cum_agg_val = val_agg.sum() / 1e9
    cum_total_wgt = wgt_total.sum() / 1e9
    cum_agg_wgt = wgt_agg.sum() / 1e9
    cum_tar_val = (cum_agg_val / cum_total_val) * 100
    cum_tar_wgt = (cum_agg_wgt / cum_total_wgt) * 100
    avg_tar_val = ann_summary["TAR_Value_Percent"].mean()
    avg_tar_wgt = ann_summary["TAR_Volume_Percent"].mean()

    print(f"\n   Cumulative 37-Year Global Seafood Exports (1988–2024): ${cum_total_val:,.2f}B | {cum_total_wgt:,.2f} MMT")
    print(f"   Cumulative Aggregated Multi-Taxa Exports: ${cum_agg_val:,.2f}B ({cum_tar_val:.2f}%) | {cum_agg_wgt:,.2f} MMT ({cum_tar_wgt:.2f}%)")
    print(f"   Mean Annual TAR by Value: {avg_tar_val:.2f}% | Mean Annual TAR by Volume: {avg_tar_wgt:.2f}%\n")

    print("5. Computing customs aggregation by regional trade bloc...")
    bloc_grp = df_exp.groupby(["Reporter_Bloc", "Customs_Granularity"]).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum")
    ).unstack(fill_value=0)
    bloc_scri = df_exp.groupby("Reporter_Bloc").apply(
        lambda g: float((g["Real_Trade_Value_USD"] * g["Subheading_SCRI"]).sum() / g["Real_Trade_Value_USD"].sum()),
        include_groups=False
    )

    bloc_summary = pd.DataFrame()
    bloc_summary["Reporter_Bloc"] = bloc_grp.index
    b_val_agg = bloc_grp["Total_Real_USD"]["Aggregated Multi-Taxa"].values
    b_val_spec = bloc_grp["Total_Real_USD"]["Taxon-Specific"].values
    b_val_tot = b_val_agg + b_val_spec

    b_wgt_agg = bloc_grp["Total_Weight_KG"]["Aggregated Multi-Taxa"].values
    b_wgt_spec = bloc_grp["Total_Weight_KG"]["Taxon-Specific"].values
    b_wgt_tot = b_wgt_agg + b_wgt_spec

    bloc_summary["Total_Real_USD_B"] = b_val_tot / 1e9
    bloc_summary["Aggregated_USD_B"] = b_val_agg / 1e9
    bloc_summary["TAR_Value_Percent"] = np.round((b_val_agg / b_val_tot) * 100, 2)
    bloc_summary["Total_Weight_MMT"] = b_wgt_tot / 1e9
    bloc_summary["Aggregated_Weight_MMT"] = b_wgt_agg / 1e9
    bloc_summary["TAR_Volume_Percent"] = np.round((b_wgt_agg / b_wgt_tot) * 100, 2)
    bloc_summary["Value_Weighted_SCRI"] = np.round(bloc_summary["Reporter_Bloc"].map(bloc_scri), 4)
    bloc_summary = bloc_summary.sort_values(by="Total_Real_USD_B", ascending=False)

    out_bloc = DATA_DERIVED / "bloc_customs_aggregation_summary.csv"
    bloc_summary.to_csv(out_bloc, index=False)
    print(f"   -> Saved Regional Bloc Aggregation Summary: {out_bloc}")

    print("6. Computing customs aggregation & SCRI compression across processing forms (Testing H1b)...")
    proc_grp = df_exp.groupby(["Processing_Form", "Customs_Granularity"]).agg(
        Total_Real_USD=("Real_Trade_Value_USD", "sum"),
        Total_Weight_KG=("Weight_KG", "sum")
    ).unstack(fill_value=0)
    proc_scri = df_exp.groupby("Processing_Form").apply(
        lambda g: float((g["Real_Trade_Value_USD"] * g["Subheading_SCRI"]).sum() / g["Real_Trade_Value_USD"].sum()),
        include_groups=False
    )

    proc_summary = pd.DataFrame()
    proc_summary["Processing_Form"] = proc_grp.index
    p_val_agg = proc_grp["Total_Real_USD"]["Aggregated Multi-Taxa"].values
    p_val_spec = proc_grp["Total_Real_USD"]["Taxon-Specific"].values
    p_val_tot = p_val_agg + p_val_spec

    p_wgt_agg = proc_grp["Total_Weight_KG"]["Aggregated Multi-Taxa"].values
    p_wgt_spec = proc_grp["Total_Weight_KG"]["Taxon-Specific"].values
    p_wgt_tot = p_wgt_agg + p_wgt_spec

    proc_summary["Total_Real_USD_B"] = p_val_tot / 1e9
    proc_summary["Aggregated_USD_B"] = p_val_agg / 1e9
    proc_summary["TAR_Value_Percent"] = np.round((p_val_agg / p_val_tot) * 100, 2)
    proc_summary["Total_Weight_MMT"] = p_wgt_tot / 1e9
    proc_summary["TAR_Volume_Percent"] = np.round((p_wgt_agg / p_wgt_tot) * 100, 2)
    proc_summary["Value_Weighted_SCRI"] = np.round(proc_summary["Processing_Form"].map(proc_scri), 4)
    proc_summary = proc_summary.sort_values(by="Total_Real_USD_B", ascending=False)

    out_proc = DATA_DERIVED / "processing_form_aggregation_summary.csv"
    proc_summary.to_csv(out_proc, index=False)
    print(f"   -> Saved Processing Form Aggregation Summary: {out_proc}")
    print(proc_summary[["Processing_Form", "TAR_Value_Percent", "Value_Weighted_SCRI"]].to_string(index=False))

    print("\n7. Exporting verified WCO HS 2028 legal amendment audit...")
    df_ng = export_wco_hs2028_legal_audit()
    print(f"   -> Saved WCO HS 2028 Legal Audit ({len(df_ng)} blocks)")

    print("=" * 80)
    print("GLOBAL CUSTOMS AGGREGATION & EMPIRICAL SCRI AUDIT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
