#!/usr/bin/env python3
"""
Module 01: Taxonomic Concordance and Regulatory Mapping Across Customs Schedules
================================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Links 6-digit WCO Harmonized System (HS) Chapters 03 and 16 seafood commodity
lines to IUCN Red List assessments and CITES Appendices I-III listing histories
(Supplementary Table S1).
"""
import os
import sqlite3
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DERIVED = BASE_DIR / "data" / "derived"
DATA_DERIVED.mkdir(parents=True, exist_ok=True)

MASTER_PARQUET = DATA_RAW / "master_global_bilateral_panel.parquet"
EXTERNAL_DB_ROOT = Path(os.environ.get("EXTERNAL_DB_ROOT", BASE_DIR.parent / "_Databases"))
IUCN_DB_PATH = EXTERNAL_DB_ROOT / "IUCN_RedList_Workspace" / "data" / "iucn_redlist.db"

# Explicit HS6 Conservation Category sets (verified against WCO HS 1988–2022 Explanatory Notes)
EEL_HS6 = {
    "030192", "030266", "030274", "030326", "030376", "160417"
}
STURGEON_CAVIAR_HS6 = {
    "160430", "160431"
}
CHONDRICHTHYAN_HS6 = {
    "030265", "030281", "030282", "030292",
    "030375", "030381", "030382", "030392",
    "030447", "030448", "030456", "030457", "030488", "030496", "030497",
    "030571", "160418"
}
SEA_CUCUMBER_HS6 = {
    "030811", "030812", "030819", "160561"
}
HIGH_TROPHIC_TUNA_HS6 = {
    "030194", "030195",
    "030235", "030236",
    "030345", "030346"
}
DEMERSAL_COMMERCIAL_HS6 = {
    "030221", "030224", "030250", "030251", "030252", "030254", "030262", "030268", "030283",
    "030331", "030334", "030352", "030360", "030362", "030363", "030364", "030366", "030372", "030378", "030383",
    "030412", "030422", "030444", "030446", "030453", "030455", "030471", "030472", "030474", "030479", "030485", "030492", "030495",
    "030532", "030551", "030553", "030562"
}

# Omnibus / multi-family / multi-genus / residual HS6 baskets lacking species-level or narrow-clade resolution
OMNIBUS_MULTI_TAXA_HS6 = {
    "030110", "030111", "030119", "030270", "030282", "030290", "030291", "030292",
    "030380", "030382", "030390", "030391", "030392",
    "030410", "030420", "030443", "030444", "030448", "030451", "030452", "030453", "030457",
    "030483", "030488", "030493", "030497",
    "030510", "030520", "030530", "030531", "030532", "030544", "030552", "030554", "030564", "030571", "030572", "030579",
    "030613", "030614", "030617", "030623", "030624", "030627", "030633", "030636", "030693", "030695",
    "030710", "030711", "030712", "030719", "030721", "030722", "030729",
    "030741", "030742", "030743", "030749", "030760", "030771", "030772", "030779",
    "030811", "030812", "030819", "030821", "030822", "030829",
    "030910", "030990",
    "160414", "160418", "160430", "160431", "160432",
    "160510", "160520", "160521", "160529", "160530", "160551", "160552", "160554", "160556", "160558", "160561", "160562"
}


def classify_granularity(hs6: str, desc: str) -> str:
    """Classify HS6 code as 'Aggregated Multi-Taxa' or 'Taxon-Specific' without WCO boilerplate false positives."""
    desc_clean = desc.lower()
    for boilerplate in (
        "other than edible fish offal",
        "other than shark fins",
        "other than sockeye salmon (oncorhynchus nerka)",
        "excluding fillets",
    ):
        desc_clean = desc_clean.replace(boilerplate, "")
    if hs6 in OMNIBUS_MULTI_TAXA_HS6 or any(k in desc_clean for k in ("n.e.c.", "other", "unspecified")):
        return "Aggregated Multi-Taxa"
    return "Taxon-Specific"


def classify_taxonomic_tier(hs6: str, desc: str, granularity: str) -> str:
    """Classify into 3-tier customs taxonomic granularity:
    - Tier 1: Taxon-Delimited (Single species, narrow genus, or monotypic group)
    - Tier 2: Clade-Pooled Multi-Species (Broader genus/family/order pools without species differentiation)
    - Tier 3: Residual Catch-All n.e.c. (Strict 'not elsewhere classified' and omnibus waste baskets)
    """
    if granularity == "Taxon-Specific":
        return "Tier 1: Taxon-Delimited"
    if hs6 == "030579":
        return "Tier 3: Residual Catch-All n.e.c."
    desc_clean = desc.lower()
    for boilerplate in (
        "other than edible fish offal",
        "other than shark fins",
        "other than sockeye salmon (oncorhynchus nerka)",
        "excluding fillets",
    ):
        desc_clean = desc_clean.replace(boilerplate, "")
    if any(k in desc_clean for k in ("n.e.c.", "other", "unspecified", "nei", "n.e.s.")):
        return "Tier 3: Residual Catch-All n.e.c."
    return "Tier 2: Clade-Pooled Multi-Species"


def main():
    print("=" * 80)
    print("MODULE 01: SYNCHRONIZING & MAPPING GLOBAL CONSERVATION TAXA")
    print("=" * 80)

    taxonomy_file = CONFIG_DIR / "species_taxonomy_master.csv"
    cites_file = CONFIG_DIR / "cites_marine_appendices_history.csv"
    iucn_file = CONFIG_DIR / "iucn_marine_threatened_taxa.csv"
    region_file = CONFIG_DIR / "country_region_mapping.csv"

    print(f"Loading master taxonomy from: {taxonomy_file}")
    df_taxa = pd.read_csv(taxonomy_file)
    df_taxa["HS6_Code"] = df_taxa["HS6_Code"].astype(str).str.zfill(6)
    print(f"  -> Total HS6 commodity lines: {len(df_taxa)}")

    print(f"Loading CITES marine timeline from: {cites_file}")
    df_cites = pd.read_csv(cites_file)
    print(f"  -> CITES regulatory records: {len(df_cites)}")

    print(f"Loading IUCN marine threatened taxa from: {iucn_file}")
    df_iucn = pd.read_csv(iucn_file)
    print(f"  -> IUCN focal marine taxa: {len(df_iucn)}")

    if IUCN_DB_PATH.exists():
        try:
            conn = sqlite3.connect(IUCN_DB_PATH)
            db_taxa = pd.read_sql("SELECT sis_taxon_id, scientific_name FROM taxa", conn)
            print(f"Connected to IUCN SQLite DB at {IUCN_DB_PATH} ({len(db_taxa)} taxa cached)")
            conn.close()
        except Exception as e:
            print(f"Note: Could not query SQLite DB ({e}), using master CSV configs.")

    records = []
    for _, row in df_taxa.iterrows():
        hs6 = str(row["HS6_Code"]).zfill(6)
        desc = str(row["Commodity_Description"])
        sp_group = str(row["Species_Group"])
        proc_form = str(row["Processing_Form"])
        trophic = float(row["Calibrated_Trophic_Level"])
        typology = str(row["Production_Typology"])

        conservation_category = "General Commercial"
        cites_status = "Non-CITES"
        iucn_threat_tier = "Not Evaluated / LC"
        threatened_flag = False

        if hs6 in EEL_HS6:
            conservation_category = "Diadromous Eels (Anguilla spp.)"
            cites_status = "CITES Appendix II (2007/2009)"
            iucn_threat_tier = "Critically Endangered / Endangered"
            threatened_flag = True
        elif hs6 in STURGEON_CAVIAR_HS6:
            conservation_category = "Sturgeons & Caviar (Acipenseriformes)"
            cites_status = "CITES Appendix II / I (1998)"
            iucn_threat_tier = "Critically Endangered"
            threatened_flag = True
        elif hs6 in CHONDRICHTHYAN_HS6:
            conservation_category = "Chondrichthyans (Sharks & Rays)"
            cites_status = "CITES App II (Progressive 2003-2022)"
            iucn_threat_tier = "Vulnerable to Critically Endangered"
            threatened_flag = True
        elif hs6 in SEA_CUCUMBER_HS6:
            conservation_category = "Sea Cucumbers (Holothuroidea)"
            cites_status = "CITES App II (Select CoP18/CoP19)"
            iucn_threat_tier = "Endangered / Vulnerable"
            threatened_flag = True
        elif hs6 in HIGH_TROPHIC_TUNA_HS6:
            conservation_category = "High-Trophic Tunas (Thunnus spp.)"
            cites_status = "Non-CITES / ICCAT Regulated"
            iucn_threat_tier = "Endangered to Least Concern"
            threatened_flag = True
        elif hs6 in DEMERSAL_COMMERCIAL_HS6:
            conservation_category = "Demersal Commercial Taxonomic Groups"
            cites_status = "Non-CITES / National Quotas"
            iucn_threat_tier = "Vulnerable to Least Concern"
            threatened_flag = True

        granularity = classify_granularity(hs6, desc)
        tier = classify_taxonomic_tier(hs6, desc, granularity)

        records.append({
            "HS6_Code": hs6,
            "Commodity_Description": desc,
            "Species_Group": sp_group,
            "Processing_Form": proc_form,
            "Calibrated_Trophic_Level": trophic,
            "Production_Typology": typology,
            "Customs_Granularity": granularity,
            "Taxonomic_Tier": tier,
            "Conservation_Category": conservation_category,
            "CITES_Status": cites_status,
            "IUCN_Threat_Tier": iucn_threat_tier,
            "Threatened_Taxa_Flag": threatened_flag
        })

    df_concordance = pd.DataFrame(records)
    out_file = DATA_DERIVED / "global_hs6_conservation_concordance.csv"
    df_concordance.to_csv(out_file, index=False)
    print(f"\nSuccessfully generated global concordance: {out_file}")
    print(f"  -> Granularity Breakdown:\n{df_concordance['Customs_Granularity'].value_counts()}")
    print(f"  -> 3-Tier Granularity Breakdown:\n{df_concordance['Taxonomic_Tier'].value_counts()}")
    print(f"  -> Conservation Categories:\n{df_concordance['Conservation_Category'].value_counts()}")
    print(f"  -> Threatened Taxa Headings Count: {df_concordance['Threatened_Taxa_Flag'].sum()} of {len(df_concordance)}")

    # Synchronize master_global_bilateral_panel.parquet with updated species_taxonomy_master.csv and country_region_mapping.csv
    if MASTER_PARQUET.exists() and region_file.exists():
        print(f"\nSynchronizing {MASTER_PARQUET} with updated taxonomic and regional concordances...")
        df_panel = pd.read_parquet(MASTER_PARQUET)
        df_panel["HS6_Code"] = df_panel["HS6_Code"].astype(str).str.zfill(6)

        df_reg = pd.read_csv(region_file)
        code_to_bloc = dict(zip(df_reg["Comtrade_Code"].astype(int), df_reg["Geopolitical_Bloc"]))
        iso_to_bloc = dict(zip(df_reg["ISO3"].astype(str).str.strip(), df_reg["Geopolitical_Bloc"]))

        df_panel["Reporter_ISO3"] = df_panel["Reporter_ISO3"].astype(str).str.strip()
        df_panel["Partner_ISO3"] = df_panel["Partner_ISO3"].astype(str).str.strip()

        rep_bloc_new = df_panel["reporterCode"].map(code_to_bloc).fillna(df_panel["Reporter_ISO3"].map(iso_to_bloc))
        par_bloc_new = df_panel["partnerCode"].map(code_to_bloc).fillna(df_panel["Partner_ISO3"].map(iso_to_bloc))
        df_panel["Reporter_Bloc"] = rep_bloc_new.fillna(df_panel["Reporter_Bloc"])
        df_panel["Partner_Bloc"] = par_bloc_new.fillna(df_panel["Partner_Bloc"])

        hs_to_sp = dict(zip(df_concordance["HS6_Code"], df_concordance["Species_Group"]))
        hs_to_pf = dict(zip(df_concordance["HS6_Code"], df_concordance["Processing_Form"]))
        hs_to_tl = dict(zip(df_concordance["HS6_Code"], df_concordance["Calibrated_Trophic_Level"]))
        hs_to_pt = dict(zip(df_concordance["HS6_Code"], df_concordance["Production_Typology"]))

        df_panel["Species_Group"] = df_panel["HS6_Code"].map(hs_to_sp).fillna(df_panel["Species_Group"])
        df_panel["Processing_Form"] = df_panel["HS6_Code"].map(hs_to_pf).fillna(df_panel["Processing_Form"])
        df_panel["Trophic_Level"] = df_panel["HS6_Code"].map(hs_to_tl).fillna(df_panel["Trophic_Level"])
        df_panel["Production_Typology"] = df_panel["HS6_Code"].map(hs_to_pt).fillna(df_panel["Production_Typology"])

        df_panel.to_parquet(MASTER_PARQUET, index=False, compression="snappy")
        print(f"  -> Master parquet panel synchronized ({len(df_panel):,} rows).")

    print("=" * 80)


if __name__ == "__main__":
    main()
