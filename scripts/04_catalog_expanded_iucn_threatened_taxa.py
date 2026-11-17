#!/usr/bin/env python3
"""
Module 04: Expanded 767-Species IUCN Marine Conservation and HS6 Mapping
========================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Constructs the 767-species global catalog of commercially traded marine and
diadromous taxa assessed as Critically Endangered, Endangered, Vulnerable, or
Near Threatened, mapping each taxon to species-specific, genus/family-aggregated,
or omnibus residual HS6 subheadings.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DERIVED = BASE_DIR / "data" / "derived"
DATA_DERIVED.mkdir(parents=True, exist_ok=True)

EXTERNAL_DB_ROOT = Path(os.environ.get("EXTERNAL_DB_ROOT", BASE_DIR.parent / "_Databases"))
IUCN_WORKSPACE = EXTERNAL_DB_ROOT / "IUCN_RedList_Workspace"
if str(IUCN_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(IUCN_WORKSPACE))

try:
    from iucn_client import IUCNRedListClient
except ImportError:
    IUCNRedListClient = None


def get_cites_status(sci_name, family, order):
    """Determines CITES regulatory status based on scientific nomenclature."""
    name_low = str(sci_name).lower()
    fam_low = str(family).lower()
    ord_low = str(order).lower()

    if "anguilla anguilla" in name_low:
        return "CITES Appendix II (2007/2009)"
    elif "anguilla" in name_low:
        return "Non-CITES"
    elif ord_low == "acipenseriformes" or "acipenser" in name_low or "huso" in name_low:
        if any(k in name_low for k in ["acipenser sturio", "acipenser brevirostrum"]):
            return "CITES Appendix I (1975/1983)"
        return "CITES Appendix II (Universal 1998)"
    elif "pristis" in name_low or "anoisopristis" in name_low or "pristidae" in fam_low:
        return "CITES Appendix I (2007)"
    elif any(k in name_low for k in [
        "sphyrna mokarran", "sphyrna lewini", "sphyrna zygaena",
        "carcharhinus longimanus", "isurus oxyrinchus", "isurus paucus",
        "lamna nasus", "alopias", "cetorhinus maximus", "rhincodon typus", "carcharodon carcharias"
    ]):
        return "CITES Appendix II (2003-2019)"
    elif any(k in name_low for k in ["carcharhinus", "prionace", "negaprion", "triaenodon", "sphyrna", "eusphyra"]):
        return "CITES Appendix II (CoP17/CoP19)"
    elif any(k in name_low for k in ["glaucostegus", "rhina", "rhynchobatus", "rhinobatos"]):
        return "CITES Appendix II (CoP18/CoP19)"
    elif "mobula" in name_low or "manta" in name_low:
        return "CITES Appendix II (CoP17 2016)"
    elif any(k in name_low for k in ["holothuria nobilis", "holothuria whitmaei", "holothuria fuscogilva", "thelenota"]):
        return "CITES Appendix II (CoP18/CoP19)"
    elif any(k in name_low for k in ["cheilinus undulatus", "aliger gigas", "strombus gigas"]):
        return "CITES Appendix II (2002-2004)"
    return "Non-CITES"


def map_customs_classification(sci_name, family, order, class_name):
    """
    Maps biological taxon to WCO Harmonized System (HS) 6-digit classification
    and determines customs reporting visibility:
    1. Species-Specific: Mono-specific WCO HS6 subheading exists for the single species
       (Thunnus maccoyii -> 030236; Thunnus obesus -> 030234; Melanogrammus aeglefinus -> 030252)
    2. Genus/Family-Aggregated: Multi-species congeneric or family WCO HS6 subheading exists
       (e.g., Thunnus orientalis pooled with T. thynnus in 030235; Gadus morhua pooled with
       G. macrocephalus & G. ogac in 030251; Merluccius spp. in 030254; Pleuronectidae/Soleidae
       flatfishes in 030229; Anguilla spp. in 030192; Acipenseriformes in 160431; Sharks/Rays
       in 030281/030282; Holothuroidea in 030811; Haliotis spp. in 030781)
    3. Omnibus-Submerged: Species is pooled into generic residual 'n.e.c.' lines (030289)
    """
    name_low = str(sci_name).lower()
    fam_low = str(family).lower()
    ord_low = str(order).lower()
    cls_low = str(class_name).lower()

    # 1. True Mono-Specific WCO HS6 Lines (Single Species Only)
    if "thunnus maccoyii" in name_low:
        return "030236", "Fresh/chilled southern bluefin tuna (Thunnus maccoyii)", "Species-Specific"
    elif "thunnus obesus" in name_low:
        return "030234", "Fresh/chilled bigeye tuna (Thunnus obesus)", "Species-Specific"
    elif "melanogrammus aeglefinus" in name_low:
        return "030252", "Fresh/chilled haddock (Melanogrammus aeglefinus)", "Species-Specific"

    # 2. Genus/Family-Aggregated Lines (Multi-Species Congeneric or Family Baskets)
    if "thunnus orientalis" in name_low or "thunnus thynnus" in name_low:
        return "030235", "Fresh/chilled Atlantic and Pacific bluefin tunas (T. thynnus, T. orientalis)", "Genus/Family-Aggregated"
    elif "gadus morhua" in name_low:
        return "030251", "Fresh/chilled cod (Gadus morhua, G. ogac, G. macrocephalus)", "Genus/Family-Aggregated"
    elif "merluccius" in name_low or "urophycis" in name_low:
        return "030254", "Fresh/chilled hake (Merluccius spp., Urophycis spp.)", "Genus/Family-Aggregated"
    elif "anguilla" in name_low or fam_low == "anguillidae":
        return "030192", "Live eels (Anguilla spp.)", "Genus/Family-Aggregated"
    elif ord_low == "acipenseriformes" or "acipenser" in name_low or "huso" in name_low:
        return "160431", "Caviar and sturgeon preparations (Acipenseriformes)", "Genus/Family-Aggregated"
    elif any(k in name_low for k in ["pristis", "anoisopristis", "rhina", "glaucostegus", "rhynchobatus", "rhinobatos"]):
        return "030289", "Other marine fish, fresh or chilled (Submerged)", "Omnibus-Submerged"
    elif cls_low == "chondrichthyes" and any(k in name_low for k in ["raja", "batis", "dipturus", "leucoraja", "bathyraja", "okamejei", "rostroraja", "amblyraja", "neoraja", "hongeo", "zearaja", "dentiraja", "spiniraja", "beringraja", "malacoraja", "fenestraja", "gurgesiella", "rajella", "cruriraja", "orbiraja", "notoraja", "brochiraja", "insentiraja", "pavoraja", "psammobatis", "sympterygia", "atlantoraja", "rioraja"]):
        return "030282", "Fresh/chilled rays and skates (Rajidae)", "Genus/Family-Aggregated"
    elif cls_low == "chondrichthyes":
        return "030281", "Fresh/chilled dogfish and other sharks", "Genus/Family-Aggregated"
    elif cls_low == "holothuroidea":
        return "030811", "Sea cucumbers (Holothuroidea), live/fresh/chilled", "Genus/Family-Aggregated"
    elif fam_low == "haliotidae" or "haliotis" in name_low:
        return "030781", "Abalone (Haliotis spp.), live/fresh/chilled", "Genus/Family-Aggregated"
    elif fam_low == "salmonidae":
        return "030219", "Other salmonidae, fresh or chilled", "Genus/Family-Aggregated"

    # 3. Omnibus-Submerged Lines (Completely Invisible in Residual n.e.c. Headings)
    return "030289", "Other fish; fresh or chilled, n.e.c. (Submerged)", "Omnibus-Submerged"


def main():
    print("=" * 80)
    print("MODULE 04: HARVESTING & CATALOGING EXPANDED IUCN THREATENED SEAFOOD TAXA")
    print("=" * 80)

    out_file = DATA_DERIVED / "expanded_iucn_threatened_taxa_catalog.csv"

    if out_file.exists():
        print(f"Updating existing 767-species IUCN catalog with verified WCO HS6 mappings: {out_file}")
        df_catalog = pd.read_csv(out_file)
        for idx, row in df_catalog.iterrows():
            sci_name = str(row["Scientific_Name"]).strip()
            cls_name = str(row["Class"]).strip()
            t_group = str(row["Taxonomic_Group"]).strip()
            fam_name = t_group if t_group in ["ANGUILLIDAE", "EPINEPHELIDAE", "SCOMBRIDAE", "HALIOTIDAE", "SALMONIDAE", "LABRIDAE"] else ""
            ord_name = t_group if t_group in ["ACIPENSERIFORMES", "GADIFORMES", "PLEURONECTIFORMES"] else ""

            cites_status = get_cites_status(sci_name, fam_name, ord_name)
            hs_code, hs_desc, visibility = map_customs_classification(sci_name, fam_name, ord_name, cls_name)

            df_catalog.loc[idx, "CITES_Status"] = cites_status
            df_catalog.loc[idx, "Primary_HS6_Code"] = str(hs_code).zfill(6)
            df_catalog.loc[idx, "HS6_Commodity_Description"] = hs_desc
            df_catalog.loc[idx, "Customs_Reporting_Visibility"] = visibility

        df_catalog["Primary_HS6_Code"] = df_catalog["Primary_HS6_Code"].astype(str).str.zfill(6)
        df_catalog.to_csv(out_file, index=False)
    else:
        if not IUCNRedListClient:
            print("Error: Could not import IUCNRedListClient and cached catalog not found.")
            sys.exit(1)
        client = IUCNRedListClient()
        target_queries = [
            ("class", "CHONDRICHTHYES", 14),
            ("order", "ACIPENSERIFORMES", 1),
            ("family", "ANGUILLIDAE", 1),
            ("class", "HOLOTHUROIDEA", 3),
            ("family", "EPINEPHELIDAE", 2),
            ("family", "SCOMBRIDAE", 1),
            ("family", "HALIOTIDAE", 1),
            ("family", "SALMONIDAE", 2),
            ("order", "GADIFORMES", 3),
            ("order", "PLEURONECTIFORMES", 3),
            ("family", "LABRIDAE", 2),
        ]
        all_assessments = []
        for q_type, name, max_pages in target_queries:
            for p in range(1, max_pages + 1):
                try:
                    if q_type == "class":
                        res = client.get_assessments_by_class(name, page=p)
                    elif q_type == "order":
                        res = client.get_assessments_by_order(name, page=p)
                    else:
                        res = client.get_assessments_by_family(name, page=p)
                    asms = res.get("assessments", [])
                    if not asms:
                        break
                    for a in asms:
                        a["query_group"] = name
                        a["query_type"] = q_type
                        all_assessments.append(a)
                except Exception:
                    break
        records = []
        seen_taxa = set()
        for a in all_assessments:
            sis_id = a.get("sis_taxon_id")
            sci_name = a.get("taxon_scientific_name", "").strip()
            cat = a.get("red_list_category_code")
            scopes = [s.get("code") for s in a.get("scopes", [])]
            if not ("1" in scopes or 1 in scopes or not scopes) or cat not in ["CR", "EN", "VU", "NT"]:
                continue
            if (sis_id, sci_name) in seen_taxa:
                continue
            seen_taxa.add((sis_id, sci_name))
            q_group = a.get("query_group", "")
            q_type = a.get("query_type", "")
            class_name = "Actinopterygii"
            order_name = q_group if q_type == "order" else ""
            family_name = q_group if q_type == "family" else ""
            if q_group == "CHONDRICHTHYES":
                class_name = "Chondrichthyes"
            elif q_group == "HOLOTHUROIDEA":
                class_name = "Holothuroidea"
            elif q_group == "HALIOTIDAE":
                class_name = "Gastropoda"
            cites_status = get_cites_status(sci_name, family_name, order_name)
            hs_code, hs_desc, visibility = map_customs_classification(sci_name, family_name, order_name, class_name)
            records.append({
                "IUCN_SIS_ID": sis_id,
                "Scientific_Name": sci_name,
                "Class": class_name,
                "Taxonomic_Group": q_group,
                "Red_List_Category": cat,
                "Criteria": a.get("criteria") or "N/A",
                "Year_Published": a.get("year_published"),
                "CITES_Status": cites_status,
                "Primary_HS6_Code": str(hs_code).zfill(6),
                "HS6_Commodity_Description": hs_desc,
                "Customs_Reporting_Visibility": visibility
            })
        df_catalog = pd.DataFrame(records).sort_values(["Class", "Taxonomic_Group", "Red_List_Category", "Scientific_Name"])
        df_catalog.to_csv(out_file, index=False)

    print("\n" + "=" * 80)
    print("EXPANDED THREATENED TAXA CATALOG COMPILED SUCCESSFULLY")
    print(f"Total Unique Threatened/NT Seafood Taxa: {len(df_catalog):,}")
    print(f"Saved to: {out_file}")
    print("=" * 80)
    print("\nBreakdown by Red List Category:")
    print(df_catalog["Red_List_Category"].value_counts())
    print("\nBreakdown by Customs Reporting Visibility:")
    print(df_catalog["Customs_Reporting_Visibility"].value_counts())
    print("\nSpecies-Specific Taxa:")
    print(df_catalog[df_catalog["Customs_Reporting_Visibility"] == "Species-Specific"][[
        "Scientific_Name", "Taxonomic_Group", "Red_List_Category", "Primary_HS6_Code", "HS6_Commodity_Description"
    ]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
