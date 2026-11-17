#!/usr/bin/env python3
"""
Module 07: Multi-Scale Sovereign Socioeconomic, Bloc, and Governance Panel
==========================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Assembles the country-year socioeconomic and institutional governance panel
across 210 reporting jurisdictions and 246 partner territories, harmonizing
UN regional groups, World Bank income tiers, SIDS/LDC classifications,
Quality of Government indicators, UN SDG 14 fisheries scores, OECD Fisheries
Support Estimates, WTO tariffs and non-tariff measures, FAO production and
border rejections, World Bank CWON fisheries wealth, Global Fishing Watch
distant-water fleet indicators, and RAM Legacy stock assessments.

Outputs:
  - data/derived/country_socioeconomic_governance_master.csv
  - data/derived/multi_scale_bloc_income_trophic_customs_matrix.csv
"""
import os
import sys
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.environ.get("EXTERNAL_DB_ROOT", os.path.join(os.path.dirname(PROJ_DIR), "_Databases"))
RAW_SOCIO_DIR = os.path.join(PROJ_DIR, "data", "raw", "socioeconomic")
DERIVED_DIR = os.path.join(PROJ_DIR, "data", "derived")

os.makedirs(RAW_SOCIO_DIR, exist_ok=True)
os.makedirs(DERIVED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Exact Institutional, Economic-Bloc, SIDS & LDC Sovereign Definitions
# ---------------------------------------------------------------------------
OECD_ISO3 = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT",
    "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA"
}

EU27_ISO3 = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
}

EFTA_ISO3 = {"ISL", "LIE", "NOR", "CHE"}
USMCA_ISO3 = {"USA", "CAN", "MEX"}
ASEAN_ISO3 = {"BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"}
RCEP_ISO3 = ASEAN_ISO3 | {"AUS", "CHN", "JPN", "KOR", "NZL"}
CPTPP_ISO3 = {"AUS", "BRN", "CAN", "CHL", "JPN", "MYS", "MEX", "NZL", "PER", "SGP", "VNM", "GBR"}
MERCOSUR_ISO3 = {"ARG", "BOL", "BRA", "PRY", "URY"}
BRICS_PLUS_ISO3 = {"BRA", "RUS", "IND", "CHN", "ZAF", "EGY", "ETH", "IRN", "ARE", "SAU", "IDN"}

AFCFTA_ISO3 = {
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
    "COM", "COG", "COD", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
    "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
    "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
    "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO",
    "TUN", "UGA", "ZMB", "ZWE"
}

SIDS_ISO3 = {
    # UN OHRLLS Small Island Developing States & Associate Island Territories
    "ATG", "ABW", "BHS", "BHR", "BRB", "BLZ", "BMU", "VGB", "CPV", "CYM",
    "COM", "COK", "CUB", "CUW", "DMA", "DOM", "TLS", "FJI", "PYF", "GRD",
    "GUM", "GNB", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "MUS", "FSM",
    "MSR", "NRU", "NCL", "NIU", "MNP", "PLW", "PNG", "PRI", "KNA", "LCA",
    "VCT", "WSM", "STP", "SYC", "SGP", "SXM", "SLB", "SUR", "TON", "TTO",
    "TCA", "TUV", "VIR", "VUT", "ASM"
}

LDC_ISO3 = {
    # UN Least Developed Countries
    "AFG", "AGO", "BGD", "BEN", "BFA", "BDI", "KHM", "CAF", "TCD", "COM",
    "COD", "DJI", "ERI", "ETH", "GMB", "GIN", "GNB", "HTI", "KIR", "LAO",
    "LSO", "LBR", "MDG", "MWI", "MLI", "MRT", "MOZ", "MMR", "NPL", "NER",
    "RWA", "STP", "SEN", "SLE", "SLB", "SOM", "SSD", "SDN", "TLS", "TGO",
    "TUV", "UGA", "TZA", "YEM", "ZMB"
}

# Track 1 High-Capacity National 8-10 Digit Customs Schedules (Immediate Jan 2028 Adopters)
TRACK1_CUSTOMS_ISO3 = EU27_ISO3 | EFTA_ISO3 | {"USA", "CAN", "JPN", "CHN", "HKG", "GBR", "AUS", "NZL", "KOR", "SGP", "TWN"}


def fetch_or_load_un_sdg14() -> pd.DataFrame:
    """Fetches or loads cached UN SDG 14 series (14.6.1 ER_REG_UNFCIM, 14.7.1 EN_SCP_FSHGDP, 14.5.1 ER_MRN_MPA)."""
    cache_path = os.path.join(RAW_SOCIO_DIR, "un_sdg14_full_harvest.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    print("Fetching full UN SDG 14 series (14.6.1 PSMA/IUU, 14.7.1 Fish % GDP, 14.5.1 MPA%)...")
    series_list = ["ER_REG_UNFCIM", "EN_SCP_FSHGDP", "ER_MRN_MPA", "ER_UNCLOS_RATACC"]
    rows = []
    for scode in series_list:
        try:
            url = f"https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg/Series/Data?seriesCode={scode}&pageSize=6000"
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json().get("data", [])
                for item in data:
                    val_str = str(item.get("value", "")).strip()
                    if val_str in ("", "NaN", "None", "nan"):
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    rows.append({
                        "UN_Code": int(str(item.get("geoAreaCode", "0")).strip()),
                        "Country_Name_SDG": item.get("geoAreaName", ""),
                        "Year": int(float(item.get("timePeriodStart", 2020))),
                        "Series_Code": scode,
                        "Value": val
                    })
                print(f"  -> {scode}: {len(data)} raw records parsed.")
        except Exception as e:
            print(f"  [WARN] SDG fetch {scode} failed: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_or_load_oecd_fse_all() -> pd.DataFrame:
    """Fetches or loads the 43-country OECD Fisheries Support Estimate (DSD_FISH_FSE@DF_FSE) in USD."""
    cache_path = os.path.join(RAW_SOCIO_DIR, "oecd_fse_43countries_usd.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    print("Fetching 43-country OECD Fisheries Support Estimate (DSD_FISH_FSE@DF_FSE) in USD...")
    url = "https://sdmx.oecd.org/public/rest/data/OECD.TAD.ARP,DSD_FISH_FSE@DF_FSE,1.0/.....USD?startPeriod=2000&endPeriod=2024"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/vnd.sdmx.data+json;version=1.0.0-wd"
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            structure = data["data"]["structure"]
            series_dims = structure["dimensions"]["series"]
            obs_dims = structure["dimensions"]["observation"]
            time_values = [v["id"] for v in obs_dims[0]["values"]]
            series_data = data["data"]["dataSets"][0].get("series", {})
            rows = []
            for series_key, s_obj in series_data.items():
                key_indices = [int(x) for x in series_key.split(":")]
                meta = {series_dims[i]["id"]: series_dims[i]["values"][idx]["id"] for i, idx in enumerate(key_indices)}
                meta_names = {f"{series_dims[i]['id']}_name": series_dims[i]["values"][idx].get("name", "") for i, idx in enumerate(key_indices)}
                for time_idx_str, obs_val in s_obj.get("observations", {}).items():
                    val = obs_val[0]
                    if val is not None:
                        rows.append({
                            **meta, **meta_names,
                            "Year": int(time_values[int(time_idx_str)]),
                            "Value_USD": float(val)
                        })
            df = pd.DataFrame(rows)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
                print(f"  -> Cached {len(df)} OECD FSE records across {df['REF_AREA'].nunique()} countries.")
                return df
    except Exception as e:
        print(f"  [WARN] Live OECD FSE fetch failed ({e}); falling back to local OECD_FSE_Fisheries_Support_Panel_Raw.csv")

    raw_local = os.path.join(DB_DIR, "OECD_Workspace", "data", "OECD_FSE_Fisheries_Support_Panel_Raw.csv")
    if os.path.exists(raw_local):
        df = pd.read_csv(raw_local)
        df = df[df["UNIT_MEASURE"] == "USD"].rename(columns={"Value": "Value_USD"})
        return df
    return pd.DataFrame()


def build_primary_rta_bloc(iso3: str) -> str:
    """Assigns a mutually exclusive primary economic/customs union bloc for clean cross-tabulation."""
    if iso3 in EU27_ISO3:
        return "EU-27 (TARIC / CFP)"
    if iso3 in EFTA_ISO3:
        return "EFTA (Norway / Iceland / Switzerland)"
    if iso3 in USMCA_ISO3:
        return "USMCA (USA / Canada / Mexico)"
    if iso3 in ASEAN_ISO3:
        return "ASEAN (AHTN Processing & Export Hub)"
    if iso3 in (RCEP_ISO3 - ASEAN_ISO3):
        return "RCEP Non-ASEAN (China / Japan / Korea / ANZ)"
    if iso3 in MERCOSUR_ISO3:
        return "Mercosur"
    if iso3 in AFCFTA_ISO3:
        return "AfCFTA (African Continental FTA)"
    if iso3 in SIDS_ISO3:
        return "Non-RTA SIDS (Caribbean / Pacific / AIMS)"
    if iso3 in BRICS_PLUS_ISO3:
        return "BRICS+ Non-RCEP (India / Russia / Middle East)"
    return "Rest of World / Other Bilateral"


def main():
    print("=" * 80)
    print("MODULE 07: BUILDING MASTER SOVEREIGN, ECONOMIC-BLOC & GOVERNANCE PANEL")
    print("=" * 80)

    # 1. Load FAO Country Groups (275 entities)
    cg_path = os.path.join(DB_DIR, "FAO", "raw_datasets", "Capture_2026.1.0", "CL_FI_COUNTRY_GROUPS.csv")
    cg = pd.read_csv(cg_path)
    cg = cg.dropna(subset=["ISO3_Code"]).copy()
    cg["ISO3"] = cg["ISO3_Code"].astype(str).str.strip().str.upper()

    income_map = {
        "High-income countries": "HIC",
        "Upper-middle income countries": "UMC",
        "Lower-middle income countries": "LMC",
        "Low-income countries": "LIC",
    }
    cg["WB_Income_Group"] = cg["EcoClass_Group_En"].map(income_map).fillna("Unclassified")
    cg["UN_Continent"] = cg["Continent_Group_En"].fillna("Other")
    cg["UN_M49_Subregion"] = cg["GeoRegion_Group_En"].fillna("Other")
    cg_clean = cg[["ISO3", "UN_Code", "Name_En", "UN_Continent", "UN_M49_Subregion", "WB_Income_Group"]].drop_duplicates("ISO3")

    # 2. Load Natural Earth Admin-0 GeoJSON for Latitude, Economy Tier, and missing Island Territories
    gj_path = os.path.join(DERIVED_DIR, "ne_110m_admin_0_countries.geojson")
    gj = json.load(open(gj_path))
    ne_rows = []
    for feat in gj["features"]:
        props = feat["properties"]
        iso3 = props.get("ISO_A3_EH") or props.get("ADM0_A3") or props.get("ISO_A3")
        if not iso3 or iso3 == "-99":
            continue
        inc_raw = str(props.get("INCOME_GRP", ""))
        if "1." in inc_raw or "2." in inc_raw:
            ne_inc = "HIC"
        elif "3." in inc_raw:
            ne_inc = "UMC"
        elif "4." in inc_raw:
            ne_inc = "LMC"
        elif "5." in inc_raw:
            ne_inc = "LIC"
        else:
            ne_inc = "Unclassified"
        lat = props.get("LABEL_Y", np.nan)
        lon = props.get("LABEL_X", np.nan)
        ne_rows.append({
            "ISO3": iso3.strip().upper(),
            "NE_Name": props.get("NAME_EN") or props.get("NAME"),
            "NE_Continent": props.get("CONTINENT"),
            "NE_Subregion": props.get("SUBREGION"),
            "NE_Income_Group": ne_inc,
            "NE_Economy_Group": props.get("ECONOMY"),
            "NE_GDP_MD": float(props.get("GDP_MD", np.nan)),
            "NE_POP_EST": float(props.get("POP_EST", np.nan)),
            "Centroid_Lat": float(lat) if lat is not None else np.nan,
            "Centroid_Lon": float(lon) if lon is not None else np.nan,
            "Abs_Latitude": abs(float(lat)) if lat is not None else np.nan
        })
    ne_df = pd.DataFrame(ne_rows).drop_duplicates("ISO3")

    # 3. Get full universe of ISO3 codes in the trade panels
    sov_tar = pd.read_csv(os.path.join(DERIVED_DIR, "sovereign_customs_aggregation_tar.csv"))
    sov_tbt = pd.read_csv(os.path.join(DERIVED_DIR, "sovereign_trophic_balance_tbt.csv"))
    bilat_panel = pd.read_parquet(
        os.path.join(DERIVED_DIR, "global_threatened_taxa_bilateral_panel.parquet"),
        columns=["Reporter_ISO3", "Partner_ISO3", "HS6_Code", "Conservation_Category", "IUCN_Threat_Tier", "Real_Trade_Value_USD", "Weight_KG", "Trade_Flow"]
    )
    all_iso3 = sorted(list(
        set(sov_tar["Reporter_ISO3"].dropna())
        | set(sov_tbt["Reporter_ISO3"].dropna())
        | set(bilat_panel["Reporter_ISO3"].dropna())
        | set(bilat_panel["Partner_ISO3"].dropna())
        | set(cg_clean["ISO3"].dropna())
    ))
    base_iso = pd.DataFrame({"ISO3": all_iso3})
    base_iso = base_iso.merge(cg_clean, on="ISO3", how="left").merge(ne_df, on="ISO3", how="left")

    # Fallback fills for territories/islands not classified by WB in FAO table
    Territory_Income_Overrides = {
        "TWN": "HIC", "HKG": "HIC", "MAC": "HIC", "SGP": "HIC", "BMU": "HIC", "CYM": "HIC",
        "ABW": "HIC", "CUW": "HIC", "PYF": "HIC", "NCL": "HIC", "FRO": "HIC", "GRL": "HIC",
        "FLK": "HIC", "GIB": "HIC", "TCA": "HIC", "VGB": "HIC", "VIR": "HIC", "GUM": "HIC",
        "MNP": "HIC", "PRI": "HIC", "REU": "HIC", "MTQ": "HIC", "GLP": "HIC", "GUF": "HIC",
        "SHN": "UMC", "COK": "UMC", "NIU": "UMC", "MSR": "UMC", "SPM": "HIC", "AIA": "HIC",
        "VEN": "UMC", "CUB": "UMC"
    }
    for iso, inc in Territory_Income_Overrides.items():
        base_iso.loc[(base_iso["ISO3"] == iso) & (~base_iso["WB_Income_Group"].isin(["HIC", "UMC", "LMC", "LIC"])), "WB_Income_Group"] = inc
    base_iso.loc[~base_iso["WB_Income_Group"].isin(["HIC", "UMC", "LMC", "LIC"]), "WB_Income_Group"] = (
        base_iso["NE_Income_Group"].where(base_iso["NE_Income_Group"].isin(["HIC", "UMC", "LMC", "LIC"]), "UMC")
    )
    base_iso["UN_Continent"] = base_iso["UN_Continent"].replace("Other", np.nan).fillna(base_iso["NE_Continent"]).fillna("Other")
    base_iso["UN_M49_Subregion"] = base_iso["UN_M49_Subregion"].replace("Other", np.nan).fillna(base_iso["NE_Subregion"]).fillna("Other")
    base_iso["Country_Name"] = base_iso["Name_En"].fillna(base_iso["NE_Name"]).fillna(base_iso["ISO3"])

    # Assign Bloc & Structural Flags
    base_iso["Is_OECD"] = base_iso["ISO3"].isin(OECD_ISO3).astype(int)
    base_iso["Is_EU27"] = base_iso["ISO3"].isin(EU27_ISO3).astype(int)
    base_iso["Is_EFTA"] = base_iso["ISO3"].isin(EFTA_ISO3).astype(int)
    base_iso["Is_USMCA"] = base_iso["ISO3"].isin(USMCA_ISO3).astype(int)
    base_iso["Is_ASEAN"] = base_iso["ISO3"].isin(ASEAN_ISO3).astype(int)
    base_iso["Is_RCEP"] = base_iso["ISO3"].isin(RCEP_ISO3).astype(int)
    base_iso["Is_CPTPP"] = base_iso["ISO3"].isin(CPTPP_ISO3).astype(int)
    base_iso["Is_AfCFTA"] = base_iso["ISO3"].isin(AFCFTA_ISO3).astype(int)
    base_iso["Is_Mercosur"] = base_iso["ISO3"].isin(MERCOSUR_ISO3).astype(int)
    base_iso["Is_BRICS_Plus"] = base_iso["ISO3"].isin(BRICS_PLUS_ISO3).astype(int)
    base_iso["Is_SIDS"] = base_iso["ISO3"].isin(SIDS_ISO3).astype(int)
    base_iso["Is_LDC"] = base_iso["ISO3"].isin(LDC_ISO3).astype(int)
    base_iso["Is_Global_South"] = (~base_iso["WB_Income_Group"].eq("HIC")).astype(int)
    base_iso["Is_Track1_2028_Customs"] = base_iso["ISO3"].isin(TRACK1_CUSTOMS_ISO3).astype(int)
    base_iso["Primary_Economic_Bloc"] = base_iso["ISO3"].map(build_primary_rta_bloc)

    # Approximate missing latitudes by M49 Subregion median
    subreg_lat = base_iso.groupby("UN_M49_Subregion")["Abs_Latitude"].transform("median")
    base_iso["Abs_Latitude"] = base_iso["Abs_Latitude"].fillna(subreg_lat).fillna(20.0)
    base_iso["Is_Tropical_Zone"] = (base_iso["Abs_Latitude"] <= 23.5).astype(int)

    # 4. Merge QoG (198 countries, 2015-2023) & OWID (247 entities) Governance & GDP per capita
    qog_path = os.path.join(DB_DIR, "Social_Science_Datasets_Workspace", "data", "government_and_institutions", "qog_bas_ts_sample_2015_2022.csv")
    qog = pd.read_csv(qog_path)
    # Map QoG UN numeric ccode to ISO3 via cg_clean UN_Code + country name fallback
    un_to_iso = dict(zip(cg_clean["UN_Code"].dropna().astype(int), cg_clean["ISO3"]))
    qog["ISO3"] = qog["ccode"].map(un_to_iso)
    # Manual overrides for any QoG ccode not in FAO UN_Code
    qog_name_overrides = {
        "Taiwan": "TWN", "Kosovo": "XKX", "United States": "USA", "United Kingdom": "GBR",
        "Russia": "RUS", "Iran": "IRN", "Vietnam": "VNM", "South Korea": "KOR",
        "Venezuela": "VEN", "Bolivia": "BOL", "Tanzania": "TZA", "Moldova": "MDA",
        "Syria": "SYR", "Laos": "LAO", "Congo, Democratic Republic": "COD", "Congo": "COG",
        "Cote d'Ivoire": "CIV", "Turkey": "TUR", "Brunei": "BRN", "Cape Verde": "CPV",
        "Eswatini": "SWZ", "Micronesia": "FSM", "Sao Tome and Principe": "STP"
    }
    for k, v in qog_name_overrides.items():
        qog.loc[qog["ISO3"].isna() & qog["cname"].str.contains(k, case=False, na=False), "ISO3"] = v

    qog_agg = qog.dropna(subset=["ISO3"]).groupby("ISO3", as_index=False).agg(
        TI_CPI_Score=("ti_cpi", "mean"),
        VDem_RuleOfLaw=("vdem_libdem", "mean"),
        UNDP_HDI=("undp_hdi", "mean"),
        WDI_GDPpc_2015USD=("wdi_gdpcapcon2015", "mean"),
        WDI_Population=("wdi_pop", "mean")
    )

    owid_path = os.path.join(DB_DIR, "Social_Science_Datasets_Workspace", "data", "macroeconomics_and_finance", "owid_democracy_governance_sample.csv")
    owid = pd.read_csv(owid_path)
    owid["ISO3"] = owid["iso_code"].astype(str).str.strip().str.upper()
    owid_sub = owid[["ISO3", "gdp_per_capita", "human_development_index", "population"]].drop_duplicates("ISO3")

    base_iso = base_iso.merge(qog_agg, on="ISO3", how="left").merge(owid_sub, on="ISO3", how="left")
    # Fill missing GDPpc and HDI from OWID and Natural Earth GDP_MD / POP_EST
    ne_gdppc = np.where(base_iso["NE_POP_EST"] > 0, (base_iso["NE_GDP_MD"] * 1e6) / base_iso["NE_POP_EST"], np.nan)
    base_iso["GDP_Per_Capita_Real_USD"] = (
        base_iso["WDI_GDPpc_2015USD"]
        .fillna(base_iso["gdp_per_capita"])
        .fillna(pd.Series(ne_gdppc, index=base_iso.index))
    )
    inc_median_gdp = base_iso.groupby("WB_Income_Group")["GDP_Per_Capita_Real_USD"].transform("median")
    base_iso["GDP_Per_Capita_Real_USD"] = base_iso["GDP_Per_Capita_Real_USD"].fillna(inc_median_gdp).fillna(8500.0)
    base_iso["Population"] = base_iso["WDI_Population"].fillna(base_iso["population"]).fillna(base_iso["NE_POP_EST"]).fillna(1e6)
    base_iso["HDI_Score"] = base_iso["UNDP_HDI"].fillna(base_iso["human_development_index"])
    inc_median_hdi = base_iso.groupby("WB_Income_Group")["HDI_Score"].transform("median")
    base_iso["HDI_Score"] = base_iso["HDI_Score"].fillna(inc_median_hdi).fillna(0.70)

    # Standardize Governance Metrics (WGI-equivalent scale [-2.5, +2.5])
    for col in ["TI_CPI_Score", "VDem_RuleOfLaw"]:
        inc_med = base_iso.groupby("WB_Income_Group")[col].transform("median")
        base_iso[col] = base_iso[col].fillna(inc_med).fillna(base_iso[col].median())

    z_cpi = (base_iso["TI_CPI_Score"] - base_iso["TI_CPI_Score"].mean()) / (base_iso["TI_CPI_Score"].std() + 1e-9)
    z_rol = (base_iso["VDem_RuleOfLaw"] - base_iso["VDem_RuleOfLaw"].mean()) / (base_iso["VDem_RuleOfLaw"].std() + 1e-9)
    z_hdi = (base_iso["HDI_Score"] - base_iso["HDI_Score"].mean()) / (base_iso["HDI_Score"].std() + 1e-9)

    base_iso["WGI_Control_of_Corruption_Eq"] = np.clip(z_cpi, -2.5, 2.5)
    base_iso["WGI_Rule_of_Law_Eq"] = np.clip(0.65 * z_cpi + 0.35 * z_rol, -2.5, 2.5)
    base_iso["WGI_Gov_Effectiveness_Eq"] = np.clip(0.55 * z_cpi + 0.45 * z_hdi, -2.5, 2.5)
    base_iso["Gov_Capacity_Index"] = (
        base_iso["WGI_Control_of_Corruption_Eq"]
        + base_iso["WGI_Rule_of_Law_Eq"]
        + base_iso["WGI_Gov_Effectiveness_Eq"]
    ) / 3.0
    # Normalized [0, 1] Customs & Digital Administrative Capacity
    g_min, g_max = base_iso["Gov_Capacity_Index"].min(), base_iso["Gov_Capacity_Index"].max()
    base_iso["Customs_Admin_Capacity_0_1"] = (base_iso["Gov_Capacity_Index"] - g_min) / (g_max - g_min + 1e-9)

    # 5. Merge UN SDG 14 Indicators (14.6.1 IUU/PSMA score, 14.7.1 Fish % GDP, 14.5.1 MPA%)
    sdg_df = fetch_or_load_un_sdg14()
    if not sdg_df.empty:
        sdg_df["ISO3"] = sdg_df["UN_Code"].map(un_to_iso)
        sdg_piv = sdg_df.dropna(subset=["ISO3"]).groupby(["ISO3", "Series_Code"])["Value"].mean().unstack()
        sdg_piv = sdg_piv.rename(columns={
            "ER_REG_UNFCIM": "SDG14_6_1_IUU_PSMA_Score",
            "EN_SCP_FSHGDP": "SDG14_7_1_Fish_GDP_Pct",
            "ER_MRN_MPA": "SDG14_5_1_MPA_KBA_Pct",
            "ER_UNCLOS_RATACC": "SDG14_c_1_UNCLOS_Score"
        }).reset_index()
        base_iso = base_iso.merge(sdg_piv, on="ISO3", how="left")
    else:
        base_iso["SDG14_6_1_IUU_PSMA_Score"] = np.nan
        base_iso["SDG14_7_1_Fish_GDP_Pct"] = np.nan
        base_iso["SDG14_5_1_MPA_KBA_Pct"] = np.nan

    inc_psma = base_iso.groupby("WB_Income_Group")["SDG14_6_1_IUU_PSMA_Score"].transform("median")
    base_iso["SDG14_6_1_IUU_PSMA_Score"] = base_iso["SDG14_6_1_IUU_PSMA_Score"].fillna(inc_psma).fillna(3.0)
    base_iso["High_PSMA_Enforcement"] = (base_iso["SDG14_6_1_IUU_PSMA_Score"] >= 4.0).astype(int)

    # 6. Merge World Bank Blue Economy & CWON Fisheries Natural Capital + Threatened Richness
    wb_path = os.path.join(DB_DIR, "World_Bank_Workspace", "data", "WorldBank_BlueEconomy_Master_Panel.csv")
    wb_df = pd.read_csv(wb_path)
    wb_df["ISO3"] = wb_df["country_iso3"].astype(str).str.strip().str.upper()
    wb_df["year"] = pd.to_numeric(wb_df["year"], errors="coerce")

    # Cross-sectional means + 1995-2020 CWON natural capital growth trajectory
    wb_cs = wb_df.groupby(["ISO3", "indicator_id"])["value"].mean().unstack().reset_index()
    wb_cs = wb_cs.rename(columns={
        "EN.FSH.THRD.NO": "WB_Threatened_Fish_Species",
        "ER.MRN.PTMR.ZS": "WB_MPA_Territorial_Pct",
        "IS.SHP.GCNW.XQ": "WB_Liner_Shipping_Index",
        "NW.NCA.FISH.PC": "CWON_Fish_Wealth_PerCap_2019USD",
        "NW.NCA.FISH.TO": "CWON_Fish_Wealth_Total_2019USD"
    })
    # Compute empirical Threatened & CITES Marine Taxa Richness from IUCN Red List SQLite + FAO capture + FishBase stocks + traded HS6
    iso2_to_iso3 = {}
    for feat in gj["features"]:
        pr = feat["properties"]
        i2 = str(pr.get("ISO_A2", "")).strip().upper()
        i3 = pr.get("ISO_A3_EH") or pr.get("ADM0_A3") or pr.get("ISO_A3")
        if i2 and i2 != "-99" and i3 and i3 != "-99":
            iso2_to_iso3[i2] = i3.strip().upper()
    try:
        import pycountry
        for c_obj in pycountry.countries:
            iso2_to_iso3.setdefault(c_obj.alpha_2, c_obj.alpha_3)
    except Exception:
        pass

    iucn_db_path = os.path.join(DB_DIR, "IUCN_RedList_Workspace", "data", "iucn_redlist.db")
    con_iucn = sqlite3.connect(iucn_db_path)
    iucn_c = pd.read_sql(
        """
        SELECT c.code, COUNT(DISTINCT a.sis_taxon_id) AS IUCN_DB_Threatened_Spp
        FROM countries c
        JOIN assessments a ON c.assessment_id = a.assessment_id
        WHERE a.latest = 1 AND a.red_list_category_code IN ('CR', 'EN', 'VU', 'NT')
        GROUP BY c.code
        """,
        con_iucn,
    )
    con_iucn.close()
    iucn_c["ISO3"] = iucn_c["code"].astype(str).str.strip().str.upper().map(iso2_to_iso3)
    iucn_iso = iucn_c.dropna(subset=["ISO3"]).groupby("ISO3", as_index=False)["IUCN_DB_Threatened_Spp"].max()

    iucn_cat = pd.read_csv(os.path.join(DERIVED_DIR, "expanded_iucn_threatened_taxa_catalog.csv"))
    iucn_thr_spp = set(iucn_cat["Scientific_Name"].astype(str).str.strip().str.lower())

    fao_cap_raw = pd.read_parquet(os.path.join(DB_DIR, "FAO", "raw_datasets", "research_ready", "global_capture.parquet"))
    fao_cap_raw["Year"] = pd.to_numeric(fao_cap_raw["Year"], errors="coerce")
    fao_cap_pos = fao_cap_raw[(fao_cap_raw["Year"] >= 1988) & (fao_cap_raw["Year"] <= 2024) & (fao_cap_raw["Capture_Tonnes"] > 0)].copy()
    fao_cap_pos["ISO3"] = fao_cap_pos["Country_ISO3"].astype(str).str.strip().str.upper()
    fao_cap_pos["sci_low"] = fao_cap_pos["Species_Scientific"].astype(str).str.strip().str.lower()
    fao_thr_only = fao_cap_pos[fao_cap_pos["sci_low"].isin(iucn_thr_spp)].copy()
    fao_thr_rich = fao_thr_only.groupby("ISO3", as_index=False)["sci_low"].nunique().rename(columns={"sci_low": "FAO_Capture_Threatened_Spp"})

    # Export 1988-2024 Country-Year Wild Threatened Capture Panel (for Suite D Stock/Capture Depletion)
    fao_thr_cy = fao_thr_only.groupby(["ISO3", "Year"], as_index=False)["Capture_Tonnes"].sum().rename(
        columns={"Capture_Tonnes": "FAO_Threatened_Capture_MT"}
    )
    fao_thr_cy = fao_thr_cy.sort_values(["ISO3", "Year"]).copy()
    fao_thr_cy["dln_Threatened_Capture"] = fao_thr_cy.groupby("ISO3")["FAO_Threatened_Capture_MT"].transform(
        lambda s: np.log(s + 1.0) - np.log(s.shift(1) + 1.0)
    )
    fao_thr_cy_path = os.path.join(DERIVED_DIR, "fao_country_year_threatened_capture_panel.csv")
    fao_thr_cy.to_csv(fao_thr_cy_path, index=False)
    print(f"[INFO] Exported FAO Threatened Wild Capture Panel: {fao_thr_cy_path} ({len(fao_thr_cy):,} country-years across {fao_thr_cy['ISO3'].nunique()} countries)")

    # Check FishBase stocks table availability
    _ = pd.read_parquet(os.path.join(DB_DIR, "FishBase_Workspace", "data", "stocks.parquet"), columns=["StockCode", "SpecCode", "IUCN_Code"])

    # Map traded threatened taxa richness per exporter from the harmonized bilateral conservation panel
    exp_only = bilat_panel[bilat_panel["Trade_Flow"] == "Export"].copy()
    exp_taxa = exp_only.groupby("Reporter_ISO3").agg(
        Traded_Threatened_HS6_Count=("HS6_Code", "nunique"),
        Traded_Threatened_Clades=("Conservation_Category", "nunique"),
        Threatened_Export_USD=("Real_Trade_Value_USD", "sum"),
        Benthic_Elasmo_Export_USD=("Real_Trade_Value_USD", lambda s: s[exp_only.loc[s.index, "Conservation_Category"].astype(str).str.contains("Sea Cucumber|Shark|Ray|Eel", case=False, na=False)].sum())
    ).reset_index().rename(columns={"Reporter_ISO3": "ISO3"})
    exp_taxa["Tropical_Benthic_Elasmo_Share"] = exp_taxa["Benthic_Elasmo_Export_USD"] / (exp_taxa["Threatened_Export_USD"] + 1.0)

    base_iso = base_iso.merge(
        wb_cs[[c for c in ["ISO3", "WB_Threatened_Fish_Species", "WB_MPA_Territorial_Pct", "WB_Liner_Shipping_Index", "CWON_Fish_Wealth_PerCap_2019USD", "CWON_Fish_Wealth_Total_2019USD"] if c in wb_cs.columns]],
        on="ISO3", how="left"
    ).merge(exp_taxa, on="ISO3", how="left").merge(iucn_iso, on="ISO3", how="left").merge(fao_thr_rich, on="ISO3", how="left")

    base_iso["IUCN_DB_Threatened_Spp"] = base_iso["IUCN_DB_Threatened_Spp"].fillna(0.0)
    base_iso["FAO_Capture_Threatened_Spp"] = base_iso["FAO_Capture_Threatened_Spp"].fillna(0.0)
    emp_rich = (
        base_iso["FAO_Capture_Threatened_Spp"]
        + base_iso["IUCN_DB_Threatened_Spp"]
        + base_iso["Traded_Threatened_HS6_Count"].fillna(0.0)
    )
    base_iso["Empirical_Threatened_Marine_Richness"] = np.where(
        base_iso["WB_Threatened_Fish_Species"].notna(),
        np.maximum(base_iso["WB_Threatened_Fish_Species"].fillna(0.0), emp_rich),
        emp_rich
    )
    base_iso["Marine_Threatened_Biodiversity_Richness"] = np.maximum(base_iso["Empirical_Threatened_Marine_Richness"], 1.0)

    # 7. Merge WTO MFN Applied Seafood Tariffs & SPS/TBT Non-Tariff Measures
    wto_mfn = pd.read_csv(os.path.join(DB_DIR, "WTO_Workspace", "data", "wto_seafood_mfn_tariffs_multicountry_2015_2024.csv"))
    name_to_iso = dict(zip(base_iso["Country_Name"].str.lower(), base_iso["ISO3"]))
    name_to_iso.update({
        "european union": "EU27", "united states of america": "USA", "united states": "USA",
        "china": "CHN", "japan": "JPN", "korea, republic of": "KOR", "republic of korea": "KOR",
        "viet nam": "VNM", "russian federation": "RUS", "united kingdom": "GBR",
        "taipei, chinese": "TWN", "hong kong, china": "HKG", "turkiye": "TUR", "türkiye": "TUR"
    })
    wto_mfn["ISO3"] = wto_mfn["Reporter_Name"].astype(str).str.strip().str.lower().map(name_to_iso)
    mfn_country = wto_mfn.dropna(subset=["ISO3"]).groupby("ISO3", as_index=False)["MFN_Applied_Duty_Pct"].mean().rename(
        columns={"MFN_Applied_Duty_Pct": "WTO_MFN_Seafood_Tariff_Pct"}
    )
    eu_mfn_val = mfn_country.loc[mfn_country["ISO3"] == "EU27", "WTO_MFN_Seafood_Tariff_Pct"]
    eu_mfn_mean = float(eu_mfn_val.iloc[0]) if not eu_mfn_val.empty else 11.4

    wto_ntm = pd.read_csv(os.path.join(DB_DIR, "WTO_Workspace", "data", "Table_Seafood_NTMs_by_Notifying_Member.csv"))
    wto_ntm["ISO3"] = wto_ntm["Notifying_Member"].astype(str).str.strip().str.lower().map(name_to_iso)
    ntm_country = wto_ntm.dropna(subset=["ISO3"]).groupby("ISO3", as_index=False)[["SPS", "TBT", "Total_Seafood_NTMs"]].sum()

    base_iso = base_iso.merge(mfn_country, on="ISO3", how="left").merge(ntm_country, on="ISO3", how="left")
    base_iso.loc[base_iso["Is_EU27"] == 1, "WTO_MFN_Seafood_Tariff_Pct"] = base_iso.loc[base_iso["Is_EU27"] == 1, "WTO_MFN_Seafood_Tariff_Pct"].fillna(eu_mfn_mean)
    inc_mfn = base_iso.groupby("WB_Income_Group")["WTO_MFN_Seafood_Tariff_Pct"].transform("median")
    base_iso["WTO_MFN_Seafood_Tariff_Pct"] = base_iso["WTO_MFN_Seafood_Tariff_Pct"].fillna(inc_mfn).fillna(10.5)
    base_iso["Total_Seafood_NTMs"] = base_iso["Total_Seafood_NTMs"].fillna(0.0)

    # 8. Merge 43-Country OECD Fisheries Support Estimate (FSE) Subsidies
    fse_df = fetch_or_load_oecd_fse_all()
    if not fse_df.empty:
        pol_col = "SUPPORT_POL_name" if "SUPPORT_POL_name" in fse_df.columns else "SUPPORT_CAT_name"
        fse_df["Pol_Lower"] = fse_df[pol_col].astype(str).str.lower()
        fse_df["Is_MCS_Management"] = fse_df["Pol_Lower"].str.contains("management|enforcement|research|monitoring|control|surveillance|decommissioning|reduction", regex=True)
        fse_df["Is_Capacity_Enhancing"] = fse_df["Pol_Lower"].str.contains("fuel|infrastructure|vessel|input|tax|capital|marketing", regex=True) & (~fse_df["Is_MCS_Management"])

        fse_tot = fse_df.groupby("REF_AREA", as_index=False).agg(
            OECD_FSE_Total_Annual_USD=("Value_USD", lambda s: s.sum() / max(1, fse_df.loc[s.index, "Year"].nunique())),
            OECD_FSE_MCS_Annual_USD=("Value_USD", lambda s: s[fse_df.loc[s.index, "Is_MCS_Management"]].sum() / max(1, fse_df.loc[s.index, "Year"].nunique())),
            OECD_FSE_CapEnh_Annual_USD=("Value_USD", lambda s: s[fse_df.loc[s.index, "Is_Capacity_Enhancing"]].sum() / max(1, fse_df.loc[s.index, "Year"].nunique()))
        ).rename(columns={"REF_AREA": "ISO3"})
        fse_tot["OECD_MCS_to_Total_Support_Share"] = fse_tot["OECD_FSE_MCS_Annual_USD"] / (fse_tot["OECD_FSE_Total_Annual_USD"] + 1.0)
        fse_tot["OECD_CapEnh_to_Total_Support_Share"] = fse_tot["OECD_FSE_CapEnh_Annual_USD"] / (fse_tot["OECD_FSE_Total_Annual_USD"] + 1.0)
        base_iso = base_iso.merge(fse_tot, on="ISO3", how="left")

    # 9. Merge FAO Border Rejections (Total + Documentary/Labelling/Fraud) & GFW Transshipment / Port Network / Dark Fleet Risk
    rej_df = pd.read_parquet(os.path.join(DB_DIR, "FAO", "raw_datasets", "research_ready", "border_rejections.parquet"))
    doc_fraud_mask = rej_df["Rejection_Cause"].astype(str).str.contains(
        "Labelling|do not match|certificate|Adulteration|Illegal|illegally|Unauthorized|mismatch|Species identification|traceability",
        case=False, na=False, regex=True
    )
    rej_df["Doc_Fraud_Rejections"] = np.where(doc_fraud_mask, rej_df["Rejections_Count"], 0.0)
    rej_exp = rej_df.groupby("Exporting_Country_ISO3", as_index=False).agg(
        FAO_Export_Border_Rejections=("Rejections_Count", "sum"),
        FAO_Doc_Fraud_Border_Rejections=("Doc_Fraud_Rejections", "sum")
    ).rename(columns={"Exporting_Country_ISO3": "ISO3"})
    rej_imp = rej_df.groupby("Importing_Country_ISO3", as_index=False)["Rejections_Count"].sum().rename(
        columns={"Importing_Country_ISO3": "ISO3", "Rejections_Count": "FAO_Import_Border_Inspections"}
    )
    rej_dyad = rej_df.groupby(["Exporting_Country_ISO3", "Importing_Country_ISO3"], as_index=False).agg(
        Dyad_Total_Rejections=("Rejections_Count", "sum"),
        Dyad_Doc_Fraud_Rejections=("Doc_Fraud_Rejections", "sum")
    ).rename(columns={"Exporting_Country_ISO3": "Exporter_ISO3", "Importing_Country_ISO3": "Importer_ISO3"})
    rej_dyad.to_csv(os.path.join(DERIVED_DIR, "fao_bilateral_border_rejections_dyad.csv"), index=False)

    gfw_eez = pd.read_parquet(os.path.join(DB_DIR, "Global_Fishing_Watch_Workspace", "data", "GFW_Global_Fishing_Effort_EEZ_Master.parquet"))
    gfw_eez_agg = gfw_eez.groupby("iso3", as_index=False).agg(
        GFW_EEZ_Fishing_Hours=("total_apparent_fishing_hours", "mean"),
        GFW_Foreign_DWF_Share=("foreign_dwf_effort_share", "mean")
    ).rename(columns={"iso3": "ISO3"})

    gfw_risk = pd.read_parquet(os.path.join(DB_DIR, "Global_Fishing_Watch_Workspace", "data", "GFW_Global_Vessel_Risk_Insights_Master.parquet"))
    gfw_risk_agg = gfw_risk.groupby("flag", as_index=False).agg(
        GFW_Flag_Dark_Hours=("total_dark_hours", "sum"),
        GFW_Flag_AIS_Gaps=("ais_gap_events_count", "sum")
    ).rename(columns={"flag": "ISO3"})

    gfw_trans = pd.read_parquet(os.path.join(DB_DIR, "Global_Fishing_Watch_Workspace", "data", "GFW_Global_Transshipment_Encounters_Master.parquet"))
    trans_fish = gfw_trans.groupby("fishing_vessel_flag", as_index=False)["estimated_transshipped_biomass_t"].sum().rename(
        columns={"fishing_vessel_flag": "ISO3", "estimated_transshipped_biomass_t": "GFW_Transshipment_FishingFlag_MT"}
    )
    trans_car = gfw_trans.groupby("carrier_flag", as_index=False)["estimated_transshipped_biomass_t"].sum().rename(
        columns={"carrier_flag": "ISO3", "estimated_transshipped_biomass_t": "GFW_Transshipment_CarrierFlag_MT"}
    )

    gfw_port = pd.read_parquet(os.path.join(DB_DIR, "Global_Fishing_Watch_Workspace", "data", "GFW_Global_Port_Network_Master.parquet"))
    port_dep = gfw_port.groupby("departure_iso3", as_index=False)["biomass_transferred_t"].sum().rename(
        columns={"departure_iso3": "ISO3", "biomass_transferred_t": "GFW_Port_Departure_MT"}
    )
    port_arr = gfw_port.groupby("arrival_iso3", as_index=False)["biomass_transferred_t"].sum().rename(
        columns={"arrival_iso3": "ISO3", "biomass_transferred_t": "GFW_Port_Arrival_MT"}
    )
    gfw_exp_df = trans_fish.merge(trans_car, on="ISO3", how="outer").merge(port_dep, on="ISO3", how="outer").merge(port_arr, on="ISO3", how="outer").fillna(0.0)
    gfw_exp_df["GFW_Transshipment_MT"] = gfw_exp_df["GFW_Transshipment_FishingFlag_MT"] + gfw_exp_df["GFW_Transshipment_CarrierFlag_MT"]
    gfw_exp_df["GFW_Port_Network_MT"] = gfw_exp_df["GFW_Port_Departure_MT"] + gfw_exp_df["GFW_Port_Arrival_MT"]
    gfw_exp_df.to_csv(os.path.join(DERIVED_DIR, "gfw_port_transshipment_exposure.csv"), index=False)

    base_iso = (
        base_iso.merge(rej_exp, on="ISO3", how="left")
        .merge(rej_imp, on="ISO3", how="left")
        .merge(gfw_eez_agg, on="ISO3", how="left")
        .merge(gfw_risk_agg, on="ISO3", how="left")
        .merge(gfw_exp_df[["ISO3", "GFW_Transshipment_MT", "GFW_Port_Network_MT"]], on="ISO3", how="left")
    )
    for zcol in [
        "FAO_Export_Border_Rejections", "FAO_Doc_Fraud_Border_Rejections", "FAO_Import_Border_Inspections",
        "GFW_Foreign_DWF_Share", "GFW_Flag_Dark_Hours", "GFW_Transshipment_MT", "GFW_Port_Network_MT"
    ]:
        base_iso[zcol] = base_iso[zcol].fillna(0.0)

    # 10. Merge FAO Annual Capture vs. Aquaculture Production Panel (1988-2024) + Build Clade-Specific Aquaculture Panel
    cas = pd.read_parquet(os.path.join(DB_DIR, "FAO", "raw_datasets", "research_ready", "country_annual_summary.parquet"))
    cas["ISO3"] = cas["Country_ISO3"].astype(str).str.strip().str.upper()
    cas["Year"] = pd.to_numeric(cas["Year"], errors="coerce")
    cas = cas[(cas["Year"] >= 1988) & (cas["Year"] <= 2024)].copy()
    cas["FAO_Total_Prod_Tonnes"] = cas["Capture_Tonnes"].fillna(0.0) + cas["Aquaculture_Tonnes"].fillna(0.0)
    cas["FAO_Aqua_Share"] = np.where(
        cas["FAO_Total_Prod_Tonnes"] > 0,
        cas["Aquaculture_Tonnes"].fillna(0.0) / cas["FAO_Total_Prod_Tonnes"],
        0.0
    )

    cas_cs = cas.groupby("ISO3", as_index=False).agg(
        Mean_Capture_Tonnes=("Capture_Tonnes", "mean"),
        Mean_Aquaculture_Tonnes=("Aquaculture_Tonnes", "mean"),
        Mean_FAO_Aqua_Share=("FAO_Aqua_Share", "mean")
    )
    base_iso = base_iso.merge(cas_cs, on="ISO3", how="left")
    base_iso["Mean_FAO_Aqua_Share"] = base_iso["Mean_FAO_Aqua_Share"].fillna(0.0)

    # Build and export clade-specific country-clade-year FAO aquaculture share panel
    fao_aqua_raw = pd.read_parquet(os.path.join(DB_DIR, "FAO", "raw_datasets", "research_ready", "global_aquaculture.parquet"))
    fao_aqua_raw["Year"] = pd.to_numeric(fao_aqua_raw["Year"], errors="coerce")
    fao_cap_sub = fao_cap_raw[(fao_cap_raw["Year"] >= 1988) & (fao_cap_raw["Year"] <= 2024)].copy()
    fao_aqua_sub = fao_aqua_raw[(fao_aqua_raw["Year"] >= 1988) & (fao_aqua_raw["Year"] <= 2024)].copy()

    def _map_fao_clade(df_in: pd.DataFrame) -> np.ndarray:
        sci = df_in["Species_Scientific"].astype(str)
        com = df_in["Species_Common"].astype(str)
        grp = df_in["ISSCAAP_Group"].astype(str)
        conds = [
            grp.eq("River eels") | sci.str.contains("Anguilla", case=False, na=False),
            sci.str.contains("Thunnus", case=False, na=False) | (grp.eq("Tunas, bonitos, billfishes") & com.str.contains("tuna", case=False, na=False)),
            grp.eq("Sharks, rays, chimaeras"),
            grp.eq("Sturgeons, paddlefishes") | sci.str.contains("Acipenser|Huso", case=False, na=False),
            sci.str.contains("Holothur|Apostichopus|Stichopus|Actinopyga|Bohadschia|Isostichopus|Thelenota", case=False, na=False)
            | com.str.contains("sea cucumber|sea-cucumber|trepang", case=False, na=False),
            grp.eq("Cods, hakes, haddocks"),
        ]
        choices = [
            "Diadromous Eels (Anguilla spp.)",
            "High-Trophic Tunas (Thunnus spp.)",
            "Chondrichthyans (Sharks & Rays)",
            "Sturgeons & Caviar (Acipenseriformes)",
            "Sea Cucumbers (Holothuroidea)",
            "Demersal Commercial Controls",
        ]
        return np.select(conds, choices, default="")

    fao_cap_sub["Clade"] = _map_fao_clade(fao_cap_sub)
    fao_aqua_sub["Clade"] = _map_fao_clade(fao_aqua_sub)
    c_clade = fao_cap_sub[fao_cap_sub["Clade"] != ""].groupby(["Country_ISO3", "Clade", "Year"], as_index=False)["Capture_Tonnes"].sum()
    a_clade = fao_aqua_sub[fao_aqua_sub["Clade"] != ""].groupby(["Country_ISO3", "Clade", "Year"], as_index=False)["Quantity_Tonnes"].sum().rename(
        columns={"Quantity_Tonnes": "Aquaculture_Tonnes"}
    )
    clade_panel = c_clade.merge(a_clade, on=["Country_ISO3", "Clade", "Year"], how="outer").fillna({"Capture_Tonnes": 0.0, "Aquaculture_Tonnes": 0.0})
    clade_panel["ISO3"] = clade_panel["Country_ISO3"].astype(str).str.strip().str.upper()
    clade_panel["Year"] = clade_panel["Year"].astype(int)
    clade_panel["Total_Clade_Tonnes"] = clade_panel["Capture_Tonnes"] + clade_panel["Aquaculture_Tonnes"]
    clade_panel["Clade_Aqua_Share"] = np.where(
        clade_panel["Total_Clade_Tonnes"] > 0,
        clade_panel["Aquaculture_Tonnes"] / clade_panel["Total_Clade_Tonnes"],
        0.0
    )
    clade_panel_out = os.path.join(DERIVED_DIR, "fao_country_clade_aquaculture_share_panel.csv")
    clade_panel[["ISO3", "Clade", "Year", "Capture_Tonnes", "Aquaculture_Tonnes", "Total_Clade_Tonnes", "Clade_Aqua_Share"]].to_csv(clade_panel_out, index=False)
    print(f"[INFO] Exported Country-Clade-Year FAO Aquaculture Share Panel: {clade_panel_out} ({len(clade_panel):,} rows)")

    # Extract RAM Legacy v4.44 focal stock status verification summary
    ram_xlsx = os.path.join(
        DB_DIR, "RAM_Legacy_Workspace", "data", "4.44", "RLSADB v4.44",
        "DB Files With Assessment Data", "RLSADB v4.44 (assessment data only).xlsx"
    )
    ram_stock = pd.read_excel(ram_xlsx, sheet_name="stock")
    ram_ts = pd.read_excel(ram_xlsx, sheet_name="timeseries_values_views")
    ram_m = ram_ts.merge(ram_stock[["stockid", "scientificname", "commonname", "region"]], on="stockid", how="left")
    ram_rows = []
    for sp_lbl, sci_name in [("Atlantic Bluefin Tuna (ICCAT eBCD)", "Thunnus thynnus"), ("Patagonian Toothfish (CCAMLR CDS)", "Dissostichus eleginoides")]:
        sub_r = ram_m[(ram_m["scientificname"] == sci_name) & (ram_m["year"].between(2000, 2020))]
        for per_lbl, mask in [("Pre-2008 (2000-2007)", sub_r["year"] <= 2007), ("Post-2008 (2009-2020)", sub_r["year"] >= 2009)]:
            p_df = sub_r[mask]
            ram_rows.append({
                "Species": sp_lbl,
                "Scientific_Name": sci_name,
                "Period": per_lbl,
                "N_Stock_Years": int(p_df["BdivBmsypref"].notna().sum()),
                "Mean_B_div_Bmsy": round(float(p_df["BdivBmsypref"].mean()), 4),
                "Mean_U_div_Umsy": round(float(p_df["UdivUmsypref"].mean()), 4) if p_df["UdivUmsypref"].notna().any() else np.nan,
            })
    ram_out = os.path.join(DERIVED_DIR, "ram_legacy_focal_stock_recovery.csv")
    pd.DataFrame(ram_rows).to_csv(ram_out, index=False)
    print(f"[INFO] Exported RAM Legacy v4.44 Stock Verification: {ram_out}")

    # Save cross-sectional sovereign profile
    cs_out = os.path.join(DERIVED_DIR, "country_cross_section_governance_concordance.csv")
    base_iso.to_csv(cs_out, index=False)
    print(f"[SUCCESS] Saved cross-sectional concordance: {cs_out} ({len(base_iso)} jurisdictions)")

    # 11. Build Longitudinal Country-Year Panel (1988-2024) joining FAO annual production, threatened capture, QoG, and WB CWON
    years = pd.DataFrame({"Year": list(range(1988, 2025))})
    panel = base_iso.assign(_k=1).merge(years.assign(_k=1), on="_k").drop(columns="_k")
    panel = panel.merge(
        cas[["ISO3", "Year", "Capture_Tonnes", "Aquaculture_Tonnes", "FAO_Total_Prod_Tonnes", "FAO_Aqua_Share"]],
        on=["ISO3", "Year"], how="left"
    ).merge(
        fao_thr_cy[["ISO3", "Year", "FAO_Threatened_Capture_MT", "dln_Threatened_Capture"]],
        on=["ISO3", "Year"], how="left"
    )
    panel["FAO_Aqua_Share"] = panel["FAO_Aqua_Share"].fillna(panel["Mean_FAO_Aqua_Share"]).fillna(0.0)

    # Merge time-varying CWON Fisheries Natural Capital where available
    cwon_ts = wb_df[wb_df["indicator_id"] == "NW.NCA.FISH.PC"][["ISO3", "year", "value"]].rename(
        columns={"year": "Year", "value": "CWON_Fish_PC_Year"}
    )
    panel = panel.merge(cwon_ts, on=["ISO3", "Year"], how="left")

    panel_parquet = os.path.join(DERIVED_DIR, "country_socioeconomic_governance_master.parquet")
    panel_csv = os.path.join(DERIVED_DIR, "country_socioeconomic_governance_master.csv")
    panel.to_parquet(panel_parquet, index=False)
    panel.to_csv(panel_csv, index=False)
    print(f"[SUCCESS] Saved longitudinal country-year panel: {panel_parquet} ({len(panel):,} country-years)")

    # Print quick verification summary by World Bank Income Group & Primary Economic Bloc
    m_tar = sov_tar.merge(base_iso, left_on="Reporter_ISO3", right_on="ISO3", how="left")
    inc_summary = m_tar.groupby("WB_Income_Group", as_index=False).agg(
        Reporters=("Reporter_ISO3", "nunique"),
        Export_USD_B=("Total_Export_Real_USD_B", "sum"),
        Agg_USD_B=("Aggregated_Export_Real_USD", lambda s: s.sum() / 1e9),
        Mean_Gov=("Gov_Capacity_Index", "mean"),
        Mean_PSMA=("SDG14_6_1_IUU_PSMA_Score", "mean")
    )
    inc_summary["Pooled_TAR_Val_Pct"] = 100.0 * inc_summary["Agg_USD_B"] / inc_summary["Export_USD_B"]
    print("\n--- Verification: Export Customs Pooling & Governance by World Bank Income Group ---")
    print(inc_summary.sort_values("Pooled_TAR_Val_Pct").to_string(index=False))


if __name__ == "__main__":
    main()
