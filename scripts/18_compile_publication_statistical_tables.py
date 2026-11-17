#!/usr/bin/env python3
"""
Module 18: Compilation of Structured Statistical Summary Tables
==============================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Compiles harmonized descriptive, econometric, CITES policy, mirror-discrepancy,
trophic-balance, price-premium, and WCO tariff-reform tables into Markdown, CSV,
and LaTeX formats in tables/.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DERIVED = BASE_DIR / "data" / "derived"
CONFIG_DIR = BASE_DIR / "config"
TABLES_DIR = BASE_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def format_display_df(df):
    disp = df.copy()
    for col in disp.columns:
        col_lower = col.lower()
        if "p_value" in col_lower or "pval" in col_lower or col_lower.endswith("_p") or col_lower == "p":
            def fmt_p(v):
                if pd.isna(v):
                    return ""
                if isinstance(v, (int, float, np.number)):
                    return "<0.0001" if v < 0.0001 else f"{v:.4f}"
                return str(v)
            disp[col] = disp[col].apply(fmt_p)
        elif col in ["Cumulative_Biomass_MT", "Mann_Whitney_U", "Cumulative_Export_MT", "Cumulative_Import_MT", "Cumulative_Net_Biomass_MT"]:
            def fmt_large(v):
                if pd.isna(v):
                    return ""
                if isinstance(v, (int, float, np.number)):
                    return f"{v:,.0f}" if col == "Mann_Whitney_U" else f"{v:,.1f}"
                return str(v)
            disp[col] = disp[col].apply(fmt_large)
    return disp


def export_table(df, base_name, title):
    csv_path = TABLES_DIR / f"{base_name}.csv"
    md_path = TABLES_DIR / f"{base_name}.md"
    tex_path = TABLES_DIR / f"{base_name}.tex"

    # Export CSV (pure numeric / structured)
    df.to_csv(csv_path, index=False)

    disp_df = format_display_df(df)

    # Export Markdown (disable scientific notation)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"### {title}\n\n")
        f.write(disp_df.to_markdown(index=False, disable_numparse=True))
        f.write("\n")

    # Export LaTeX
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(f"% {title}\n")
        f.write(disp_df.to_latex(index=False, escape=True))

    print(f"   Exported {base_name} (.csv, .md, .tex)")


def main():
    print("=" * 80)
    print("MODULE 18: COMPILING 10 GLOBAL STATISTICAL TABLES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TABLE 1: Panel Descriptive Statistics Across 5-Year Epochs
    # -------------------------------------------------------------------------
    print("Compiling Table 1: Panel Descriptive Statistics...")
    ann_tar = pd.read_csv(DATA_DERIVED / "annual_global_customs_aggregation_tar.csv")
    ann_trade = pd.read_csv(DATA_DERIVED / "annual_global_conservation_trade_summary.csv")

    # Group into epochs
    def get_epoch(yr):
        if yr <= 1995: return "1988–1995"
        elif yr <= 2000: return "1996–2000"
        elif yr <= 2005: return "2001–2005"
        elif yr <= 2010: return "2006–2010"
        elif yr <= 2015: return "2011–2015"
        elif yr <= 2020: return "2016–2020"
        else: return "2021–2024"

    ann_tar["Epoch"] = ann_tar["Year"].apply(get_epoch)
    ann_trade["Epoch"] = ann_trade["Year"].apply(get_epoch)

    t1_tar = ann_tar.groupby("Epoch").agg(
        Years=("Year", "count"),
        Cumulative_Export_USD_B=("Total_Export_Real_USD_B", "sum"),
        Cumulative_Weight_MMT=("Total_Export_Weight_MMT", "sum"),
        Mean_TAR_Value_Pct=("TAR_Value_Percent", "mean"),
        Mean_TAR_Vol_Pct=("TAR_Volume_Percent", "mean")
    ).reset_index()

    t1_trade = ann_trade[ann_trade["Trade_Flow"] == "Export"].groupby("Epoch").agg(
        Active_Reporters=("Active_Reporters", "mean"),
        Active_Partners=("Active_Partners", "mean"),
        Annual_Bilateral_Pairs=("Bilateral_Pairs", "mean")
    ).reset_index()

    t1 = pd.merge(t1_tar, t1_trade, on="Epoch")
    t1_total = pd.DataFrame([{
        "Epoch": "Total / Mean (1988–2024)",
        "Years": int(t1["Years"].sum()),
        "Cumulative_Export_USD_B": ann_tar["Total_Export_Real_USD_B"].sum(),
        "Cumulative_Weight_MMT": ann_tar["Total_Export_Weight_MMT"].sum(),
        "Mean_TAR_Value_Pct": ann_tar["TAR_Value_Percent"].mean(),
        "Mean_TAR_Vol_Pct": ann_tar["TAR_Volume_Percent"].mean(),
        "Active_Reporters": t1["Active_Reporters"].mean(),
        "Active_Partners": t1["Active_Partners"].mean(),
        "Annual_Bilateral_Pairs": t1["Annual_Bilateral_Pairs"].mean()
    }])
    t1 = pd.concat([t1, t1_total], ignore_index=True).round(2)
    export_table(t1, "Table_01_Panel_Descriptive_Statistics", "Table 1. Global Seafood Trade Panel Descriptive Statistics (1988–2024)")

    # -------------------------------------------------------------------------
    # TABLE 2: Customs Aggregation Gap Across 7 WCO Revisions
    # -------------------------------------------------------------------------
    print("Compiling Table 2: Customs Aggregation Gap Across WCO Revisions...")
    t2_raw = ann_tar.groupby("HS_Era").agg(
        Years_Spanned=("Year", "count"),
        Cumulative_Export_Real_USD_B=("Total_Export_Real_USD_B", "sum"),
        Aggregated_Real_USD_B=("Aggregated_Real_USD_B", "sum"),
        Taxon_Specific_Real_USD_B=("Specific_Real_USD_B", "sum"),
        Mean_TAR_Value_Percent=("TAR_Value_Percent", "mean"),
        Mean_TAR_Volume_Percent=("TAR_Volume_Percent", "mean")
    ).reset_index()
    t2_total = pd.DataFrame([{
        "HS_Era": "All Editions Combined (1988–2024)",
        "Years_Spanned": int(ann_tar["Year"].count()),
        "Cumulative_Export_Real_USD_B": ann_tar["Total_Export_Real_USD_B"].sum(),
        "Aggregated_Real_USD_B": ann_tar["Aggregated_Real_USD_B"].sum(),
        "Taxon_Specific_Real_USD_B": ann_tar["Specific_Real_USD_B"].sum(),
        "Mean_TAR_Value_Percent": ann_tar["TAR_Value_Percent"].mean(),
        "Mean_TAR_Volume_Percent": ann_tar["TAR_Volume_Percent"].mean()
    }])
    t2_raw = pd.concat([t2_raw, t2_total], ignore_index=True).round(2)
    export_table(t2_raw, "Table_02_Customs_Aggregation_WCO_Revisions", "Table 2. The Global Customs Aggregation Gap Across 7 WCO Revisions")

    # -------------------------------------------------------------------------
    # TABLE 3: Marine Taxa Conservation Concordance & Economic Footprint
    # -------------------------------------------------------------------------
    print("Compiling Table 3: Marine Taxa Conservation Footprint...")
    df_taxa = pd.read_csv(DATA_DERIVED / "annual_global_conservation_trade_summary.csv")
    t3 = df_taxa[df_taxa["Trade_Flow"] == "Export"].groupby("Conservation_Category").agg(
        Time_Series_Years=("Year", "count"),
        Cumulative_Real_USD_B=("Total_Real_USD_M", lambda x: x.sum() / 1000.0),
        Cumulative_Biomass_MT=("Total_Weight_MT", "sum"),
        Mean_Annual_USD_M=("Total_Real_USD_M", "mean"),
        Mean_Unit_Price_USD_per_KG=("Real_Unit_Value_USD_per_KG", "mean"),
        Max_Unit_Price_USD_per_KG=("Real_Unit_Value_USD_per_KG", "max")
    ).reset_index().round(2)
    export_table(t3, "Table_03_Marine_Taxa_Conservation_Footprint", "Table 3. Marine Taxa Conservation Concordance & 37-Year Economic Footprint")

    # -------------------------------------------------------------------------
    # TABLE 4: Anthropogenic Allee Effect Econometric Regressions
    # -------------------------------------------------------------------------
    print("Compiling Table 4: Anthropogenic Allee Effect Regressions...")
    t4 = pd.read_csv(DATA_DERIVED / "global_allee_effect_econometric_results.csv").round(4)
    export_table(t4, "Table_04_Anthropogenic_Allee_Effect_Regressions", "Table 4. Econometric Estimates of the Anthropogenic Allee Effect (AAE)")

    # -------------------------------------------------------------------------
    # TABLE 5: Quasi-Experimental ITS Segmented Regressions (Newey-West HAC)
    # -------------------------------------------------------------------------
    print("Compiling Table 5: Quasi-Experimental ITS Regressions...")
    t5 = pd.read_csv(DATA_DERIVED / "global_interrupted_time_series_results.csv").round(4)
    export_table(t5, "Table_05_ITS_Segmented_Regressions_HAC", "Table 5. Quasi-Experimental Interrupted Time Series (ITS) Estimates (Newey-West HAC)")

    # -------------------------------------------------------------------------
    # TABLE 6: Bilateral Mirror Discrepancy (MDI) Integrity Across Taxa
    # -------------------------------------------------------------------------
    print("Compiling Table 6: Bilateral Mirror Discrepancy Across Taxa...")
    t6 = pd.read_csv(DATA_DERIVED / "global_mirror_discrepancy_by_taxa.csv").round(3)
    export_table(t6, "Table_06_Bilateral_Mirror_Discrepancies_Taxa", "Table 6. Bilateral Mirror Discrepancy Index (MDI) Across Conservation Taxa")

    # -------------------------------------------------------------------------
    # TABLE 7: Customs Granularity Comparison in MDI Integrity
    # -------------------------------------------------------------------------
    print("Compiling Table 7: Customs Granularity MDI Comparison...")
    t7 = pd.read_csv(DATA_DERIVED / "global_mirror_discrepancy_granularity_comparison.csv").round(3)
    export_table(t7, "Table_07_Customs_Granularity_MDI_Integrity", "Table 7. Impact of Customs Granularity on Bilateral Mirror Integrity")

    # -------------------------------------------------------------------------
    # TABLE 8: 37-Year Trophic Balance of Trade (TBT) by Regional Bloc
    # -------------------------------------------------------------------------
    print("Compiling Table 8: Trophic Balance of Trade Across Regional Blocs...")
    t8 = pd.read_csv(DATA_DERIVED / "bloc_trophic_balance_of_trade.csv").round(3)
    export_table(t8, "Table_08_Trophic_Balance_of_Trade_Blocs", "Table 8. 37-Year Cumulative Trophic Balance of Trade (TBT) Across 8 Regional Blocs")

    # -------------------------------------------------------------------------
    # TABLE 9: Wild vs Farmed Unit Value Premiums Across Processing Forms
    # -------------------------------------------------------------------------
    print("Compiling Table 9: Wild vs Farmed Price Premiums...")
    t9 = pd.read_csv(DATA_DERIVED / "wild_vs_farmed_price_premium_tests.csv")
    t9["Price_Ratio_Wild_to_Farm"] = t9["Price_Ratio_Wild_to_Farm"].round(2)
    t9["Rank_Biserial_Correlation_r"] = t9["Rank_Biserial_Correlation_r"].round(3)
    export_table(t9, "Table_09_Wild_vs_Farmed_Price_Premiums", "Table 9. Real Export Unit Value Premiums: Wild Capture vs. Aquaculture")

    # -------------------------------------------------------------------------
    # TABLE 10: Prioritized 2028 WCO Harmonized System Reform Matrix
    # -------------------------------------------------------------------------
    print("Compiling Table 10: 2028 WCO Harmonized System Reform Matrix (K* = 22 MILP Schedule)...")
    dr_df = pd.read_csv(DATA_DERIVED / "wco_2028_dirichlet_robustness_sensitivity.csv").set_index("Cluster_ID")
    f_m01 = dr_df.loc["M01", "Selection_Frequency_At_K22_Pct"]
    f_m04 = dr_df.loc["M04", "Selection_Frequency_At_K22_Pct"]
    f_m03 = dr_df.loc["M03", "Selection_Frequency_At_K22_Pct"]
    f_m08 = dr_df.loc["M08", "Selection_Frequency_At_K22_Pct"]
    f_m05 = dr_df.loc["M05", "Selection_Frequency_At_K22_Pct"]
    f_m07 = dr_df.loc["M07", "Selection_Frequency_At_K22_Pct"]
    matrix_data = [
        {
            "Reform_Module_And_Cost": "Paired Shark Meat & Fin Modules (4+3=7)",
            "Focal_Taxon": "Pelagic & Coastal Sharks (Carcharhinidae, Sphyrnidae, Lamnidae)",
            "IUCN_Status": "Critically Endangered / Endangered",
            "CITES_Listing": "Appendix II (CoP16, CoP17, CoP19; 78 spp.)",
            "Current_Customs_Structure": "Pooled fresh, frozen & filleted dogfish/other shark meat and generic shark fin headings",
            "Proposed_2028_HS_Structure": "4 carcass/meat subheadings + 3 fresh, frozen & dried fin subheadings (Prionace, Carcharhinus, Sphyrna, Isurus)",
            "Pooling_And_Robustness": f"55.0 -> 37.9 -> 1.0 ({f_m01:.1f}%)",
            "Conservation_Policy_Justification": "Eliminates generic shark meat and fin aggregation; statutory raw-to-processed coupling (Equation 13) prevents carcass-versus-fin tariff jumping"
        },
        {
            "Reform_Module_And_Cost": "Mobulid Ray Module (2)",
            "Focal_Taxon": "Mobulid Rays (Mobula spp. - Manta & Devil Rays)",
            "IUCN_Status": "Endangered / Vulnerable",
            "CITES_Listing": "Appendix II (CoP17; 9 spp.)",
            "Current_Customs_Structure": "Pooled skate and ray meat/wing baskets and generic dried fish offal headings",
            "Proposed_2028_HS_Structure": "2 dedicated six-digit subheadings for mobulid meat/wings and dried prebranchial gill plates",
            "Pooling_And_Robustness": f"18.0 -> 17.5 -> 1.0 ({f_m04:.1f}%)",
            "Conservation_Policy_Justification": "Stops unrecorded trade of dried gill plates concealed within generic batoid meat or dried fish offal residual baskets"
        },
        {
            "Reform_Module_And_Cost": "Rhinopristiformes Module (3)",
            "Focal_Taxon": "Rhinopristiformes (Sawfishes, Guitarfishes & Wedgefishes)",
            "IUCN_Status": "Critically Endangered (Highest Risk)",
            "CITES_Listing": "Appendix I / II (CoP14, CoP18; 10 spp.)",
            "Current_Customs_Structure": "Pooled skate/ray meat, unclassified marine finfish, and generic dried fin baskets",
            "Proposed_2028_HS_Structure": "3 dedicated six-digit subheadings for Rhinidae, Glaucostegidae, and Pristidae meat, white fins, and live specimens",
            "Pooling_And_Robustness": f"41.0 -> 27.3 -> 1.0 ({f_m03:.1f}%)",
            "Conservation_Policy_Justification": "Separates Critically Endangered wedgefishes and guitarfishes from pooled batoid meat headings and generic shark fin baskets"
        },
        {
            "Reform_Module_And_Cost": "Sturgeon & Caviar Module (3)",
            "Focal_Taxon": "Sturgeons & Caviar (Acipenseriformes)",
            "IUCN_Status": "Critically Endangered",
            "CITES_Listing": "Appendix I / II (Universal CoP10; 25 spp.)",
            "Current_Customs_Structure": "Order-wide pooled caviar heading and generic live, fresh, and frozen finfish baskets",
            "Proposed_2028_HS_Structure": "3 six-digit subheadings splitting wild-stock vs. aquaculture caviar (Huso, Acipenser) and sturgeon meat",
            "Pooling_And_Robustness": f"25.0 -> 8.3 -> 1.0 ({f_m08:.1f}%)",
            "Conservation_Policy_Justification": "Discriminates between legal aquaculture caviar/meat and illicit poaching from residual Caspian, Black Sea, and Amur wild stocks"
        },
        {
            "Reform_Module_And_Cost": "Paired Live & Prepared Eel Modules (2+2=4)",
            "Focal_Taxon": "Anguillid Eels (Anguilla anguilla, A. japonica, A. rostrata)",
            "IUCN_Status": "Critically Endangered / Endangered",
            "CITES_Listing": "Appendix II (A. anguilla, CoP14; 1 sp.)",
            "Current_Customs_Structure": "Genus-wide live, fresh, and frozen whole eel headings and pooled prepared/smoked fillet headings",
            "Proposed_2028_HS_Structure": "2 live/whole subheadings (glass eel fry vs. adult) + 2 species-resolved prepared and smoked fillet subheadings",
            "Pooling_And_Robustness": f"5.0 -> 2.5 -> 1.0 ({f_m05:.1f}%)",
            "Conservation_Policy_Justification": "Statutory raw-to-processed coupling (Equation 13) prevents displacement of European glass eels via Asian grow-out ponds into prepared fillet exports"
        },
        {
            "Reform_Module_And_Cost": "Sea Cucumber Module (3)",
            "Focal_Taxon": "Sea Cucumbers (Holothuroidea - Teatfish & Look-Alikes)",
            "IUCN_Status": "Endangered / Vulnerable",
            "CITES_Listing": "Appendix II (CoP18, CoP19; 2 spp.)",
            "Current_Customs_Structure": "Class-wide live, frozen, dried/salted beche-de-mer, and prepared sea cucumber headings",
            "Proposed_2028_HS_Structure": "3 six-digit subheadings separating dried/salted beche-de-mer and frozen lines (Holothuria teatfish, Thelenota, Stichopus)",
            "Pooling_And_Robustness": f"12.0 -> 4.0 -> 1.0 ({f_m07:.1f}%)",
            "Conservation_Policy_Justification": "Addresses highest bilateral mirror discrepancy (57.05% severe mismatch) and congeneric substitution in Indo-Pacific reef fisheries"
        }
    ]
    t10 = pd.DataFrame(matrix_data)
    export_table(t10, "Table_10_Prioritized_2028_WCO_HS_Reform_Matrix", "Table 10. Prioritized 2028 WCO Harmonized System Reform Matrix for CITES-Listed Marine Taxa (K* = 22 Pareto-Optimal MILP Schedule)")

    # -------------------------------------------------------------------------
    # TABLE 18 (Supplementary Table S18): Multi-Scale Sovereign, Income, Continent, Bloc & Subregion Matrix
    # -------------------------------------------------------------------------
    print("Compiling Supplementary Table S18: Multi-Scale Sovereign, Income-Tier, Continent, Bloc & SIDS/LDC Matrix...")
    t18 = pd.read_csv(DATA_DERIVED / "multi_scale_bloc_income_trophic_customs_matrix.csv")
    export_table(t18, "Table_18_Multi_Scale_Sovereign_Bloc_Income_Matrix", "Supplementary Table S18. Multi-Scale Sovereign, Income-Tier, Continental, Economic-Bloc, SIDS/LDC & UN M49 Subregional Trade-Conservation Matrix (1988-2024)")

    # -------------------------------------------------------------------------
    # TABLE 19 (Supplementary Table S19): Partial-Derivative Econometric Suites A-D Summary
    # -------------------------------------------------------------------------
    print("Compiling Supplementary Table S19: Partial-Derivative Econometric Suites A-D & CBDR Reform Exposure...")
    sa = pd.read_csv(DATA_DERIVED / "sovereign_customs_consolidation_decomposition.csv")
    sb = pd.read_csv(DATA_DERIVED / "sovereign_allee_income_aquaculture_moderation.csv")
    sc1 = pd.read_csv(DATA_DERIVED / "governance_moderated_gravity_ppml_summary.csv")
    sc2 = pd.read_csv(DATA_DERIVED / "governance_moderated_ppml_and_evasion_results.csv")
    sd = pd.read_csv(DATA_DERIVED / "brander_taylor_cwon_natural_capital_results.csv")

    s19_rows = []
    _sa0 = sa.iloc[0]
    _suiteA_lbl = f"Suite A: Sovereign Customs Consolidation (Eq. 2c, WLS + Shapley R2={_sa0['Total_WLS_R2']:.3f})"
    _suiteA_tgt = (f"TAR_Value_Percent (Biogeography {_sa0['Shapley_Biogeographic_Pct']:.1f}%, "
                   f"Governance {_sa0['Shapley_Governance_Capacity_Pct']:.1f}%, Processing {_sa0['Shapley_Processing_Aquaculture_Pct']:.1f}%)")
    for _, r in sa[sa["Model"] == "WLS_LogTradeWeighted_TAR_Pct"].iterrows():
        s19_rows.append({
            "Econometric_Suite": _suiteA_lbl,
            "Target_Outcome_Or_Clade": _suiteA_tgt,
            "Partial_Derivative_Or_Parameter": r["Parameter"],
            "Estimate": r["Estimate"],
            "Std_Error": r["Std_Error_HC3"],
            "Stat": r["z_or_t_stat"],
            "P_Value": round(float(r["p_value"]), 5),
            "N_Obs": int(r["N_Countries"])
        })
    for _, r in sb.iterrows():
        s19_rows.append({
            "Econometric_Suite": "Suite B: Coupled ODE Scarcity Elasticity & Moderation (Eq. 4c & S4d)",
            "Target_Outcome_Or_Clade": r["Clade_Scope"],
            "Partial_Derivative_Or_Parameter": "d ln P / d ln Q [Beta_ln_Q_Base]",
            "Estimate": r["Beta_ln_Q_Base"],
            "Std_Error": r["SE_ln_Q_Base"],
            "Stat": round(float(r["Beta_ln_Q_Base"]) / max(float(r["SE_ln_Q_Base"]), 1e-6), 3) if pd.notna(r["SE_ln_Q_Base"]) else np.nan,
            "P_Value": round(float(r["p_ln_Q_Base"]), 5) if pd.notna(r["p_ln_Q_Base"]) else np.nan,
            "N_Obs": int(r["N_Obs"])
        })
        if pd.notna(r["Theta2_ln_Q_x_ExporterAquaShare"]) and pd.notna(r["SE_Theta2_Aqua"]):
            s19_rows.append({
                "Econometric_Suite": "Suite B: Coupled ODE Aquaculture Attenuation Cross-Partial (Eq. 4c & S4d)",
                "Target_Outcome_Or_Clade": r["Clade_Scope"],
                "Partial_Derivative_Or_Parameter": f"d^2 ln P / (d ln Q * d {r['Aqua_Measure']}) [theta_2]",
                "Estimate": r["Theta2_ln_Q_x_ExporterAquaShare"],
                "Std_Error": r["SE_Theta2_Aqua"],
                "Stat": round(float(r["Theta2_ln_Q_x_ExporterAquaShare"]) / max(float(r["SE_Theta2_Aqua"]), 1e-6), 3),
                "P_Value": round(float(r["p_Theta2_Aqua"]), 5) if pd.notna(r["p_Theta2_Aqua"]) else np.nan,
                "N_Obs": int(r["N_Obs"])
            })
        if pd.notna(r["Theta1_ln_Q_x_ImporterLnGDPpc"]) and pd.notna(r["SE_Theta1_GDPpc"]):
            s19_rows.append({
                "Econometric_Suite": "Suite B: Coupled ODE Income-Stratification Cross-Partial (Eq. 4c & S4d)",
                "Target_Outcome_Or_Clade": r["Clade_Scope"],
                "Partial_Derivative_Or_Parameter": "d^2 ln P / (d ln Q * d ln y_j) [theta_1]",
                "Estimate": r["Theta1_ln_Q_x_ImporterLnGDPpc"],
                "Std_Error": r["SE_Theta1_GDPpc"],
                "Stat": round(float(r["Theta1_ln_Q_x_ImporterLnGDPpc"]) / max(float(r["SE_Theta1_GDPpc"]), 1e-6), 3),
                "P_Value": round(float(r["p_Theta1_GDPpc"]), 5) if pd.notna(r["p_Theta1_GDPpc"]) else np.nan,
                "N_Obs": int(r["N_Obs"])
            })
    for _, r in sc1.iterrows():
        s19_rows.append({
            "Econometric_Suite": "Suite C1: Governance-Moderated Structural Gravity PPML (Eq. 6d)",
            "Target_Outcome_Or_Clade": r["Regime"],
            "Partial_Derivative_Or_Parameter": "d^2 ln E[Q] / (d[Treat*Post] * d CorridorGov) [phi_1]",
            "Estimate": r["CrossPartial_TreatPost_x_CorridorGov"],
            "Std_Error": r["SE_CrossPartial_Gov"],
            "Stat": round(float(r["CrossPartial_TreatPost_x_CorridorGov"]) / max(float(r["SE_CrossPartial_Gov"]), 1e-6), 3),
            "P_Value": round(float(r["p_CrossPartial_Gov"]), 5),
            "N_Obs": int(r["N_Cells"])
        })
    for _, r in sc2[sc2["Corridor_Stratum"] == "All_Global_Corridors"].iterrows():
        m_tag = str(r["Model"]).replace("Unilateral_Orphan_Extensive_LPM", "Unilateral_Unmatched_Extensive_LPM")
        s19_rows.append({
            "Econometric_Suite": f"Suite C2: Fisman-Wei Tariff vs. CITES Evasion ({m_tag}, Eq. 9d)",
            "Target_Outcome_Or_Clade": "All Global Corridors (Dyad + Year FE)",
            "Partial_Derivative_Or_Parameter": r["Parameter"],
            "Estimate": r["Estimate"],
            "Std_Error": r["Std_Error"],
            "Stat": r["t_stat"],
            "P_Value": round(float(r["p_value"]), 6),
            "N_Obs": int(r["N_Obs"])
        })
    for _, r in sd.iterrows():
        s19_rows.append({
            "Econometric_Suite": f"Suite D: Brander-Taylor-CWON Natural Capital & Trophic Drain ({r['Model']}, Eq. 11c)",
            "Target_Outcome_Or_Clade": r["Model"],
            "Partial_Derivative_Or_Parameter": r["Parameter"],
            "Estimate": r["Estimate"],
            "Std_Error": r["Std_Error_HC3"],
            "Stat": r["t_stat"],
            "P_Value": round(float(r["p_value"]), 5),
            "N_Obs": int(r["N_Obs"])
        })
    t19 = pd.DataFrame(s19_rows)
    export_table(t19, "Table_19_Partial_Derivative_Econometric_Suites_A_to_D", "Supplementary Table S19. Structural Partial-Derivative & Cross-Partial Econometric Suites A-D (Sovereign Governance, Aquaculture Attenuation, Tariff-vs.-CITES Evasion & CWON Natural Capital)")

    # -------------------------------------------------------------------------
    # TABLE 20 (Supplementary Table S20): Calibrated Coupled Bioeconomic ODE Equilibria & Stability Regimes
    # -------------------------------------------------------------------------
    print("Compiling Supplementary Table S20: Calibrated Coupled Bioeconomic ODE Equilibria & Stability Regimes...")
    t20 = pd.read_csv(DATA_DERIVED / "bioeconomic_ode_bifurcation_regimes.csv")
    export_table(t20, "Table_20_Coupled_Bioeconomic_ODE_Jacobian_Bifurcations", "Supplementary Table S20. Calibrated Coupled Wild-Aquaculture Bioeconomic Differential Equation (ODE) Equilibria, 2x2 Jacobian Eigenvalues & Stability Regimes")

    print("\nAll global statistical tables (including S18-S20) successfully compiled in tables/ directory!")
    print("=" * 80)


if __name__ == "__main__":
    main()
