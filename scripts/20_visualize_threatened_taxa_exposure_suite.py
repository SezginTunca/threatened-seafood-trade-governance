#!/usr/bin/env python3
"""
Module 20: Expanded 767-Species Taxonomic Resolution Figures (Figs. S5-S6)
==========================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Generates Supplementary Figures S5 and S6 and the interactive species customs
exposure summary from the 767-taxon conservation audit.
"""
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DERIVED = BASE_DIR / "data" / "derived"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_CSV = DATA_DERIVED / "expanded_iucn_threatened_taxa_catalog.csv"
SUMMARY_CSV = DATA_DERIVED / "threatened_taxa_customs_invisibility_summary.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Lucida Grande"],
    "font.size": 11.5,
    "axes.titlesize": 13.0,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.0,
    "figure.titlesize": 14.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

# Styling constants
COLOR_CR = "#B71C1C"  # Deep Crimson
COLOR_EN = "#E65100"  # Vibrant Amber/Orange
COLOR_VU = "#F57F17"  # Golden Yellow
COLOR_NT = "#33691E"  # Forest Olive
COLOR_SPEC = "#1B5E20"  # Dark Emerald
COLOR_GEN = "#0D47A1"  # Deep Royal Blue
COLOR_SUB = "#4A148C"  # Rich Indigo/Purple
TEXT_DARK = "#0F172A"
BG_LIGHT = "#F8FAFC"

TAXON_DISPLAY_LABELS = {
    "CHONDRICHTHYES": "Chondrichthyes (Sharks & Rays)",
    "SALMONIDAE": "Salmonidae (Salmons & Trouts)",
    "ACIPENSERIFORMES": "Acipenseriformes (Sturgeons)",
    "HALIOTIDAE": "Haliotidae (Abalones)",
    "EPINEPHELIDAE": "Epinephelidae (Groupers)",
    "LABRIDAE": "Labridae (Wrasses)",
    "HOLOTHUROIDEA": "Holothuroidea (Sea Cucumbers)",
    "ANGUILLIDAE": "Anguillidae (Freshwater Eels)",
    "SCOMBRIDAE": "Scombridae (Tunas & Mackerels)",
    "PLEURONECTIFORMES": "Pleuronectiformes (Flatfishes)",
    "GADIFORMES": "Gadiformes (Cods & Hakes)",
}


def plot_figure_s05():
    """Generates Supplementary Figure S5: The Global Marine Conservation Invisibility Funnel."""
    print("Generating Supplementary Figure S5: Conservation Invisibility Funnel (600 DPI)...")
    df_taxa = pd.read_csv(CATALOG_CSV)

    fig = plt.figure(figsize=(18.5, 12.2), dpi=600, facecolor="#FFFFFF")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.30, wspace=0.36)

    # -------------------------------------------------------------
    # Panel a: The Global Invisibility Funnel
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(BG_LIGHT)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for s in ax_a.spines.values():
        s.set_visible(True)
        s.set_color("#CBD5E1")
        s.set_linewidth(0.9)

    ax_a.set_title("a  The Global Marine Conservation Classification Funnel",
                   fontsize=13.0, fontweight="bold", loc="left", pad=12, color=TEXT_DARK)

    # Draw funnel steps with downward connecting arrows
    stages = [
        ("Evaluated Marine & Diadromous Seafood Taxa", "767 Commercial Taxa (11 Major Orders)", "#1565C0", 0.87, 0.88),
        ("Strictly Threatened Lineages (CR, EN, VU)", "602 Species (78.5% Strictly Threatened)", "#D84315", 0.68, 0.76),
        ("Catch-All or Genus-Aggregated in Customs", "764 Species (99.61% Customs Non-Identifiability)", "#6A1B9A", 0.49, 0.64),
        ("Species-Specific Customs Monitoring", "3 Species (0.39% Direct Coverage)", "#2E7D32", 0.30, 0.52),
    ]

    box_half_h = 0.062
    for idx, (label, stat, color, y, width) in enumerate(stages):
        rect = patches.FancyBboxPatch(
            (0.5 - width / 2, y - box_half_h), width, box_half_h * 2,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=color, edgecolor="#FFFFFF", linewidth=1.6, alpha=0.94, zorder=3
        )
        ax_a.add_patch(rect)
        ax_a.text(0.5, y + 0.018, label, fontsize=11.0, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=4)
        ax_a.text(0.5, y - 0.024, stat, fontsize=10.5, fontweight="bold", color="#FFF9C4", ha="center", va="center", zorder=4)

        if idx < len(stages) - 1:
            next_y = stages[idx + 1][3]
            ax_a.annotate(
                "", xy=(0.5, next_y + box_half_h + 0.012), xytext=(0.5, y - box_half_h - 0.010),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=2.0, color="#475569"),
                zorder=5
            )

    # Bottom Callout Card
    callout = patches.FancyBboxPatch(
        (0.06, 0.035), 0.88, 0.16,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor="#ECEFF1", edgecolor="#90A4AE", linewidth=1.3, zorder=3
    )
    ax_a.add_patch(callout)
    ax_a.text(0.5, 0.152, "CORE EMPIRICAL FINDING", fontsize=11.0, fontweight="bold", color="#B71C1C", ha="center", va="center", zorder=4)
    ax_a.text(
        0.5, 0.082,
        "99.61% of all evaluated threatened and near-threatened marine seafood taxa\n"
        "lack individual species-level declarations under WCO Harmonized System trade codes.",
        fontsize=10.2, fontweight="bold", color=TEXT_DARK, ha="center", va="center", linespacing=1.35, zorder=4
    )

    # -------------------------------------------------------------
    # Panel b: Customs Visibility by Extinction Risk Category
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(BG_LIGHT)
    ax_b.set_title("b  Extinction Risk Composition by Customs Visibility",
                   fontsize=13.0, fontweight="bold", loc="left", pad=12, color=TEXT_DARK)

    ct = pd.crosstab(df_taxa["Customs_Reporting_Visibility"], df_taxa["Red_List_Category"])
    ct = ct[["CR", "EN", "VU", "NT"]]
    # Ensure consistent vertical order from largest to smallest (bottom to top)
    desired_order = ["Genus/Family-Aggregated", "Omnibus-Submerged", "Species-Specific"]
    ct = ct.reindex([r for r in desired_order if r in ct.index])

    y_pos = np.arange(len(ct))
    bar_h = 0.52

    lefts = np.zeros(len(ct))
    colors_tier = [COLOR_CR, COLOR_EN, COLOR_VU, COLOR_NT]
    tier_labels = ["Critically Endangered (CR)", "Endangered (EN)", "Vulnerable (VU)", "Near Threatened (NT)"]

    for idx, col in enumerate(["CR", "EN", "VU", "NT"]):
        vals = ct[col].values
        ax_b.barh(y_pos, vals, left=lefts, height=bar_h, color=colors_tier[idx], label=tier_labels[idx], edgecolor="#FFFFFF", linewidth=0.9)
        for i, val in enumerate(vals):
            if val >= 18:
                ax_b.text(lefts[i] + val / 2, y_pos[i], f"{val}", ha="center", va="center", color="#FFFFFF", fontweight="bold", fontsize=10.0)
        lefts += vals

    for i, tot in enumerate(lefts):
        if tot < 10:
            ax_b.text(tot + 10, y_pos[i], f"{int(tot)} spp. (1 EN, 2 VU)", va="center", ha="left", fontsize=10.0, fontweight="bold", color=TEXT_DARK)
        else:
            ax_b.text(tot + 10, y_pos[i], f"{int(tot)} spp.", va="center", ha="left", fontsize=10.0, fontweight="bold", color=TEXT_DARK)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(ct.index, fontsize=11.0, fontweight="bold", color=TEXT_DARK)
    ax_b.set_xlabel("Number of Assessed Taxa", fontsize=11.5, fontweight="bold", color=TEXT_DARK)
    ax_b.set_xlim(0, 770)
    ax_b.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#B0BEC5", fontsize=9.8)
    ax_b.grid(axis="x", linestyle="--", alpha=0.4, color="#9E9E9E")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------
    # Panel c: Taxonomic Group Distribution
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(BG_LIGHT)
    ax_c.set_title("c  Threatened & Near-Threatened Species by Taxonomic Group",
                   fontsize=13.0, fontweight="bold", loc="left", pad=12, color=TEXT_DARK)

    grp_counts = df_taxa["Taxonomic_Group"].value_counts().sort_values(ascending=True)
    y_pos_c = np.arange(len(grp_counts))
    clean_grp_labels = [TAXON_DISPLAY_LABELS.get(str(g), str(g).title()) for g in grp_counts.index]

    bars = ax_c.barh(y_pos_c, grp_counts.values, height=0.62, color="#0277BD", edgecolor="#FFFFFF", linewidth=0.8)
    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels(clean_grp_labels, fontsize=10.5, fontweight="bold", color=TEXT_DARK)
    ax_c.set_xlabel("Total Assessed Taxa Count", fontsize=11.5, fontweight="bold", color=TEXT_DARK)
    ax_c.set_xlim(0, 620)
    ax_c.grid(axis="x", linestyle="--", alpha=0.4, color="#9E9E9E")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    for bar, val in zip(bars, grp_counts.values):
        ax_c.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2, f"{val} spp.",
                  va="center", ha="left", fontsize=10.0, fontweight="bold", color=TEXT_DARK)

    # -------------------------------------------------------------
    # Panel d: CITES Coverage Breakdown
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(BG_LIGHT)
    ax_d.set_title("d  Multilateral CITES Regulatory Coverage Gap",
                   fontsize=13.0, fontweight="bold", loc="left", pad=12, color=TEXT_DARK)

    n_non_cites = int((df_taxa["CITES_Status"] == "Non-CITES").sum())
    n_app2 = int(df_taxa["CITES_Status"].str.contains(r"Appendix II\b", regex=True).sum())
    n_app1 = int(df_taxa["CITES_Status"].str.contains(r"Appendix I(?:\b|/)", regex=True).sum())

    cites_vals = [n_non_cites, n_app2, n_app1]
    cites_labels = [
        f"Non-CITES Regulated\n({n_non_cites} spp., {n_non_cites/len(df_taxa)*100:.1f}%)",
        f"CITES Appendix II\n({n_app2} spp., {n_app2/len(df_taxa)*100:.1f}%)",
        f"CITES Appendix I / Combined\n({n_app1} spp., {n_app1/len(df_taxa)*100:.1f}%)",
    ]

    colors_d = ["#C62828", "#1565C0", "#2E7D32"]
    wedges, texts, autotexts = ax_d.pie(
        cites_vals,
        labels=cites_labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 5.0 else "",
        pctdistance=0.76,
        labeldistance=1.14,
        startangle=145,
        colors=colors_d,
        wedgeprops=dict(width=0.45, edgecolor="#FFFFFF", linewidth=2.2),
        textprops=dict(fontsize=10.5, fontweight="bold", color=TEXT_DARK)
    )
    for at in autotexts:
        at.set_color("#FFFFFF")
        at.set_fontsize(10.5)
        at.set_fontweight("bold")

    ax_d.text(0, 0, f"Total\n{len(df_taxa)}\nSpecies", ha="center", va="center",
              fontsize=12.5, fontweight="bold", color=TEXT_DARK)

    out_png = FIGURES_DIR / "Figure_S05_Taxonomic_Invisibility_Funnel.png"
    out_pdf = FIGURES_DIR / "Figure_S05_Taxonomic_Invisibility_Funnel.pdf"
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"   -> Saved: {out_png.name} & {out_pdf.name}")


def plot_figure_s06():
    """Generates Supplementary Figure S6: Threatened Family Exposure Matrix across Customs Lines."""
    print("Generating Supplementary Figure S6: Threatened Family Exposure Matrix (600 DPI)...")
    df_taxa = pd.read_csv(CATALOG_CSV, dtype={"Primary_HS6_Code": str})

    fig, ax = plt.subplots(figsize=(16.5, 9.2), dpi=600, facecolor="#FFFFFF")
    ax.set_facecolor(BG_LIGHT)

    ct = pd.crosstab(df_taxa["Taxonomic_Group"], df_taxa["Customs_Reporting_Visibility"])
    ct["Total"] = ct.sum(axis=1)
    ct = ct.sort_values("Total", ascending=True)

    y_pos = np.arange(len(ct))
    bar_h = 0.58
    clean_labels = [TAXON_DISPLAY_LABELS.get(str(g), str(g).title()) for g in ct.index]

    ax.barh(y_pos, ct.get("Species-Specific", 0), height=bar_h, color=COLOR_SPEC,
            label="Species-Specific Line (Monitored: 3 spp., 0.39%)", edgecolor="#FFFFFF", linewidth=0.8)
    ax.barh(y_pos, ct.get("Genus/Family-Aggregated", 0), left=ct.get("Species-Specific", 0), height=bar_h, color=COLOR_GEN,
            label="Genus/Family-Aggregated Line (Pooled Multi-Species: 673 spp., 87.74%)", edgecolor="#FFFFFF", linewidth=0.8)
    left_sub = ct.get("Species-Specific", 0) + ct.get("Genus/Family-Aggregated", 0)
    ax.barh(y_pos, ct.get("Omnibus-Submerged", 0), left=left_sub, height=bar_h, color=COLOR_SUB,
            label="Catch-All Residual Basket ('Other Fish' n.e.c.: 91 spp., 11.86%)", edgecolor="#FFFFFF", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_labels, fontsize=11.0, fontweight="bold", color=TEXT_DARK)
    ax.set_xlabel("Number of Threatened and Near-Threatened Species", fontsize=12.0, fontweight="bold", color=TEXT_DARK)
    ax.set_xlim(0, 615)
    ax.set_title(
        "Taxonomic Exposure Matrix: Distribution of Threatened Seafood Taxa across Customs Specificity Tiers\n"
        "Distribution of 767 Threatened and Near-Threatened Marine Taxa Across Three WCO Customs Reporting Tiers",
        fontsize=13.5, fontweight="bold", pad=16, color=TEXT_DARK
    )

    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#B0BEC5", fontsize=10.8)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#9E9E9E")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, tot in enumerate(ct["Total"]):
        ax.text(tot + 8, y_pos[i], f"{tot} spp.", va="center", ha="left", fontsize=10.5, fontweight="bold", color=TEXT_DARK)

    out_png = FIGURES_DIR / "Figure_S06_Threatened_Family_Exposure_Matrix.png"
    out_pdf = FIGURES_DIR / "Figure_S06_Threatened_Family_Exposure_Matrix.pdf"
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"   -> Saved: {out_png.name} & {out_pdf.name}")


def generate_interactive_html():
    """Generates a standalone, searchable HTML explorer for all 767 threatened taxa."""
    print("Generating Interactive HTML Explorer...")
    df_taxa = pd.read_csv(CATALOG_CSV, dtype={"Primary_HS6_Code": str})
    df_taxa["Primary_HS6_Code"] = df_taxa["Primary_HS6_Code"].astype(str).str.zfill(6)

    taxa_json = df_taxa.to_json(orient="records")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IUCN-Threatened Seafood Taxa & Customs Invisibility Explorer</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary: #0D47A1;
      --primary-dark: #002171;
      --bg: #F4F6F9;
      --card-bg: #FFFFFF;
      --text: #212529;
      --text-muted: #6C757D;
      --border: #DEE2E6;
      --cr: #B71C1C;
      --en: #E65100;
      --vu: #F57F17;
      --nt: #33691E;
      --monitored: #1B5E20;
      --aggregated: #0D47A1;
      --submerged: #6A1B9A;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .header {{
      background: linear-gradient(135deg, #0A192F 0%, #172A45 100%);
      color: #FFFFFF;
      padding: 28px 36px;
      border-radius: 12px;
      margin-bottom: 24px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    .header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.5px; }}
    .header p {{ font-size: 14px; color: #B0BEC5; max-width: 900px; }}
    
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      padding: 20px;
      border-radius: 10px;
      border: 1px solid var(--border);
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .stat-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
    .stat-val {{ font-size: 28px; font-weight: 700; color: var(--primary-dark); }}
    .stat-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

    .controls {{
      background: var(--card-bg);
      padding: 20px 24px;
      border-radius: 10px;
      border: 1px solid var(--border);
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
    }}
    .search-input {{
      flex: 1;
      min-width: 260px;
      padding: 10px 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 14px;
      outline: none;
    }}
    .search-input:focus {{ border-color: var(--primary); }}
    select {{
      padding: 10px 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 14px;
      background: #FFFFFF;
      outline: none;
    }}

    .table-container {{
      background: var(--card-bg);
      border-radius: 10px;
      border: 1px solid var(--border);
      overflow-x: auto;
      box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #F8F9FA;
      padding: 12px 16px;
      font-weight: 600;
      color: var(--text);
      border-bottom: 2px solid var(--border);
      white-space: nowrap;
    }}
    td {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
    }}
    tr:hover {{ background-color: #F8F9FA; }}
    
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      color: #FFFFFF;
    }}
    .badge-cr {{ background: var(--cr); }}
    .badge-en {{ background: var(--en); }}
    .badge-vu {{ background: var(--vu); color: #000; }}
    .badge-nt {{ background: var(--nt); }}
    
    .badge-vis {{
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
    }}
    .vis-spec {{ background: #E8F5E9; color: var(--monitored); border: 1px solid #A5D6A7; }}
    .vis-gen {{ background: #E3F2FD; color: var(--aggregated); border: 1px solid #90CAF9; }}
    .vis-sub {{ background: #F3E5F5; color: var(--submerged); border: 1px solid #CE93D8; }}
  </style>
</head>
<body>

  <div class="header">
    <h1>IUCN-Threatened Seafood Taxa & Customs Invisibility Explorer</h1>
    <p>Comprehensive biological-customs concordance mapping 767 threatened and near-threatened commercial marine and diadromous species to the World Customs Organization (WCO) Harmonized System (HS 6-digit) classification.</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-title">Total Assessed Taxa</div>
      <div class="stat-val">767</div>
      <div class="stat-sub">Across 11 Major Marine Orders</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Strict Threatened (CR/EN/VU)</div>
      <div class="stat-val" style="color: var(--cr);">602</div>
      <div class="stat-sub">78.5% of Assessed Taxa</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Customs Invisibility Rate</div>
      <div class="stat-val" style="color: var(--submerged);">99.61%</div>
      <div class="stat-sub">764 of 767 Species Invisible</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Species-Level Monitored</div>
      <div class="stat-val" style="color: var(--monitored);">3</div>
      <div class="stat-sub">Only 0.39% Direct Monitoring</div>
    </div>
  </div>

  <div class="controls">
    <input type="text" id="searchInput" class="search-input" placeholder="Search scientific name, family, or HS code...">
    <select id="groupFilter">
      <option value="">All Taxonomic Groups</option>
      <option value="CHONDRICHTHYES">Chondrichthyes (Sharks & Rays)</option>
      <option value="SALMONIDAE">Salmonidae (Salmonids)</option>
      <option value="ACIPENSERIFORMES">Acipenseriformes (Sturgeons)</option>
      <option value="HALIOTIDAE">Haliotidae (Abalones)</option>
      <option value="EPINEPHELIDAE">Epinephelidae (Groupers)</option>
      <option value="HOLOTHUROIDEA">Holothuroidea (Sea Cucumbers)</option>
      <option value="ANGUILLIDAE">Anguillidae (Eels)</option>
      <option value="SCOMBRIDAE">Scombridae (Tunas)</option>
      <option value="LABRIDAE">Labridae (Wrasses)</option>
      <option value="GADIFORMES">Gadiformes (Cods & Hakes)</option>
      <option value="PLEURONECTIFORMES">Pleuronectiformes (Flatfishes, Halibuts & Soles)</option>
    </select>
    <select id="catFilter">
      <option value="">All Red List Categories</option>
      <option value="CR">Critically Endangered (CR)</option>
      <option value="EN">Endangered (EN)</option>
      <option value="VU">Vulnerable (VU)</option>
      <option value="NT">Near Threatened (NT)</option>
    </select>
    <select id="visFilter">
      <option value="">All Customs Visibility Tiers</option>
      <option value="Species-Specific">Species-Specific (Monitored)</option>
      <option value="Genus/Family-Aggregated">Genus/Family-Aggregated</option>
      <option value="Omnibus-Submerged">Omnibus-Submerged (100% Invisible)</option>
    </select>
  </div>

  <div class="table-container">
    <table id="taxaTable">
      <thead>
        <tr>
          <th>SIS ID</th>
          <th>Scientific Name</th>
          <th>Class</th>
          <th>Taxonomic Group</th>
          <th>IUCN Status</th>
          <th>CITES Listing</th>
          <th>HS6 Code</th>
          <th>Customs Visibility</th>
        </tr>
      </thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>

  <script>
    const taxaData = {taxa_json};

    function renderTable(data) {{
      const tbody = document.getElementById('tableBody');
      tbody.innerHTML = '';
      data.forEach(row => {{
        const tr = document.createElement('tr');
        
        let catClass = 'badge-cr';
        if (row.Red_List_Category === 'EN') catClass = 'badge-en';
        else if (row.Red_List_Category === 'VU') catClass = 'badge-vu';
        else if (row.Red_List_Category === 'NT') catClass = 'badge-nt';

        let visClass = 'vis-sub';
        if (row.Customs_Reporting_Visibility === 'Species-Specific') visClass = 'vis-spec';
        else if (row.Customs_Reporting_Visibility === 'Genus/Family-Aggregated') visClass = 'vis-gen';

        tr.innerHTML = `
          <td><code>${{row.IUCN_SIS_ID}}</code></td>
          <td><strong><em>${{row.Scientific_Name}}</em></strong></td>
          <td>${{row.Class}}</td>
          <td>${{row.Taxonomic_Group}}</td>
          <td><span class="badge ${{catClass}}">${{row.Red_List_Category}}</span></td>
          <td>${{row.CITES_Status}}</td>
          <td><code>${{row.Primary_HS6_Code}}</code></td>
          <td><span class="badge-vis ${{visClass}}">${{row.Customs_Reporting_Visibility}}</span></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterData() {{
      const query = document.getElementById('searchInput').value.toLowerCase();
      const group = document.getElementById('groupFilter').value;
      const cat = document.getElementById('catFilter').value;
      const vis = document.getElementById('visFilter').value;

      const filtered = taxaData.filter(d => {{
        const matchesQuery = !query || 
          d.Scientific_Name.toLowerCase().includes(query) ||
          d.Taxonomic_Group.toLowerCase().includes(query) ||
          d.Primary_HS6_Code.includes(query) ||
          String(d.IUCN_SIS_ID).includes(query);
        
        const matchesGroup = !group || d.Taxonomic_Group === group;
        const matchesCat = !cat || d.Red_List_Category === cat;
        const matchesVis = !vis || d.Customs_Reporting_Visibility === vis;

        return matchesQuery && matchesGroup && matchesCat && matchesVis;
      }});

      renderTable(filtered);
    }}

    document.getElementById('searchInput').addEventListener('input', filterData);
    document.getElementById('groupFilter').addEventListener('change', filterData);
    document.getElementById('catFilter').addEventListener('change', filterData);
    document.getElementById('visFilter').addEventListener('change', filterData);

    // Initial render
    renderTable(taxaData);
  </script>
</body>
</html>
"""
    out_html = FIGURES_DIR / "interactive_threatened_taxa_customs_explorer.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   -> Saved: {out_html.name}")


# Keep backward-compatible aliases for function imports
plot_figure_s13 = plot_figure_s05
plot_figure_s14 = plot_figure_s06


def main():
    print("=" * 80)
    print("MODULE 20: GENERATING EXPANDED VISUALIZATION SUITE & INTERACTIVE EXPLORER")
    print("=" * 80)

    plot_figure_s05()
    plot_figure_s06()
    generate_interactive_html()

    print("\n" + "=" * 80)
    print("ALL VISUAL ASSETS GENERATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
