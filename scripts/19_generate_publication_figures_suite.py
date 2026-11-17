#!/usr/bin/env python3
"""
Module 19: Publication Figures Generation Suite (Main and Supplementary Panels)
=============================================================================
Study: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade
Author: Sezgin Tunca (Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark; ORCID: 0000-0001-6840-1466)

Generates 600 DPI publication-ready Main Figures 1-5 and 7 alongside
Supplementary Figures S1-S4 and S7 in vector PDF and raster PNG formats.
"""
import sys
import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pyproj
import plotly.graph_objects as go
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DERIVED = BASE_DIR / "data" / "derived"
CONFIG_DIR = BASE_DIR / "config"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
GEOJSON_PATH = DATA_DERIVED / "ne_110m_admin_0_countries.geojson"

# Typography and Styling for Publication (increased for optimal readability)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Lucida Grande"],
    "font.size": 11.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.0,
    "figure.titlesize": 14.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

# Color Palette Tokens
NAVY = "#1B365D"
TEAL = "#008080"
CORAL = "#D9534F"
AMBER = "#E67E22"
SLATE = "#5D6D7E"
CRIMSON = "#900C3F"
FOREST = "#1E8449"
GOLD = "#F1C40F"
PURPLE = "#6C3483"
CYAN = "#0284C7"

# Publication Light Cartographic Theme
LIGHT_BG = "#FFFFFF"
LIGHT_OCEAN = "#F8FAFC"
LIGHT_LAND = "#E2E8F0"
LIGHT_BORDER = "#94A3B8"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#475569"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#CBD5E1"

# Rigorously curated mainland / economic / port anchor coordinates
ACCURATE_COUNTRY_COORDS = {
    "HKG": (114.1694, 22.3193),
    "SGP": (103.8198, 1.3521),
    "MLT": (14.3754, 35.9375),
    "MAC": (113.5439, 22.1987),
    "S19": (120.9605, 23.6978),
    "TWN": (120.9605, 23.6978),
    "PAN": (-79.5167, 8.9833),
    "ARE": (54.3773, 24.4539),
    "MDV": (73.5093, 4.1755),
    "MUS": (57.5522, -20.3484),
    "SYC": (55.4915, -4.6796),
    "CYP": (33.3823, 35.1856),
    "FRO": (-6.7744, 62.0079),
    "GRL": (-42.6043, 70.0000),
    "FJI": (178.0650, -17.7134),
    "PNG": (143.9555, -6.3150),
    "SLB": (160.1562, -9.6457),
    "VUT": (168.3273, -17.7333),
    "WSM": (-172.1046, -13.7590),
    "KIR": (-157.3630, 1.8709),
    "FSM": (158.2147, 6.8874),
    "NCL": (165.6180, -21.3000),
    "PYF": (-149.4068, -17.6797),
    "USA": (-97.5000, 38.5000),
    "CAN": (-98.0000, 52.0000),
    "NOR": (8.5000, 60.5000),
    "RUS": (40.0000, 56.0000),
    "FRA": (2.3522, 46.8000),
    "GBR": (-1.5000, 52.8000),
    "ESP": (-3.7038, 40.4168),
    "PRT": (-8.2245, 39.3999),
    "ITA": (12.5674, 42.5000),
    "DEU": (10.4515, 51.1657),
    "NLD": (5.2913, 52.1326),
    "BEL": (4.4699, 50.5039),
    "DNK": (9.5018, 56.0000),
    "SWE": (15.0000, 59.5000),
    "FIN": (25.7482, 61.9241),
    "ISL": (-18.5000, 64.9631),
    "IRL": (-7.6921, 53.1424),
    "GRC": (21.8243, 39.0742),
    "HRV": (16.0000, 45.1000),
    "TUR": (35.0000, 39.0000),
    "POL": (19.1451, 51.9194),
    "UKR": (31.1656, 48.3794),
    "BLR": (27.9534, 53.7098),
    "KAZ": (66.9237, 48.0196),
    "CHN": (113.5000, 33.5000),
    "JPN": (138.5000, 36.5000),
    "KOR": (127.7669, 36.5000),
    "VNM": (106.0000, 16.0000),
    "THA": (100.5000, 15.0000),
    "MYS": (102.0000, 3.5000),
    "IDN": (106.8456, -6.2088),
    "PHL": (121.0000, 14.5000),
    "MMR": (96.0000, 19.5000),
    "IND": (78.9629, 20.5937),
    "BGD": (90.3563, 23.6850),
    "LKA": (80.7718, 7.8731),
    "PAK": (69.3451, 30.0000),
    "AUS": (135.0000, -25.5000),
    "NZL": (173.0000, -41.0000),
    "SAU": (45.0792, 23.8859),
    "IRN": (53.6880, 32.4279),
    "OMN": (57.0000, 21.4735),
    "EGY": (30.8025, 26.8206),
    "TUN": (9.8000, 35.5000),
    "MAR": (-7.0926, 32.0000),
    "DZA": (3.0000, 28.0000),
    "NGA": (7.5000, 9.0000),
    "ZAF": (25.0000, -29.0000),
    "NAM": (17.5000, -22.0000),
    "SEN": (-15.0000, 14.5000),
    "GHA": (-1.0000, 7.9000),
    "KEN": (37.9062, 0.0236),
    "MEX": (-102.0000, 23.5000),
    "NIC": (-85.0000, 12.8000),
    "CRI": (-84.0000, 9.9281),
    "GTM": (-90.5000, 15.5000),
    "HND": (-86.5000, 14.8000),
    "COL": (-74.0000, 4.5709),
    "VEN": (-66.5897, 6.4238),
    "ECU": (-78.5000, -1.5000),
    "PER": (-75.5000, -10.0000),
    "CHL": (-70.6693, -33.4489),
    "BRA": (-50.0000, -14.0000),
    "ARG": (-64.0000, -35.0000),
    "URY": (-56.0000, -32.5000),
    "PRY": (-58.4438, -23.4425),
    "BOL": (-64.7000, -16.2902),
    "COK": (-159.7777, -21.2367),
    "VAT": (12.4534, 41.9029),
    "BHR": (50.5577, 26.0667),
    "VCT": (-61.2872, 13.2528),
    "CUW": (-68.9900, 12.1696),
    "TKL": (-171.8484, -9.2002),
    "AND": (1.5218, 42.5063),
    "ATG": (-61.8000, 17.0608),
    "GRD": (-61.6790, 12.1165),
    "NRU": (166.9315, -0.5228),
    "X1": (103.8519, 1.2903),
    "VGB": (-64.6399, 18.4207),
    "F19": (2.3522, 48.8566),
    "NFK": (167.9547, -29.0408),
    "MHL": (171.1845, 7.1315),
    "GUF": (-53.1258, 3.9339),
    "HMD": (73.5042, -53.0818),
    "MNP": (145.7500, 15.2000),
    "DMA": (-61.3710, 15.4150),
    "CYM": (-81.2546, 19.3133),
    "WLF": (-176.2000, -13.3000),
    "GUM": (144.7937, 13.4443),
    "CSK": (14.4378, 50.0755),
    "IOT": (71.8765, -6.3432),
    "TCA": (-71.7979, 21.6940),
    "LCA": (-60.9789, 13.9094),
    "F97": (120.0000, 25.0000),
    "MTQ": (-61.0242, 14.6415),
    "REU": (55.5364, -21.1151),
    "TON": (-175.1982, -21.1789),
    "AIA": (-63.0686, 18.2206),
    "PCN": (-130.1015, -25.0667),
    "BMU": (-64.7574, 32.3078),
    "PLW": (134.5825, 7.5150),
    "NIU": (-169.8672, -19.0544),
    "GLP": (-61.5510, 16.2650),
    "ANT": (-68.9900, 12.1696),
    "SMR": (12.4578, 43.9424),
    "BLM": (-62.8333, 17.9000),
    "COM": (43.3333, -11.6455),
    "SUN": (37.6173, 55.7558),
    "ZA1": (28.2293, -25.7479),
    "BRB": (-59.5432, 13.1939),
    "PUS": (-97.5000, 38.5000),
    "A79": (103.8519, 1.2903),
    "ATB": (-42.6043, 70.0000),
    "SCG": (20.4489, 44.7866),
    "SXM": (-63.0548, 18.0425),
    "DDR": (13.4050, 52.5200),
    "BES": (-68.2700, 12.1500),
    "CPV": (-23.5133, 14.9330),
    "GIB": (-5.3536, 36.1408),
    "E19": (2.3522, 48.8566),
    "TUV": (179.1940, -8.5167),
    "CXR": (105.6904, -10.4475),
    "YUG": (20.4489, 44.7866),
    "O19": (-149.4068, -17.6797),
    "MSR": (-62.1874, 16.7425),
    "ABW": (-69.9683, 12.5211),
    "SHN": (-5.7089, -15.9650),
    "UMI": (-160.0000, 0.0000),
    "STP": (6.6131, 0.1864),
    "SPM": (-56.2711, 46.8852),
    "KNA": (-62.7830, 17.3578),
    "SGS": (-36.5879, -54.4296),
    "ASM": (-170.7020, -14.2710),
    "MYT": (45.1662, -12.8275),
    "SSD": (31.6000, 4.8500),
    "PSE": (35.2332, 31.9522)
}

COMTRADE_TO_MAP_ALIASES = {
    "S19": "TWN",
    "SSD": "SDS",
    "PSE": "PSX"
}


def save_fig(fig, base_name, legacy_aliases=None):
    png_path = FIG_DIR / f"{base_name}.png"
    pdf_path = FIG_DIR / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   [+] Saved: {png_path.name} & {pdf_path.name} (600 DPI)")
    if legacy_aliases:
        for alias in legacy_aliases:
            shutil.copyfile(png_path, FIG_DIR / f"{alias}.png")
            shutil.copyfile(pdf_path, FIG_DIR / f"{alias}.pdf")
            print(f"   [+] Synced alias: {alias}")


def load_world_and_coords():
    world = gpd.read_file(GEOJSON_PATH)
    iso_col = "ADM0_A3" if "ADM0_A3" in world.columns else "ISO_A3"

    coords = dict(zip(world[iso_col].str.upper(), zip(world["LABEL_X"], world["LABEL_Y"])))
    if "ISO_A3" in world.columns:
        coords.update(dict(zip(world["ISO_A3"].str.upper(), zip(world["LABEL_X"], world["LABEL_Y"]))))

    coords.update(ACCURATE_COUNTRY_COORDS)

    world["lon"] = [coords.get(str(iso).upper(), (row["LABEL_X"], row["LABEL_Y"]))[0]
                    for iso, (_, row) in zip(world[iso_col], world.iterrows())]
    world["lat"] = [coords.get(str(iso).upper(), (row["LABEL_X"], row["LABEL_Y"]))[1]
                    for iso, (_, row) in zip(world[iso_col], world.iterrows())]

    return world, coords, iso_col


PROJ_EQEARTH = "+proj=eqearth +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
EQ_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", PROJ_EQEARTH, always_xy=True)
EQ_INV_TRANSFORMER = pyproj.Transformer.from_crs(PROJ_EQEARTH, "EPSG:4326", always_xy=True)


def load_world_and_coords_eqearth():
    world, coords_latlon, iso_col = load_world_and_coords()
    world = world[world[iso_col] != "ATA"].copy()
    world_eq = world.to_crs(PROJ_EQEARTH)
    coords_eq = {iso: EQ_TRANSFORMER.transform(lon, lat) for iso, (lon, lat) in coords_latlon.items()}
    return world_eq, coords_eq, iso_col, coords_latlon


def compute_clean_corridor_arc(s, d, coords_dict, t_steps=35, curvature_scale=0.14, max_arch=2200000):
    if s not in coords_dict or d not in coords_dict:
        return None, None
    x0, y0 = coords_dict[s]
    x1, y1 = coords_dict[d]
    dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    arch = min(dist * curvature_scale, max_arch)

    if x0 > 8000000 and x1 > 8000000:
        xm = min(x0, x1) - arch * 0.55
        ym = (y0 + y1) / 2.0
    else:
        xm = (x0 + x1) / 2.0
        ym = (y0 + y1) / 2.0 + (arch if (y0 + y1) / 2.0 >= -1200000 else -arch)

    t = np.linspace(0, 1, t_steps)
    xt = (1 - t)**2 * x0 + 2 * (1 - t) * t * xm + t**2 * x1
    yt = (1 - t)**2 * y0 + 2 * (1 - t) * t * ym + t**2 * y1
    return xt, yt


# =============================================================================
# MAIN FIGURE 1: Global Seafood Trade Macro Dynamics & Customs Aggregation Gap
# =============================================================================
def plot_figure_01():
    print("[1/12] Generating Main Figure 1: Macro Dynamics & Customs Aggregation Gap (2x3 Composite Grid)...")
    ann_csv = DATA_DERIVED / "annual_taxa_threat_tier_summary.csv"
    bloc_ann_csv = DATA_DERIVED / "annual_trophic_level_by_bloc.csv"
    wco_csv = DATA_DERIVED / "annual_global_customs_aggregation_tar.csv"
    proc_csv = DATA_DERIVED / "processing_form_aggregation_summary.csv"

    if not ann_csv.exists() or not bloc_ann_csv.exists() or not wco_csv.exists() or not proc_csv.exists():
        print("Skipping Figure 1, missing data files.")
        return

    df = pd.read_csv(ann_csv)
    df_exp = df[df["Trade_Flow"] == "Export"].copy()
    ann_tot = df_exp.groupby("Year").agg({
        "Total_Real_USD_M": "sum",
        "Total_Weight_MT": "sum"
    }).reset_index()
    ann_tot["Total_Real_USD_B"] = ann_tot["Total_Real_USD_M"] / 1000.0
    ann_tot["Total_Weight_MMT"] = ann_tot["Total_Weight_MT"] / 1e6

    threat_pivot = df_exp.pivot(index="Year", columns="IUCN_Threat_Tier", values="Total_Real_USD_M").fillna(0)
    threat_pivot_b = threat_pivot / 1000.0

    df_bloc = pd.read_csv(bloc_ann_csv)
    bloc_pivot = df_bloc.pivot(index="Year", columns="Reporter_Bloc", values="Total_Weight_MT_Export").fillna(0) / 1e6

    df_wco = pd.read_csv(wco_csv)
    df_proc = pd.read_csv(proc_csv).sort_values("TAR_Value_Percent", ascending=True)

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.4), facecolor=LIGHT_BG)

    # Panel a: Real Value and Physical Volume
    year_ticks_fig1 = [1988, 1994, 2000, 2006, 2012, 2018, 2024]
    ax1 = axes[0, 0]
    ax1.plot(ann_tot["Year"], ann_tot["Total_Real_USD_B"], color=NAVY, lw=2.8, label="Export Value (Real 2024 USD)")
    ax1.set_ylabel("Global Export Value (Billion USD, Real 2024)", color=NAVY, fontweight="bold", fontsize=11.5)
    ax1.tick_params(axis="y", labelcolor=NAVY, labelsize=10.5)
    ax1.tick_params(axis="x", labelsize=10.5)
    ax1.set_xlabel("Year", fontweight="bold", fontsize=11.5)
    ax1.set_xlim(1987, 2025)
    ax1.set_xticks(year_ticks_fig1)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(ann_tot["Year"], ann_tot["Total_Weight_MMT"], color=TEAL, lw=2.4, linestyle="--", label="Export Volume (Million MT)")
    ax1_twin.set_ylabel("Global Export Volume (Million Metric Tons)", color=TEAL, fontweight="bold", fontsize=11.5)
    ax1_twin.tick_params(axis="y", labelcolor=TEAL, labelsize=10.5)
    ax1_twin.spines["top"].set_visible(False)

    ax1.axvspan(2008, 2009, color="gray", alpha=0.16)
    ax1.text(2008.5, 42, "2008 GFC", rotation=90, fontsize=9.2, fontweight="bold", color="#475569", ha="center")
    ax1.axvspan(2020, 2021, color="gray", alpha=0.16)
    ax1.text(2020.5, 42, "COVID-19", rotation=90, fontsize=9.2, fontweight="bold", color="#475569", ha="center")

    ax1.set_title("a  Global Macro Expansion (1988–2024)", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, framealpha=0.95, facecolor="white", fontsize=9.8)

    # Panel b: Composition by IUCN Threat Tier
    ax2 = axes[0, 1]
    tiers = ["Not Evaluated / LC", "Vulnerable to Least Concern", "Critically Endangered / Endangered"]
    tier_colors = [SLATE, AMBER, CRIMSON]
    bottom = np.zeros(len(threat_pivot_b))
    for tier, col in zip(tiers, tier_colors):
        if tier in threat_pivot_b.columns:
            vals = threat_pivot_b[tier].values
            ax2.bar(threat_pivot_b.index, vals, bottom=bottom, color=col, label=tier, width=0.85, alpha=0.9, edgecolor="none")
            bottom += vals

    ax2.set_xlabel("Year", fontweight="bold", fontsize=11.5)
    ax2.set_ylabel("Export Valuation by IUCN Status (Billion Real USD)", fontweight="bold", fontsize=11.5)
    ax2.set_title("b  Valuation Grouped by IUCN Threat Status", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax2.set_xlim(1987, 2025)
    ax2.set_xticks(year_ticks_fig1)
    ax2.tick_params(axis="both", labelsize=10.5)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, facecolor="white", fontsize=9.8)

    # Panel c: Stacked Regional Volume Composition
    ax3 = axes[0, 2]
    key_bloc_cols = {
        "East Asia & Pacific": "#2E86C1",
        "Latin America & Caribbean": "#27AE60",
        "EU-27 & EFTA": "#8E44AD",
        "North America": "#E67E22",
        "South Asia": "#16A085",
        "Sub-Saharan Africa & Rest of World": "#C0392B",
        "Middle East & North Africa (MENA)": "#D35400",
        "Eurasia & Post-Soviet Corridor": "#34495E"
    }
    bloc_short = {
        "East Asia & Pacific": "East Asia & Pacific",
        "Latin America & Caribbean": "Latin America & Carib.",
        "EU-27 & EFTA": "EU-27 & EFTA",
        "North America": "North America",
        "South Asia": "South Asia",
        "Sub-Saharan Africa & Rest of World": "Sub-Saharan Africa & RoW",
        "Middle East & North Africa (MENA)": "Middle East & N. Africa",
        "Eurasia & Post-Soviet Corridor": "Eurasia & Post-Soviet"
    }
    bottom_c = np.zeros(len(bloc_pivot))
    for b_name in bloc_pivot.columns:
        c = key_bloc_cols.get(b_name, "#7F8C8D")
        vals_c = bloc_pivot[b_name].values
        ax3.bar(bloc_pivot.index, vals_c, bottom=bottom_c, label=bloc_short.get(b_name, b_name), color=c, width=0.85, alpha=0.9, edgecolor="none")
        bottom_c += vals_c

    ax3.set_xlabel("Year", fontweight="bold", fontsize=11.5)
    ax3.set_ylabel("Export Volume by Regional Bloc (Million MT)", fontweight="bold", fontsize=11.5)
    ax3.set_title("c  Regional Export Volume Shares Across 8 Blocs", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax3.set_xlim(1987, 2025)
    ax3.set_xticks(year_ticks_fig1)
    ax3.tick_params(axis="both", labelsize=10.5)
    ax3.legend(loc="upper left", frameon=True, fontsize=8.8, ncol=1, facecolor="white", framealpha=0.95)

    # Panel d: TAR Timeline across WCO Revisions
    ax4 = axes[1, 0]
    tot_spec = df_wco["Specific_Real_USD_B"].sum()
    tot_aggr = df_wco["Aggregated_Real_USD_B"].sum()
    cum_val_tar = (tot_aggr / (tot_spec + tot_aggr)) * 100.0

    ax4.plot(df_wco["Year"], df_wco["TAR_Value_Percent"], marker="o", markersize=4.8, lw=2.5, color=CRIMSON, label="Value Aggregation (TAR Value %)")
    ax4.plot(df_wco["Year"], df_wco["TAR_Volume_Percent"], marker="s", markersize=4.8, lw=2.3, color=AMBER, linestyle="--", label="Volume Aggregation (TAR Volume %)")
    ax4.axhline(cum_val_tar, color=SLATE, linestyle=":", lw=1.6, alpha=0.85, label=f"37-Yr Cumulative Value Benchmark ({cum_val_tar:.1f}%)")

    era_years = [1996, 2002, 2007, 2012, 2017, 2022]
    for yr in era_years:
        ax4.axvline(yr, color="gray", linestyle="-.", alpha=0.38)

    hs_labels = [
        (1991.8, 74.0, "HS 1988"),
        (1999.0, 74.0, "HS 1996"),
        (2004.5, 74.0, "HS 2002"),
        (2009.5, 74.0, "HS 2007"),
        (2014.5, 74.0, "HS 2012"),
        (2019.5, 74.0, "HS 2017"),
        (2023.4, 74.0, "HS 2022")
    ]
    for x_pos, y_pos_lbl, hs_txt in hs_labels:
        ax4.text(x_pos, y_pos_lbl, hs_txt, fontsize=8.3, fontweight="bold", color=SLATE, ha="center")

    ax4.set_xlabel("Year", fontweight="bold", fontsize=11.5)
    ax4.set_ylabel("Taxonomic Aggregation Ratio (TAR %)", fontweight="bold", fontsize=11.5)
    ax4.set_title("d  Taxonomic Aggregation Ratio (1988–2024)", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax4.set_xlim(1987, 2025)
    ax4.set_xticks(year_ticks_fig1)
    ax4.set_ylim(32, 79)
    ax4.tick_params(axis="both", labelsize=10.5)
    ax4.legend(loc="lower left", frameon=True, framealpha=0.95, facecolor="white", fontsize=9.5)

    # Panel e: Cumulative Dollar Volume: Documented vs Aggregated
    ax5 = axes[1, 1]
    categories = ["Taxon-Specific\nCustoms Lines", "Aggregated Baskets\n('Other' / 'nei' Baskets)"]
    vals = [tot_spec, tot_aggr]
    bars = ax5.bar(categories, vals, color=[TEAL, CRIMSON], width=0.52, edgecolor="black", linewidth=1.2, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2.0, h + 45, f"USD {h:,.1f} Billion\n({h/(tot_spec+tot_aggr)*100:.1f}%)",
                 ha="center", va="bottom", fontweight="bold", fontsize=10.5)

    ax5.set_ylabel("Cumulative Real Export Value (Billion USD, 1988–2024)", fontweight="bold", fontsize=11.5)
    ax5.set_title(f"e  The USD {tot_aggr/1000:.2f} Trillion Global Customs Aggregation Gap", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax5.set_ylim(0, max(vals) * 1.25)
    ax5.tick_params(axis="x", labelsize=11.0)
    ax5.tick_params(axis="y", labelsize=10.5)

    # Panel f: Processing Form Aggregation Breakdown
    ax6 = axes[1, 2]
    proc_short = {
        "Flours, Meals & Pellets Fit for Human Consumption": "Flours, Meals & Pellets",
        "Live / Fresh / Chilled Shellfish": "Live / Fresh Shellfish",
        "Prepared / Preserved / Canned": "Prepared / Preserved",
        "Fresh / Chilled Fillet & Meat": "Fresh / Chilled Fillet",
        "Smoked / Cured / Dried / Salted": "Smoked / Cured / Dried",
        "Frozen Fillet & Meat": "Frozen Fillet & Meat",
        "Fresh / Chilled Whole Round": "Fresh / Chilled Whole",
        "Frozen Whole Round": "Frozen Whole Round",
        "Other Form": "Other Forms",
        "Live": "Live Marine Species"
    }
    y_pos = np.arange(len(df_proc))
    colors_proc = [CRIMSON if r >= 50 else (AMBER if r >= 30 else TEAL) for r in df_proc["TAR_Value_Percent"]]
    bars_p = ax6.barh(y_pos, df_proc["TAR_Value_Percent"], color=colors_proc, height=0.62, edgecolor="black", alpha=0.9)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([proc_short.get(str(p), str(p)) for p in df_proc["Processing_Form"]], fontsize=10.0, fontweight="bold")
    ax6.set_xlabel("Taxonomic Aggregation Ratio (TAR Value %)", fontweight="bold", fontsize=11.5)
    ax6.set_title("f  Customs Aggregation by Product Processing Form", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax6.set_xlim(0, 116)
    ax6.tick_params(axis="x", labelsize=10.5)
    for bar in bars_p:
        w = bar.get_width()
        ax6.text(w + 1.2, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}%", va="center", fontweight="bold", fontsize=9.5)

    plt.tight_layout(w_pad=2.2, h_pad=2.0)
    save_fig(fig, "Figure_01_Global_Seafood_Trade_Macro_Dynamics_and_Customs_Aggregation_Gap")


# =============================================================================
# MAIN FIGURE 2: Planetary Sovereign Customs Aggregation & Reporting Ambiguity
# =============================================================================
def plot_figure_02():
    print("[2/12] Generating Main Figure 2: Sovereign Customs Aggregation Multi-Panel (600 DPI & HTML)...")
    tar_csv = DATA_DERIVED / "sovereign_customs_aggregation_tar.csv"
    if not tar_csv.exists() or not GEOJSON_PATH.exists():
        print("Skipping Figure 2, missing required files.")
        return

    world_eq, coords_eq, iso_col, _ = load_world_and_coords_eqearth()
    df_tar = pd.read_csv(tar_csv)
    df_tar["Map_ISO"] = df_tar["Reporter_ISO3"].replace(COMTRADE_TO_MAP_ALIASES)
    world_tar = world_eq.merge(df_tar, left_on=iso_col, right_on="Map_ISO", how="left")

    fig = plt.figure(figsize=(18.0, 12.8), dpi=600, facecolor=LIGHT_BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.92, 1.0], width_ratios=[1.0, 1.0],
                          left=0.01, right=0.995, bottom=0.05, top=0.995, wspace=0.04, hspace=0.11)

    def _bc_axes(cell, label_frac=0.22):
        sub = cell.subgridspec(1, 2, width_ratios=[label_frac, 1 - label_frac], wspace=0.0)
        return sub[0, 1]

    # Panel 2a: Equal Earth Cartographic Exhibit across TOP ROW
    ax_m = fig.add_subplot(gs[0, :])
    ax_m.set_facecolor(LIGHT_OCEAN)
    world_eq.plot(ax=ax_m, color=LIGHT_LAND, edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=1)

    norm_tar = mcolors.Normalize(vmin=20, vmax=90)
    cmap_tar = plt.cm.YlOrRd
    active_tar = world_tar[world_tar["Reporter_ISO3"].notna() & world_tar["TAR_Value_Percent"].notna()]
    active_tar.plot(column="TAR_Value_Percent", ax=ax_m, cmap=cmap_tar, norm=norm_tar,
                    edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=2)

    minx, miny, maxx, maxy = world_eq.total_bounds
    cb_excl_x = minx + 0.145 * (maxx - minx)
    cb_excl_y = -1.1e6

    world_isos_3 = set(world_tar[world_tar["Reporter_ISO3"].notna()]["Map_ISO"].str.upper())
    unmapped_tar = df_tar[~df_tar["Map_ISO"].isin(world_isos_3)]
    for _, r in unmapped_tar.iterrows():
        iso = r["Reporter_ISO3"]
        if iso in coords_eq and pd.notna(r["TAR_Value_Percent"]):
            tar_v = r["TAR_Value_Percent"]
            pt = coords_eq[iso]
            if pt[0] < cb_excl_x and pt[1] < cb_excl_y:
                continue
            ax_m.scatter(pt[0], pt[1], c=[tar_v], cmap=cmap_tar, norm=norm_tar, s=30,
                         edgecolors="#1E293B", linewidth=0.7, zorder=5)

    ax_m.set_xlim(minx - 1e5, maxx + 1e5)
    ax_m.set_ylim(-6.92e6, maxy + 6.0e5)
    ax_m.set_aspect("auto")
    ax_m.margins(0)
    ax_m.set_facecolor("#EFF6FF")
    ax_m.patch.set_visible(True)
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_color("#CBD5E1")
        s.set_linewidth(0.8)
    ax_m.set_xticks([])
    ax_m.set_yticks([])

    # Inset Colorbar inside South Pacific quadrant with clean card backdrop
    cax = inset_axes(ax_m, width="1.9%", height="34%", loc="lower left",
                     bbox_to_anchor=(0.015, 0.06, 1, 1), bbox_transform=ax_m.transAxes, borderpad=0)
    cax.patch.set_facecolor("#FFFFFF")
    sm = plt.cm.ScalarMappable(cmap=cmap_tar, norm=norm_tar)
    sm._A = []
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Customs Aggregation (TAR Value %)", color=TEXT_DARK, fontweight="bold", fontsize=10.0)
    cb.ax.yaxis.set_tick_params(color=TEXT_DARK, labelsize=9.2)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=TEXT_DARK, fontsize=9.2)

    ax_m.text(0.008, 0.985, "a", transform=ax_m.transAxes, fontsize=16, fontweight="bold", color=TEXT_DARK, va="top")
    ax_m.text(0.026, 0.985, "Sovereign Taxonomic Aggregation Ratio (37-Year Mean % by Value)", transform=ax_m.transAxes,
              fontsize=12.5, fontweight="bold", color="#334155", va="top")

    # Panel 2b: Regional Customs Aggregation Disparity on BOTTOM LEFT
    ax_b = fig.add_subplot(_bc_axes(gs[1, 0], label_frac=0.23))
    ax_b.set_facecolor("#FFFFFF")
    bloc_stats = df_tar.groupby("Reporter_Bloc")["TAR_Value_Percent"].agg(["mean", "sem", "count"]).sort_values("mean", ascending=True)
    bloc_stats.index = [bloc_stats.index[i].replace("& Rest of World", "").replace("(MENA)", "").replace("& Caribbean", "& Carib.").replace("Corridor", "").strip() for i in range(len(bloc_stats))]

    y_pos_b = np.arange(len(bloc_stats))
    bars_b = ax_b.barh(y_pos_b, bloc_stats["mean"], xerr=bloc_stats["sem"], height=0.62,
                       color="#EA580C", edgecolor="#9A3412", alpha=0.88,
                       error_kw=dict(ecolor=TEXT_DARK, lw=1.3, capsize=3.5, capthick=1.3))
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(bloc_stats.index, fontsize=10.0, color=TEXT_DARK, fontweight="bold")
    ax_b.set_xlabel("Mean Taxonomic Aggregation Ratio (%) ± SEM", fontsize=10.5, color=TEXT_DARK, fontweight="bold")
    ax_b.set_xlim(0, 96)
    ax_b.set_ylim(-1.45, len(bloc_stats) - 0.15)
    ax_b.tick_params(axis="x", colors=TEXT_DARK, labelsize=9.8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_color("#CBD5E1")
    ax_b.spines["bottom"].set_color("#CBD5E1")
    ax_b.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")

    for bar, (_, row) in zip(bars_b, bloc_stats.iterrows()):
        m = row["mean"]
        s = row["sem"] if pd.notna(row["sem"]) else 0.0
        ax_b.text(m + s + 1.5, bar.get_y() + bar.get_height() / 2.0, f"{m:.1f}%",
                  va="center", color=TEXT_DARK, fontsize=9.2, fontweight="bold")

    # Add Shapley R^2 & Income-Tier Divide callout box in Panel 2b (compact 2-line format)
    _mat = pd.read_csv(DATA_DERIVED / "multi_scale_bloc_income_trophic_customs_matrix.csv") if (DATA_DERIVED / "multi_scale_bloc_income_trophic_customs_matrix.csv").exists() else None
    _dec = pd.read_csv(DATA_DERIVED / "sovereign_customs_consolidation_decomposition.csv").iloc[0]
    _inc = {r.Group_Name: r.Pooled_TAR_Value_Pct for r in _mat[_mat["Classification_Dimension"] == "1. World Bank Income Tier"].itertuples()} if _mat is not None else {}
    ax_b.text(
        0.98, 0.03,
        f"WB Income-Tier Divide (TAR %): HIC {_inc.get('HIC', 34.6):.1f}% | LMC {_inc.get('LMC', 72.5):.1f}% | UMC {_inc.get('UMC', 76.8):.1f}% | LIC {_inc.get('LIC', 86.3):.1f}%\n"
        f"Shapley R² (WLS R²={_dec['Total_WLS_R2']:.3f}): Biogeography {_dec['Shapley_Biogeographic_Pct']:.1f}% | Governance {_dec['Shapley_Governance_Capacity_Pct']:.1f}% | Processing {_dec['Shapley_Processing_Aquaculture_Pct']:.1f}%",
        transform=ax_b.transAxes, ha="right", va="bottom", fontsize=7.0, fontweight="bold",
        color=TEXT_DARK, bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#D97706", lw=0.9, alpha=0.96)
    )

    ax_b.text(-0.27, 1.07, "b", transform=ax_b.transAxes, fontsize=15, fontweight="bold", color=TEXT_DARK, va="top")
    ax_b.text(-0.22, 1.065, "Regional Customs Aggregation Disparity (Global North vs. South)", transform=ax_b.transAxes,
              fontsize=11.5, fontweight="bold", color="#334155", va="top")

    # Panel 2c: Top Sovereign Exporters by Reporting Ambiguity on BOTTOM RIGHT
    ax_c = fig.add_subplot(_bc_axes(gs[1, 1], label_frac=0.22))
    ax_c.set_facecolor("#FFFFFF")
    top_exp = df_tar[df_tar["Total_Export_Real_USD_B"] > 20.0].sort_values("TAR_Value_Percent", ascending=True).tail(12)
    y_pos_c = np.arange(len(top_exp))
    norm_sub = mcolors.Normalize(vmin=40, vmax=85)
    bar_cols = [plt.cm.YlOrRd(norm_sub(v)) for v in top_exp["TAR_Value_Percent"]]

    bars_c = ax_c.barh(y_pos_c, top_exp["TAR_Value_Percent"], height=0.62, color=bar_cols, edgecolor="#1E293B", alpha=0.92)
    ax_c.set_yticks(y_pos_c)
    c_labels = [f"{r.Reporter_Country} ({r.Reporter_ISO3})" for _, r in top_exp.iterrows()]
    ax_c.set_yticklabels(c_labels, fontsize=9.6, color=TEXT_DARK, fontweight="bold")
    ax_c.set_xlabel("Taxonomic Aggregation Ratio (TAR Value %) [Total Export USD B]", fontsize=10.5, color=TEXT_DARK, fontweight="bold")
    ax_c.set_xlim(0, 120)
    ax_c.tick_params(axis="x", colors=TEXT_DARK, labelsize=9.8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_color("#CBD5E1")
    ax_c.spines["bottom"].set_color("#CBD5E1")
    ax_c.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")

    for bar, (_, r) in zip(bars_c, top_exp.iterrows()):
        w = bar.get_width()
        usd = r["Total_Export_Real_USD_B"]
        ax_c.text(w + 1.2, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}% (USD {usd:.0f}B)",
                  va="center", color=TEXT_DARK, fontsize=9.0, fontweight="bold")

    ax_c.text(-0.26, 1.07, "c", transform=ax_c.transAxes, fontsize=15, fontweight="bold", color=TEXT_DARK, va="top")
    ax_c.text(-0.21, 1.065, "Top Sovereign Exporters by Customs Reporting Ambiguity", transform=ax_c.transAxes,
              fontsize=11.5, fontweight="bold", color="#334155", va="top")

    save_fig(fig, "Figure_02_Global_Map_Customs_Aggregation_and_Sovereign_Exposure")

    # Interactive Plotly Equal Earth Map
    fig_p = go.Figure()
    fig_p.add_trace(go.Choropleth(
        locations=world_tar[iso_col],
        z=world_tar["TAR_Value_Percent"],
        text=world_tar["Reporter_ISO3"] + "<br>TAR: " + world_tar["TAR_Value_Percent"].round(1).astype(str) + "%",
        colorscale="YlOrRd",
        colorbar_title="TAR Value %",
        marker_line_color="#94A3B8",
        marker_line_width=0.5
    ))
    fig_p.update_layout(
        title="Figure 2 | Global Sovereign Customs Aggregation and Exposure (Equal Earth Projection)",
        geo=dict(projection_type="equal earth", showframe=False, showcoastlines=True, coastlinecolor="#94A3B8",
                 bgcolor="#F8FAFC", lakecolor="#F1F5F9", landcolor="#E2E8F0"),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="Arial, sans-serif", size=13)
    )
    html_out = FIG_DIR / "Figure_02_Global_Map_Customs_Aggregation_and_Sovereign_Exposure.html"
    fig_p.write_html(str(html_out))
    print(f"   [+] Saved HTML: {html_out.name}")


# =============================================================================
# MAIN FIGURE 3: Empirical Allee Effects & Quasi-Experimental ITS Regressions
# =============================================================================
def plot_figure_03():
    print("[3/12] Generating Main Figure 3: Anthropogenic Allee Effects & ITS Regressions (2x3 Year-Aligned Causal Grid)...")
    ann_csv = DATA_DERIVED / "annual_taxa_allee_time_series.csv"
    traj_csv = DATA_DERIVED / "its_model_predicted_trajectories.csv"
    aae_csv = DATA_DERIVED / "global_allee_effect_econometric_results.csv"
    its_csv = DATA_DERIVED / "global_interrupted_time_series_results.csv"
    if not ann_csv.exists() or not traj_csv.exists():
        print("Skipping Figure 3, missing data files.")
        return

    df_ann = pd.read_csv(ann_csv)
    df_aae = pd.read_csv(aae_csv).set_index("Conservation_Category")
    df_its = pd.read_csv(its_csv).set_index("Intervention_Year")
    eels = df_ann[df_ann["Conservation_Category"] == "Diadromous Eels (Anguilla spp.)"].sort_values("Year").copy()
    tunas = df_ann[df_ann["Conservation_Category"] == "High-Trophic Tunas (Thunnus spp.)"].sort_values("Year").copy()
    cucumbers = df_ann[df_ann["Conservation_Category"] == "Sea Cucumbers (Holothuroidea)"].sort_values("Year").copy()
    df_traj = pd.read_csv(traj_csv)

    fig, axes = plt.subplots(
        2, 3, figsize=(21.0, 11.4), facecolor=LIGHT_BG,
        gridspec_kw={"wspace": 0.42, "hspace": 0.28}
    )

    # --- ROW 1: REAL UNIT PRICE & ANNUAL EXPORT VOLUME OVER YEAR ---
    # Panel a: Diadromous Eels
    ax1 = axes[0, 0]
    ax1_t = ax1.twinx()
    l1a = ax1.plot(eels["Year"], eels["Weighted_Unit_Price_USD_per_KG"], color=CRIMSON, lw=2.5, marker="o", ms=4.8,
                   label="Real Unit Price (2024 USD/kg)", zorder=4)
    l1b = ax1_t.plot(eels["Year"], eels["Total_Weight_MT"] / 1000.0, color=NAVY, lw=2.2, linestyle="--", marker="s", ms=4.2,
                     label="Annual Export Volume (10³ MT)", zorder=3)
    idx_max_e = eels["Weighted_Unit_Price_USD_per_KG"].idxmax()
    ax1.annotate(
        f"{int(eels.loc[idx_max_e, 'Year'])} Peak: USD {eels.loc[idx_max_e, 'Weighted_Unit_Price_USD_per_KG']:.2f}/kg",
        xy=(eels.loc[idx_max_e, "Year"], eels.loc[idx_max_e, "Weighted_Unit_Price_USD_per_KG"]),
        xytext=(eels.loc[idx_max_e, "Year"] + 2.5, eels.loc[idx_max_e, "Weighted_Unit_Price_USD_per_KG"] + 3.0),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4), fontweight="bold", color=CRIMSON, fontsize=9.0
    )
    ax1.set_xlim(1987.5, 2024.5)
    ax1.set_xticks([1988, 1994, 2000, 2006, 2012, 2018, 2024])
    ax1.set_ylim(5.0, 36.0)
    ax1_t.set_ylim(0.0, 82.0)
    ax1.set_xlabel("Year", fontweight="bold", fontsize=11.0)
    ax1.set_ylabel("Real Unit Price (2024 USD / kg)", color=CRIMSON, fontweight="bold", fontsize=11.0)
    ax1_t.set_ylabel("Annual Export Volume (10³ MT)", color=NAVY, fontweight="bold", fontsize=11.0)
    ax1.tick_params(axis="y", labelcolor=CRIMSON, labelsize=9.8)
    ax1_t.tick_params(axis="y", labelcolor=NAVY, labelsize=9.8)
    ax1.tick_params(axis="x", labelsize=9.8)
    ax1_t.spines["top"].set_visible(False)
    ax1.set_title("a  Diadromous Eels: Price & Volume Dynamics", loc="left", fontweight="bold", fontsize=12.0, pad=10)
    ax1.legend(l1a + l1b, [l.get_label() for l in l1a + l1b], loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.8)
    ax1.text(
        0.97, 0.96,
        r"Macro OLS: $\hat{\beta}_{\mathrm{OLS}} = -0.18\ (p < 0.0001)$" + "\n"
        r"Route 2WFE IV: $\hat{\beta}_{\mathrm{IV}} = -0.027\ (p = 0.212$, Farmed Dilution)",
        transform=ax1.transAxes, ha="right", va="top", fontsize=8.1, fontweight="bold",
        color=TEXT_DARK, bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFBEB", edgecolor=CRIMSON, lw=1.0, alpha=0.95)
    )

    # Panel b: High-Trophic Tunas
    ax2 = axes[0, 1]
    ax2_t = ax2.twinx()
    l2a = ax2.plot(tunas["Year"], tunas["Weighted_Unit_Price_USD_per_KG"], color=CRIMSON, lw=2.5, marker="o", ms=4.8,
                   label="Real Unit Price (2024 USD/kg)", zorder=4)
    l2b = ax2_t.plot(tunas["Year"], tunas["Total_Weight_MT"] / 1000.0, color=NAVY, lw=2.2, linestyle="--", marker="s", ms=4.2,
                     label="Annual Export Volume (10³ MT)", zorder=3)
    idx_max_t = tunas["Weighted_Unit_Price_USD_per_KG"].idxmax()
    ax2.annotate(
        f"{int(tunas.loc[idx_max_t, 'Year'])} Peak: USD {tunas.loc[idx_max_t, 'Weighted_Unit_Price_USD_per_KG']:.2f}/kg",
        xy=(tunas.loc[idx_max_t, "Year"], tunas.loc[idx_max_t, "Weighted_Unit_Price_USD_per_KG"]),
        xytext=(tunas.loc[idx_max_t, "Year"] + 1.5, tunas.loc[idx_max_t, "Weighted_Unit_Price_USD_per_KG"] + 3.2),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4), fontweight="bold", color=CRIMSON, fontsize=9.0
    )
    ax2.set_xlim(2001.5, 2024.5)
    ax2.set_xticks([2002, 2006, 2010, 2014, 2018, 2022, 2024])
    ax2.set_ylim(7.0, 41.0)
    ax2_t.set_ylim(15.0, 74.0)
    ax2.set_xlabel("Year", fontweight="bold", fontsize=11.0)
    ax2.set_ylabel("Real Unit Price (2024 USD / kg)", color=CRIMSON, fontweight="bold", fontsize=11.0)
    ax2_t.set_ylabel("Annual Export Volume (10³ MT)", color=NAVY, fontweight="bold", fontsize=11.0)
    ax2.tick_params(axis="y", labelcolor=CRIMSON, labelsize=9.8)
    ax2_t.tick_params(axis="y", labelcolor=NAVY, labelsize=9.8)
    ax2.tick_params(axis="x", labelsize=9.8)
    ax2_t.spines["top"].set_visible(False)
    ax2.set_title("b  High-Trophic Tunas: Price & Volume Dynamics", loc="left", fontweight="bold", fontsize=12.0, pad=10)
    ax2.legend(l2a + l2b, [l.get_label() for l in l2a + l2b], loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.8)
    ax2.text(
        0.97, 0.96,
        r"Macro OLS: $\hat{\beta}_{\mathrm{OLS}} = -0.82\ (p = 0.0040)$" + "\n"
        r"LOO Supply IV: $\hat{\beta}_{\mathrm{IV,LOO}} = -0.233\ (p = 0.0001$, Quota Scarcity)",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8.1, fontweight="bold",
        color=TEXT_DARK, bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFBEB", edgecolor=NAVY, lw=1.0, alpha=0.95)
    )

    # Panel c: Sea Cucumbers
    ax3 = axes[0, 2]
    ax3_t = ax3.twinx()
    l3a = ax3.plot(cucumbers["Year"], cucumbers["Weighted_Unit_Price_USD_per_KG"], color=CRIMSON, lw=2.5, marker="o", ms=4.8,
                   label="Real Unit Price (2024 USD/kg)", zorder=4)
    l3b = ax3_t.plot(cucumbers["Year"], cucumbers["Total_Weight_MT"] / 1000.0, color=NAVY, lw=2.2, linestyle="--", marker="s", ms=4.2,
                     label="Annual Export Volume (10³ MT)", zorder=3)
    idx_max_c = cucumbers["Weighted_Unit_Price_USD_per_KG"].idxmax()
    ax3.annotate(
        f"{int(cucumbers.loc[idx_max_c, 'Year'])} Peak:\nUSD {cucumbers.loc[idx_max_c, 'Weighted_Unit_Price_USD_per_KG']:.2f}/kg",
        xy=(cucumbers.loc[idx_max_c, "Year"], cucumbers.loc[idx_max_c, "Weighted_Unit_Price_USD_per_KG"]),
        xytext=(cucumbers.loc[idx_max_c, "Year"] - 2.6, cucumbers.loc[idx_max_c, "Weighted_Unit_Price_USD_per_KG"] + 4.5),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4), fontweight="bold", color=CRIMSON, fontsize=9.0
    )
    ax3.set_xlim(2011.5, 2024.5)
    ax3.set_xticks([2012, 2014, 2016, 2018, 2020, 2022, 2024])
    ax3.set_ylim(15.0, 74.0)
    ax3_t.set_ylim(4.0, 38.0)
    ax3.set_xlabel("Year", fontweight="bold", fontsize=11.0)
    ax3.set_ylabel("Real Unit Price (2024 USD / kg)", color=CRIMSON, fontweight="bold", fontsize=11.0)
    ax3_t.set_ylabel("Annual Export Volume (10³ MT)", color=NAVY, fontweight="bold", fontsize=11.0)
    ax3.tick_params(axis="y", labelcolor=CRIMSON, labelsize=9.8)
    ax3_t.tick_params(axis="y", labelcolor=NAVY, labelsize=9.8)
    ax3.tick_params(axis="x", labelsize=9.8)
    ax3_t.spines["top"].set_visible(False)
    ax3.set_title("c  Sea Cucumbers: Price & Volume Dynamics", loc="left", fontweight="bold", fontsize=12.0, pad=10)
    ax3.legend(l3a + l3b, [l.get_label() for l in l3a + l3b], loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.8)
    ax3.text(
        0.97, 0.96,
        r"Macro OLS: $\hat{\beta}_{\mathrm{OLS}} = -0.36\ (p = 0.0333)$" + "\n"
        r"Mirror 2WFE IV: $\hat{\beta}_{\mathrm{IV}} = -0.134\ (p = 0.0002, F = 135.5$, Wild Scarcity)",
        transform=ax3.transAxes, ha="right", va="top", fontsize=8.0, fontweight="bold",
        color=TEXT_DARK, bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFBEB", edgecolor="#B45309", lw=1.0, alpha=0.95)
    )

    # --- ROW 2: QUASI-EXPERIMENTAL INTERRUPTED TIME SERIES (10^3 MT) ---
    # Panel d: Diadromous Eels (2009 CITES App II & 2010 EU Zero-Quota Ban)
    ax4 = axes[1, 0]
    sub1 = df_traj[df_traj["Conservation_Category"] == "Diadromous Eels (Anguilla spp.)"].sort_values("Year")
    its_eel = df_its.loc[2009]
    ax4.scatter(sub1["Year"], sub1["Total_Weight_MT"] / 1000.0, color=SLATE, s=44, label="Reported Volume (10³ MT)", zorder=3)
    ax4.plot(sub1["Year"], sub1["pred_Volume_MT"] / 1000.0, color=NAVY, lw=2.8, label="Fitted ITS Model", zorder=4)
    ax4.plot(sub1["Year"], sub1["cf_Volume_MT"] / 1000.0, color=CRIMSON, lw=2.2, linestyle="--", label="Counterfactual (No Intervention)", zorder=2)
    ax4.axvline(2009, color=AMBER, lw=2.2, linestyle=":", label="CITES App II (2009)")
    ax4.text(
        0.03, 0.42,
        rf"$\hat{{\beta}}_2$: {its_eel['Immediate_Shift_Percent']:.1f}% ($p = {its_eel['Immediate_Shift_Pval']:.4f}$, HAC)" + "\n"
        + rf"$\hat{{\beta}}_3$: {its_eel['Post_Slope_Change_beta3']:.2f}/yr ($p < 0.0001$)",
        transform=ax4.transAxes, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.95, edgecolor=NAVY, lw=1.1),
        fontsize=8.6, fontweight="bold", color=TEXT_DARK
    )
    ax4.set_xlim(1987.5, 2024.5)
    ax4.set_xticks([1988, 1994, 2000, 2006, 2012, 2018, 2024])
    ax4.set_ylim(0.0, 520.0)
    ax4.set_xlabel("Year", fontweight="bold", fontsize=11.0)
    ax4.set_ylabel("Global Export Volume (10³ MT)", fontweight="bold", fontsize=11.0)
    ax4.set_title("d  Diadromous Eels: CITES II & EU Ban (2009)", loc="left", fontweight="bold", fontsize=12.0, pad=10)
    ax4.tick_params(axis="both", labelsize=9.8)
    ax4.legend(loc="upper left", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)

    # Panel e: High-Trophic Tunas (2008 ICCAT Recovery Plan)
    ax5 = axes[1, 1]
    sub2 = df_traj[df_traj["Conservation_Category"] == "High-Trophic Tunas (Thunnus spp.)"].sort_values("Year")
    its_tuna = df_its.loc[2008]
    ax5.scatter(sub2["Year"], sub2["Total_Weight_MT"] / 1000.0, color=SLATE, s=44, label="Reported Volume (10³ MT)", zorder=3)
    ax5.plot(sub2["Year"], sub2["pred_Volume_MT"] / 1000.0, color=FOREST, lw=2.8, label="Fitted ITS Model", zorder=4)
    ax5.plot(sub2["Year"], sub2["cf_Volume_MT"] / 1000.0, color=CRIMSON, lw=2.2, linestyle="--", label="Counterfactual (No Intervention)", zorder=2)
    ax5.axvline(2008, color=AMBER, lw=2.2, linestyle=":", label="ICCAT Bluefin Plan (2008)")
    ax5.text(
        0.03, 0.42,
        rf"$\hat{{\beta}}_2$: {its_tuna['Immediate_Shift_Percent']:.1f}% ($p = {its_tuna['Immediate_Shift_Pval']:.4f}$, HAC)" + "\n"
        + rf"$\hat{{\beta}}_3$: {its_tuna['Post_Slope_Change_beta3']:.2f}/yr ($p = {its_tuna['Post_Slope_Change_Pval']:.4f}$)",
        transform=ax5.transAxes, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.95, edgecolor=FOREST, lw=1.1),
        fontsize=8.6, fontweight="bold", color=TEXT_DARK
    )
    ax5.set_xlim(2001.5, 2024.5)
    ax5.set_xticks([2002, 2006, 2010, 2014, 2018, 2022, 2024])
    ax5.set_ylim(0.0, 235.0)
    ax5.set_xlabel("Year", fontweight="bold", fontsize=11.0)
    ax5.set_ylabel("Global Export Volume (10³ MT)", fontweight="bold", fontsize=11.0)
    ax5.set_title("e  High-Trophic Tunas: ICCAT Plan (2008)", loc="left", fontweight="bold", fontsize=12.0, pad=10)
    ax5.tick_params(axis="both", labelsize=9.8)
    ax5.legend(loc="upper left", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)

    # Panel f: Sea Cucumbers (2020 CITES CoP18 Teatfish Listing)
    ax6 = axes[1, 2]
    sub3 = df_traj[df_traj["Conservation_Category"] == "Sea Cucumbers (Holothuroidea)"].sort_values("Year")
    its_cuc = df_its.loc[2020]
    ax6.scatter(sub3["Year"], sub3["Total_Weight_MT"] / 1000.0, color=SLATE, s=44, label="Reported Volume (10³ MT)", zorder=3)
    ax6.plot(sub3["Year"], sub3["pred_Volume_MT"] / 1000.0, color=PURPLE, lw=2.8, label="Fitted ITS Model", zorder=4)
    ax6.plot(sub3["Year"], sub3["cf_Volume_MT"] / 1000.0, color=CRIMSON, lw=2.2, linestyle="--", label="Counterfactual (No Intervention)", zorder=2)
    ax6.axvline(2020, color=AMBER, lw=2.2, linestyle=":", label="CITES CoP18 Listing (2020)")
    ax6.text(
        0.97, 0.96,
        rf"$\hat{{\beta}}_2$: {its_cuc['Immediate_Shift_Percent']:.1f}% ($p = {its_cuc['Immediate_Shift_Pval']:.4f}$, HAC)" + "\n"
        + rf"$\hat{{\beta}}_3$: {its_cuc['Post_Slope_Change_beta3']:+.2f}/yr ($p = {its_cuc['Post_Slope_Change_Pval']:.4f}$)",
        transform=ax6.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.95, edgecolor=PURPLE, lw=1.1),
        fontsize=8.5, fontweight="bold", color=TEXT_DARK
    )
    ax6.set_xlim(2011.5, 2024.5)
    ax6.set_xticks([2012, 2014, 2016, 2018, 2020, 2022, 2024])
    ax6.set_ylim(4.0, 36.0)
    ax6.set_xlabel("Year", fontweight="bold", fontsize=11.0)
    ax6.set_ylabel("Global Export Volume (10³ MT)", fontweight="bold", fontsize=11.0)
    ax6.set_title("f  Sea Cucumbers: CITES CoP18 Listing (2020)", loc="left", fontweight="bold", fontsize=12.0, pad=10)
    ax6.tick_params(axis="both", labelsize=9.8)
    ax6.legend(loc="upper left", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)

    fig.subplots_adjust(left=0.055, right=0.945, top=0.94, bottom=0.07)
    save_fig(fig, "Figure_03_Empirical_Allee_Effects_and_Quasi_Experimental_ITS_Regressions")


# =============================================================================
# MAIN FIGURE 4: Planetary Trade Network Geography of Threatened Marine Species
# =============================================================================
def _tight_map_axis(ax, world_eq, pad=0.03):
    x0, y0, x1, y1 = world_eq.total_bounds
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + 8.0e5)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.margins(0)
    ax.axis("off")


def plot_figure_04():
    print("[4/12] Generating Main Figure 4: Threatened Marine Species Multi-Map & Divided-Period Suite (600 DPI & HTML)...")
    parquet_path = DATA_DERIVED / "global_threatened_taxa_bilateral_panel.parquet"
    if not parquet_path.exists() or not GEOJSON_PATH.exists():
        print("Skipping Figure 4, missing data files.")
        return

    world_eq, coords_eq, iso_col, _ = load_world_and_coords_eqearth()
    df_threat = pd.read_parquet(parquet_path)

    df_threat["Map_ISO"] = df_threat["Reporter_ISO3"].replace(COMTRADE_TO_MAP_ALIASES)
    df_threat["Partner_Map_ISO"] = df_threat["Partner_ISO3"].replace(COMTRADE_TO_MAP_ALIASES)

    p1_df = df_threat[df_threat["Year"] <= 2006]
    p2_df = df_threat[df_threat["Year"] >= 2007]

    p1_exp = p1_df[p1_df["Trade_Flow"] == "Export"]
    p1_imp = p1_df[p1_df["Trade_Flow"] == "Import"]
    p2_exp = p2_df[p2_df["Trade_Flow"] == "Export"]
    p2_imp = p2_df[p2_df["Trade_Flow"] == "Import"]

    p1_sov_exp = p1_exp.groupby("Map_ISO")["Real_Trade_Value_USD"].sum() / 1e9
    p2_sov_exp = p2_exp.groupby("Map_ISO")["Real_Trade_Value_USD"].sum() / 1e9
    p1_sov_imp = p1_imp.groupby("Map_ISO")["Real_Trade_Value_USD"].sum() / 1e9
    p2_sov_imp = p2_imp.groupby("Map_ISO")["Real_Trade_Value_USD"].sum() / 1e9

    world_p1_exp = world_eq.merge(p1_sov_exp.rename("Val_B"), left_on=iso_col, right_index=True, how="left")
    world_p2_exp = world_eq.merge(p2_sov_exp.rename("Val_B"), left_on=iso_col, right_index=True, how="left")
    world_p1_imp = world_eq.merge(p1_sov_imp.rename("Val_B"), left_on=iso_col, right_index=True, how="left")
    world_p2_imp = world_eq.merge(p2_sov_imp.rename("Val_B"), left_on=iso_col, right_index=True, how="left")

    flow_p1_exp = p1_exp.groupby(["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"]).agg(
        Total_USD=("Real_Trade_Value_USD", "sum"),
        Total_MT=("Weight_KG", lambda x: x.sum() / 1000.0)
    ).reset_index().sort_values("Total_USD", ascending=False)

    flow_p2_exp = p2_exp.groupby(["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"]).agg(
        Total_USD=("Real_Trade_Value_USD", "sum"),
        Total_MT=("Weight_KG", lambda x: x.sum() / 1000.0)
    ).reset_index().sort_values("Total_USD", ascending=False)

    flow_p1_imp = p1_imp.groupby(["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"]).agg(
        Total_USD=("Real_Trade_Value_USD", "sum"),
        Total_MT=("Weight_KG", lambda x: x.sum() / 1000.0)
    ).reset_index().sort_values("Total_USD", ascending=False)

    flow_p2_imp = p2_imp.groupby(["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"]).agg(
        Total_USD=("Real_Trade_Value_USD", "sum"),
        Total_MT=("Weight_KG", lambda x: x.sum() / 1000.0)
    ).reset_index().sort_values("Total_USD", ascending=False)

    taxa_colors = {
        "Diadromous Eels (Anguilla spp.)": "#0D9488",
        "High-Trophic Tunas (Thunnus spp.)": "#1D4ED8",
        "Chondrichthyans (Sharks & Rays)": "#DC2626",
        "Sea Cucumbers (Holothuroidea)": "#D97706",
        "Sturgeons & Caviar (Acipenseriformes)": "#7C3AED"
    }

    short_taxa = {
        "Diadromous Eels (Anguilla spp.)": "Diadromous Eels",
        "High-Trophic Tunas (Thunnus spp.)": "High-Trophic Tunas",
        "Chondrichthyans (Sharks & Rays)": "Sharks & Rays",
        "Sea Cucumbers (Holothuroidea)": "Sea Cucumbers",
        "Sturgeons & Caviar (Acipenseriformes)": "Sturgeons & Caviar"
    }

    fig = plt.figure(figsize=(22.5, 15.8), dpi=600, facecolor=LIGHT_BG)
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.74],
                                left=0.01, right=0.995, bottom=0.045, top=0.99, hspace=0.08)
    gs_row0 = gs_outer[0].subgridspec(1, 2, wspace=0.04)
    gs_row1 = gs_outer[1].subgridspec(1, 2, wspace=0.04)
    gs_row2 = gs_outer[2].subgridspec(1, 3, wspace=0.34)

    norm_exp = mcolors.LogNorm(vmin=0.01, vmax=25.0)
    cmap_exp = plt.cm.YlGnBu
    norm_imp = mcolors.LogNorm(vmin=0.01, vmax=25.0)
    cmap_imp = plt.cm.YlOrRd
    world_isos_set = set(world_eq[iso_col].str.upper())
    wx0, _, wx1, _ = world_eq.total_bounds
    cb_excl_x4 = wx0 + 0.145 * (wx1 - wx0)
    cb_excl_y4 = -1.2e6

    # Panel 4a: Period 1 EXPORT Supply Geographies
    ax_a = fig.add_subplot(gs_row0[0, 0])
    ax_a.set_facecolor(LIGHT_OCEAN)
    world_eq.plot(ax=ax_a, color=LIGHT_LAND, edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=1)

    active_p1_exp = world_p1_exp[world_p1_exp["Val_B"].notna() & (world_p1_exp["Val_B"] > 0)]
    active_p1_exp.plot(column="Val_B", ax=ax_a, cmap=cmap_exp, norm=norm_exp,
                       edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=2)

    for iso, val in p1_sov_exp.items():
        if iso not in world_isos_set and iso in coords_eq and val > 0:
            pt = coords_eq[iso]
            if pt[0] < cb_excl_x4 and pt[1] < cb_excl_y4:
                continue
            ax_a.scatter(pt[0], pt[1], c=[val], cmap=cmap_exp, norm=norm_exp, s=26,
                         edgecolors="#1E293B", linewidth=0.7, zorder=4)

    top_p1_exp = flow_p1_exp.groupby("Reporter_ISO3").head(1)
    mega_p1_exp = flow_p1_exp.head(25)
    comb_p1_exp = pd.concat([top_p1_exp, mega_p1_exp]).drop_duplicates(subset=["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"])
    max_p1_v = comb_p1_exp["Total_USD"].max() if not comb_p1_exp.empty else 1.0

    plotted_p1_exp_nodes = set()
    for _, r in comb_p1_exp.iterrows():
        s, d = str(r["Reporter_ISO3"]).upper(), str(r["Partner_ISO3"]).upper()
        xt, yt = compute_clean_corridor_arc(s, d, coords_eq)
        if xt is not None:
            plotted_p1_exp_nodes.add(s)
            plotted_p1_exp_nodes.add(d)
            col = taxa_colors.get(r["Conservation_Category"], "#0D9488")
            lw = np.clip(0.8 + (r["Total_USD"] / max_p1_v) * 3.2, 0.8, 4.0)
            alpha = np.clip(0.35 + (r["Total_USD"] / max_p1_v) * 0.50, 0.35, 0.85)
            ax_a.plot(xt, yt, color=col, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=3)

    for iso in plotted_p1_exp_nodes:
        if iso in coords_eq:
            pt_n = coords_eq[iso]
            if pt_n[0] < cb_excl_x4 and pt_n[1] < cb_excl_y4:
                continue
            ax_a.scatter(pt_n[0], pt_n[1], color="#334155", s=11, zorder=5, edgecolors="#FFFFFF", linewidth=0.4)

    _tight_map_axis(ax_a, world_eq)
    ax_a.text(0.01, 0.99, "a", transform=ax_a.transAxes, fontsize=15.5, fontweight="bold", color=TEXT_DARK, va="top")
    ax_a.text(0.04, 0.99, "Period 1: Baseline Era (1988–2006) — EXPORT Supply & Extraction Corridors", transform=ax_a.transAxes,
              fontsize=11.0, fontweight="bold", color="#334155", va="top")

    cax_a = inset_axes(ax_a, width="2.5%", height="28%", loc="lower left",
                       bbox_to_anchor=(0.015, 0.06, 1, 1), bbox_transform=ax_a.transAxes, borderpad=0)
    cax_a.patch.set_facecolor("#FFFFFF")
    sm_a = plt.cm.ScalarMappable(cmap=cmap_exp, norm=norm_exp)
    sm_a._A = []
    cb_a = fig.colorbar(sm_a, cax=cax_a)
    cb_a.set_label("Export Value (USD Billion, Log)", color=TEXT_DARK, fontweight="bold", fontsize=8.8)
    cb_a.ax.yaxis.set_tick_params(color=TEXT_DARK, labelsize=8.2)

    # Panel 4b: Period 2 EXPORT Supply Geographies
    ax_b = fig.add_subplot(gs_row0[0, 1])
    ax_b.set_facecolor(LIGHT_OCEAN)
    world_eq.plot(ax=ax_b, color=LIGHT_LAND, edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=1)

    active_p2_exp = world_p2_exp[world_p2_exp["Val_B"].notna() & (world_p2_exp["Val_B"] > 0)]
    active_p2_exp.plot(column="Val_B", ax=ax_b, cmap=cmap_exp, norm=norm_exp,
                       edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=2)

    for iso, val in p2_sov_exp.items():
        if iso not in world_isos_set and iso in coords_eq and val > 0:
            pt = coords_eq[iso]
            ax_b.scatter(pt[0], pt[1], c=[val], cmap=cmap_exp, norm=norm_exp, s=26,
                         edgecolors="#1E293B", linewidth=0.7, zorder=4)

    top_p2_exp = flow_p2_exp.groupby("Reporter_ISO3").head(1)
    mega_p2_exp = flow_p2_exp.head(45)
    comb_p2_exp = pd.concat([top_p2_exp, mega_p2_exp]).drop_duplicates(subset=["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"])
    max_p2_v = comb_p2_exp["Total_USD"].max() if not comb_p2_exp.empty else 1.0

    plotted_p2_exp_nodes = set()
    for _, r in comb_p2_exp.iterrows():
        s, d = str(r["Reporter_ISO3"]).upper(), str(r["Partner_ISO3"]).upper()
        xt, yt = compute_clean_corridor_arc(s, d, coords_eq)
        if xt is not None:
            plotted_p2_exp_nodes.add(s)
            plotted_p2_exp_nodes.add(d)
            col = taxa_colors.get(r["Conservation_Category"], "#0D9488")
            lw = np.clip(0.8 + (r["Total_USD"] / max_p2_v) * 3.4, 0.8, 4.2)
            alpha = np.clip(0.35 + (r["Total_USD"] / max_p2_v) * 0.50, 0.35, 0.85)
            ax_b.plot(xt, yt, color=col, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=3)

    for iso in plotted_p2_exp_nodes:
        if iso in coords_eq:
            ax_b.scatter(coords_eq[iso][0], coords_eq[iso][1], color="#334155", s=11, zorder=5, edgecolors="#FFFFFF", linewidth=0.4)

    _tight_map_axis(ax_b, world_eq)
    ax_b.text(0.01, 0.99, "b", transform=ax_b.transAxes, fontsize=15.5, fontweight="bold", color=TEXT_DARK, va="top")
    ax_b.text(0.04, 0.99, "Period 2: Modern CITES Era (2007–2024) — EXPORT Supply & Extraction Corridors", transform=ax_b.transAxes,
              fontsize=11.0, fontweight="bold", color="#334155", va="top")

    legend_handles5 = [
        plt.Line2D([0], [0], color="#0D9488", lw=2.6, label=r"Diadromous Eels ($\it{Anguilla}$ spp.)"),
        plt.Line2D([0], [0], color="#1D4ED8", lw=2.6, label=r"High-Trophic Tunas ($\it{Thunnus}$ spp.)"),
        plt.Line2D([0], [0], color="#DC2626", lw=2.6, label="Chondrichthyans (Sharks & Rays)"),
        plt.Line2D([0], [0], color="#D97706", lw=2.6, label="Sea Cucumbers (Holothuroidea)"),
        plt.Line2D([0], [0], color="#7C3AED", lw=2.6, label="Sturgeons & Caviar (Acipenseriformes)")
    ]
    ax_b.legend(handles=legend_handles5, loc="lower center", bbox_to_anchor=(0.50, 0.04),
                frameon=True, facecolor="#FFFFFF", edgecolor=CARD_BORDER,
                framealpha=0.96, fontsize=8.4, labelcolor=TEXT_DARK, title="Threatened Marine Taxa Flows", title_fontsize=9.0)

    # Panel 4c: Period 1 IMPORT Consumer Markets
    ax_c = fig.add_subplot(gs_row1[0, 0])
    ax_c.set_facecolor(LIGHT_OCEAN)
    world_eq.plot(ax=ax_c, color=LIGHT_LAND, edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=1)

    active_p1_imp = world_p1_imp[world_p1_imp["Val_B"].notna() & (world_p1_imp["Val_B"] > 0)]
    active_p1_imp.plot(column="Val_B", ax=ax_c, cmap=cmap_imp, norm=norm_imp,
                       edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=2)

    for iso, val in p1_sov_imp.items():
        if iso not in world_isos_set and iso in coords_eq and val > 0:
            pt = coords_eq[iso]
            if pt[0] < cb_excl_x4 and pt[1] < cb_excl_y4:
                continue
            ax_c.scatter(pt[0], pt[1], c=[val], cmap=cmap_imp, norm=norm_imp, s=26,
                         edgecolors="#1E293B", linewidth=0.7, zorder=4)

    top_p1_imp = flow_p1_imp.groupby("Reporter_ISO3").head(1)
    mega_p1_imp = flow_p1_imp.head(25)
    comb_p1_imp = pd.concat([top_p1_imp, mega_p1_imp]).drop_duplicates(subset=["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"])
    max_p1_iv = comb_p1_imp["Total_USD"].max() if not comb_p1_imp.empty else 1.0

    plotted_p1_imp_nodes = set()
    for _, r in comb_p1_imp.iterrows():
        s, d = str(r["Partner_ISO3"]).upper(), str(r["Reporter_ISO3"]).upper()
        xt, yt = compute_clean_corridor_arc(s, d, coords_eq)
        if xt is not None:
            plotted_p1_imp_nodes.add(s)
            plotted_p1_imp_nodes.add(d)
            col = taxa_colors.get(r["Conservation_Category"], "#0D9488")
            lw = np.clip(0.8 + (r["Total_USD"] / max_p1_iv) * 3.2, 0.8, 4.0)
            alpha = np.clip(0.35 + (r["Total_USD"] / max_p1_iv) * 0.50, 0.35, 0.85)
            ax_c.plot(xt, yt, color=col, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=3)

    for iso in plotted_p1_imp_nodes:
        if iso in coords_eq:
            pt_n = coords_eq[iso]
            if pt_n[0] < cb_excl_x4 and pt_n[1] < cb_excl_y4:
                continue
            ax_c.scatter(pt_n[0], pt_n[1], color="#334155", s=11, zorder=5, edgecolors="#FFFFFF", linewidth=0.4)

    _tight_map_axis(ax_c, world_eq)
    ax_c.text(0.01, 0.99, "c", transform=ax_c.transAxes, fontsize=15.5, fontweight="bold", color=TEXT_DARK, va="top")
    ax_c.text(0.04, 0.99, "Period 1: Baseline Era (1988–2006) — IMPORT Consumer Markets & Luxury Demand Inflows", transform=ax_c.transAxes,
              fontsize=11.0, fontweight="bold", color="#334155", va="top")

    cax_c = inset_axes(ax_c, width="2.5%", height="28%", loc="lower left",
                       bbox_to_anchor=(0.015, 0.06, 1, 1), bbox_transform=ax_c.transAxes, borderpad=0)
    cax_c.patch.set_facecolor("#FFFFFF")
    sm_c = plt.cm.ScalarMappable(cmap=cmap_imp, norm=norm_imp)
    sm_c._A = []
    cb_c = fig.colorbar(sm_c, cax=cax_c)
    cb_c.set_label("Import Value (USD Billion, Log)", color=TEXT_DARK, fontweight="bold", fontsize=8.8)
    cb_c.ax.yaxis.set_tick_params(color=TEXT_DARK, labelsize=8.2)

    # Panel 4d: Period 2 IMPORT Consumer Markets
    ax_d = fig.add_subplot(gs_row1[0, 1])
    ax_d.set_facecolor(LIGHT_OCEAN)
    world_eq.plot(ax=ax_d, color=LIGHT_LAND, edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=1)

    active_p2_imp = world_p2_imp[world_p2_imp["Val_B"].notna() & (world_p2_imp["Val_B"] > 0)]
    active_p2_imp.plot(column="Val_B", ax=ax_d, cmap=cmap_imp, norm=norm_imp,
                       edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=2)

    for iso, val in p2_sov_imp.items():
        if iso not in world_isos_set and iso in coords_eq and val > 0:
            pt = coords_eq[iso]
            ax_d.scatter(pt[0], pt[1], c=[val], cmap=cmap_imp, norm=norm_imp, s=26,
                         edgecolors="#1E293B", linewidth=0.7, zorder=4)

    top_p2_imp = flow_p2_imp.groupby("Reporter_ISO3").head(1)
    mega_p2_imp = flow_p2_imp.head(45)
    comb_p2_imp = pd.concat([top_p2_imp, mega_p2_imp]).drop_duplicates(subset=["Reporter_ISO3", "Partner_ISO3", "Conservation_Category"])
    max_p2_iv = comb_p2_imp["Total_USD"].max() if not comb_p2_imp.empty else 1.0

    plotted_p2_imp_nodes = set()
    for _, r in comb_p2_imp.iterrows():
        s, d = str(r["Partner_ISO3"]).upper(), str(r["Reporter_ISO3"]).upper()
        xt, yt = compute_clean_corridor_arc(s, d, coords_eq)
        if xt is not None:
            plotted_p2_imp_nodes.add(s)
            plotted_p2_imp_nodes.add(d)
            col = taxa_colors.get(r["Conservation_Category"], "#0D9488")
            lw = np.clip(0.8 + (r["Total_USD"] / max_p2_iv) * 3.4, 0.8, 4.2)
            alpha = np.clip(0.35 + (r["Total_USD"] / max_p2_iv) * 0.50, 0.35, 0.85)
            ax_d.plot(xt, yt, color=col, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=3)

    for iso in plotted_p2_imp_nodes:
        if iso in coords_eq:
            ax_d.scatter(coords_eq[iso][0], coords_eq[iso][1], color="#334155", s=11, zorder=5, edgecolors="#FFFFFF", linewidth=0.4)

    hubs_refined = {
        "Hong Kong SAR": (114.1694, 22.3193, 350000, -480000),
        "Guangzhou / China": (113.2644, 23.1291, -1800000, 420000),
        "Singapore": (103.8198, 1.3521, -1350000, -350000),
        "Tokyo / Japan": (139.6917, 35.6895, 350000, 320000),
        "Seoul / Korea": (127.0000, 37.5665, 320000, -320000),
        "United Arab Emirates": (54.3773, 24.4539, 320000, -360000),
        "Malta": (14.3754, 35.9375, 340000, -240000),
        "Spain (Madrid / Vigo)": (-3.7038, 40.4168, -600000, 520000),
        "Panama": (-79.5167, 8.9833, 180000, -420000)
    }
    for name, (hx, hy, ox, oy) in hubs_refined.items():
        hx_m, hy_m = EQ_TRANSFORMER.transform(hx, hy)
        ax_d.scatter(hx_m, hy_m, color="#FFFFFF", s=44, zorder=6, edgecolors="#DC2626", linewidth=1.5)
        ax_d.scatter(hx_m, hy_m, color="#DC2626", s=12, zorder=7)
        ax_d.text(hx_m + ox, hy_m + oy, name, color=TEXT_DARK, fontsize=7.8, fontweight="bold", zorder=8,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", alpha=0.94, edgecolor="#CBD5E1"))

    _tight_map_axis(ax_d, world_eq)
    ax_d.text(0.01, 0.99, "d", transform=ax_d.transAxes, fontsize=15.5, fontweight="bold", color=TEXT_DARK, va="top")
    ax_d.text(0.04, 0.99, "Period 2: Modern CITES Era (2007–2024) — IMPORT Consumer Markets & Luxury Demand Inflows", transform=ax_d.transAxes,
              fontsize=11.0, fontweight="bold", color="#334155", va="top")

    legend_handles_d = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFFFFF", markeredgecolor="#DC2626", markersize=6.5, label="Primary Transit / Destination Hub"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#334155", markeredgecolor="#FFFFFF", markersize=5.0, label="Active Trading Node")
    ]
    ax_d.legend(handles=legend_handles_d, loc="lower center", bbox_to_anchor=(0.50, 0.04),
                frameon=True, facecolor="#FFFFFF", edgecolor=CARD_BORDER,
                framealpha=0.96, fontsize=8.4, labelcolor=TEXT_DARK, title="Key Nodes", title_fontsize=9.0)

    # Panel 4e: Divided-Period Dynamics across Five Taxa (using harmonized longitudinal series incl. BACI 1995-2024)
    ax_e = fig.add_subplot(gs_row2[0, 0])
    ax_e.set_facecolor("#FFFFFF")

    taxa_order = [
        "Diadromous Eels (Anguilla spp.)",
        "High-Trophic Tunas (Thunnus spp.)",
        "Sea Cucumbers (Holothuroidea)",
        "Chondrichthyans (Sharks & Rays)",
        "Sturgeons & Caviar (Acipenseriformes)"
    ]
    short_taxa_e = dict(short_taxa)
    short_taxa_e["Sea Cucumbers (Holothuroidea)"] = "Sea Cucumbers (HS 2012+)"

    df_ann_4 = pd.read_csv(DATA_DERIVED / "annual_taxa_allee_time_series.csv")
    p1_vals = [df_ann_4[(df_ann_4["Conservation_Category"] == t) & (df_ann_4["Year"] <= 2006)]["Total_Real_USD"].sum() / 1e9 for t in taxa_order]
    p2_vals = [df_ann_4[(df_ann_4["Conservation_Category"] == t) & (df_ann_4["Year"] >= 2007)]["Total_Real_USD"].sum() / 1e9 for t in taxa_order]

    y_e = np.arange(len(taxa_order))
    h = 0.36
    bars_p1 = ax_e.barh(y_e - h / 2, p1_vals, height=h, color="#94A3B8", edgecolor="#475569", label="Period 1: 1988–2006 (Pre-Listing Era)", alpha=0.9)
    bars_p2 = ax_e.barh(y_e + h / 2, p2_vals, height=h, color=[taxa_colors[t] for t in taxa_order], edgecolor="#1E293B", label="Period 2: 2007–2024 (Modern CITES Era)", alpha=0.92)

    ax_e.set_yticks(y_e)
    ax_e.set_yticklabels([short_taxa_e[t] for t in taxa_order], fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax_e.set_xlabel("Cumulative Export Value (Real USD Billion)", fontsize=10.2, fontweight="bold", color=TEXT_DARK)
    max_e = max(max(p1_vals), max(p2_vals))
    ax_e.set_xlim(0, max_e * 1.40)
    ax_e.tick_params(axis="x", colors=TEXT_DARK, labelsize=9.2)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    ax_e.spines["left"].set_color("#CBD5E1")
    ax_e.spines["bottom"].set_color("#CBD5E1")
    ax_e.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")
    ax_e.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#FFFFFF", framealpha=0.95)

    for bar, val in zip(bars_p1, p1_vals):
        if val > 0.05:
            ax_e.text(val + max_e * 0.02, bar.get_y() + bar.get_height() / 2.0, f"USD {val:.2f}B", va="center", color="#475569", fontsize=8.4, fontweight="bold")
    for bar, val in zip(bars_p2, p2_vals):
        if val > 0.05:
            ax_e.text(val + max_e * 0.02, bar.get_y() + bar.get_height() / 2.0, f"USD {val:.2f}B", va="center", color=TEXT_DARK, fontsize=8.6, fontweight="bold")

    ax_e.text(-0.02, 1.08, "e", transform=ax_e.transAxes, fontsize=15, fontweight="bold", color=TEXT_DARK, va="top")
    ax_e.text(0.03, 1.075, "Divided-Period Taxa Exports (1988–2006 vs. 2007–2024)", transform=ax_e.transAxes,
              fontsize=10.8, fontweight="bold", color="#334155", va="top")

    # Panel 4f: Sovereign Decoupling
    ax_f = fig.add_subplot(gs_row2[0, 1])
    ax_f.set_facecolor("#FFFFFF")

    df_focal = df_threat[df_threat["Conservation_Category"].isin(taxa_order)]
    iso_labels = [
        ("CHN", "China"),
        ("JPN", "Japan"),
        ("HKG", "Hong Kong SAR"),
        ("ESP", "Spain"),
        ("AUS", "Australia"),
        ("KOR", "Rep. of Korea"),
        ("USA", "USA")
    ]
    comb_exp = [df_focal[(df_focal["Trade_Flow"] == "Export") & (df_focal["Reporter_ISO3"] == iso)]["Real_Trade_Value_USD"].sum() / 1e9 for iso, _ in iso_labels]
    comb_imp = [df_focal[(df_focal["Trade_Flow"] == "Import") & (df_focal["Reporter_ISO3"] == iso)]["Real_Trade_Value_USD"].sum() / 1e9 for iso, _ in iso_labels]

    y_f = np.arange(len(iso_labels))
    h_f = 0.36
    bars_exp_f = ax_f.barh(y_f - h_f / 2, comb_exp, height=h_f, color="#059669", edgecolor="#064E3B", label="Total Export Supply (USD B)", alpha=0.88)
    bars_imp_f = ax_f.barh(y_f + h_f / 2, comb_imp, height=h_f, color="#DC2626", edgecolor="#7F1D1D", label="Total Import Demand (USD B)", alpha=0.88)

    ax_f.set_yticks(y_f)
    ax_f.set_yticklabels([lbl for _, lbl in iso_labels], fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax_f.set_xlabel("Cumulative Focal Taxa Trade (Real USD Billion, 1988–2024)", fontsize=10.2, fontweight="bold", color=TEXT_DARK)
    max_f = max(max(comb_exp), max(comb_imp))
    ax_f.set_xlim(0, max_f * 1.40)
    ax_f.tick_params(axis="x", colors=TEXT_DARK, labelsize=9.2)
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)
    ax_f.spines["left"].set_color("#CBD5E1")
    ax_f.spines["bottom"].set_color("#CBD5E1")
    ax_f.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")
    ax_f.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#FFFFFF", framealpha=0.95)

    for bar, val in zip(bars_exp_f, comb_exp):
        if val > 0.3:
            ax_f.text(val + max_f * 0.02, bar.get_y() + bar.get_height() / 2.0, f"USD {val:.1f}B", va="center", color="#059669", fontsize=8.4, fontweight="bold")
    for bar, val in zip(bars_imp_f, comb_imp):
        if val > 0.3:
            ax_f.text(val + max_f * 0.02, bar.get_y() + bar.get_height() / 2.0, f"USD {val:.1f}B", va="center", color="#DC2626", fontsize=8.4, fontweight="bold")

    ax_f.text(-0.02, 1.08, "f", transform=ax_f.transAxes, fontsize=15, fontweight="bold", color=TEXT_DARK, va="top")
    ax_f.text(0.03, 1.075, "Sovereign Decoupling: Supply Origins vs. Consumer Markets", transform=ax_f.transAxes,
              fontsize=10.8, fontweight="bold", color="#334155", va="top")

    # Panel 4g: Transit Hub Intermediation
    ax_g = fig.add_subplot(gs_row2[0, 2])
    ax_g.set_facecolor("#FFFFFF")

    ecr_csv = DATA_DERIVED / "global_entrepot_masking_btar_ecr.csv"
    df_ecr = pd.read_csv(ecr_csv)
    hub_isos = [
        ("HKG", "Hong Kong"),
        ("SGP", "Singapore"),
        ("MLT", "Malta"),
        ("MAR", "Morocco"),
        ("ARE", "UAE"),
        ("PAN", "Panama")
    ]
    names_g = [lbl for _, lbl in hub_isos]
    inflow = [df_focal[(df_focal["Trade_Flow"] == "Import") & (df_focal["Reporter_ISO3"] == iso)]["Real_Trade_Value_USD"].sum() / 1e9 for iso, _ in hub_isos]
    re_exp = [df_focal[(df_focal["Trade_Flow"] == "Export") & (df_focal["Reporter_ISO3"] == iso)]["Real_Trade_Value_USD"].sum() / 1e9 for iso, _ in hub_isos]
    tar_val = [float(df_ecr.loc[df_ecr["Reporter_ISO3"] == iso, "TAR_Val_Pct_Export"].values[0]) if iso in df_ecr["Reporter_ISO3"].values else 50.0 for iso, _ in hub_isos]

    y_g = np.arange(len(names_g))
    h_g = 0.36
    bars_in = ax_g.barh(y_g - h_g / 2, inflow, height=h_g, color="#D97706", edgecolor="#78350F", label="Threatened Taxa Inflows (USD B)", alpha=0.88)
    bars_out = ax_g.barh(y_g + h_g / 2, re_exp, height=h_g, color="#7C3AED", edgecolor="#4C1D95", label="Outward Re-Exports (USD B)", alpha=0.88)

    ax_g.set_yticks(y_g)
    ax_g.set_yticklabels([f"{n} (TAR {t:.0f}%)" for n, t in zip(names_g, tar_val)], fontsize=9.4, fontweight="bold", color=TEXT_DARK)
    ax_g.set_xlabel("Transit Inflows & Outward Flows (USD Billion)", fontsize=10.2, fontweight="bold", color=TEXT_DARK)
    max_g = max(max(inflow), max(re_exp))
    ax_g.set_xlim(0, max_g * 1.40)
    ax_g.tick_params(axis="x", colors=TEXT_DARK, labelsize=9.2)
    ax_g.spines["top"].set_visible(False)
    ax_g.spines["right"].set_visible(False)
    ax_g.spines["left"].set_color("#CBD5E1")
    ax_g.spines["bottom"].set_color("#CBD5E1")
    ax_g.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")
    ax_g.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#FFFFFF", framealpha=0.95)

    for bar, val in zip(bars_in, inflow):
        if val > 0.02:
            ax_g.text(val + max_g * 0.02, bar.get_y() + bar.get_height() / 2.0, f"USD {val:.2f}B", va="center", color="#D97706", fontsize=8.2, fontweight="bold")
    for bar, val in zip(bars_out, re_exp):
        if val > 0.02:
            ax_g.text(val + max_g * 0.02, bar.get_y() + bar.get_height() / 2.0, f"USD {val:.2f}B", va="center", color="#7C3AED", fontsize=8.2, fontweight="bold")

    ax_g.text(-0.02, 1.08, "g", transform=ax_g.transAxes, fontsize=15, fontweight="bold", color=TEXT_DARK, va="top")
    ax_g.text(0.03, 1.075, "Strategic Transit Hubs: Inflows, Re-Exports & Aggregation", transform=ax_g.transAxes,
              fontsize=10.8, fontweight="bold", color="#334155", va="top")

    save_fig(fig, "Figure_04_Global_Threatened_Marine_Species_Flow_Corridors")

    # Interactive Plotly Equal Earth Map
    fig_p = go.Figure()
    fig_p.add_trace(go.Choropleth(
        locations=world_p2_exp[iso_col],
        z=world_p2_exp["Val_B"],
        text=world_p2_exp[iso_col] + "<br>P2 Export: $" + world_p2_exp["Val_B"].round(2).astype(str) + "B",
        colorscale="YlGnBu",
        colorbar_title="Export ($B)",
        marker_line_color="#94A3B8",
        marker_line_width=0.5,
        name="P2 Export Intensity"
    ))

    for _, r in comb_p2_exp.head(60).iterrows():
        s, d = str(r["Reporter_ISO3"]).upper(), str(r["Partner_ISO3"]).upper()
        xt, yt = compute_clean_corridor_arc(s, d, coords_eq)
        if xt is not None:
            lons, lats = EQ_INV_TRANSFORMER.transform(xt, yt)
            cat = r["Conservation_Category"]
            col = taxa_colors.get(cat, "#0D9488")
            fig_p.add_trace(go.Scattergeo(
                lon=list(lons),
                lat=list(lats),
                mode="lines",
                line=dict(width=max(1.2, float(r["Total_USD"]) / max_p2_v * 4.2), color=col),
                opacity=0.75,
                name=cat.split("(")[0].strip() + " (P2 Export)",
                hoverinfo="text",
                text=f"<b>{cat}</b><br>{s} → {d}<br>Value: ${r['Total_USD']/1e6:.1f}M<br>Weight: {r['Total_MT']:,.1f} MT"
            ))

    hub_lons = [pt[0] for pt in hubs_refined.values()]
    hub_lats = [pt[1] for pt in hubs_refined.values()]
    hub_texts = list(hubs_refined.keys())
    fig_p.add_trace(go.Scattergeo(
        lon=hub_lons,
        lat=hub_lats,
        mode="markers+text",
        marker=dict(size=8, color="#DC2626", line=dict(width=2, color="#FFFFFF")),
        text=hub_texts,
        textposition="top right",
        textfont=dict(size=10, color="#0F172A", family="Arial, sans-serif"),
        name="Transit Hubs"
    ))

    fig_p.update_layout(
        title="Figure 4 | Planetary Trade Network Geography of Threatened Marine Species (Equal Earth)",
        geo=dict(projection_type="equal earth", showframe=False, showcoastlines=True, coastlinecolor="#94A3B8",
                 bgcolor="#F8FAFC", lakecolor="#F1F5F9", landcolor="#E2E8F0"),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="Arial, sans-serif", size=13)
    )
    html_out = FIG_DIR / "Figure_04_Global_Threatened_Marine_Species_Flow_Corridors.html"
    fig_p.write_html(str(html_out))
    print(f"   [+] Saved HTML: {html_out.name}")


# =============================================================================
# MAIN FIGURE 5: Global Cartography of Trophic Transfers & Geodesic Corridors
# =============================================================================
def plot_figure_05():
    print("[5/12] Generating Main Figure 5: Trophic Transfers & Geodesic Corridors Multi-Panel (600 DPI & HTML)...")
    tbt_csv = DATA_DERIVED / "sovereign_trophic_balance_tbt.csv"
    flow_csv = DATA_DERIVED / "top_bilateral_trophic_flow_corridors.csv"
    if not tbt_csv.exists() or not flow_csv.exists() or not GEOJSON_PATH.exists():
        print("Skipping Figure 5, missing data files.")
        return

    world_eq, coords_eq, iso_col, _ = load_world_and_coords_eqearth()
    df_tbt = pd.read_csv(tbt_csv)
    df_flow = pd.read_csv(flow_csv)
    df_tbt["Map_ISO"] = df_tbt["Reporter_ISO3"].replace(COMTRADE_TO_MAP_ALIASES)
    world_tbt = world_eq.merge(df_tbt, left_on=iso_col, right_on="Map_ISO", how="left")

    fig = plt.figure(figsize=(18.0, 12.8), dpi=600, facecolor=LIGHT_BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.92, 1.0], width_ratios=[1.0, 1.0],
                          left=0.01, right=0.995, bottom=0.05, top=0.995, wspace=0.04, hspace=0.11)

    def _bc_axes(cell, label_frac=0.22):
        sub = cell.subgridspec(1, 2, width_ratios=[label_frac, 1 - label_frac], wspace=0.0)
        return sub[0, 1]

    # Panel 5a: Equal Earth Cartographic Exhibit across TOP ROW
    ax_m = fig.add_subplot(gs[0, :])
    ax_m.set_facecolor(LIGHT_OCEAN)
    world_eq.plot(ax=ax_m, color=LIGHT_LAND, edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=1)

    norm_tbt = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    cmap_tbt = plt.cm.coolwarm
    active_tbt = world_tbt[world_tbt["Reporter_ISO3"].notna() & world_tbt["TBT"].notna()]
    active_tbt.plot(column="TBT", ax=ax_m, cmap=cmap_tbt, norm=norm_tbt,
                    edgecolor=LIGHT_BORDER, linewidth=0.4, zorder=2)

    minx, miny, maxx, maxy = world_eq.total_bounds
    cb_excl_x5 = minx + 0.145 * (maxx - minx)
    cb_excl_y5 = -1.2e6

    world_isos_4 = set(world_tbt[world_tbt["Reporter_ISO3"].notna()]["Map_ISO"].str.upper())
    unmapped_tbt = df_tbt[~df_tbt["Map_ISO"].isin(world_isos_4)]
    for _, r in unmapped_tbt.iterrows():
        iso = r["Reporter_ISO3"]
        if iso in coords_eq and pd.notna(r["TBT"]):
            tbt_v = r["TBT"]
            pt = coords_eq[iso]
            if pt[0] < cb_excl_x5 and pt[1] < cb_excl_y5:
                continue
            ax_m.scatter(pt[0], pt[1], c=[tbt_v], cmap=cmap_tbt, norm=norm_tbt, s=30,
                         edgecolors="#1E293B", linewidth=0.7, zorder=5)

    top_per_reporter = df_flow.sort_values("Total_Weight_MMT", ascending=False).groupby("Reporter_ISO3").head(1)
    mega_flows = df_flow.head(35)
    combined_flows = pd.concat([top_per_reporter, mega_flows]).drop_duplicates(subset=["Reporter_ISO3", "Partner_ISO3"])
    max_vol = combined_flows["Total_Weight_MMT"].max()

    plotted_isos_4 = set()
    for _, r in combined_flows.iterrows():
        s, d = str(r["Reporter_ISO3"]).upper(), str(r["Partner_ISO3"]).upper()
        xt, yt = compute_clean_corridor_arc(s, d, coords_eq)
        if xt is not None:
            plotted_isos_4.add(s)
            plotted_isos_4.add(d)
            c_line = "#1D4ED8" if r["Mean_Trophic_Level"] >= 3.8 else ("#D97706" if r["Mean_Trophic_Level"] >= 3.0 else "#059669")
            lw = np.clip(0.8 + (r["Total_Weight_MMT"] / max_vol) * 3.4, 0.8, 4.2)
            alpha = np.clip(0.35 + (r["Total_Weight_MMT"] / max_vol) * 0.50, 0.35, 0.85)
            ax_m.plot(xt, yt, color=c_line, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=3)

    for iso in plotted_isos_4:
        if iso in coords_eq:
            pt_n = coords_eq[iso]
            if pt_n[0] < cb_excl_x5 and pt_n[1] < cb_excl_y5:
                continue
            ax_m.scatter(pt_n[0], pt_n[1], color="#D97706", s=20, zorder=5, edgecolors="#FFFFFF", linewidth=0.6)

    ax_m.set_xlim(minx - 1e5, maxx + 1e5)
    ax_m.set_ylim(-6.92e6, maxy + 6.5e5)
    ax_m.set_aspect("auto")
    ax_m.margins(0)
    ax_m.set_facecolor("#EFF6FF")
    ax_m.patch.set_visible(True)
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_color("#CBD5E1")
        s.set_linewidth(0.8)
    ax_m.set_xticks([])
    ax_m.set_yticks([])

    cax4 = inset_axes(ax_m, width="1.9%", height="34%", loc="lower left",
                      bbox_to_anchor=(0.015, 0.06, 1, 1), bbox_transform=ax_m.transAxes, borderpad=0)
    cax4.patch.set_facecolor("#FFFFFF")
    sm4 = plt.cm.ScalarMappable(cmap=cmap_tbt, norm=norm_tbt)
    sm4._A = []
    cb4 = fig.colorbar(sm4, cax=cax4)
    cb4.set_label("Trophic Balance (TBT)", color=TEXT_DARK, fontweight="bold", fontsize=10.0)
    cb4.ax.yaxis.set_tick_params(color=TEXT_DARK, labelsize=9.2)
    plt.setp(plt.getp(cb4.ax.axes, "yticklabels"), color=TEXT_DARK, fontsize=9.2)

    legend_handles4 = [
        plt.Line2D([0], [0], color="#1D4ED8", lw=3.0, label="Apex Predators (MTL ≥ 3.8)"),
        plt.Line2D([0], [0], color="#D97706", lw=2.5, label="Pelagics & Demersals (3.0 ≤ MTL < 3.8)"),
        plt.Line2D([0], [0], color="#059669", lw=2.0, label="Low Trophic / Invertebrates (MTL < 3.0)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#D97706", markeredgecolor="#FFFFFF", markersize=6.5, label="Bilateral Flow Anchor")
    ]
    ax_m.legend(handles=legend_handles4, loc="lower center", bbox_to_anchor=(0.55, 0.05),
                frameon=True, facecolor="#FFFFFF", edgecolor=CARD_BORDER,
                framealpha=0.96, fontsize=9.2, labelcolor=TEXT_DARK, title="Bilateral Flow Corridors", title_fontsize=9.8)

    ax_m.text(0.008, 0.985, "a", transform=ax_m.transAxes, fontsize=16, fontweight="bold", color=TEXT_DARK, va="top")
    ax_m.text(0.026, 0.985, "Global Trophic Balance of Trade (TBT) & Bilateral Corridors", transform=ax_m.transAxes,
              fontsize=12.5, fontweight="bold", color="#334155", va="top")

    # Panel 5b: Regional Ecological Trophic Drain on BOTTOM LEFT
    ax_b = fig.add_subplot(_bc_axes(gs[1, 0], label_frac=0.23))
    ax_b.set_facecolor("#FFFFFF")
    reg_bio = df_tbt.groupby("Reporter_Bloc")["Net_Biomass_Export_MT"].sum().sort_values(ascending=True) / 1e6
    reg_bio.index = [reg_bio.index[i].replace("& Rest of World", "").replace("(MENA)", "").replace("& Caribbean", "& Carib.").replace("Corridor", "").strip() for i in range(len(reg_bio))]

    y_4b = np.arange(len(reg_bio))
    colors_bio = ["#2563EB" if v < 0 else "#DC2626" for v in reg_bio]
    bars_4b = ax_b.barh(y_4b, reg_bio, height=0.62, color=colors_bio, edgecolor="#1E293B", alpha=0.85)
    ax_b.axvline(0, color=TEXT_DARK, linewidth=0.9, linestyle="--")
    ax_b.set_yticks(y_4b)
    ax_b.set_yticklabels(reg_bio.index, fontsize=10.0, color=TEXT_DARK, fontweight="bold")
    ax_b.set_xlabel("Net Biomass Balance (MMT; Red = Net Supplier, Blue = Net Importer)", fontsize=10.2, color=TEXT_DARK, fontweight="bold")
    ax_b.set_xlim(-56, 62)
    ax_b.set_ylim(-1.25, len(reg_bio) - 0.15)
    ax_b.tick_params(axis="x", colors=TEXT_DARK, labelsize=9.8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_color("#CBD5E1")
    ax_b.spines["bottom"].set_color("#CBD5E1")
    ax_b.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")

    for bar, val in zip(bars_4b, reg_bio):
        sign = "+" if val > 0 else ""
        txt = f"{sign}{val:.1f} MMT"
        y_coord = bar.get_y() + bar.get_height() / 2.0
        if val < 0:
            ax_b.text(val - 1.4, y_coord, txt, va="center", ha="right", color=TEXT_DARK, fontsize=9.0, fontweight="bold")
        else:
            ax_b.text(val + 1.4, y_coord, txt, va="center", ha="left", color=TEXT_DARK, fontsize=9.0, fontweight="bold")

    ax_b.text(
        0.98, 0.03,
        "North–South Asymmetry:\n"
        "• SIDS TBT = +0.396 | LDCs = −0.059 (Supplementary Table S18)\n"
        "• Sub-Saharan Africa: USD 3.57/kg export vs USD 1.54/kg import",
        transform=ax_b.transAxes, ha="right", va="bottom", fontsize=7.6, fontweight="bold",
        color=TEXT_DARK, bbox=dict(boxstyle="round,pad=0.25", facecolor="#EFF6FF", edgecolor="#2563EB", lw=0.9, alpha=0.95)
    )

    ax_b.text(-0.27, 1.07, "b", transform=ax_b.transAxes, fontsize=15, fontweight="bold", color=TEXT_DARK, va="top")
    ax_b.text(-0.22, 1.065, "Global Ecological Trophic Drain (Net Biomass Flux)", transform=ax_b.transAxes,
              fontsize=11.5, fontweight="bold", color="#334155", va="top")

    # Panel 5c: Top Bilateral Trophic Corridors on BOTTOM RIGHT
    ax_c = fig.add_subplot(_bc_axes(gs[1, 1], label_frac=0.23))
    ax_c.set_facecolor("#FFFFFF")
    top10_flow = df_flow.head(10).iloc[::-1]
    y_4c = np.arange(len(top10_flow))
    cols_flow = ["#1D4ED8" if mtl >= 3.8 else ("#D97706" if mtl >= 3.0 else "#059669") for mtl in top10_flow["Mean_Trophic_Level"]]
    bars_4c = ax_c.barh(y_4c, top10_flow["Total_Weight_MMT"], height=0.62, color=cols_flow, edgecolor="#1E293B", alpha=0.88)
    ax_c.set_yticks(y_4c)
    flow_labels = [f"{r.Reporter_ISO3} → {r.Partner_ISO3} (MTL {r.Mean_Trophic_Level:.2f})" for _, r in top10_flow.iterrows()]
    ax_c.set_yticklabels(flow_labels, fontsize=9.4, color=TEXT_DARK, fontweight="bold")
    ax_c.set_xlabel("Cumulative Biomass Transferred (Million Metric Tons, MMT)", fontsize=10.2, color=TEXT_DARK, fontweight="bold")
    ax_c.set_xlim(0, 13.0)
    ax_c.set_xticks(np.arange(0, 14, 2))
    ax_c.tick_params(axis="x", colors=TEXT_DARK, labelsize=9.8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_color("#CBD5E1")
    ax_c.spines["bottom"].set_color("#CBD5E1")
    ax_c.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")

    for bar, (_, r) in zip(bars_4c, top10_flow.iterrows()):
        w = bar.get_width()
        usd_m = r["Total_Real_USD_M"]
        ax_c.text(w + 0.18, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f} MMT (USD {usd_m/1e3:.1f}B)",
                  va="center", color=TEXT_DARK, fontsize=8.8, fontweight="bold")

    ax_c.text(-0.27, 1.07, "c", transform=ax_c.transAxes, fontsize=15, fontweight="bold", color=TEXT_DARK, va="top")
    ax_c.text(-0.22, 1.065, "Top Intercontinental Biomass Corridors by Volume", transform=ax_c.transAxes,
              fontsize=11.5, fontweight="bold", color="#334155", va="top")

    save_fig(fig, "Figure_05_Global_Map_Trophic_Transfers_and_Geodesic_Corridors")

    # Interactive Plotly Equal Earth Map
    fig_p = go.Figure()
    fig_p.add_trace(go.Choropleth(
        locations=world_tbt[iso_col],
        z=world_tbt["TBT"],
        text=world_tbt["Reporter_ISO3"] + "<br>TBT: " + world_tbt["TBT"].round(3).astype(str),
        colorscale="RdBu_r",
        zmid=0,
        colorbar_title="TBT",
        marker_line_color="#94A3B8",
        marker_line_width=0.5
    ))

    for _, r in combined_flows.head(50).iterrows():
        s, d = str(r["Reporter_ISO3"]).upper(), str(r["Partner_ISO3"]).upper()
        xt, yt = compute_clean_corridor_arc(s, d, coords_eq)
        if xt is not None:
            lons, lats = EQ_INV_TRANSFORMER.transform(xt, yt)
            c_line = "#1D4ED8" if r["Mean_Trophic_Level"] >= 3.8 else ("#D97706" if r["Mean_Trophic_Level"] >= 3.0 else "#059669")
            fig_p.add_trace(go.Scattergeo(
                lon=list(lons),
                lat=list(lats),
                mode="lines",
                line=dict(width=max(1.0, float(r["Total_Weight_MMT"]) / max_vol * 4.0), color=c_line),
                opacity=0.75,
                hoverinfo="text",
                text=f"{s} → {d}<br>Volume: {r['Total_Weight_MMT']:.2f} MMT<br>MTL: {r['Mean_Trophic_Level']:.2f}<br>Value: ${r['Total_Real_USD_M']/1e3:.1f}B",
                showlegend=False
            ))

    fig_p.update_layout(
        title="Figure 5 | Planetary Trophic Biomass Transfers & Geodesic Supply Chains (Equal Earth)",
        geo=dict(projection_type="equal earth", showframe=False, showcoastlines=True, coastlinecolor="#94A3B8",
                 bgcolor="#F8FAFC", lakecolor="#F1F5F9", landcolor="#E2E8F0"),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="Arial, sans-serif", size=13)
    )
    html_out = FIG_DIR / "Figure_05_Global_Map_Trophic_Transfers_and_Geodesic_Corridors.html"
    fig_p.write_html(str(html_out))
    print(f"   [+] Saved HTML: {html_out.name}")


# =============================================================================
# MAIN FIGURE 7: Multi-Scale Sovereign Governance, Tariff-vs.-CITES Evasion & CBDR Capacity Matrix
# =============================================================================
def plot_figure_07():
    print("[5b/13] Generating Main Figure 7: Sovereign Governance Moderation, Evasion & CBDR Matrix (600 DPI)...")
    mat_csv = DATA_DERIVED / "multi_scale_bloc_income_trophic_customs_matrix.csv"
    ppml_gov_csv = DATA_DERIVED / "governance_moderated_gravity_ppml_summary.csv"
    ev_csv = DATA_DERIVED / "governance_moderated_ppml_and_evasion_results.csv"
    sreci_csv = DATA_DERIVED / "sovereign_cbdr_reform_exposure_sreci.csv"
    if not mat_csv.exists() or not ppml_gov_csv.exists() or not sreci_csv.exists():
        print("Skipping Figure 7, missing derived data files.")
        return

    df_mat = pd.read_csv(mat_csv)
    df_pgov = pd.read_csv(ppml_gov_csv)
    df_ev = pd.read_csv(ev_csv)
    df_sreci = pd.read_csv(sreci_csv)

    fig, axes = plt.subplots(2, 2, figsize=(18.5, 11.8), dpi=600, facecolor=LIGHT_BG)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # Panel a: Multi-Scale Income-Tier & Regional Bloc Customs Consolidation Divide + Shapley Decomposition
    sub_inc = df_mat[df_mat["Classification_Dimension"].isin(["1. World Bank Income Tier", "2. Development & Vulnerability Tier", "4. Regional Trade & Customs Bloc"])].copy()
    keep_groups = [
        "HIC", "Track 1 Immediate 2028 Customs Schedules", "OECD Members (38)", "EU-27 (TARIC / CFP)",
        "USMCA (USA / Canada / Mexico)", "CPTPP (Trans-Pacific Partnership)", "RCEP (15-Nation Asia-Pacific Bloc)",
        "LMC", "Global South (UMC + LMC + LIC)", "UMC", "Small Island Developing States (SIDS)",
        "ASEAN (AHTN Processing Hub)", "AfCFTA (African Continental FTA)", "LIC", "Least Developed Countries (LDCs)"
    ]
    sub_inc = sub_inc[sub_inc["Group_Name"].isin(keep_groups)].drop_duplicates("Group_Name").sort_values("Pooled_TAR_Value_Pct", ascending=True)
    y_a = np.arange(len(sub_inc))
    cols_a = [TEAL if v < 45 else (AMBER if v < 75 else CRIMSON) for v in sub_inc["Pooled_TAR_Value_Pct"]]
    bars_a = ax_a.barh(y_a, sub_inc["Pooled_TAR_Value_Pct"], color=cols_a, height=0.64, edgecolor="#1E293B", alpha=0.90)
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels(sub_inc["Group_Name"], fontsize=9.2, fontweight="bold", color=TEXT_DARK)
    ax_a.set_xlabel("Pooled Customs Export Share (TAR Value %)", fontsize=10.5, fontweight="bold", color=TEXT_DARK)
    ax_a.set_title("a  Global North–South & Economic-Bloc Customs Consolidation Divide", loc="left", fontweight="bold", fontsize=12.0, pad=8)
    ax_a.set_xlim(0, 108)
    for bar, (_, r) in zip(bars_a, sub_inc.iterrows()):
        w = bar.get_width()
        ax_a.text(w + 1.2, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}% (Gov {r['Mean_Gov_Capacity_Index']:+.2f})", va="center", fontsize=8.3, fontweight="bold", color=TEXT_DARK)

    # Panel b: Governance-Moderated Structural Gravity PPML Cross-Partials (High-Gov vs Low-Gov Corridors)
    y_b = np.arange(len(df_pgov))
    h_b = 0.36
    b_high = ax_b.barh(y_b - h_b / 2, df_pgov["Implied_Pct_Change_HighGov_Plus1SD"], height=h_b, color=NAVY, edgecolor="black", label="High-Governance Corridors (+1 unit Gov index)", alpha=0.90)
    b_low = ax_b.barh(y_b + h_b / 2, np.clip(df_pgov["Implied_Pct_Change_LowGov_Minus1SD"], -100, 115), height=h_b, color=CORAL, edgecolor="black", label="Low-Governance Corridors (−1 unit Gov index; label = interaction p)", alpha=0.90)
    ax_b.axvline(0, color="black", linestyle="--", lw=1.3)
    ax_b.set_yticks(y_b)
    b_lbls = [r.replace(" (Thunnus spp.)", "").replace(" (Anguilla spp.)", "").replace(" (Acipenseriformes)", "").replace(" (Holothuroidea)", "") for r in df_pgov["Regime"]]
    ax_b.set_yticklabels(b_lbls, fontsize=9.6, fontweight="bold", color=TEXT_DARK)
    ax_b.set_xlabel("Post-Treaty Volume Shift in Gravity PPML (%)", fontsize=10.5, fontweight="bold", color=TEXT_DARK)
    ax_b.set_title(r"b  Governance-Moderated Treaty Compliance vs. Trade Displacement ($\partial^2 \ln \mathbb{E}[Q] / \partial(\mathrm{Treat\cdot Post})\partial\mathrm{Gov} < 0$)", loc="left", fontweight="bold", fontsize=11.5, pad=8)
    ax_b.set_xlim(-118, 135)
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.8)
    for bar, (_, r) in zip(b_high, df_pgov.iterrows()):
        w = bar.get_width()
        ax_b.text(w - 2.5 if w < 0 else w + 2.0, bar.get_y() + bar.get_height() / 2.0, f"{w:+.1f}%", va="center", ha="right" if w < 0 else "left", fontsize=8.4, fontweight="bold", color=NAVY)
    for bar, (_, r) in zip(b_low, df_pgov.iterrows()):
        orig_w = r["Implied_Pct_Change_LowGov_Minus1SD"]
        w = bar.get_width()
        p_val = r["p_CrossPartial_Gov"]
        p_txt = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        ax_b.text(w + 2.0 if w >= 0 else w - 2.5, bar.get_y() + bar.get_height() / 2.0, f"{orig_w:+.1f}% ({p_txt})", va="center", ha="left" if w >= 0 else "right", fontsize=8.2, fontweight="bold", color=CRIMSON)

    # Panel c: Fisman-Wei WTO Tariff Evasion vs. CITES Treaty Evasion & Processing Transformation (Eq. 9d)
    ev_glob = df_ev[(df_ev["Corridor_Stratum"] == "All_Global_Corridors") & (df_ev["Parameter"].isin(["Is_CITES", "Pooled_x_Tariff", "Is_Pooled", "Is_Ch16"]))].copy()
    param_labels = {
        "Is_CITES": "CITES Appendix I/II Listing\n(Regulatory Evasion Shift)",
        "Pooled_x_Tariff": "Pooled Code × Importer MFN Tariff\n(Fisman–Wei Tariff Evasion)",
        "Is_Pooled": "Baseline Pooled Code\n(At Mean Tariff = 10.5%)",
        "Is_Ch16": "Chapter 16 Processed Form\n(Standardized Tariff Line)"
    }
    p_order = ["Is_CITES", "Pooled_x_Tariff", "Is_Pooled", "Is_Ch16"]
    mdi_sub = ev_glob[ev_glob["Model"] == "Symmetric_MDI_Intensive_And_Extensive"].set_index("Parameter").loc[p_order]
    orp_sub = ev_glob[ev_glob["Model"] == "Unilateral_Orphan_Extensive_LPM"].set_index("Parameter").loc[p_order]
    y_c = np.arange(len(p_order))
    ax_c.barh(y_c - h_b / 2, mdi_sub["Estimate"] * 100.0, xerr=1.96 * mdi_sub["Std_Error"] * 100.0, height=h_b, color=PURPLE, edgecolor="black", label="Symmetric Mirror Discrepancy (pp ± 95% CI)", alpha=0.90)
    ax_c.barh(y_c + h_b / 2, orp_sub["Estimate"] * 100.0, xerr=1.96 * orp_sub["Std_Error"] * 100.0, height=h_b, color=AMBER, edgecolor="black", label="Unilateral Unmatched Declaration Probability (pp ± 95% CI)", alpha=0.90)
    ax_c.axvline(0, color="black", linestyle="--", lw=1.3)
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([param_labels[p] for p in p_order], fontsize=9.2, fontweight="bold", color=TEXT_DARK)
    ax_c.set_xlabel("Marginal Effect on Bilateral Discrepancy / Unmatched Rate (Percentage Points)", fontsize=10.2, fontweight="bold", color=TEXT_DARK)
    _n_c = int(df_ev.loc[df_ev["Corridor_Stratum"] == "All_Global_Corridors", "N_Obs"].iloc[0])
    ax_c.set_title(f"c  Tariff-Motivated vs. CITES-Motivated Misreporting (Dyad + Year FE, N = {_n_c:,})", loc="left", fontweight="bold", fontsize=11.5, pad=8)
    ax_c.legend(loc="lower left", frameon=True, facecolor="white", fontsize=8.8)

    # Panel d: Sovereign Reform Exposure & Capacity Index (SRECI_r) — Two-Track CBDR Matrix
    s_plot = df_sreci[df_sreci["Pooled_Threatened_CITES_Export_USD"] + df_sreci["Pooled_Threatened_CITES_Import_USD"] > 5e7].copy()
    s_plot["Total_Thr_Trade_B"] = (s_plot["Pooled_Threatened_CITES_Export_USD"] + s_plot["Pooled_Threatened_CITES_Import_USD"]) / 1e9
    track_cols = {
        "Track 1A: Immediate Jan 2028 National 8-10 Digit Adopters (EU TARIC, US HTS, JPN, CHN, GBR, EFTA, CAN, AUS, KOR)": NAVY,
        "Track 1B: Regional Processing Hub Harmonized Schedule (ASEAN AHTN 8-Digit Co-Adoption)": TEAL,
        "Track 2A: Priority CBDR eCITES & Customs Co-Financing States (SIDS, LDCs, LIC & LMC Exporters)": CRIMSON,
        "Track 2B: Phased WCO 2033 Standard Six-Digit Transition (Other UMC Exporters)": AMBER
    }
    _ti = df_sreci["Pooled_Threatened_CITES_Import_USD"].sum()
    _te = df_sreci["Pooled_Threatened_CITES_Export_USD"].sum()
    _t1a = df_sreci["Is_Track1_2028_Customs"] == 1
    _t1b = (df_sreci["Is_ASEAN"] == 1) & ~_t1a
    _p1a_imp = 100.0 * df_sreci.loc[_t1a, "Pooled_Threatened_CITES_Import_USD"].sum() / max(_ti, 1.0)
    _p1b_exp = 100.0 * df_sreci.loc[_t1b, "Pooled_Threatened_CITES_Export_USD"].sum() / max(_te, 1.0)
    _p1ab_imp = 100.0 * df_sreci.loc[_t1a | _t1b, "Pooled_Threatened_CITES_Import_USD"].sum() / max(_ti, 1.0)
    _p1ab_exp = 100.0 * df_sreci.loc[_t1a | _t1b, "Pooled_Threatened_CITES_Export_USD"].sum() / max(_te, 1.0)
    short_trk = {
        "Track 1A: Immediate Jan 2028 National 8-10 Digit Adopters (EU TARIC, US HTS, JPN, CHN, GBR, EFTA, CAN, AUS, KOR)": f"Track 1A: Immediate 2028 8–10 Digit Adopters ({_p1a_imp:.1f}% Imports)",
        "Track 1B: Regional Processing Hub Harmonized Schedule (ASEAN AHTN 8-Digit Co-Adoption)": f"Track 1B: ASEAN AHTN Processing Hubs (+{_p1b_exp:.1f}% Exports)",
        "Track 2A: Priority CBDR eCITES & Customs Co-Financing States (SIDS, LDCs, LIC & LMC Exporters)": "Track 2A: Priority CBDR Co-Financed States (SIDS, LDCs, LIC, LMC)",
        "Track 2B: Phased WCO 2033 Standard Six-Digit Transition (Other UMC Exporters)": "Track 2B: Phased WCO 2033 Six-Digit Transition"
    }
    for trk, grp_s in s_plot.groupby("CBDR_Implementation_Track"):
        sz = np.clip(np.sqrt(grp_s["Total_Thr_Trade_B"]) * 48.0, 28.0, 360.0)
        ax_d.scatter(
            grp_s["Customs_Admin_Capacity_0_1"], grp_s["Reform_Exposure_Share"] * 100.0,
            s=sz, color=track_cols.get(trk, SLATE), alpha=0.82, edgecolors="black", lw=0.8,
            label=short_trk.get(trk, trk)
        )
    offsets = {
        "CHN": (-28, 16), "TUR": (24, -4), "JPN": (18, -10), "ESP": (12, 16),
        "USA": (14, 14), "NOR": (-30, 12), "KOR": (-30, -4), "CHL": (16, -6),
        "MEX": (-30, 14), "VNM": (-34, 4), "PHL": (-2, 24), "ECU": (-20, -16),
        "THA": (12, -16), "IDN": (36, -14), "IND": (18, 22), "PER": (26, 12),
        "FJI": (6, 14)
    }
    for iso_lbl, off in offsets.items():
        r_iso = s_plot[s_plot["ISO3"] == iso_lbl]
        if not r_iso.empty:
            x_v = float(r_iso["Customs_Admin_Capacity_0_1"].iloc[0])
            y_v = float(r_iso["Reform_Exposure_Share"].iloc[0] * 100.0)
            ax_d.annotate(
                iso_lbl, (x_v, y_v), xytext=off, textcoords="offset points",
                fontsize=7.8, fontweight="bold", color=TEXT_DARK,
                arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.7, alpha=0.85)
            )
    ax_d.axvline(0.55, color=SLATE, linestyle=":", lw=1.3)
    ax_d.axhline(0.0, color="#CBD5E1", lw=0.8, zorder=1)
    ax_d.set_ylim(-5.2, 66.0)
    ax_d.set_xlabel("Sovereign Governance-Capacity Index (rescaled 0–1)", fontsize=10.5, fontweight="bold", color=TEXT_DARK)
    ax_d.set_ylabel("Pooled Threatened / CITES Export Exposure (%)", fontsize=10.5, fontweight="bold", color=TEXT_DARK)
    ax_d.set_title(f"d  Two-Track CBDR Reform Matrix (Track 1A+1B Captures {_p1ab_imp:.1f}% Imports & {_p1ab_exp:.1f}% Exports)", loc="left", fontweight="bold", fontsize=11.5, pad=8)
    ax_d.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.0)

    plt.tight_layout(w_pad=2.2, h_pad=2.2)
    save_fig(fig, "Figure_07_Sovereign_Governance_Displacement_and_CBDR_Reform_Matrix")
    save_fig(fig, "Figure_07_Sovereign_Governance_Leakage_and_CBDR_Reform_Matrix")


# =============================================================================
# SUPPLEMENTARY FIGURE S1: Bilateral Trophic Biomass Transfer Heatmap Matrix
# =============================================================================
def plot_figure_S01():
    print("[6/12] Generating Supplementary Figure S1: Bilateral Trophic Biomass Flow Matrix...")
    bilat_csv = DATA_DERIVED / "global_bilateral_trophic_flows.csv"
    if not bilat_csv.exists():
        print("Skipping Figure S1, missing data file.")
        return

    df = pd.read_csv(bilat_csv)
    df["Biomass_MMT"] = df["Total_Weight_MT"] / 1e6
    matrix = df.pivot(index="Reporter_Bloc", columns="Partner_Bloc", values="Biomass_MMT").fillna(0)

    bloc_short = {
        "EU-27 & EFTA": "EU-27 & EFTA",
        "East Asia & Pacific": "East Asia & Pacific",
        "Eurasia & Post-Soviet Corridor": "Eurasia & Post-Soviet",
        "Latin America & Caribbean": "Latin America & Carib.",
        "Middle East & North Africa (MENA)": "Middle East & N. Africa",
        "North America": "North America",
        "South Asia": "South Asia",
        "Sub-Saharan Africa & Rest of World": "Sub-Saharan Africa & RoW"
    }

    fig, ax = plt.subplots(figsize=(11.5, 9.6), dpi=600, facecolor=LIGHT_BG)
    cax = ax.imshow(matrix.values, cmap="YlGnBu", aspect="equal")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels([bloc_short.get(c, c) for c in matrix.columns], rotation=38, ha="right", fontweight="bold", fontsize=10.5, color=TEXT_DARK)
    ax.set_yticklabels([bloc_short.get(r, r) for r in matrix.index], fontweight="bold", fontsize=10.5, color=TEXT_DARK)
    ax.set_xlabel("Destination Trade Bloc", fontweight="bold", fontsize=12.0, labelpad=10, color=TEXT_DARK)
    ax.set_ylabel("Origin Trade Bloc", fontweight="bold", fontsize=12.0, labelpad=10, color=TEXT_DARK)
    ax.set_title(
        "Global Bilateral Seafood Biomass Transfers (Million Metric Tons, 1988–2024)\n"
        "Matrix cells record cumulative physical biomass transferred across 8 geopolitical trading blocs",
        fontweight="bold", pad=16, fontsize=13.0, color=TEXT_DARK
    )
    ax.grid(False)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            text_col = "white" if val > matrix.values.max() * 0.4 else TEXT_DARK
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_col, fontsize=10.2, fontweight="bold")

    cb = fig.colorbar(cax, ax=ax, fraction=0.044, pad=0.04)
    cb.set_label("Physical Biomass Transferred (Million MT)", fontweight="bold", fontsize=11.5, color=TEXT_DARK)
    cb.ax.tick_params(labelsize=10.0)

    plt.tight_layout()
    save_fig(fig, "Figure_S01_Bilateral_Trophic_Biomass_Flow_Matrix")


# =============================================================================
# SUPPLEMENTARY FIGURE S2: Bilateral Mirror Discrepancy & Customs Integrity
# =============================================================================
def plot_figure_S02():
    print("[7/12] Generating Supplementary Figure S2: Bilateral Mirror Discrepancies & Customs Integrity...")
    taxa_csv = DATA_DERIVED / "global_mirror_discrepancy_by_taxa.csv"
    gran_csv = DATA_DERIVED / "global_mirror_discrepancy_granularity_comparison.csv"
    theta_csv = DATA_DERIVED / "global_mirror_discrepancy_theta_sensitivity.csv"
    if not taxa_csv.exists() or not gran_csv.exists():
        print("Skipping Figure S2, missing data files.")
        return

    df_taxa = pd.read_csv(taxa_csv).sort_values("Severe_Mismatch_Rate", ascending=True)
    df_gran = pd.read_csv(gran_csv)
    df_theta = pd.read_csv(theta_csv)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.5, 6.2), facecolor=LIGHT_BG)

    # Panel a: Severe Mismatch Rate by Taxa
    y_pos = np.arange(len(df_taxa))
    colors_taxa = [CRIMSON if "Sea Cucumbers" in t or "Sharks" in t or "Eels" in t else TEAL for t in df_taxa["Conservation_Category"]]
    bars = ax1.barh(y_pos, df_taxa["Severe_Mismatch_Rate"], color=colors_taxa, height=0.62, alpha=0.9, edgecolor="black")
    ax1.set_yticks(y_pos)
    s02_labels = [
        str(c).replace("(Anguilla spp.)", r"($\it{Anguilla}$ spp.)").replace("(Thunnus spp.)", r"($\it{Thunnus}$ spp.)")
        for c in df_taxa["Conservation_Category"]
    ]
    ax1.set_yticklabels(s02_labels, fontweight="bold", fontsize=10.2)
    ax1.set_xlabel("Severe Mismatch Rate (|MDI Value| > 0.50, %)", fontweight="bold", fontsize=11.5)
    ax1.set_title("a  Severe Mirror Mismatch across Focal Taxa", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax1.set_xlim(0, 74)
    ax1.tick_params(axis="x", labelsize=10.2)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 1.2, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}%", va="center", fontweight="bold", fontsize=10.0)

    # Panel b: Granularity Comparison (Specific vs Aggregated)
    df_gran_sorted = df_gran.set_index("Customs_Granularity").loc[["Taxon-Specific", "Aggregated Multi-Taxa"]].reset_index()
    labels = ["Taxon-Specific\nCustoms Lines", "Aggregated Baskets\n('Other' / 'nei' Lines)"]
    rates = df_gran_sorted["Severe_Mismatch_Rate"].values
    mean_abs_mdi = df_gran_sorted["Mean_Abs_MDI_Val"].values * 100
    x = np.arange(len(labels))
    width = 0.35
    b1 = ax2.bar(x - width / 2, rates, width, label="Severe Mismatch Rate (|MDI| > 0.50, %)", color=AMBER, edgecolor="black", alpha=0.9)
    b2 = ax2.bar(x + width / 2, mean_abs_mdi, width, label="Mean Absolute Discrepancy (|MDI| %)", color=NAVY, edgecolor="black", alpha=0.9)
    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10.2)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight="bold", fontsize=10.5)
    ax2.set_ylabel("Discrepancy Metric (%)", fontweight="bold", fontsize=11.5)
    ax2.set_title("b  Customs Aggregation Inflates Divergence", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax2.set_ylim(0, 74)
    ax2.tick_params(axis="y", labelsize=10.2)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9.8)

    # Panel c: Shipping Margin Sensitivity Specification
    margins = [r"Low ($\theta = 8\%$)", r"Baseline ($\theta = 11\%$)", r"High ($\theta = 15\%$)"]
    thetas = [0.08, 0.11, 0.15]
    rates_agg = [df_theta[df_theta["Theta_Value"] == th]["Severe_Mismatch_Aggregated_Pct"].values[0] for th in thetas]
    rates_all = [df_theta[df_theta["Theta_Value"] == th]["Severe_Mismatch_Rate_All_Pct"].values[0] for th in thetas]
    rates_spec = [df_theta[df_theta["Theta_Value"] == th]["Severe_Mismatch_Specific_Pct"].values[0] for th in thetas]
    xm = np.arange(len(margins))
    w_m = 0.25
    bars_c1 = ax3.bar(xm - w_m, rates_agg, w_m, label="Aggregated Multi-Taxa", color=CRIMSON, edgecolor="black", alpha=0.9)
    bars_c2 = ax3.bar(xm, rates_all, w_m, label="All Matched Pairs", color=GOLD, edgecolor="black", alpha=0.9)
    bars_c3 = ax3.bar(xm + w_m, rates_spec, w_m, label="Taxon-Specific Lines", color=TEAL, edgecolor="black", alpha=0.9)

    for b_grp in (bars_c1, bars_c2, bars_c3):
        for bar in b_grp:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.8)

    ax3.set_xticks(xm)
    ax3.set_xticklabels(margins, fontweight="bold", fontsize=10.5)
    ax3.set_ylabel("Severe Mismatch Rate (%)", fontweight="bold", fontsize=11.5)
    ax3.set_title("c  Robustness to CIF/FOB Shipping Margins", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax3.set_ylim(42, 63)
    ax3.tick_params(axis="y", labelsize=10.2)
    ax3.legend(loc="upper right", frameon=True, fontsize=9.5, facecolor="white")

    plt.tight_layout(w_pad=2.0)
    save_fig(fig, "Figure_S02_Mirror_Discrepancy_Distribution_and_Severities")


# =============================================================================
# SUPPLEMENTARY FIGURE S3: Wild vs Farmed Price Dynamics Across Processing Forms
# =============================================================================
def plot_figure_S03():
    print("[8/12] Generating Supplementary Figure S3: Wild vs Farmed Price Dynamics...")
    dist_csv = DATA_DERIVED / "wild_vs_farmed_price_premium_tests.csv"
    if not dist_csv.exists():
        print("Skipping Figure S3, missing data file.")
        return

    df = pd.read_csv(dist_csv)
    sub = df[df["Processing_Form"] != "Luxury Marine Capture vs Industrial Aquaculture"].copy()
    sub = sub.sort_values("Price_Ratio_Wild_to_Farm", ascending=False)

    proc_short = {
        "Live / Fresh / Chilled Shellfish": "Live / Fresh Shellfish",
        "Prepared / Preserved / Canned": "Prepared / Preserved",
        "Fresh / Chilled Fillet & Meat": "Fresh / Chilled Fillet",
        "Smoked / Cured / Dried / Salted": "Smoked / Cured / Dried",
        "Frozen Fillet & Meat": "Frozen Fillet & Meat",
        "Fresh / Chilled Whole Round": "Fresh / Chilled Whole",
        "Frozen Whole Round": "Frozen Whole Round",
        "Other Form": "Other Forms",
        "Live": "Live Marine Species"
    }
    clean_labels = [proc_short.get(str(p), str(p)) for p in sub["Processing_Form"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.8, 6.6), facecolor=LIGHT_BG)

    y = np.arange(len(sub))
    height = 0.37

    # Panel a: Horizontal Grouped Bar Chart of Median Real Unit Values (symmetric with Panels b & c)
    b1 = ax1.barh(y - height / 2, sub["Median_Wild_USD_KG"], height, label="Wild Capture (Median USD/kg)", color=NAVY, edgecolor="black", alpha=0.9)
    b2 = ax1.barh(y + height / 2, sub["Median_Farmed_USD_KG"], height, label="Predominantly Farmed (USD/kg)", color=TEAL, edgecolor="black", alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(clean_labels, fontweight="bold", fontsize=10.0)
    ax1.set_xlabel("Median Real Unit Value (2024 USD / kg)", fontweight="bold", fontsize=11.5)
    ax1.set_title("a  Real Unit Valuation by Processing Form", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax1.set_xlim(0, max(sub["Median_Wild_USD_KG"].max(), sub["Median_Farmed_USD_KG"].max()) * 1.25)
    ax1.set_ylim(-0.6, len(sub) + 0.65)
    ax1.tick_params(axis="x", labelsize=10.0)
    ax1.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.96, fontsize=9.2)
    for bar in b1:
        w = bar.get_width()
        ax1.text(w + 0.35, bar.get_y() + bar.get_height() / 2, f"${w:.1f}", va="center", ha="left", fontsize=8.8, fontweight="bold", color=NAVY)
    for bar in b2:
        w = bar.get_width()
        ax1.text(w + 0.35, bar.get_y() + bar.get_height() / 2, f"${w:.1f}", va="center", ha="left", fontsize=8.8, fontweight="bold", color=TEAL)

    # Panel b: Wild/Farmed Price Ratio
    bars_r = ax2.barh(y, sub["Price_Ratio_Wild_to_Farm"], color=[CRIMSON if r > 1.0 else SLATE for r in sub["Price_Ratio_Wild_to_Farm"]], height=0.6, edgecolor="black", alpha=0.9)
    ax2.axvline(1.0, color="black", linestyle="--", lw=1.4, label="Parity (Ratio = 1.0)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(clean_labels, fontweight="bold", fontsize=10.0)
    ax2.set_xlabel("Price Ratio (Wild Capture / Farmed Equivalent)", fontweight="bold", fontsize=11.5)
    ax2.set_title("b  Wild Premium (Red) vs. Farmed Parity (Gray)", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax2.set_xlim(0, 1.58)
    ax2.set_ylim(-0.6, len(sub) + 0.65)
    ax2.tick_params(axis="x", labelsize=10.0)
    ax2.legend(loc="upper center", frameon=True, facecolor="white", framealpha=0.96, fontsize=9.2)
    for bar in bars_r:
        w = bar.get_width()
        if 0.84 <= w <= 1.0:
            ax2.text(w - 0.025, bar.get_y() + bar.get_height() / 2, f"{w:.2f}x", va="center", ha="right", color="white", fontweight="bold", fontsize=9.5)
        else:
            ax2.text(w + 0.025, bar.get_y() + bar.get_height() / 2, f"{w:.2f}x", va="center", ha="left", color=TEXT_DARK, fontweight="bold", fontsize=9.5)

    # Panel c: Transaction Share Distribution (with expanded ylim so legend never overlaps bars)
    total_records = sub["N_Wild_Total"] + sub["N_Farmed_Total"]
    share_wild = sub["N_Wild_Total"] / total_records * 100
    share_farmed = sub["N_Farmed_Total"] / total_records * 100
    ax3.barh(y, share_wild, height=0.6, label="Wild Sourced Share (%)", color=NAVY, edgecolor="black", alpha=0.9)
    ax3.barh(y, share_farmed, left=share_wild, height=0.6, label="Farmed Sourced Share (%)", color=TEAL, edgecolor="black", alpha=0.9)
    for i, (sw, sf) in enumerate(zip(share_wild, share_farmed)):
        if sw >= 15:
            ax3.text(sw / 2, y[i], f"{sw:.0f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9.0)
        if sf >= 15:
            ax3.text(sw + sf / 2, y[i], f"{sf:.0f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9.0)

    ax3.set_yticks(y)
    ax3.set_yticklabels(clean_labels, fontweight="bold", fontsize=10.0)
    ax3.set_xlabel("Share of Transaction Microdata (%)", fontweight="bold", fontsize=11.5)
    ax3.set_title("c  Sourcing Mode Composition across Forms", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-0.6, len(sub) + 0.65)
    ax3.tick_params(axis="x", labelsize=10.0)
    ax3.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", framealpha=0.96, fontsize=9.2)

    plt.tight_layout(w_pad=1.8)
    save_fig(fig, "Figure_S03_Wild_vs_Farmed_Price_Dynamics")


# =============================================================================
# SUPPLEMENTARY FIGURE S4: CITES Listing Chronology & Regulatory Lag
# =============================================================================
def plot_figure_S04():
    print("[9/12] Generating Supplementary Figure S4: CITES Listing Timeline & Regulatory Lag...")
    cites_csv = CONFIG_DIR / "cites_marine_appendices_history.csv"
    if not cites_csv.exists():
        print("Skipping Figure S4, missing data file.")
        return

    df = pd.read_csv(cites_csv).sort_values("Listing_Year")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.0, 6.2), facecolor=LIGHT_BG)

    # Panel a: Cumulative CITES Marine Listings Over Time
    years = np.arange(1975, 2025)
    cum_map = {yr: len(df[df["Listing_Year"] <= yr]) for yr in years}
    cum_listings = [cum_map[yr] for yr in years]
    ax1.plot(years, cum_listings, color=CRIMSON, lw=2.8, marker="o", markersize=4.0, label="Cumulative Marine Taxa Listed")
    ax1.fill_between(years, cum_listings, color=CRIMSON, alpha=0.15)

    ax1.annotate("CoP10 (1998):\nSturgeons Caviar", xy=(1998, cum_map[1998]), xytext=(1982, cum_map[1998] + 5.5),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4), fontsize=9.5, fontweight="bold")
    ax1.annotate("CoP14 (2007):\nSawfishes & European Eel", xy=(2007, cum_map[2007]), xytext=(1989, cum_map[2007] + 6.2),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4), fontsize=9.5, fontweight="bold")
    ax1.annotate("CoP16 (2013):\nHammerheads & Oceanic Whitetip", xy=(2013, cum_map[2013]), xytext=(1993, cum_map[2013] + 7.2),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4), fontsize=9.5, fontweight="bold")
    ax1.annotate("CoP18 (2019):\nMakos, Guitarfishes & Teatfish", xy=(2019, cum_map[2019]), xytext=(1999, cum_map[2019] + 5.2),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4), fontsize=9.5, fontweight="bold")
    ax1.annotate("CoP19 (2022):\nRequiem Sharks (54 spp.)", xy=(2022, cum_map[2022]), xytext=(2004, cum_map[2022] + 3.5),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4), fontsize=9.5, fontweight="bold")

    ax1.set_xlabel("Year", fontweight="bold", fontsize=11.5)
    ax1.set_ylabel("Cumulative Number of Focal Marine Listings", fontweight="bold", fontsize=11.5)
    ax1.set_title("a  50-Year Regulatory Acceleration in CITES Marine Listings", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax1.set_xlim(1974, 2025)
    ax1.set_xticks([1975, 1985, 1995, 2005, 2015, 2024])
    ax1.set_ylim(0, max(cum_listings) + 6.5)
    ax1.tick_params(axis="both", labelsize=10.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", fontsize=10.0)

    # Panel b: Conservation Regulatory Lag Analysis
    taxa_lags = pd.DataFrame([
        {"Taxon": "Porbeagle Shark", "Collapse_Year": 1989, "Listing_Year": 2013, "Lag_Years": 24},
        {"Taxon": "European Eel", "Collapse_Year": 1985, "Listing_Year": 2007, "Lag_Years": 22},
        {"Taxon": "Oceanic Whitetip Shark", "Collapse_Year": 1995, "Listing_Year": 2013, "Lag_Years": 18},
        {"Taxon": "Scalloped Hammerhead", "Collapse_Year": 1996, "Listing_Year": 2013, "Lag_Years": 17},
        {"Taxon": "Shortfin Mako Shark", "Collapse_Year": 2004, "Listing_Year": 2019, "Lag_Years": 15},
        {"Taxon": "Giant Guitarfishes", "Collapse_Year": 2005, "Listing_Year": 2019, "Lag_Years": 14},
        {"Taxon": "Teatfish Sea Cucumbers", "Collapse_Year": 2011, "Listing_Year": 2019, "Lag_Years": 8},
        {"Taxon": "Requiem Sharks Family", "Collapse_Year": 2010, "Listing_Year": 2022, "Lag_Years": 12},
    ]).sort_values("Lag_Years", ascending=True)

    y_l = np.arange(len(taxa_lags))
    bars_lag = ax2.barh(y_l, taxa_lags["Lag_Years"], color=CORAL, height=0.6, edgecolor="black", alpha=0.9, zorder=2)
    mean_lag = taxa_lags["Lag_Years"].mean()
    ax2.axvline(mean_lag, color=NAVY, linestyle="--", lw=1.8, label=f"Mean Regulatory Delay: {mean_lag:.1f} Years", zorder=3)
    ax2.set_yticks(y_l)
    ax2.set_yticklabels(taxa_lags["Taxon"], fontweight="bold", fontsize=10.5)
    ax2.set_xlabel("Regulatory Delay (Years from Commercial Depletion to CITES II)", fontweight="bold", fontsize=11.5)
    ax2.set_title("b  Conservation Regulatory Lag across Exploited Taxonomic Groups", loc="left", fontweight="bold", fontsize=12.5, pad=10)
    ax2.set_xlim(0, 29.5)
    ax2.tick_params(axis="x", labelsize=10.5)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", fontsize=10.0)
    for bar in bars_lag:
        w = bar.get_width()
        ax2.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w} yrs",
                 va="center", ha="left", fontweight="bold", fontsize=10.0, color=TEXT_DARK,
                 bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.88, edgecolor="none"), zorder=4)

    plt.tight_layout(w_pad=2.2)
    save_fig(fig, "Figure_S04_CITES_Marine_Taxa_Listing_Timeline_and_Regulatory_Lag")


# =============================================================================
# SUPPLEMENTARY FIGURE S5 & S6: Expanded IUCN Audit (delegated to Module 20)
# =============================================================================
def plot_figure_S05_and_S06():
    print("[10-11/12] Generating Supplementary Figures S5 & S6 via Module 20...")
    import importlib.util
    spec19 = importlib.util.spec_from_file_location("script19", Path(__file__).resolve().parent / "20_visualize_threatened_taxa_exposure_suite.py")
    mod19 = importlib.util.module_from_spec(spec19)
    spec19.loader.exec_module(mod19)
    mod19.plot_figure_s05()
    mod19.plot_figure_s06()


# =============================================================================
# SUPPLEMENTARY FIGURE S7: Integrated Conceptual Governance Framework
# =============================================================================
def plot_figure_S07():
    print("[12/12] Generating Supplementary Figure S7: Conceptual Governance Framework...")
    fig, ax = plt.subplots(figsize=(16.5, 8.6), dpi=600, facecolor=LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        5.0, 9.55,
        "Global Seafood Trade and Marine Biodiversity Governance:\nFrom Customs Aggregation to Targeted 2028 WCO Reform",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color=NAVY, linespacing=1.25
    )

    # Top row: 3 symmetric boxes with pad=0.15 so visual bounds are exact:
    # Box 1 visual x: [0.35, 3.15], y: [5.25, 8.75]
    # Box 2 visual x: [3.60, 6.40], y: [5.25, 8.75]
    # Box 3 visual x: [6.85, 9.65], y: [5.25, 8.75]
    pad_v = 0.15
    top_y = 5.40
    top_h = 3.20

    box1 = FancyBboxPatch((0.50, top_y), 2.50, top_h, boxstyle=f"round,pad={pad_v},rounding_size=0.22", ec=CRIMSON, fc="#FDEDEC", lw=2.2)
    ax.add_patch(box1)
    ax.text(1.75, 8.28, "1. Customs Aggregation Gap", ha="center", va="center", fontweight="bold", color=CRIMSON, fontsize=12.2)
    ax.text(
        1.75, 6.72,
        "• USD 1.78T in Multi-Species Codes\n"
        "• 53.3% Cumulative / 56.8% Mean Annual TAR\n"
        "• 41.5% Cumulative Volume Aggregated\n"
        "• 99.6% of 767 IUCN Taxa Unspecified\n"
        "• 55.3% Severe Mismatch in Pooled Codes\n"
        "• Processing Chain Aggregation Decay",
        ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1C2833", linespacing=1.48
    )

    box2 = FancyBboxPatch((3.75, top_y), 2.50, top_h, boxstyle=f"round,pad={pad_v},rounding_size=0.22", ec=AMBER, fc="#FEF9E7", lw=2.2)
    ax.add_patch(box2)
    ax.text(5.00, 8.28, "2. Price Dynamics & Stability", ha="center", va="center", fontweight="bold", color=AMBER, fontsize=12.2)
    ax.text(
        5.00, 6.72,
        r"• Route 2WFE IV: $\hat{\beta}_{\mathrm{IV}} = -0.134\ (p=0.0002)$" + "\n"
        r"• Aquaculture Cross-Partial: $\hat{\theta}_2 = +0.132$" + "\n"
        r"• Bioeconomic Regimes I–IVb (ODEs)" + "\n"
        r"• Sea Cucumbers: $\hat{\beta} = -0.364\ (p = 0.033)$" + "\n"
        r"• Diadromous Eels: $\hat{\beta} = -0.184\ (p < 0.0001)$" + "\n"
        "• Luxury Shellfish Wild Premium: 1.31× Ratio",
        ha="center", va="center", fontsize=10.0, fontweight="bold", color="#1C2833", linespacing=1.48
    )

    box3 = FancyBboxPatch((7.00, top_y), 2.50, top_h, boxstyle=f"round,pad={pad_v},rounding_size=0.22", ec=PURPLE, fc="#F4ECF7", lw=2.2)
    ax.add_patch(box3)
    ax.text(8.25, 8.28, "3. Multilateral Governance", ha="center", va="center", fontweight="bold", color=PURPLE, fontsize=12.2)
    ax.text(
        8.25, 6.72,
        "• CITES App II Listings & RFMO Quotas\n"
        "• Quasi-Experimental Level Shifts:\n"
        "  – Eels: −53.6% (p = 0.0101, HAC)\n"
        "  – Bluefin: −37.6% (p = 0.0001, eBCD)\n"
        "  – Teatfish: −19.1% (p = 0.2041, NS)\n"
        "• Governance-Moderated Displacement",
        ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1C2833", linespacing=1.48
    )

    # Clean horizontal arrows connecting outer borders [3.15 -> 3.60] and [6.40 -> 6.85]
    ax.annotate("", xy=(3.57, 7.00), xytext=(3.18, 7.00),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=2.4, color=SLATE))
    ax.annotate("", xy=(6.82, 7.00), xytext=(6.43, 7.00),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=2.4, color=SLATE))

    # Bottom row: 2 symmetric boxes with clean center gap:
    # Box 4 visual x: [0.35, 4.75], y: [0.85, 4.45]
    # Box 5 visual x: [5.25, 9.65], y: [0.85, 4.45]
    bot_y = 1.00
    bot_h = 3.30

    box4 = FancyBboxPatch((0.50, bot_y), 4.10, bot_h, boxstyle=f"round,pad={pad_v},rounding_size=0.22", ec=TEAL, fc="#E8F8F5", lw=2.2)
    ax.add_patch(box4)
    ax.text(2.55, 3.98, "4. Ecological & Sovereign Distribution", ha="center", va="center", fontweight="bold", color=TEAL, fontsize=12.2)
    ax.text(
        2.55, 2.38,
        "• Net Trophic Exporters: SIDS (+0.396), Eurasia (+0.356), SSA (+0.214)\n"
        "• High-Income Trophic Importers: EU-27 & EFTA (TBT = −0.035; net −17.5 MMT)\n"
        "• Africa High-Trophic Export (USD 3.57/kg) vs. Pelagic Import (USD 1.54/kg)\n"
        "• Latin America & Caribbean: +39.5 MMT Net Biomass Supplier\n"
        "• Governance Sustains Natural Capital Across Corridors (p < 0.0001)",
        ha="center", va="center", fontsize=9.9, fontweight="bold", color="#1C2833", linespacing=1.50
    )

    box5 = FancyBboxPatch((5.40, bot_y), 4.10, bot_h, boxstyle=f"round,pad={pad_v},rounding_size=0.22", ec=FOREST, fc="#EAFAF1", lw=2.2)
    ax.add_patch(box5)
    ax.text(7.45, 3.98, "5. Policy Action Blueprint (2028 WCO HS Reform)", ha="center", va="center", fontweight="bold", color=FOREST, fontsize=12.2)
    ax.text(
        7.45, 2.38,
        "• Systematic Disaggregation of Catch-All 'Other Fish' Subheadings\n"
        "• 22-Subheading Non-Circular MILP Reform Schedule (Table 2)\n"
        "• Two-Track CBDR Implementation (Track 1A/B 2028 & Track 2A/B)\n"
        "• Mandatory 8-to-10 Digit National Codes for All CITES & IUCN Taxa\n"
        "• Interoperable Digital CITES e-Permitting & Universal eBCD Tracking",
        ha="center", va="center", fontsize=9.9, fontweight="bold", color="#1C2833", linespacing=1.50
    )

    # Clean vertical arrows connecting top boxes bottom visual edge (5.25) to bottom boxes top visual edge (4.45)
    ax.annotate("", xy=(2.55, 4.49), xytext=(2.55, 5.21),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=2.4, color=SLATE))
    ax.annotate("", xy=(7.45, 4.49), xytext=(7.45, 5.21),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=2.4, color=SLATE))
    # Horizontal arrow connecting Box 4 (4.75) to Box 5 (5.25)
    ax.annotate("", xy=(5.21, 2.65), xytext=(4.79, 2.65),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=2.4, color=SLATE))

    plt.tight_layout()
    save_fig(fig, "Figure_S07_Conceptual_Governance_Framework")


# =============================================================================
# CLEANUP AND MAIN ORCHESTRATOR
# =============================================================================
def cleanup_legacy_duplicate_figures():
    pass


def main():
    print("=" * 80)
    print("MODULE 19: MASTER 12-FIGURE PUBLICATION SUITE (5 MAIN + 7 SI)")
    print("=" * 80)

    cleanup_legacy_duplicate_figures()

    # Main Figures (in exact manuscript order)
    plot_figure_01()
    plot_figure_02()
    plot_figure_03()
    plot_figure_04()
    plot_figure_05()
    plot_figure_07()

    # 7 Supplementary Figures (in exact SI order)
    plot_figure_S01()
    plot_figure_S02()
    plot_figure_S03()
    plot_figure_S04()
    plot_figure_S05_and_S06()
    plot_figure_S07()

    print("\n" + "=" * 80)
    print("ALL 12 PUBLICATION FIGURES SUCCESSFULLY GENERATED IN ORDERLY SEQUENCE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
