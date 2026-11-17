# Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24310941.svg)](https://doi.org/10.5281/zenodo.24310941)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](https://www.python.org/)

## Overview

This repository provides the core analytical, econometric, bioeconomic, optimization, and visualization codebase for the study:

> **Tunca, S.** (2026). *Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade*. Submitted to *Nature Sustainability*.  
> **Author:** Sezgin Tunca (`sezgin.tunca@gmail.com`), Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark ([ORCID: 0000-0001-6840-1466](https://orcid.org/0000-0001-6840-1466))  
> **Zenodo Archive:** [10.5281/zenodo.24310941](https://doi.org/10.5281/zenodo.24310941)  
> **GitHub Repository:** <https://github.com/SezginTunca/threatened-seafood-trade-governance>

International seafood trade generates over four-fifths of the economic value of legal cross-border trade in wild fauna and flora, yet customs schedules classify marine species largely as bulk commodities rather than biological entities. Linking global bilateral trade declarations with IUCN extinction-risk assessments, CITES regulatory timelines, World Customs Organization (WCO) Harmonized System (HS) nomenclature audits, marine life-history traits, and sovereign institutional indicators, the 20 modules in this codebase implement end-to-end empirical identification, dynamical bioeconomic simulation, and actionable customs reform optimization.

> [!IMPORTANT]
> **Data-Sharing Policy:** **No raw or derived datasets are distributed in this repository or its Zenodo archive.** Only the 20 core analytical Python scripts (`01`–`20`) and execution documentation are shared. All primary trade, species assessment, treaty, nomenclature, fisheries production, and governance series are publicly accessible from their institutional sources listed in the [Public Data Acquisition Guide](#public-data-acquisition-guide), and all aggregated summary statistics supporting the findings appear in the Manuscript and Supplementary Information (Supplementary Tables S1–S20).

---

## Repository Structure

```text
├── scripts/
│   ├── 01_sync_and_map_global_conservation_taxa.py          # HS6 x IUCN Red List x CITES concordance
│   ├── 02_build_global_conservation_panel.py                # Bilateral panel & real 2024 USD deflation (Eq. 3)
│   ├── 03_calculate_customs_aggregation_and_scri.py         # TAR, SCRI, and WCO HS 2028 legal audit (Eqs. 1-2b, S1a)
│   ├── 04_catalog_expanded_iucn_threatened_taxa.py          # Expanded 767-species IUCN threatened marine catalog
│   ├── 05_evaluate_threatened_taxa_customs_resolution.py    # Taxonomic invisibility & commercial exposure (Tables S5-S6)
│   ├── 06_build_sovereign_spatial_corridors.py              # Sovereign TAR/TBT & bilateral trade corridors
│   ├── 07_build_sovereign_socioeconomic_governance_panel.py # 210-reporter, 246-partner governance & WTO/OECD/GFW panel
│   ├── 08_estimate_scarcity_pricing_and_dual_iv_models.py   # Dual-IV scarcity elasticities & wild premiums (Eqs. 4-5, S1b-S1c)
│   ├── 09_estimate_bioeconomic_optimal_control_and_levies.py# Pontryagin border levies & stochastic SDE trajectories
│   ├── 10_simulate_coupled_bioeconomic_ode_bifurcations.py  # Wild-aquaculture ODE bifurcations (Eqs. S4a-S4d, Fig. S8)
│   ├── 11_estimate_cits_and_gravity_ppml_models.py          # Newey-West ITS, CITS, Gravity PPML & eel displacement (Eqs. 6-6c)
│   ├── 12_calculate_mirror_discrepancies_and_evasion_fe.py  # Mirror Discrepancy Index, BTAR, ECR & evasion FE (Eqs. 7-10)
│   ├── 13_estimate_stackelberg_customs_inspection_game.py   # Structural GMM decomposition & Stackelberg border deterrence
│   ├── 14_evaluate_trophic_transfers_and_lwe_balances.py    # Mean Trophic Level, TBT & LWE balances (Eqs. 11-11b)
│   ├── 15_estimate_sovereign_bloc_governance_models.py      # Shapley R2, 2WFE interactions, tariff-vs-CITES & SRECI (Eqs. 2c, 4c, 6d, 9d, 11c, 13b)
│   ├── 16_run_wco_tariff_milp_optimization.py               # Binary MILP disaggregation & Pareto frontier (Eqs. 12-13a, S5a-S5c, Fig. 6, Tables S7-S8)
│   ├── 17_run_bilevel_port_network_interdiction.py          # Bilevel chokepoint port interdiction & MC Shapley allocation
│   ├── 18_compile_publication_statistical_tables.py         # Formatted summary tables across all analytical pillars
│   ├── 19_generate_publication_figures_suite.py             # Main Figures 1-5, 7 and Supplementary Figs. S1-S4, S7 (600 DPI)
│   └── 20_visualize_threatened_taxa_exposure_suite.py       # Taxonomic funnel and family exposure figures (Figs. S5-S6)
├── .gitignore
├── .zenodo.json
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── run_all.sh
```

---

## Systematic Code Inventory

### Stage 1: Concordance, Panel Construction & Sovereign Governance (`01`–`07`)
- **`01_sync_and_map_global_conservation_taxa.py`**: Maps all 315 six-digit HS Chapters 03 and 16 seafood subheadings to IUCN Red List conservation categories and historical CITES Appendices I–III listings.
- **`02_build_global_conservation_panel.py`**: Deflates nominal bilateral trade values to constant 2024 US dollars (`CPI_2024 = 313.0`, Eq. 3), filters keystroke price outliers, and constructs clean analytical panels.
- **`03_calculate_customs_aggregation_and_scri.py`**: Quantifies value- and volume-based Taxonomic Aggregation Ratios (`TAR`, Eqs. 1–2), empirical Species Customs Resolution Index (`SCRI`, Eq. 2b and Eq. S1a), processing-tier compression, and the line-by-line legal audit of adopted WCO HS 2028 amendments.
- **`04_catalog_expanded_iucn_threatened_taxa.py`** & **`05_evaluate_threatened_taxa_customs_resolution.py`**: Catalog 767 commercially traded threatened and Near Threatened marine species, classify them into species-specific, genus/family-aggregated, and omnibus residual HS6 headings, and export Supplementary Tables S5 and S6.
- **`06_build_sovereign_spatial_corridors.py`** & **`07_build_sovereign_socioeconomic_governance_panel.py`**: Construct sovereign aggregation/trophic indices, bilateral geodesic corridors, and the country-year governance panel uniting UN M49, World Bank income/SIDS/LDC tiers, QoG governance capacity, UN SDG 14.6.1 PSMA scores, OECD Fisheries Support Estimates, WTO MFN tariffs, FAO production and border rejections, Global Fishing Watch distant-water fishing shares, World Bank CWON fisheries natural capital, and RAM Legacy v4.44 stock assessments.

### Stage 2: Econometrics, Trophic Accounting & MILP Optimization (`08`, `11`, `12`, `14`–`16`, `18`)
- **`08_estimate_scarcity_pricing_and_dual_iv_models.py`**: Estimates aggregate Newey–West HAC log-log price-volume elasticities across UN Comtrade (`1988`–`2024`) and CEPII BACI HS92 (`1995`–`2024`), converged route-by-year two-way fixed-effects Mirror-IV, Leave-One-Out Supply-IV, and overidentified Dual-IV 2SLS regressions (Eqs. 4–4b, Eqs. S1b–S1c), plus Mann–Whitney $U$ and rank-biserial wild-versus-farmed price tests (Eq. 5).
- **`11_estimate_cits_and_gravity_ppml_models.py`**: Implements segmented Newey–West HAC ITS (Eq. 6), differenced Comparative ITS with Prais–Winsten AR(1) correction (Eq. 6b), zero-retaining Structural Gravity PPML corridor regressions (Eq. 6c), and Chapter 16 (`HS 160417`) eel displacement decompositions.
- **`12_calculate_mirror_discrepancies_and_evasion_fe.py`**: Computes symmetric CIF/FOB-adjusted value and weight Mirror Discrepancy Indices (`MDI`, Eqs. 7–9), Bilateral Taxonomic Aggregation Asymmetry and Entrepot Masking Ratios (Eq. 10), and high-dimensional exporter-importer-year fixed-effects evasion and one-sided unmatched declaration models (Eqs. 9b–9c).
- **`14_evaluate_trophic_transfers_and_lwe_balances.py`**: Computes export and import Mean Trophic Levels (`MTL`), Trophic Balance of Trade (`TBT`, Eq. 11), and FAO Live-Weight Equivalent (`LWE`) robustness balances (Eq. 11b).
- **`15_estimate_sovereign_bloc_governance_models.py`**: Estimates sovereign Shapley–Owen $R^2$ variance decompositions (Eq. 2c), 2WFE Dual-IV aquaculture and income interactions (Eq. 4c), governance-moderated Gravity PPML displacement (Eq. 6d), 2012–2024 WTO applied tariff versus CITES mirror evasion models (Eq. 9d), natural-capital depletion regressions (Eq. 11c), and Sovereign Reform Exposure and Capacity Index tracks (`SRECI`, Eq. 13b).
- **`16_run_wco_tariff_milp_optimization.py`**: Solves the constrained `0–1` binary mixed-integer linear program with statutory processing-form pairing constraints (`Eqs. 12–13a`, `S5a–S5c`), continuous LP relaxation duality bounds, four IUCN weighting schemes, 1,000 Dirichlet simplex draws, Figure 6, and Supplementary Tables S7 and S8.
- **`18_compile_publication_statistical_tables.py`**: Builds structured summary tables across empirical pillars.

### Stage 3: Bioeconomic Dynamics, Game Theory & Port Interdiction (`09`, `10`, `13`, `17`)
- **`09_estimate_bioeconomic_optimal_control_and_levies.py`** & **`10_simulate_coupled_bioeconomic_ode_bifurcations.py`**: Solve Pontryagin optimal border scarcity levies with 2,000 Monte Carlo HAC parameter draws, stochastic Itô differential equations (`M = 1,000` sample paths), coupled FishBase-calibrated wild–aquaculture ODEs, Jacobian eigenvalues, saddle-node bifurcation curves (`Eqs. S4a–S4d`), and Supplementary Fig. S8.
- **`13_estimate_stackelberg_customs_inspection_game.py`**: Fits structural nonlinear least-squares parameters with 2,000 parametric bootstrap draws and solves two-stage Stackelberg border inspection deterrence thresholds under digital catch certification.
- **`17_run_bilevel_port_network_interdiction.py`**: Solves the single-level MILP reformulation of bilevel port network interdiction across 11,390 threatened-species corridors (208 jurisdictions) with taxon-specific transit-hub deflection elasticities and 5,000 Monte Carlo permutation Shapley values.

### Stage 4: Publication Figures (`19`–`20`)
- **`19_generate_publication_figures_suite.py`** & **`20_visualize_threatened_taxa_exposure_suite.py`**: Render all 600 DPI publication Main Figures (`1–5`, `7`) and Supplementary Figures (`S1–S7`) in `.png` and vector `.pdf` formats.

---

## Public Data Acquisition Guide

All underlying data sources are publicly accessible from institutional repositories without proprietary redistribution:
1. **UN Comtrade (`1988`–`2024`)**: Bilateral seafood declarations under HS Chapters 03 and 16 (<https://comtradeplus.un.org/>).
2. **CEPII BACI HS92 (`1995`–`2024`)**: Reconciled bilateral trade flows (<http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37>).
3. **IUCN Red List & CITES**: Species extinction-risk assessments (<https://www.iucnredlist.org/>) and CITES Appendices I–III listing histories (<https://cites.org/>).
4. **World Customs Organization (WCO)**: Official Harmonized System nomenclature texts and HS 2028 amendments (<https://www.wcoomd.org/>).
5. **FishBase & SeaLifeBase**: Species trophic levels and von Bertalanffy growth/mortality parameters (<https://www.fishbase.se/>, <https://www.sealifebase.ca/>).
6. **Institutional Governance, Production & Stock Databases**: FAO FishStatJ and border rejections (<https://www.fao.org/fishery/en/statistics>), World Bank WDI/WGI/CWON (<https://databank.worldbank.org/>), Quality of Government (<https://www.gu.se/en/quality-government>), WTO Tariff Download Facility (<https://tariffdata.wto.org/>), OECD Fisheries Support Estimate (<https://www.oecd.org/>), Global Fishing Watch (<https://globalfishingwatch.org/>), RAM Legacy Stock Assessment Database v4.44 (<https://www.ramlegacy.org/>), and Natural Earth sovereign boundaries (<https://www.naturalearthdata.com/>).

External institutional databases can be pointed to via `export EXTERNAL_DB_ROOT=/path/to/databases`, while bilateral trade panels should be placed under `data/raw/` before running replication.

---

## Computational Environment & Replication

Verified under **Python 3.12**:

```bash
git clone https://github.com/SezginTunca/threatened-seafood-trade-governance.git
cd threatened-seafood-trade-governance

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

bash run_all.sh
```

---

## Citation & License

This codebase is distributed under the [MIT License](LICENSE).

```bibtex
@software{tunca_2026_threatened_seafood_trade_governance,
  author       = {Tunca, Sezgin},
  title        = {{Replication Codebase: Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24310941},
  url          = {https://doi.org/10.5281/zenodo.24310941}
}
```
