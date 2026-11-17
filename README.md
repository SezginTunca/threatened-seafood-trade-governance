# Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24310941.svg)](https://doi.org/10.5281/zenodo.24310941)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](https://www.python.org/)

## Overview

This repository provides exclusively the **13 core analytical Python modules (`01`–`13`)** supporting the research article:

> **Tunca, S.** (2026). *Customs aggregation and scarcity pricing drive a trillion-dollar global threatened seafood trade*. Submitted to *Nature Sustainability*.  
> **Author:** Sezgin Tunca (`sezgin.tunca@gmail.com`), Department of Business and Sustainability, University of Southern Denmark, Odense, Denmark ([ORCID: 0000-0001-6840-1466](https://orcid.org/0000-0001-6840-1466))  
> **Zenodo Archive:** [10.5281/zenodo.24310941](https://doi.org/10.5281/zenodo.24310941)  
> **GitHub Repository:** <https://github.com/SezginTunca/threatened-seafood-trade-governance>

International seafood trade generates over four-fifths of the economic value of legal cross-border trade in wild fauna and flora, yet customs schedules classify marine species largely as bulk commodities rather than biological entities. Linking global bilateral trade declarations with IUCN extinction-risk assessments, CITES regulatory timelines, World Customs Organization (WCO) Harmonized System (HS) nomenclature audits, marine life-history traits, and sovereign institutional indicators, these 13 sequential analytical modules implement every econometric regression, bioeconomic differential-equation solver, game-theoretic calibration, and constrained optimization model reported in the Manuscript (`Equations 1–13b`, `Tables 1–2`) and Supplementary Information (`Equations S1a–S5c`, `Supplementary Tables S1–S20`).

> [!IMPORTANT]
> **Data-Sharing & Scope Policy:** **No raw or derived datasets are distributed in this repository or its Zenodo archive**, and no auxiliary figure-rendering or document-formatting utilities are included. Only the 13 core analytical Python modules (`01`–`13`) are shared. All primary datasets are publicly accessible from the institutional repositories listed in the [Public Data Acquisition Guide](#public-data-acquisition-guide), and all aggregated empirical summaries supporting the study appear in the Manuscript and Supplementary Information.

---

## Repository Structure

```text
├── scripts/
│   ├── 01_sync_and_map_global_conservation_taxa.py            # HS6 x IUCN Red List x CITES concordance (Table S1)
│   ├── 02_build_global_conservation_panel.py                  # Bilateral panel & real 2024 USD deflation (Eq. 3; Tables S2-S4)
│   ├── 03_calculate_customs_aggregation_and_scri.py           # TAR, SCRI & WCO HS 2028 legal audit (Eqs. 1-2b, S1a; Table S13)
│   ├── 04_catalog_expanded_iucn_threatened_taxa.py            # Expanded 767-species IUCN threatened marine catalog
│   ├── 05_evaluate_threatened_taxa_customs_resolution.py      # Taxonomic resolution & commercial exposure audit (Table S10)
│   ├── 06_build_sovereign_socioeconomic_governance_panel.py   # Sovereign TAR/TBT, corridors & 270-jurisdiction panel (Table S18)
│   ├── 07_estimate_scarcity_pricing_and_dual_iv_models.py     # Dual-IV scarcity elasticities & wild premiums (Eqs. 4-5, S1b-S1c; Table 1, S9, S14)
│   ├── 08_simulate_coupled_bioeconomic_ode_and_control.py     # Coupled ODE bifurcations & Pontryagin control (Eqs. S4a-S4d; Table S20)
│   ├── 09_estimate_cits_and_gravity_ppml_models.py            # CITS, Prais-Winsten, PPML & eel displacement (Eqs. 6-6c, S2-S3; Table 1, S5, S15-S16)
│   ├── 10_calculate_mirror_discrepancies_and_evasion_models.py# Mirror Discrepancy, evasion FE & Stackelberg game (Eqs. 7-10; Tables S7, S17)
│   ├── 11_evaluate_trophic_transfers_and_lwe_balances.py      # Mean Trophic Level, TBT & LWE balances (Eqs. 11-11b; Table S8)
│   ├── 12_estimate_sovereign_bloc_governance_models.py        # Shapley R2, 2WFE interactions, tariff-vs-CITES & SRECI (Eqs. 2c, 4c, 6d, 9d, 11c, 13b; Table S19)
│   └── 13_solve_wco_milp_and_bilevel_port_interdiction.py     # Non-circular MILP (K*=22) & bilevel port interdiction (K*=10) (Eqs. 12-13a, S5a-S5c; Table 2, S6, S11-S12)
├── .gitignore
├── .zenodo.json
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── run_all.sh
```

---

## Core Analytical Modules Mapped to Manuscript & Supplementary Information

1. **`01_sync_and_map_global_conservation_taxa.py`** — Links 315 six-digit WCO HS Chapters 03 and 16 subheadings to IUCN Red List extinction-risk assessments and CITES Appendices I–III listing timelines (**Supplementary Table S1**).
2. **`02_build_global_conservation_panel.py`** — Constructs the harmonized bilateral panel across UN Comtrade (`1988`–`2024`) and CEPII BACI HS92 (`1995`–`2024`), applies US BLS CPI-U deflation (`2024 = 313.0`, **Equation 3**), filters keystroke outliers, and summarizes descriptive epochs (**Supplementary Tables S2–S4**).
3. **`03_calculate_customs_aggregation_and_scri.py`** — Quantifies value- and volume-weighted Taxonomic Aggregation Ratios (**Equations 1–2**), cardinality-based Species Customs Resolution Index (`SCRI`, **Equation 2b**, **Equation S1a**), processing-tier pooling, and line-by-line legal verification of adopted WCO HS 2028 amendments (**Supplementary Table S13**).
4. **`04_catalog_expanded_iucn_threatened_taxa.py`** & **`05_evaluate_threatened_taxa_customs_resolution.py`** — Compile the 767-species IUCN threatened (`CR`, `EN`, `VU`) and Near Threatened (`NT`) marine catalog and evaluate species-specific, genus/family-aggregated, and omnibus residual customs resolution (**Supplementary Table S10**).
5. **`06_build_sovereign_socioeconomic_governance_panel.py`** — Computes national TAR/TBT indices and bilateral threatened-species corridors alongside the 270-jurisdiction governance panel uniting UN M49 regions, World Bank income tiers, SIDS/LDC status, QoG institutional capacity, UN SDG 14.6.1 PSMA scores, WTO applied tariffs and NTMs, OECD Fisheries Support Estimates, FAO capture/aquaculture production and border rejections, Global Fishing Watch distant-water fishing shares, World Bank CWON fisheries natural capital, and RAM Legacy v4.44 stock assessments (**Supplementary Table S18**).
6. **`07_estimate_scarcity_pricing_and_dual_iv_models.py`** — Estimates Newey–West HAC log-log elasticities (**Equation 4**), converged route-by-year two-way fixed-effects Mirror-IV, Leave-One-Out Supply-IV, and overidentified Dual-IV 2SLS models (**Equation 4b**, **Equations S1b–S1c**, **Main Table 1**, **Supplementary Table S14**), plus Mann–Whitney $U$ and rank-biserial wild-versus-farmed price tests (**Equation 5**, **Supplementary Table S9**).
7. **`08_simulate_coupled_bioeconomic_ode_and_control.py`** — Solves FishBase-calibrated coupled wild–aquaculture differential equations (**Equations S4a–S4d**), Jacobian eigenvalue stability, saddle-node tipping thresholds, Pontryagin optimal border scarcity levies ($\tau_k^*$), and 30-year Itô stochastic quasi-extinction probabilities (**Supplementary Table S20**).
8. **`09_estimate_cits_and_gravity_ppml_models.py`** — Estimates Newey–West HAC segmented ITS (**Equation 6**, **Equation S2**), log-link Quasi-Poisson GLM sensitivity (**Equation S3**, **Supplementary Table S5**), differenced Comparative ITS with Prais–Winsten AR(1) correction (**Equation 6b**), zero-retaining Structural Gravity PPML policy regressions (**Equation 6c**, **Main Table 1**, **Supplementary Table S15**), and anguillid eel cross-chapter (`Ch. 03` vs. `Ch. 16 HS 160417`) displacement (**Supplementary Table S16**).
9. **`10_calculate_mirror_discrepancies_and_evasion_models.py`** — Computes symmetric CIF/FOB-adjusted value and weight Mirror Discrepancy Indices (**Equations 7–9**, **Supplementary Table S7**), Bilateral Taxonomic Aggregation Asymmetry and Entrepot Masking Ratios (**Equation 10**), high-dimensional Exporter×Importer×Year fixed-effects strategic evasion and extensive-margin unmatched-declaration models (**Equations 9b–9c**, **Supplementary Table S17**), and nonlinear GMM Stackelberg border-inspection calibration (**SI Section S3.3**).
10. **`11_evaluate_trophic_transfers_and_lwe_balances.py`** — Quantifies export and import Mean Trophic Levels (`MTL`), net Trophic Balance of Trade (`TBT`, **Equation 11**), and FAO Live-Weight Equivalent (`LWE`) robustness across 210 reporting jurisdictions and eight regional trade blocs (**Equation 11b**, **Supplementary Table S8**).
11. **`12_estimate_sovereign_bloc_governance_models.py`** — Runs Suite A Shapley–Owen $R^2$ decompositions of sovereign pooling (**Equation 2c**), Suite B 2WFE Dual-IV income and aquaculture interactions (**Equation 4c**), Suite C governance-moderated Gravity PPML and 2012–2024 WTO applied tariff versus CITES evasion models (**Equations 6d, 9d**), Suite D CWON natural-capital and FAO wild-capture growth panels (**Equation 11c**), and Sovereign Reform Exposure and Capacity Index tracks (**Equation 13b**, **Supplementary Table S19**).
12. **`13_solve_wco_milp_and_bilevel_port_interdiction.py`** — Solves the non-circular `0–1` binary MILP tariff disaggregation problem ($K^* = 22$, **Equations 12–13**, **Main Table 2**, **Supplementary Tables S6, S11, S12**), continuous LP relaxation bounds and shadow prices (**Equation 13a**), statutory processing-form continuity (**Equations S5a–S5c**), 1,000-draw Dirichlet preference sensitivity, and the single-level MILP reformulation of bilevel chokepoint port interdiction ($K^* = 10$) with 5,000 Monte Carlo Shapley permutations (**Methods**, **SI Section S3.4**).

---

## Public Data Acquisition Guide

All underlying data sources are publicly accessible from their institutional repositories:
1. **UN Comtrade (`1988`–`2024`)**: <https://comtradeplus.un.org/>
2. **CEPII BACI HS92 (`1995`–`2024`)**: <http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37>
3. **IUCN Red List & CITES**: <https://www.iucnredlist.org/> and <https://cites.org/>
4. **World Customs Organization (WCO)**: <https://www.wcoomd.org/>
5. **FishBase & SeaLifeBase**: <https://www.fishbase.se/> and <https://www.sealifebase.ca/>
6. **Institutional Governance, Production & Stock Databases**: FAO FishStatJ (<https://www.fao.org/fishery/en/statistics>), World Bank WDI/WGI/CWON (<https://databank.worldbank.org/>), Quality of Government (<https://www.gu.se/en/quality-government>), WTO Tariff Download Facility (<https://tariffdata.wto.org/>), OECD (<https://www.oecd.org/>), Global Fishing Watch (<https://globalfishingwatch.org/>), and RAM Legacy Stock Assessment Database v4.44 (<https://www.ramlegacy.org/>).

Place bilateral panels in `data/raw/` and point `EXTERNAL_DB_ROOT` to local institutional database downloads before executing `bash run_all.sh`.

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

Released under the [MIT License](LICENSE).

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
