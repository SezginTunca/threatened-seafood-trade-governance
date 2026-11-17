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

International seafood trade generates over four-fifths of the economic value of legal cross-border commerce in wild fauna and flora, yet customs schedules classify marine species largely as bulk commodities rather than biological entities. Linking global bilateral trade declarations with IUCN extinction-risk assessments, CITES regulatory timelines, World Customs Organization (WCO) Harmonized System (HS) nomenclature audits, marine life-history traits, and sovereign institutional indicators, these 13 sequential analytical modules implement every econometric regression, bioeconomic differential-equation solver, game-theoretic calibration, and constrained optimization model reported in the Manuscript (`Equations 1–13b`, `Tables 1–2`) and Supplementary Information (`Equations S1a–S5c`, `Supplementary Tables S1–S20`).

---

## Data-Sharing Policy & Scope

> [!IMPORTANT]
> **Data Distribution Notice:**
> **In strict adherence to repository scope, institutional terms of use, and open-science best practices, no raw microdata or processed empirical datasets are distributed in this GitHub repository or in the Zenodo archive.**
>
> This distribution policy ensures full respect for the proprietary and institutional terms of third-party public data providers (UN Comtrade, CEPII, IUCN, FAO, World Bank, WTO, OECD, GFW, and RAM Legacy). All aggregated empirical summaries, regression tables, coefficient estimates, taxonomic resolution matrices, and optimization frontiers necessary to verify and evaluate the conclusions of this study are published directly in the main article (`Tables 1–2`) and the Supplementary Information (`Supplementary Tables S1–S20`).
>
> Researchers seeking to reconstruct the full empirical panels from scratch can obtain the underlying primary data directly from the official institutional repositories detailed below.

---

## Description of the Data Used

The empirical, bioeconomic, and optimization frameworks in this study integrate twelve primary institutional datasets spanning international commerce, conservation status, international treaties, biological life histories, and sovereign governance:

### 1. Global Bilateral Trade Declarations (UN Comtrade & CEPII BACI HS92)
* **UN Comtrade Database** ([https://comtradeplus.un.org/](https://comtradeplus.un.org/)):
  * **Coverage:** Annual bilateral trade declarations across 210 reporting jurisdictions spanning 37 years (`1988`–`2024`).
  * **Nomenclature:** All 315 six-digit subheadings under Harmonized System (HS) Chapter 03 (*Fish and crustaceans, molluscs and other aquatic invertebrates*; 249 subheadings across revisions HS 1988, 1996, 2002, 2007, 2012, 2017, and 2022) and Chapter 16 (*Preparations of meat, of fish or of crustaceans, molluscs or other aquatic invertebrates*; 66 subheadings under HS 1604 and 1605).
  * **Variables:** Exporter ISO3, Importer ISO3, HS 6-digit commodity code, trade flow (export, import, re-export), trade value (current USD, CIF for imports, FOB for exports), net mass (kg), and supplementary quantity units.
  * **Processing:** Export and import values are deflated to constant 2024 USD using the US Bureau of Labor Statistics Consumer Price Index for All Urban Consumers (CPI-U, base $2024 = 313.0$, `Equation 3`). Keystroke recording errors are filtered at the 99.9th percentile of annual unit values.
* **CEPII BACI HS92 Bilateral Database** ([http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37)):
  * **Coverage:** Harmonized bilateral trade matrix from `1995` to `2024` under the HS 1992 classification.
  * **Function:** Provides reconciled transaction values and quantities with estimated CIF/FOB transport-cost wedges, serving as the benchmark for Structural Gravity PPML estimations and bilateral mirror trade discrepancy models (`Equations 7–10`).

### 2. Extinction Risk & Conservation Status (IUCN Red List of Threatened Species)
* **Source:** IUCN Red List of Threatened Species, API v4, Release 2024-2 ([https://www.iucnredlist.org/](https://www.iucnredlist.org/)).
* **Coverage:** Complete taxonomic and conservation assessments for 767 marine and diadromous species.
* **Threat Categories:** Critically Endangered (CR, $n=77$), Endangered (EN, $n=182$), Vulnerable (VU, $n=282$), and Near Threatened (NT, $n=226$).
* **Extracted Variables:** Global threat category, quantitative assessment criteria (e.g., A2bd, B1ab), population trajectory (decreasing, stable, increasing, unknown), depth distribution, FAO major fishing area ranges, and documented biological use and trade stressor codes.

### 3. International Endangered Species Treaty Schedules (CITES Appendices)
* **Source:** CITES Secretariat & Checklist of CITES Species ([https://cites.org/](https://cites.org/) and [https://checklist.cites.org/](https://checklist.cites.org/)).
* **Coverage:** Historical listings from Conference of the Parties CoP1 (`1975`) through CoP19 (Panama City, `2022`; entering into force `2023`–`2024`).
* **Variables:** Appendix classification (Appendix I, II, or III), exact entry-into-force dates, taxonomic listing resolution (species-specific, whole genus, whole family, or entire order, including Acipenseriformes, Anguillidae, Sphyrnidae, Carcharhinidae, Mobulidae, and Holothuriidae), annual export quotas, and trade reservation notices.

### 4. Customs Classification & Legal Nomenclature (World Customs Organization - WCO)
* **Source:** World Customs Organization ([https://www.wcoomd.org/](https://www.wcoomd.org/)).
* **Coverage:** Historical legal texts of the Harmonized System across all six completed review cycles (HS 1988, 1996, 2002, 2007, 2012, 2017, and 2022), plus the formally adopted Seventh Review Cycle amendments for HS 2028 (WCO Council Document NG0304Ba, approved June 2024).
* **Attributes:** Complete statutory descriptions for all 315 HS6 subheadings in Chapters 03 and 16, taxonomic scope of statutory descriptions, processing-form hierarchies (live, fresh/chilled, frozen, fillets, dried, salted, smoked, prepared/preserved), and legal explanatory notes.

### 5. Marine Biological Life Histories & Trophic Ecology (FishBase & SeaLifeBase)
* **Sources:** FishBase ([https://www.fishbase.se/](https://www.fishbase.se/)) and SeaLifeBase ([https://www.sealifebase.ca/](https://www.sealifebase.ca/)).
* **Parameters:** Fractional trophic level ($TL \in [2.0, 5.0]$) derived from stomach-content and food-web models, von Bertalanffy growth parameters (asymptotic length $L_\infty$, curvature parameter $K_v$), natural mortality rate $M$, age at first maturity $t_m$, intrinsic rate of population increase $r$, carrying capacity $K$, and reproductive resilience ratings.
* **Role:** Feeds the wild-aquaculture differential-equation bifurcations (`Equations S4a–S4d`) and national Mean Trophic Level (MTL) and Trophic Balance of Trade (TBT) balances (`Equations 11–11b`).

### 6. Fisheries Production, Food Balances & Live-Weight Conversion (FAO FishStatJ)
* **Source:** Food and Agriculture Organization of the United Nations (FAO, [https://www.fao.org/fishery/en/statistics](https://www.fao.org/fishery/en/statistics)).
* **Modules:**
  * *Global Production:* Capture fishery landings and aquaculture production tonnage by species, reporting state, and FAO fishing zone (`1950`–`2024`).
  * *Technical Conversion Factors (FAO Fisheries Circular No. 847):* Species- and preservation-specific Live-Weight Equivalent (LWE) conversion coefficients used to transform processed, filleted, dried, and canned customs trade weights back into primary live biomass equivalents (`Equation 11b`).
  * *Border Interceptions:* Regulatory rejection and food safety consignment interception records across major importing markets.

### 7. Macroeconomic Indicators, Governance & Fisheries Wealth (World Bank & QoG)
* **World Bank DataBank** ([https://databank.worldbank.org/](https://databank.worldbank.org/)):
  * *World Development Indicators (WDI):* Real GDP per capita (constant 2015 USD), total population, land area, and trade openness across 210 sovereign jurisdictions.
  * *Worldwide Governance Indicators (WGI):* Six standardized governance dimensions: Regulatory Quality, Rule of Law, Government Effectiveness, Control of Corruption, Political Stability, and Voice and Accountability.
  * *Changing Wealth of Nations (CWON 2021):* Fisheries natural capital asset values, net resource rents, and capitalized marine wealth accounts (`Equation 11c`).
* **Quality of Government (QoG) Institute Standard Dataset** ([https://www.gu.se/en/quality-government](https://www.gu.se/en/quality-government)):
  * Sovereign institutional quality, judicial impartiality, and administrative bureaucratic capacity.

### 8. Trade Policy, Tariffs & Non-Tariff Measures (WTO & UNCTAD)
* **Source:** World Trade Organization Tariff Download Facility ([https://tariffdata.wto.org/](https://tariffdata.wto.org/)) and Integrated Trade Intelligence Portal (I-TIP).
* **Metrics:** Most-Favored-Nation (MFN) applied ad valorem and specific tariffs, bound tariff rates, preferential trade agreement duty schedules, and counts of active Sanitary and Phytosanitary (SPS) and Technical Barriers to Trade (TBT) measures for Chapters 03 and 16.

### 9. Distant-Water Fleet Tracking & Stock Assessment Status (GFW & RAM Legacy)
* **Global Fishing Watch (GFW)** ([https://globalfishingwatch.org/](https://globalfishingwatch.org/)):
  * Automatic Identification System (AIS) tracking of refrigerated cargo (carrier) vessels, at-sea transshipment events, distant-water fishing (DWF) fleet operations inside foreign EEZs and on the high seas, and foreign vessel port visits.
* **RAM Legacy Stock Assessment Database (v4.44)** ([https://www.ramlegacy.org/](https://www.ramlegacy.org/)):
  * Time series of stock biomass relative to maximum sustainable yield ($B/B_{MSY}$), fishing mortality rate relative to $F_{MSY}$ ($U/U_{MSY}$), and biological reference points across commercially exploited global fish stocks.

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

## Local Directory Configuration & Replication

If executing the scripts locally with downloaded institutional microdata, organize local directories as follows:

```text
├── data/
│   ├── raw/
│   │   ├── comtrade/               # Raw UN Comtrade annual bilateral CSV files (1988-2024)
│   │   ├── baci/                   # CEPII BACI HS92 CSV files (1995-2024)
│   │   ├── iucn/                   # IUCN Red List marine species JSON/CSV extracts
│   │   ├── cites/                  # CITES Checklist species & Appendices CoP1-CoP19 records
│   │   ├── wco/                    # WCO Harmonized System nomenclature legal texts & HS 2028 amendments
│   │   └── external/               # FAO FishStatJ, World Bank (WDI, WGI, CWON), WTO, OECD, GFW, RAM Legacy
```

Before running, set the environmental variable pointing to your external microdata root:

```bash
export EXTERNAL_DB_ROOT="/path/to/your/external/databases"
```

### Computational Environment Setup

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

This replication codebase is distributed under the terms of the [MIT License](LICENSE).

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
