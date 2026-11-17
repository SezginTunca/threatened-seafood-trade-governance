#!/usr/bin/env bash
# ==============================================================================
# Analytical Replication Pipeline
# Study: "Customs aggregation and scarcity pricing drive a trillion-dollar global
#         threatened seafood trade"
# Target Journal: Nature Sustainability
# Author: Sezgin Tunca (ORCID: 0000-0001-6840-1466)
# ==============================================================================
set -euo pipefail

echo "========================================================================"
echo " Starting Threatened Seafood Trade Governance Analytical Pipeline (01-20)"
echo "========================================================================"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Stage 1: Concordances, panel construction, customs resolution & sovereign panel
python3 scripts/01_sync_and_map_global_conservation_taxa.py
python3 scripts/02_build_global_conservation_panel.py
python3 scripts/03_calculate_customs_aggregation_and_scri.py
python3 scripts/04_catalog_expanded_iucn_threatened_taxa.py
python3 scripts/05_evaluate_threatened_taxa_customs_resolution.py
python3 scripts/06_build_sovereign_spatial_corridors.py
python3 scripts/07_build_sovereign_socioeconomic_governance_panel.py

# Stage 2: Econometric identification, trophic accounting, MILP & summary tables
python3 scripts/08_estimate_scarcity_pricing_and_dual_iv_models.py
python3 scripts/11_estimate_cits_and_gravity_ppml_models.py
python3 scripts/12_calculate_mirror_discrepancies_and_evasion_fe.py
python3 scripts/14_evaluate_trophic_transfers_and_lwe_balances.py
python3 scripts/15_estimate_sovereign_bloc_governance_models.py
python3 scripts/16_run_wco_tariff_milp_optimization.py
python3 scripts/18_compile_publication_statistical_tables.py

# Stage 3: Bioeconomic control, ODE bifurcations, Stackelberg & port interdiction
python3 scripts/09_estimate_bioeconomic_optimal_control_and_levies.py
python3 scripts/10_simulate_coupled_bioeconomic_ode_bifurcations.py
python3 scripts/13_estimate_stackelberg_customs_inspection_game.py
python3 scripts/17_run_bilevel_port_network_interdiction.py

# Stage 4: Publication display figures
python3 scripts/19_generate_publication_figures_suite.py
python3 scripts/20_visualize_threatened_taxa_exposure_suite.py

echo "========================================================================"
echo " Analytical Pipeline Execution Complete."
echo "========================================================================"
