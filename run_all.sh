#!/usr/bin/env bash
# ==============================================================================
# Analytical Replication Pipeline (13 Core Modules)
# Study: "Customs aggregation and scarcity pricing drive a trillion-dollar global
#         threatened seafood trade"
# Target Journal: Nature Sustainability
# Author: Sezgin Tunca (ORCID: 0000-0001-6840-1466)
# ==============================================================================
set -euo pipefail

echo "========================================================================"
echo " Starting Core Analytical Replication Pipeline (Modules 01-13)          "
echo "========================================================================"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python3 scripts/01_sync_and_map_global_conservation_taxa.py
python3 scripts/02_build_global_conservation_panel.py
python3 scripts/03_calculate_customs_aggregation_and_scri.py
python3 scripts/04_catalog_expanded_iucn_threatened_taxa.py
python3 scripts/05_evaluate_threatened_taxa_customs_resolution.py
python3 scripts/06_build_sovereign_socioeconomic_governance_panel.py
python3 scripts/07_estimate_scarcity_pricing_and_dual_iv_models.py
python3 scripts/08_simulate_coupled_bioeconomic_ode_and_control.py
python3 scripts/09_estimate_cits_and_gravity_ppml_models.py
python3 scripts/10_calculate_mirror_discrepancies_and_evasion_models.py
python3 scripts/11_evaluate_trophic_transfers_and_lwe_balances.py
python3 scripts/12_estimate_sovereign_bloc_governance_models.py
python3 scripts/13_solve_wco_milp_and_bilevel_port_interdiction.py

echo "========================================================================"
echo " Core Analytical Pipeline Complete.                                     "
echo "========================================================================"
