# Apply patch (v6_2_5 hotfix)
This hotfix makes the ITSC regression/partial-effect exports:
- Always run in --paper_mode (auto --itsc_package), and
- No longer depend on statsmodels (numpy HC3 robust OLS used).

## Install
Copy these files over your scripts folder (the one you actually execute):
- odd_risk_pipeline/diagnose_tau_gate_flow.py
- odd_risk_pipeline/eval_risk_models.py

Example:
cp odd_risk_pipeline_itsc_patch_v6_2_5/odd_risk_pipeline/diagnose_tau_gate_flow.py  $SCRIPTS/odd_risk_pipeline_test_antigravity/odd_risk_pipeline/
cp odd_risk_pipeline_itsc_patch_v6_2_5/odd_risk_pipeline/eval_risk_models.py        $SCRIPTS/odd_risk_pipeline_test_antigravity/odd_risk_pipeline/

## Run
Use --paper_mode and check outputs under:
$out_dir/<run_name>/<split>/


v6_2_6 hotfix: adds interaction_table.csv export (density interaction terms) without requiring statsmodels.
