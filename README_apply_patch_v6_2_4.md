# ITSC Patch v6.2.3 (hotfix)

This patch fixes a runtime NameError in `eval_risk_models.py` (undefined `suf`).

## Apply
Copy the two files into your repo (overwrite):
- `odd_risk_pipeline/eval_risk_models.py`
- `odd_risk_pipeline/diagnose_tau_gate_flow.py`

In your environment, if you run scripts from `$SCRIPTS/odd_risk_pipeline_test_antigravity/odd_risk_pipeline/`, overwrite the files in that folder.

## Verify
Run:
```bash
grep -n "file suffix for submission artifacts" -n odd_risk_pipeline/eval_risk_models.py -n
```
You should see a block that assigns `suf = ...`.
