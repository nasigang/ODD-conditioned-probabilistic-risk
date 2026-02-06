# ITSC patch v6_2 (quantile-tail main risk, unified eval+diagnose)

This patch aligns **eval_risk_models.py** and **diagnose_tau_gate_flow.py** to use the **same tail-event definition** and to produce an ITSC-friendly set of outputs.

## What changed (high-level)

- **Main risk (paper body)**: *Quantile-based tail event*\
  `Risk_q(X) = P(TTC <= t_q | X, gate=1) * p_gate(X)`\
  where `t_q` is the `q`-quantile of the **TRAIN** TTC distribution (default `q=0.1`).

- **Physical risk (appendix)**: `P(TTC <= s*(v))` remains available but is exported as **appendix outputs**.

- **Gate budget**: all “pass rates” are defined ONLY via `--gate_target_recall` (no `p_gate>=0.5`).

- **PIT**: only **censoring-aware randomized PIT** is used for submission artifacts.

## Where to copy

Copy the patched files into your repo, overwriting the originals:

- `odd_risk_pipeline/eval_risk_models.py`
- `odd_risk_pipeline/diagnose_tau_gate_flow.py`

(An unchanged `odd_risk_pipeline/risk_pipeline/risk.py` is included only so the zip has the expected structure.)

## Recommended commands

### (1) Eval (submission mode)

```bash
python odd_risk_pipeline/eval_risk_models.py \
  --csv "$OUT/gssm_inputs_train_minimal_v2_noleak.csv" \
  --run "$OUT/runs/run_concat_minimalv6_1" \
  --split val \
  --paper_mode \
  --tail_q 0.10 \
  --tail_q_source train_candidates \
  --gate_target_recall 0.90 \
  --uncertainty_samples 512 \
  --save_dir "$OUT/eval_itsc_v6_2" \
  --save_preds
```

### (2) Diagnose + ITSC package (Strategy A/B)

```bash
python odd_risk_pipeline/diagnose_tau_gate_flow.py \
  --csv "$OUT/gssm_inputs_train_minimal_v2_noleak.csv" \
  --run "$OUT/runs/run_concat_minimalv6_1" \
  --split val \
  --paper_mode \
  --tail_q 0.10 \
  --tail_q_source train_candidates \
  --gate_target_recall 0.90 \
  --flow_diag_dir "$OUT/flow_diag_itsc_v6_2" \
  --out_dir "$OUT/diagnose_itsc_v6_2" \
  --itsc_package \
  --save_preds
```

### Appendix (optional): physical s*(v) diagnostics

```bash
python odd_risk_pipeline/diagnose_tau_gate_flow.py \
  --csv "$OUT/gssm_inputs_train_minimal_v2_noleak.csv" \
  --run "$OUT/runs/run_concat_minimalv6_1" \
  --split val \
  --main_risk_def quantile_tail \
  --appendix_physical \
  --tau_sweep 0.5 1.0 \
  --gate_target_recall 0.90 \
  --flow_diag_dir "$OUT/flow_diag_itsc_v6_2" \
  --out_dir "$OUT/diagnose_itsc_v6_2" \
  --save_preds
```

## Key outputs (where to look)

### eval_risk_models.py outputs (under `--save_dir/<run_name>/`)

- `eval_<split>_tailq0p10.json` : headline metrics for **quantile-tail** risk (paper body)
- `preds_<split>_tailq0p10.npz` : per-sample arrays (optional, with `--save_preds`)
- `pit_hist_<split>_tailq0p10.png` : randomized PIT histogram (submission PIT)
- `eval_summary_<split>_*.csv` : convenient multi-run summary table

Inside `eval_...json`:
- `risk_*` metrics are for **quantile-tail main risk**.
- `appendix_physical` contains the same metrics but for **physical s*(v)** risk.

### diagnose_tau_gate_flow.py outputs

Under `--out_dir/<run_name>/<split>/`:

- `diagnose_<split>_tailq0p10.json` : core diagnose summary (paper body)
- `gate_budget_tailq0p10.csv` : gate budget table (pass-rate/enrichment) using the **target-recall threshold**
- `p_event_calibration_tailq0p10.csv/png` : reliability diagram for `p_event` (tail probability, conditional on gate)
- `risk_calibration_tailq0p10.csv/png` : reliability for overall `risk = p_gate*p_event`

Under `--flow_diag_dir/<run_name>/<split>/`:

- `pit_hist_tailq0p10.png` : randomized PIT histogram (same definition as eval)
- `sigma_log_tailq0p10.npy` : per-sample uncertainty proxy (std of logTTC)

If `--itsc_package`:

- `density_sensitivity_tailq0p10.csv/png` : density sensitivity with bootstrap 95% CI (Strategy B)
- `proxy_regression_sigma_log_tailq0p10.csv` : multivariate proxy regression for sigma_log (+ interactions)
- `proxy_regression_taillogit_tailq0p10.csv` : multivariate proxy regression for logit(p_event)
- `partial_effect_density_sigma_tailq0p10.csv/png` : partial effect curve for density

