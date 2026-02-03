# StrategyC v5: Expert Warp Mixing + Strict Censoring

## What changed (why)
- **Expert now sees some "harder" tail samples**: we append warped samples into each Expert batch.
  This directly attacks PIT bias caused by Expert training only on easy raw safe data.
- **One-direction warp**: not a fixed direction. For each sample we pick the *dominant* approach direction among {front, rear, side} and warp only that one.
- **Physics-gated labeling for Expert**: after warping, we recompute a conservative TTC proxy from (range/closing) and update `y_expert` in logTTC-space with **strict censoring** at `ttc_cap`.

## Apply patch
Unzip at your repo root (where `train_gate_expert_flow.py` exists):

```bash
unzip -o odd_risk_strategyC_expertwarp_v5_patch.zip -d .
```

## Train (baseline vs expert-warp)
**Baseline (no Expert warp):**
```bash
python train_gate_expert_flow.py \
  --csv "$OUT"/gssm_inputs_train_minimal_v3_noleak.csv \
  --out "$OUT"/runs/run_concat_baseline \
  --ttc_floor 0.05 --ttc_cap 10.0 \
  --warp_p 0.35 \
  --expert_warp_p 0.0
```

**Expert-warp (recommended starting point):**
```bash
python train_gate_expert_flow.py \
  --csv "$OUT"/gssm_inputs_train_minimal_v3_noleak.csv \
  --out "$OUT"/runs/run_concat_expertwarp_v5 \
  --ttc_floor 0.05 --ttc_cap 10.0 \
  --warp_p 0.35 \
  --warp_speed_scale_min 1.05 --warp_speed_scale_max 1.25 \
  --expert_warp_p 0.10 \
  --expert_warp_closing_add_min 0.5 --expert_warp_closing_add_max 6.0
```

## Diagnose / Eval
Use your existing evaluation scripts the same way.
The key indicator you should see improve first is:
- **PIT mean** moving toward 0.5 (less left-skew), and
- **ttc_pdf_low_vs_high** showing less tail collapse.

## Tuning knobs (practical)
If training becomes unstable (NLL explodes):
- lower `--expert_warp_p` (0.05)
- lower `--expert_warp_closing_add_max` (3.0)
- keep `--expert_warp_speed_scale_max` near 1.0 (disable ego speed scaling for Expert warp)
