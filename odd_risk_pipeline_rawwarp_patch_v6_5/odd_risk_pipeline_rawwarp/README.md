# Raw-space Warp ODD Risk Pipeline (2-Stage)

This package implements the same checklist + safety pins, but upgrades warp to true raw-space physical warp:

- Dataset preserves raw features (no z-score in preprocessing).
- Gate warp is applied in raw space (m/s, m/s^2, m/s^3, rad/s), then scaled with train mean/std.
- Expert is trained on scaled real-only samples (no warp).
- Risk is computed via Flow CDF:
  R = sigmoid(gate) * P(TTC <= s*(v) | X, gate=1)

## Run
```bash
python train_gate_expert_flow.py --csv /path/to/prefixed_onehot_framelevel.csv --out runs/rawwarp_demo
```

## Requirements on CSV
- Frame-level inputs with prefixes: x__ego_speed_mps, x__ego_accel_mps2, ...
- ODD one-hot columns: c__odd_weather=Rain etc.
- Targets/labels: y_soft (TTCI), n_edges, n_flagged, x__density (optional), min_ttc_est (optional).
