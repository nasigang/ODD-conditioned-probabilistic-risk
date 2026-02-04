PATCH: Proxy-Gate (hide label-defining kinematics from Gate model, keep for warp) + strict feature-order checks + gate strictness options

What changed
------------
(B) Gate "perfect AUC" fix without breaking warp:
- We KEEP x__min_range_* and x__max_closing_speed_* in x_gate_raw so warp + relabeling can use them.
- But we MASK them (mean-fill by default) BEFORE scaling+feeding the Gate network.
  => Gate learns to predict y_gate from ODD/perception/ego proxies instead of trivially reproducing the label definition.

New CLI flags (train_gate_expert_flow.py)
-----------------------------------------
--gate_input_mask_regex   (default masks min_range_* and max_closing_speed_*)
--gate_input_mask_fill    mean|zero  (mean recommended)
--disable_gate_input_mask (not recommended)

(C) Strict feature-order guard
------------------------------
- training saves out/model_meta.json with gate/expert feature order + mask settings.
- eval_risk_models.py and diagnose_tau_gate_flow.py load model_meta.json and throw a FATAL error
  if current dataset feature order differs. This prevents silent misalignment.

(D) Make expert_mask less censored (increase uncensored ratio)
--------------------------------------------------------------
Already available via Gate-label options:
--gate_candidate_range_m
--gate_closing_thr_mps
--gate_ttc_max_s
Tune these to reduce censored dominance in expert_mask. (Start: ttc_max_s 4.0~6.0)

How to apply
------------
Copy these files into your odd_risk_pipeline project, overwriting existing ones:
- risk_pipeline/train.py
- risk_pipeline/schema.py
- train_gate_expert_flow.py
- eval_risk_models.py
- diagnose_tau_gate_flow.py

Then retrain.
