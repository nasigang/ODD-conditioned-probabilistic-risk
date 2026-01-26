#!/usr/bin/env python3
"""
10_visualize_sensitivity.py

Visualizes how the learned Risk model responds to changes in physical parameters
(Closing Speed) while keeping the ODD context fixed.

This helps verify if the model has learned "Physics" (Risk increases with Speed)
or is just overfitting to segment IDs.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from risk_pipeline.data import RiskCSVDataset, SegmentBalancedBatchSampler, make_dataloader
from risk_pipeline.models import GateMLP, ConditionalSpline1DFlow
from risk_pipeline.preprocess import load_preprocess_state, transform_dataframe
from risk_pipeline.risk import RiskConfig, compute_risk
from risk_pipeline.train import TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to training output (runs/odd_risk_demo)")
    ap.add_argument("--csv", type=str, required=True, help="Path to training CSV (gssm_inputs_train.csv)")
    ap.add_argument("--segment_id", type=str, default=None, help="Specific segment to analyze (optional)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = "cpu"  # faster for inference on small batch

    # 1. Load State & Models
    # ----------------------
    print(f"Loading state from {run_dir}...")
    state = load_preprocess_state(str(run_dir / "preprocess_state.json"))
    
    # We need to infer model args (hidden/depth) if they were saved, 
    # but for now we'll try to load with the default or guess.
    # ideally we should have saved a 'config.json' with model hypers.
    # Assuming the user just ran with the recent defaults (Hidden 64, Depth 2, Dropout 0.5)
    # But for inference, dropout doesn't matter (eval mode).
    # Hidden/Depth MUST match.
    # Let's try the new defaults: 64, 2.
    
    # Heuristic: Try to load state dict. If size mismatch, try old defaults.
    try:
        gate = GateMLP(len(state.schema.x_gate_cols_in_order()), hidden=64, depth=2).to(device)
        gate.load_state_dict(torch.load(run_dir / "gate.pt", map_location=device))
        print("Loaded Gate (64, 2)")
    except RuntimeError:
        print("Param mismatch for (64,2), trying old defaults (256, 3)...")
        gate = GateMLP(len(state.schema.x_gate_cols_in_order()), hidden=256, depth=3).to(device)
        gate.load_state_dict(torch.load(run_dir / "gate.pt", map_location=device))
        print("Loaded Gate (256, 3)")
        
    try:
        flow = ConditionalSpline1DFlow(len(state.schema.x_expert_cols_in_order()), hidden=64, depth=2).to(device)
        flow.load_state_dict(torch.load(run_dir / "expert_flow.pt", map_location=device))
        print("Loaded Flow (64, 2)")
    except RuntimeError:
        print("Param mismatch for (64,2), trying old defaults (256, 2)...")
        flow = ConditionalSpline1DFlow(len(state.schema.x_expert_cols_in_order()), hidden=256, depth=2).to(device)
        flow.load_state_dict(torch.load(run_dir / "expert_flow.pt", map_location=device))
        print("Loaded Flow (256, 2)")

    gate.eval()
    flow.eval()

    # 2. Load Data & Prepare One Sample
    # ---------------------------------
    print("Loading Data...")
    df = pd.read_csv(args.csv, nrows=5000) # Read enough rows to find a good candidate
    df_t = transform_dataframe(df, state, ttc_floor=0.05, ttc_cap=10.0)
    ds = RiskCSVDataset(df_t, state, ttc_floor=0.05, ttc_cap=10.0)
    
    # Pick a sample with high traffic density to make it interesting
    # We look for a segment with many objects
    if args.segment_id:
        indices = ds.get_segment_indices().get(args.segment_id, [])
        if not indices:
            raise ValueError(f"Segment {args.segment_id} not found in first 5000 rows")
        idx = indices[0]
    else:
        # heuristic: find index with max interaction count (if available in raw df) or just random
        idx = 0 
        
    print(f"Analyzing Sample Index: {idx} (Segment: {ds.tensors.segment_ids[idx]})")
    
    sample = ds[idx]
    
    # 3. Sensitivity Analysis
    # -----------------------
    # Variable: Closing Speed [0m/s ... 30m/s]
    # Fixed: Gate Features, Expert Context
    
    n_steps = 50
    v_speeds = np.linspace(0, 30, n_steps).astype(np.float32)
    
    # Expand sample to batch
    x_gate = sample["x_gate_raw"].unsqueeze(0).repeat(n_steps, 1).to(device)
    x_expert = sample["x_expert_raw"].unsqueeze(0).repeat(n_steps, 1).to(device)
    
    # Apply Scaling (Manual, as per train loop)
    # We need the means/stds from state
    # Need to reconstruct the logic from RiskCSVDataset.get_scale_tensors... 
    # simpler: just pull from state directly
    
    def get_scaler(cols):
        m = torch.tensor([state.scaler.means.get(c, 0.0) for c in cols], dtype=torch.float32, device=device)
        s = torch.tensor([state.scaler.stds.get(c, 1.0) for c in cols], dtype=torch.float32, device=device)
        return m, s

    gm, gs = get_scaler(state.schema.x_gate_cols_in_order())
    em, es = get_scaler(state.schema.x_expert_cols_in_order())
    
    # Scale static features
    x_gate_s = (x_gate - gm) / (gs + 1e-6)
    x_expert_s = (x_expert - em) / (es + 1e-6)
    
    # Overwrite Closing Speed in inputs? 
    # NOTE: Closing speed is NOT usually a direct input feature to strict models 
    # UNLESS we included it in x__.
    # But it IS used in s*(v) calculation for compute_risk.
    # Let's check schema.
    
    c_speed_col = None
    cols = state.schema.x_expert_cols_in_order()
    for i, c in enumerate(cols):
        if "closing_speed" in c:
            c_speed_col = i
            break
            
    if c_speed_col is not None:
        print(f"Detected closing speed feature at index {c_speed_col}. Varying it...")
        # Update x_expert tensor as well (if the model uses it)
        # We need to unscale -> modify -> rescale to be safe, or just modify raw and rescale
        # But we already scaled. Let's just modify x_expert_raw directly? No we have scaled.
        # Efficient way:
        # x_expert is scaled. v is raw.
        # scaled_v = (v - mu) / sigma
        
        mu_v = state.scaler.means.get(cols[c_speed_col], 0.0)
        sigma_v = state.scaler.stds.get(cols[c_speed_col], 1.0)
        
        v_scaled = (torch.tensor(v_speeds) - mu_v) / (sigma_v + 1e-6)
        x_expert_s[:, c_speed_col] = v_scaled
    
    # 4. Predict
    # ----------
    # Gate
    logits = gate(x_gate_s)
    
    # Flow
    # compute_risk needs: gate_logits, flow, x_expert_scaled, raw_closing_speed, target_std, cfg
    
    v_close_tensor = torch.tensor(v_speeds, dtype=torch.float32, device=device)
    
    rcfg = RiskConfig(tau=0.7, a_max=6.0, ttc_floor=0.05, ttc_cap=10.0)
    
    with torch.no_grad():
        risk, p_gate = compute_risk(logits, flow, x_expert_s, v_close_tensor, state.target_std, rcfg)
        
    # 5. Plot
    # -------
    plt.figure(figsize=(10, 6))
    plt.plot(v_speeds, risk.numpy(), label="Total Risk", linewidth=3)
    plt.plot(v_speeds, p_gate.numpy(), '--', label="P(Gate | ODD)", alpha=0.7)
    
    # Also plot the tail probability P(TTC < s* | ...)
    p_tail = risk / (p_gate + 1e-9)
    plt.plot(v_speeds, p_tail.numpy(), ':', label="P(Crash | Gate)", alpha=0.7)
    
    plt.xlabel("Closing Speed (m/s)")
    plt.ylabel("Probability")
    plt.title(f"Risk Sensitivity to Closing Speed\nSegment: {ds.tensors.segment_ids[idx]}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = run_dir / "sensitivity_plot.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
