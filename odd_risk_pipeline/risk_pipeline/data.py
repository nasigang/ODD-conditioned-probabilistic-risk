from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .preprocess import PreprocessState


@dataclass
class DatasetTensors:
    # Raw (unscaled) feature tensors
    x_gate_raw: torch.Tensor
    x_expert_raw: torch.Tensor
    # Targets
    y_gate: torch.Tensor
    y_expert: torch.Tensor
    expert_mask: torch.Tensor
    censored_mask: torch.Tensor
    # Per-sample raw scalars for risk thresholding
    raw_speed_mps: torch.Tensor
    raw_closing_speed_mps: torch.Tensor
    # Book-keeping
    segment_ids: List[str]
    frame_label: torch.Tensor


class RiskCSVDataset(Dataset):
    """Frame-level dataset backed by a transformed pandas DataFrame.

    Notes
    -----
    - This dataset assumes you already called `transform_dataframe` so that:
        * all selected feature columns are numeric
        * y_gate, y_expert, expert_mask are present
    - We keep the *raw* (unscaled) feature vectors so that raw-space warp can be
      applied BEFORE scaling.
    """

    def __init__(self, df: pd.DataFrame, state: PreprocessState, *, ttc_floor: float, ttc_cap: float):
        # Keep a temporary DataFrame only during tensor materialization.
        # This can be large (frame-level, many columns), so we free it after
        # building tensors to avoid OOM on modest machines.
        self.df = df.reset_index(drop=True)
        self.state = state
        self.schema = state.schema
        self.ttc_floor = float(ttc_floor)
        self.ttc_cap = float(ttc_cap)

        # Cache column order for raw tensors (needed for feature masking / ablations)
        self.x_gate_cols = self.schema.x_gate_cols_in_order()
        self.x_expert_cols = self.schema.x_expert_cols_in_order()

        self.tensors = self._build_tensors()
        self._n = len(self.tensors.segment_ids)

        # Segment -> indices maps (for segment-balanced batching)
        self._seg_map = self._build_segment_indices()
        self._seg_pos_map = self._build_segment_pos_indices()

        # Free the heavy DataFrame (we keep only torch tensors + segment ids).
        self.df = None

        # For raw-space warp: feature name -> index inside x_gate_raw
        self.gate_feature_index = {name: i for i, name in enumerate(self.schema.x_gate_cols_in_order())}

    def get_x_gate_colnames(self) -> List[str]:
        return list(self.x_gate_cols)

    def get_x_expert_colnames(self) -> List[str]:
        return list(self.x_expert_cols)

    def get_frame_labels(self) -> np.ndarray:
        """Frame label per row, aligned with tensor indices."""
        return self.tensors.frame_label.cpu().numpy()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x_gate_raw": self.tensors.x_gate_raw[idx],
            "x_expert_raw": self.tensors.x_expert_raw[idx],
            "y_gate": self.tensors.y_gate[idx],
            "y_expert": self.tensors.y_expert[idx],
            "expert_mask": self.tensors.expert_mask[idx],
            "censored_mask": self.tensors.censored_mask[idx],
            "raw_speed_mps": self.tensors.raw_speed_mps[idx],
            "raw_closing_speed_mps": self.tensors.raw_closing_speed_mps[idx],
            "segment_id": self.tensors.segment_ids[idx],
            # Keep as a tensor scalar so collate can stack safely.
            "frame_label": self.tensors.frame_label[idx],
        }

    def _build_tensors(self) -> DatasetTensors:
        sch = self.schema

        x_gate_cols = self.x_gate_cols
        x_expert_cols = self.x_expert_cols

        # --- Raw (unscaled) features ---
        # v2: Gate and Expert inputs are *disjoint* by design.
        # v1: x_expert_cols may include gate columns; this still works (expert will see them).
        x_gate_raw = torch.tensor(self.df[x_gate_cols].to_numpy(np.float32), dtype=torch.float32)
        x_expert_raw = torch.tensor(self.df[x_expert_cols].to_numpy(np.float32), dtype=torch.float32)

        # --- Targets / masks ---
        y_gate = torch.tensor(self.df["y_gate"].to_numpy(np.float32), dtype=torch.float32)
        y_expert = torch.tensor(self.df["y_expert"].to_numpy(np.float32), dtype=torch.float32)
        expert_mask = torch.tensor(self.df["expert_mask"].to_numpy(np.float32), dtype=torch.float32)

        # --- Raw scalar channels (for s*(·)) ---
        # 1) ego speed
        if "x__ego_speed_mps" in self.df.columns:
            rs = pd.to_numeric(self.df["x__ego_speed_mps"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        else:
            rs = np.zeros((len(self.df),), dtype=np.float32)
        rs = np.clip(rs, 0.0, 80.0)

        # 2) closing speed (preferred) - derived from rel_pos/rel_vel in 07, so it is a *cause*-side feature
        if "x__max_closing_speed_any_mps" in self.df.columns:
            rc = pd.to_numeric(self.df["x__max_closing_speed_any_mps"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        elif "x__closing_speed_mps_max" in self.df.columns:
            rc = pd.to_numeric(self.df["x__closing_speed_mps_max"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        else:
            rc = np.zeros((len(self.df),), dtype=np.float32)
        rc = np.clip(rc, 0.0, 80.0)

        raw_speed_mps = torch.tensor(rs, dtype=torch.float32)
        raw_closing_speed_mps = torch.tensor(rc, dtype=torch.float32)

        seg_ids = self.df["segment_id"].astype(str).tolist()
        frame_label = torch.tensor(self.df["frame_label"].to_numpy(np.int64), dtype=torch.int64)

        # --- Censoring mask ---
        # Identify samples where TTC was clipped to ttc_cap (right-censored).
        # y_expert = (log(ttc) - mu) / sigma
        # y_censored = (log(ttc_cap) - mu) / sigma
        y_c_val = (np.log(self.ttc_cap + 1e-9) - getattr(self.state.target_std, 'mu_y', getattr(self.state.target_std, 'mean'))) / (getattr(self.state.target_std, 'sigma_y', getattr(self.state.target_std, 'std')) + 1e-6)
        # Use a small epsilon for float comparison safety
        censored_mask = (y_expert >= (y_c_val - 1e-4)).float()

        return DatasetTensors(
            x_gate_raw=x_gate_raw,
            x_expert_raw=x_expert_raw,
            y_gate=y_gate,
            y_expert=y_expert,
            expert_mask=expert_mask,
            censored_mask=censored_mask,
            raw_speed_mps=raw_speed_mps,
            raw_closing_speed_mps=raw_closing_speed_mps,
            segment_ids=seg_ids,
            frame_label=frame_label,
        )

    def _build_segment_indices(self) -> Dict[str, List[int]]:
        seg_map: Dict[str, List[int]] = {}
        for i, sid in enumerate(self.tensors.segment_ids):
            seg_map.setdefault(sid, []).append(i)
        return seg_map

    def _build_segment_pos_indices(self) -> Dict[str, List[int]]:
        seg_pos: Dict[str, List[int]] = {}
        m = self.tensors.expert_mask.numpy() > 0.5
        for i, sid in enumerate(self.tensors.segment_ids):
            if m[i]:
                seg_pos.setdefault(sid, []).append(i)
        return seg_pos

    def get_segment_indices(self) -> Dict[str, List[int]]:
        return self._seg_map

    def get_segment_pos_indices(self) -> Dict[str, List[int]]:
        return self._seg_pos_map

    def get_scale_tensors(self, device: str) -> Dict[str, torch.Tensor]:
        """Return mean/std tensors for z-score scaling (train split only)."""
        s = self.state

        # We only z-score continuous columns. For binary / one-hot columns we use
        # mean=0, std=1 so that scaling becomes an identity transform.
        gate_cols = self.schema.x_gate_cols_in_order()
        expert_cols = self.schema.x_expert_cols_in_order()

        gm = np.array([s.scaler.means.get(c, 0.0) for c in gate_cols], dtype=np.float32)
        gs = np.array([s.scaler.stds.get(c, 1.0) for c in gate_cols], dtype=np.float32)
        em = np.array([s.scaler.means.get(c, 0.0) for c in expert_cols], dtype=np.float32)
        es = np.array([s.scaler.stds.get(c, 1.0) for c in expert_cols], dtype=np.float32)

        gate_mean = torch.tensor(gm, dtype=torch.float32, device=device)
        gate_std = torch.tensor(gs, dtype=torch.float32, device=device)
        expert_mean = torch.tensor(em, dtype=torch.float32, device=device)
        expert_std = torch.tensor(es, dtype=torch.float32, device=device)
        return {
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "expert_mean": expert_mean,
            "expert_std": expert_std,
        }


def _sample_with_min_frame_gap(rng: np.random.Generator, cand: List[int], frame_labels: np.ndarray, k: int, min_gap: int) -> List[int]:
    """Sample up to k indices from cand with a minimum |frame_label| gap.

    If constraints cannot be satisfied, falls back to simple random sampling.
    """
    if k <= 0:
        return []
    if min_gap is None or min_gap <= 0 or len(cand) <= 1:
        replace = len(cand) < k
        return rng.choice(cand, k, replace=replace).tolist()

    cand_arr = np.asarray(cand, dtype=np.int64)
    fl = frame_labels[cand_arr]
    order = np.argsort(fl)
    cand_sorted = cand_arr[order]
    fl_sorted = fl[order]

    # Try a few random starts; greedy forward pick ensures consecutive picks are >= min_gap.
    n = len(cand_sorted)
    for _ in range(6):
        start = int(rng.integers(0, n))
        picked = [int(cand_sorted[start])]
        last_fl = float(fl_sorted[start])

        # walk forward circularly
        i = (start + 1) % n
        while len(picked) < k and i != start:
            if float(fl_sorted[i]) - last_fl >= min_gap:
                picked.append(int(cand_sorted[i]))
                last_fl = float(fl_sorted[i])
            i = (i + 1) % n

        if len(picked) == k:
            return picked

    # Fallback: no constraint
    replace = len(cand) < k
    return rng.choice(cand, k, replace=replace).tolist()


class SegmentBalancedBatchSampler(Sampler[List[int]]):
    """Batch sampler that draws M segments and K frames per segment."""

    def __init__(
        self,
        seg_map: Dict[str, List[int]],
        frame_labels: np.ndarray,
        m_segments: int,
        k_frames: int,
        steps_per_epoch: int,
        *,
        seed: Optional[int] = None,
        min_frame_gap: int = 0,
    ):
        self.seg_map = seg_map
        self.frame_labels = frame_labels
        self.seg_ids = np.array(list(seg_map.keys()))
        self.m = int(m_segments)
        self.k = int(k_frames)
        self.steps = int(steps_per_epoch)
        self.min_frame_gap = int(min_frame_gap)
        self.seed = None if seed is None else int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        # Make sampling reproducible when a seed is provided, but still vary
        # across epochs so that we don't replay the exact same batches.
        self._epoch += 1
        if self.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed + self._epoch)
        
        # Infinite cyclic buffer of shuffled segments for "replace=False" behavior across batches
        SHUFFLE_BUFFER_SIZE = 10000 
        # Just simple cyclic iterator
        seg_pool = self.seg_ids.copy()
        rng.shuffle(seg_pool)
        pool_idx = 0
        
        for _ in range(self.steps):
            batch: List[int] = []
            
            # Select M unique segments
            chosen_segs = []
            while len(chosen_segs) < self.m:
                remaining = self.m - len(chosen_segs)
                available = len(seg_pool) - pool_idx
                
                if available >= remaining:
                    chosen_segs.extend(seg_pool[pool_idx : pool_idx + remaining])
                    pool_idx += remaining
                else:
                    chosen_segs.extend(seg_pool[pool_idx:])
                    # Reset pool
                    rng.shuffle(seg_pool)
                    pool_idx = 0
            
            for sid in chosen_segs:
                cand = self.seg_map[sid]
                batch.extend(_sample_with_min_frame_gap(rng, cand, self.frame_labels, self.k, self.min_frame_gap))
            yield batch


class SegmentBalancedPosBatchSampler(Sampler[List[int]]):
    """Segment-balanced sampler that only draws frames where expert_mask==1."""

    def __init__(
        self,
        seg_pos_map: Dict[str, List[int]],
        frame_labels: np.ndarray,
        m_segments: int,
        k_frames: int,
        steps_per_epoch: int,
        *,
        seed: Optional[int] = None,
        min_frame_gap: int = 0,
    ):
        self.seg_map = seg_pos_map
        self.frame_labels = frame_labels
        self.seg_ids = np.array(list(seg_pos_map.keys()))
        self.m = int(m_segments)
        self.k = int(k_frames)
        self.steps = int(steps_per_epoch)
        self.min_frame_gap = int(min_frame_gap)
        self.seed = None if seed is None else int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        self._epoch += 1
        if self.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed + self._epoch)
        if len(self.seg_ids) == 0:
            return
            
        seg_pool = self.seg_ids.copy()
        rng.shuffle(seg_pool)
        pool_idx = 0
        
        for _ in range(self.steps):
            batch: List[int] = []
            
            # Select M unique segments
            chosen_segs = []
            # We must be careful because some segments might have 0 positive samples (though we filtered seg_pos_map keys)
            
            while len(chosen_segs) < self.m:
                remaining = self.m - len(chosen_segs)
                available = len(seg_pool) - pool_idx
                
                if available >= remaining:
                    chosen_segs.extend(seg_pool[pool_idx : pool_idx + remaining])
                    pool_idx += remaining
                else:
                    chosen_segs.extend(seg_pool[pool_idx:])
                    rng.shuffle(seg_pool)
                    pool_idx = 0

            for sid in chosen_segs:
                cand = self.seg_map.get(sid, [])
                if not cand:
                    continue
                batch.extend(_sample_with_min_frame_gap(rng, cand, self.frame_labels, self.k, self.min_frame_gap))
                    
            if batch:
                yield batch


def make_dataloader(dataset: RiskCSVDataset, batch_sampler: Sampler[List[int]], num_workers: int = 0) -> DataLoader:
    def _collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = batch_list[0].keys()
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [b[k] for b in batch_list]
            # Keep non-numeric identifiers as python lists.
            if k in {"segment_id"}:
                out[k] = vals
                continue

            v0 = vals[0]
            if torch.is_tensor(v0):
                out[k] = torch.stack(vals, dim=0)
            else:
                # Fallback: try to materialize numeric scalars as tensors.
                # This prevents `torch.stack` errors when a dataset returns
                # python ints/floats for book-keeping keys.
                try:
                    arr = np.asarray(vals)
                    if np.issubdtype(arr.dtype, np.floating):
                        out[k] = torch.tensor(arr, dtype=torch.float32)
                    else:
                        out[k] = torch.tensor(arr, dtype=torch.int64)
                except Exception:
                    out[k] = vals
        return out

    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=_collate, num_workers=num_workers, pin_memory=True)
