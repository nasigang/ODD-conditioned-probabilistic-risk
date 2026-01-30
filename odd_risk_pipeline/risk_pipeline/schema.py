from __future__ import annotations

"""Feature schema utilities.

This project uses a *strict feature-role separation* to mitigate segment-fingerprinting:

- Gate (Stage-1): routing / candidate detection
  - ODD one-hots (time/weather)
  - Near-* binary flags (intersection, stop sign, ...)
  - Directionality / interaction hints (approach counts, min-range, max-closing-speed)

- Expert (Stage-2 Flow): TTC distribution conditioned on *ego dynamics* + *threat/visibility/density*.
  - x__ (ego dynamics only)
  - c_cond (threat/visibility/density), implemented as additional conditioning features

Important
---------
- Identifiers (segment_id, frame_label) and target/proxy columns (label, y_soft, x__best_ttci, ...)
  are NEVER used as model inputs.
- This module only decides "which columns are inputs" and their deterministic order.
  Scaling/standardization is handled in preprocess.py.
"""

from dataclasses import dataclass, asdict
from typing import List, Sequence, Optional
import json
import re


# -------------------------------
# Schema dataclass
# -------------------------------


@dataclass
class FeatureSchema:
    # Stage-1 Gate inputs
    x_gate_cont: List[str]
    x_gate_bin: List[str]
    x_gate_onehot: List[str]

    # Stage-2 Expert inputs (for Flow conditioning)
    # New (v2) separation
    x_expert_x_cont: Optional[List[str]] = None  # ego dynamics ("x__")
    x_expert_c_cont: Optional[List[str]] = None  # conditioning ("c_cond")

    # Legacy (v1) expert-only extras (expert saw gate+extras). Kept for backward compatibility.
    x_expert_extra_cont: Optional[List[str]] = None

    # Identifiers / targets
    id_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None

    # Optional explicit Flow split (for FiLM / coupling conditioning experiments)
    flow_x_cols: Optional[List[str]] = None
    flow_c_cols: Optional[List[str]] = None

    # --------------------------
    # Deterministic ordering
    # --------------------------

    def x_gate_cols_in_order(self) -> List[str]:
        return list(self.x_gate_cont) + list(self.x_gate_bin) + list(self.x_gate_onehot)

    def x_expert_cols_in_order(self) -> List[str]:
        """Expert conditioning columns in a fixed order.

        v2 schema: concat(x_expert_x_cont, x_expert_c_cont)
        v1 schema: concat(gate_cols, x_expert_extra_cont)
        """
        if self.x_expert_x_cont is not None or self.x_expert_c_cont is not None:
            return list(self.x_expert_x_cont or []) + list(self.x_expert_c_cont or [])
        # legacy fallback
        return self.x_gate_cols_in_order() + list(self.x_expert_extra_cont or [])

    def flow_x_cols_in_order(self) -> List[str]:
        if self.flow_x_cols:
            return list(self.flow_x_cols)
        # default split: ego dynamics as x-part if v2
        if self.x_expert_x_cont is not None:
            return list(self.x_expert_x_cont)
        return self.x_expert_cols_in_order()

    def flow_c_cols_in_order(self) -> List[str]:
        if self.flow_c_cols:
            return list(self.flow_c_cols)
        if self.x_expert_c_cont is not None:
            return list(self.x_expert_c_cont)
        return []

    def flow_cond_cols_in_order(self) -> List[str]:
        return self.flow_x_cols_in_order() + self.flow_c_cols_in_order()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "FeatureSchema":
        d = json.loads(s)
        if not isinstance(d, dict):
            raise TypeError("FeatureSchema JSON must decode to a dict.")
        # forward/backward compatible: ignore unknown keys
        try:
            from dataclasses import fields

            allowed = {f.name for f in fields(FeatureSchema)}
            d = {k: v for k, v in d.items() if k in allowed}
        except Exception:
            pass
        # normalize missing fields
        d.setdefault("x_expert_x_cont", None)
        d.setdefault("x_expert_c_cont", None)
        d.setdefault("x_expert_extra_cont", d.get("x_expert_extra_cont", None))
        d.setdefault("id_cols", d.get("id_cols", []))
        d.setdefault("target_cols", d.get("target_cols", []))
        return FeatureSchema(**d)


# -------------------------------
# Column selection
# -------------------------------


def _match_regex(cols: Sequence[str], pattern: str) -> List[str]:
    rgx = re.compile(pattern)
    return [c for c in cols if rgx.match(c)]


def _targets_and_forbidden(cols: Sequence[str]) -> List[str]:
    target_cols = [
        c
        for c in [
            "y_soft",
            "label",
            "min_ttc_est",
            "log_min_ttc",
            "x__best_ttci",
            "x__best_ttc",
            "x__min_ttc",
        ]
        if c in cols
    ]
    return target_cols


def build_schema_from_columns(
    columns: Sequence[str],
    context_mode: str = "all",
    *,
    schema_profile: str = "auto",
) -> FeatureSchema:
    """Build a feature schema from CSV headers.

    schema_profile
    --------------
    - "minimal_v2": strict lists aligned with your current research plan.
    - "auto": heuristic-based (legacy behavior).

    Notes
    -----
    Even in minimal_v2, we intersect lists with available columns so the code
    stays robust when some optional features are missing.
    """

    cols = list(columns)

    id_cols = [c for c in ["segment_id", "frame_label"] if c in cols]
    target_cols = _targets_and_forbidden(cols)
    forbidden = set(target_cols)

    profile = (schema_profile or "auto").lower().strip()

    if profile == "minimal_v2":
        # -----------------
        # Expert (Flow)
        # -----------------
        expert_x = [
            "x__ego_speed_mps",
            "x__ego_accel_mps2",
            "x__ego_yaw_rate_rps",
            "x__ego_jerk_mps3",
            "x__ego_speed_lag1",
            "x__ego_accel_lag1",
            "x__ego_speed_lag2",
            "x__ego_accel_lag2",
        ]

        expert_c = [
            # threat strength
            "x__min_range_front_m",
            "x__min_range_left_m",
            "x__min_range_right_m",
            "x__min_range_rear_m",
            "x__min_range_any_m",
            "x__max_closing_speed_front_mps",
            "x__max_closing_speed_left_mps",
            "x__max_closing_speed_right_mps",
            "x__max_closing_speed_rear_mps",
            "x__max_closing_speed_any_mps",
            # intersection distance as continuous
            "x__dist_to_intersection_lane_m",
            # density / interaction
            "x__density",
            "x__dyn_label_count_30m",
            # perception / visibility
            "x__mean_lidar_points_in_box_dyn_30m",
            "x__occlusion_low_points_ratio_dyn_30m",
        ]

        # -------------
        # Gate inputs
        # -------------
        gate_bin = [
            "x__is_near_intersection",
            "x__is_near_stop_sign",
            "x__is_near_crosswalk",
            "x__is_near_speed_bump",
            "x__is_near_driveway",
            "x__is_near_signalized_intersection",
        ]

        gate_onehot_prefix = (
            r"^c__odd_time=",
            r"^c__odd_weather=",
        )
        gate_onehot = [c for c in cols if any(re.match(p, c) for p in gate_onehot_prefix)]

        # directionality routing hints (numeric)
        gate_cont = [
            # approach counts
            "x__appr_cnt_any",
            "x__appr_cnt_front",
            "x__appr_cnt_left",
            "x__appr_cnt_right",
            "x__appr_cnt_rear",
            # strength
            "x__min_range_any_m",
            "x__max_closing_speed_any_mps",
            "x__max_closing_speed_front_mps",
            "x__max_closing_speed_rear_mps",
            "x__max_closing_speed_side_mps",  # derived (max(left,right))
        ]

        # intersect with available columns and filter forbidden
        def _keep(lst: List[str]) -> List[str]:
            out: List[str] = []
            for c in lst:
                if c in forbidden:
                    continue
                if c in cols:
                    out.append(c)
                    continue
                # allow virtual/derived columns that will be created in preprocess.transform_dataframe
                if c == 'x__max_closing_speed_side_mps' and (
                    ('x__max_closing_speed_left_mps' in cols) or ('x__max_closing_speed_right_mps' in cols)
                ):
                    out.append(c)
                # tolerate export variant names for approach-count columns
                if c.startswith('x__appr_cnt_'):
                    alt = c.replace('x__appr_cnt_', 'x_appr_cnt_', 1)
                    if alt in cols:
                        out.append(c)
            return out

        schema = FeatureSchema(
            x_gate_cont=_keep(gate_cont),
            x_gate_bin=_keep(gate_bin),
            x_gate_onehot=sorted([c for c in gate_onehot if c not in forbidden]),
            x_expert_x_cont=_keep(expert_x),
            x_expert_c_cont=_keep(expert_c),
            x_expert_extra_cont=[],
            id_cols=id_cols,
            target_cols=target_cols,
            flow_x_cols=_keep(expert_x),
            flow_c_cols=_keep(expert_c),
        )

        return schema

    # -----------------------------
    # Legacy heuristic profile
    # -----------------------------

    mode = (context_mode or "all").lower().strip()
    if mode not in {"all", "odd_only", "none"}:
        raise ValueError(f"Invalid context_mode={context_mode!r}. Use 'all'/'odd_only'/'none'.")

    x_cols = [c for c in cols if c.startswith("x__") and c not in forbidden]
    c_cols = [c for c in cols if c.startswith("c__") and c not in forbidden]

    c_onehot_all = sorted([c for c in c_cols if "=" in c])
    c_non_onehot = [c for c in c_cols if "=" not in c]
    c_bin_all = [c for c in c_non_onehot if c.startswith("c__has_")]
    c_cont_all = [c for c in c_non_onehot if c not in c_bin_all]

    if mode == "none":
        x_gate_onehot = []
        c_bin = []
        c_cont = []
    elif mode == "odd_only":
        allow_oh = (
            r"^c__odd_time=",
            r"^c__odd_weather=",
            r"^c__odd_category=",
            r"^c__traffic_density_bin=",
            r"^c__infra_density_bin=",
            r"^c__occlusion_bin=",
            r"^c__speed_bin=",
        )
        x_gate_onehot = [c for c in c_onehot_all if any(re.match(p, c) for p in allow_oh)]
        c_bin = list(c_bin_all)
        c_cont = []
    else:
        x_gate_onehot = list(c_onehot_all)
        c_bin = list(c_bin_all)
        c_cont = list(c_cont_all)

    expert_extra_patterns = (
        r"^x__density$",
        r"^x__appr_cnt_",
        r"^x__min_range_",
        r"^x__max_closing_speed_",
        r"^x__closing_speed_",
    )
    x_expert_extra_cont: List[str] = []
    for p in expert_extra_patterns:
        x_expert_extra_cont.extend(_match_regex(x_cols, p + r".*"))

    for c in ["n_edges", "n_flagged"]:
        if c in cols and c not in forbidden:
            x_expert_extra_cont.append(c)

    x_expert_extra_cont = sorted(set(x_expert_extra_cont))

    expert_extra_set = set([c for c in x_expert_extra_cont if c.startswith("x__")])
    x_gate_candidates = [c for c in x_cols if c not in expert_extra_set]

    gate_bin_prefixes = (
        "x__is_",
        "x__has_",
        "x__near_",
    )
    x_gate_bin = sorted([c for c in x_gate_candidates if c.startswith(gate_bin_prefixes)])
    x_gate_cont = sorted([c for c in x_gate_candidates if c not in set(x_gate_bin)])

    x_gate_bin = sorted(set(x_gate_bin + c_bin))
    x_gate_cont = sorted(set(x_gate_cont + c_cont))

    return FeatureSchema(
        x_gate_cont=x_gate_cont,
        x_gate_bin=x_gate_bin,
        x_gate_onehot=x_gate_onehot,
        x_expert_x_cont=None,
        x_expert_c_cont=None,
        x_expert_extra_cont=x_expert_extra_cont,
        id_cols=id_cols,
        target_cols=target_cols,
        flow_x_cols=None,
        flow_c_cols=None,
    )


def save_schema(schema: FeatureSchema, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(schema.to_json())


def load_schema(path: str) -> FeatureSchema:
    with open(path, "r", encoding="utf-8") as f:
        return FeatureSchema.from_json(f.read())
