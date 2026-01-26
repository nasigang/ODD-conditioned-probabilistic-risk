from __future__ import annotations

"""Feature schema utilities.

The training CSV is expected to follow a simple naming convention:
  - Frame-level (raw) features:   x__*
  - ODD / segment context:        c__*
  - Targets / labels:             y_soft, label, ... (never used as inputs)

This module builds a schema *from CSV columns only* (no data access), using
conservative heuristics to:
  - prevent leakage (drop x__best_ttci / y_soft / label / ...)
  - keep Gate inputs "causal" (exclude interaction-field aggregates)
  - keep Expert additional inputs (interaction-field aggregates) separated

If you change upstream feature names, you should only need to update the
regex/prefix rules below.
"""

from dataclasses import dataclass, asdict
from typing import List, Sequence
import json
import re


@dataclass
class FeatureSchema:
    # Stage-1 Gate inputs
    x_gate_cont: List[str]
    x_gate_bin: List[str]
    x_gate_onehot: List[str]

    # Stage-2 Expert extra inputs (appended to Gate inputs)
    x_expert_extra_cont: List[str]

    # Identifiers / targets
    id_cols: List[str]
    target_cols: List[str]

    # ------------------------------------------------------------------
    # Deterministic column ordering helpers
    # ------------------------------------------------------------------

    def x_gate_cols_in_order(self) -> List[str]:
        """Return Gate input columns in a fixed order.

        Order: continuous -> binary -> one-hot.
        We keep the order stable so that scaler statistics and warp feature
        indices remain consistent across Train/Val/Test.
        """
        return list(self.x_gate_cont) + list(self.x_gate_bin) + list(self.x_gate_onehot)

    def x_expert_cols_in_order(self) -> List[str]:
        """Return Expert conditioning columns in a fixed order.

        Expert receives (Gate inputs) + (expert-only interaction aggregates).
        """
        return self.x_gate_cols_in_order() + list(self.x_expert_extra_cont)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "FeatureSchema":
        return FeatureSchema(**json.loads(s))


def _match_regex(cols: Sequence[str], pattern: str) -> List[str]:
    rgx = re.compile(pattern)
    return [c for c in cols if rgx.match(c)]


def build_schema_from_columns(columns: Sequence[str], context_mode: str = "all") -> FeatureSchema:
    cols = list(columns)

    mode = (context_mode or "all").lower().strip()
    if mode not in {"all", "odd_only", "none"}:
        raise ValueError(
            f"Invalid context_mode={context_mode!r}. Use one of: 'all', 'odd_only', 'none'."
        )

    id_cols = [c for c in ["segment_id", "frame_label"] if c in cols]

    # Targets / leakage candidates (must never be used as inputs)
    target_cols = [c for c in [
        "y_soft",
        "label",
        "min_ttc_est",
        "log_min_ttc",
        "x__best_ttci",
        "x__best_ttc",
        "x__min_ttc",
    ] if c in cols]
    forbidden = set(target_cols)

    # ------------------------------------------------------------------
    # Identify feature columns
    # ------------------------------------------------------------------

    x_cols = [c for c in cols if c.startswith("x__") and c not in forbidden]
    c_cols = [c for c in cols if c.startswith("c__") and c not in forbidden]

    # --------------------------
    # Context (c__*) selection
    # --------------------------
    # c__ one-hot: e.g., c__odd_weather=Rain
    c_onehot_all = sorted([c for c in c_cols if "=" in c])

    # c__ binary-ish flags: e.g., c__has_crosswalk
    c_non_onehot = [c for c in c_cols if "=" not in c]
    c_bin_all = [c for c in c_non_onehot if c.startswith("c__has_")]
    c_cont_all = [c for c in c_non_onehot if c not in c_bin_all]

    # NOTE: 모든 c__는 segment 내 상수인 경우가 많아 "fingerprint"로 악용될 수 있습니다.
    # 그래서 context_mode를 통해 스키마 단계에서 필터링할 수 있게 합니다.
    if mode == "none":
        x_gate_onehot = []
        c_bin = []
        c_cont = []
    elif mode == "odd_only":
        # Keep only ODD one-hots + simple map presence flags.
        # (Segment-level continuous stats/topology are dropped)
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

    # Expert-only interaction field aggregates (derived from interaction graph)
    expert_extra_patterns = (
        r"^x__density$",
        r"^x__appr_cnt_",
        r"^x__min_range_",
        r"^x__max_closing_speed_",
        r"^x__closing_speed_",  # future-proof alias
    )
    x_expert_extra_cont = []
    for p in expert_extra_patterns:
        x_expert_extra_cont.extend(_match_regex(x_cols, p + r".*"))

    # counts are not x__ prefixed
    for c in ["n_edges", "n_flagged"]:
        if c in cols and c not in forbidden:
            x_expert_extra_cont.append(c)

    x_expert_extra_cont = sorted(set(x_expert_extra_cont))

    # Gate candidates: all x__ except expert-extra
    expert_extra_set = set([c for c in x_expert_extra_cont if c.startswith("x__")])
    x_gate_candidates = [c for c in x_cols if c not in expert_extra_set]

    # Binary heuristics for gate: common indicator prefixes
    gate_bin_prefixes = (
        "x__is_",
        "x__has_",
        "x__near_",
    )
    x_gate_bin = sorted([c for c in x_gate_candidates if c.startswith(gate_bin_prefixes)])
    x_gate_cont = sorted([c for c in x_gate_candidates if c not in set(x_gate_bin)])

    # Include ODD / segment context in Gate by default
    x_gate_bin = sorted(set(x_gate_bin + c_bin))
    x_gate_cont = sorted(set(x_gate_cont + c_cont))

    return FeatureSchema(
        x_gate_cont=x_gate_cont,
        x_gate_bin=x_gate_bin,
        x_gate_onehot=x_gate_onehot,
        x_expert_extra_cont=x_expert_extra_cont,
        id_cols=id_cols,
        target_cols=sorted(set(target_cols)),
    )


def save_schema(schema: FeatureSchema, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(schema.to_json())


def load_schema(path: str) -> FeatureSchema:
    with open(path, "r", encoding="utf-8") as f:
        return FeatureSchema.from_json(f.read())
