from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import pandas as pd


@dataclass
class FeatureRoleMap:
    """Record of feature-role remapping performed at CSV/DataFrame stage.

    moved: new_c_col -> old_x_col mapping
    dropped_x: x__ columns removed after copying to c__

    Notes
    -----
    - This is a *data-layer* operation. It does not change model code by itself.
    - Intended usage: apply on the exported CSV before schema building.
    """

    moved: Dict[str, str]
    dropped_x: List[str]
    prefix: str = "ctx__"
    regex: str = ""
    # Python 3.8 compatibility: avoid PEP604 union syntax.
    explicit: Optional[List[str]] = None


def _parse_csv_list(s: str) -> List[str]:
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def apply_x_to_c(
    df: pd.DataFrame,
    *,
    x_to_c: str = "",
    x_to_c_regex: str = "",
    c_prefix: str = "ctx__",
    drop_x: bool = True,
) -> Tuple[pd.DataFrame, FeatureRoleMap]:
    """
    - Select x__ columns by explicit list and/or regex (match against both full col and base name).
    - Create c__{c_prefix}{base} columns with same values.
    - Optionally drop the original x__ columns to prevent Expert/Flow from seeing them.
    """
    """Copy selected x__ columns into c__ columns.

    Creates: c__{c_prefix}{base} where base is x__ column name without 'x__'.

    Why
    ---
    - To prevent Flow likelihood from directly using high-entropy ODD/context-like
      features (segment fingerprints) while still allowing conditioning via c__.

    Args
    ----
    df:
      Input dataframe.
    x_to_c:
      Comma-separated list of column names; can use either full "x__foo" or base "foo".
    x_to_c_regex:
      Regex applied to either full column name or base.
    c_prefix:
      Prefix placed after "c__" (default "ctx__" -> c__ctx__foo).
    drop_x:
      If True, drop original x__ columns after copying.

    Returns
    -------
    (df_new, role_map)
    """

    df = df.copy()
    explicit = _parse_csv_list(x_to_c)
    explicit_set = set(explicit)
    rgx = re.compile(x_to_c_regex) if x_to_c_regex else None

    moved: Dict[str, str] = {}
    to_drop: List[str] = []

    for col in list(df.columns):
        if not col.startswith("x__"):
            continue
        base = col[len("x__") :]

        hit = False
        if col in explicit_set or base in explicit_set:
            hit = True
        if (not hit) and (rgx is not None):
            if rgx.search(col) or rgx.search(base):
                hit = True
        if not hit:
            continue

        new_c = f"c__{c_prefix}{base}"
        if new_c not in df.columns:
            df[new_c] = df[col]
        moved[new_c] = col
        if drop_x:
            to_drop.append(col)

    if to_drop:
        df = df.drop(columns=to_drop)

    role_map = FeatureRoleMap(
        moved=moved,
        dropped_x=to_drop,
        prefix=c_prefix,
        regex=x_to_c_regex or "",
        explicit=explicit,
    )
    return df, role_map


def save_role_map(role_map: FeatureRoleMap, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(role_map), f, indent=2, ensure_ascii=False)


def load_role_map(path: str) -> FeatureRoleMap:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return FeatureRoleMap(
        moved=d.get("moved", {}),
        dropped_x=d.get("dropped_x", []),
        prefix=d.get("prefix", "ctx__"),
        regex=d.get("regex", ""),
        explicit=d.get("explicit", []),
    )
