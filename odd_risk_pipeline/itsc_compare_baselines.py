#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITSC baseline comparison pack (v6_2_10)

Minimum but strong baselines for ITSC (without external "SOTA"):
- 2-stage ablations from your model outputs (gate-only / expert-only / 2-stage risk)
- Survival baselines (LogNormal AFT, Weibull AFT) under right-censoring at ttc_cap
- Strong classifier baseline (HistGradientBoostingClassifier) for y_event = 1[TTC<=t_q]

Tail threshold t_q is computed on TRAIN split by default. If --use_eval_tq is set and --eval_dir points to an eval output folder, t_q is loaded from eval_{split}_*.json to exactly match your model's evaluation definition.
If tail_q_source=train_candidates, we restrict quantile computation to rows with expert_mask==1
(if such a column exists); otherwise we fall back to all train rows.

Censoring is defined as TTC clipped at ttc_cap (right-censoring).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from scipy.optimize import minimize
from scipy.stats import norm, kstest


ID_COLS = {
    "segment_id","frame_label","scenario_id","track_id","agent_id","timestamp_micros",
    "logfile","log_file","frame","t","idx",
}
LABEL_COLS_PREFIX = ("y__",)
EXCLUDE_PREFIX = ("pred__",)
DEFAULT_BINS = 10


def log(msg: str) -> None:
    print(msg, flush=True)


def load_split(run_dir: Path) -> dict:
    p = run_dir / "segment_split.json"
    if not p.exists():
        raise FileNotFoundError(f"segment_split.json not found in {run_dir}")
    with open(p, "r") as f:
        return json.load(f)


def pick_split_df(df: pd.DataFrame, segs) -> pd.DataFrame:
    if not segs:
        return df.iloc[0:0].copy()
    if "segment_id" not in df.columns:
        raise KeyError("CSV must contain segment_id")
    sset = set(str(s) for s in segs)
    return df[df["segment_id"].astype(str).isin(sset)].copy()


def extract_ttc_seconds(df: pd.DataFrame) -> np.ndarray:
    """Prefer min_ttc_est if exists, else TTC from y_soft (=1/TTC)."""
    if "min_ttc_est" in df.columns:
        return pd.to_numeric(df["min_ttc_est"], errors="coerce").to_numpy(np.float64)
    if "y_soft" in df.columns:
        y_soft = pd.to_numeric(df["y_soft"], errors="coerce").to_numpy(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.clip(y_soft, 1e-9, np.inf)
    raise KeyError("CSV must contain min_ttc_est or y_soft")


def build_candidate_mask(df: pd.DataFrame) -> np.ndarray:
    if "expert_mask" in df.columns:
        return (pd.to_numeric(df["expert_mask"], errors="coerce").fillna(0.0).to_numpy(np.float64) > 0.5)
    for alt in ["c__expert_mask","x__expert_mask","gate_expert_mask","is_candidate","candidate_mask"]:
        if alt in df.columns:
            return (pd.to_numeric(df[alt], errors="coerce").fillna(0.0).to_numpy(np.float64) > 0.5)
    return np.ones((len(df),), dtype=bool)


def clip_and_censor(ttc: np.ndarray, ttc_floor: float, ttc_cap: float):
    ttc = np.asarray(ttc, dtype=np.float64)
    ttc = np.clip(ttc, ttc_floor, ttc_cap)
    cens = ttc >= (ttc_cap - 1e-12)
    return ttc, cens


def compute_tail_threshold_tq(df_train: pd.DataFrame, tail_q: float, tail_q_source: str,
                             ttc_floor: float, ttc_cap: float) -> float:
    ttc_train_raw = extract_ttc_seconds(df_train)
    ttc_train, _ = clip_and_censor(ttc_train_raw, ttc_floor, ttc_cap)
    m = np.ones((len(df_train),), dtype=bool)
    if tail_q_source == "train_candidates":
        m = build_candidate_mask(df_train)
        if int(m.sum()) < 10:
            log(f"[WARN] Not enough candidate rows for tail_q_source=train_candidates (n={int(m.sum())}). Falling back to all train rows.")
            m = np.ones((len(df_train),), dtype=bool)
    tq = float(np.quantile(ttc_train[m], float(tail_q)))
    return float(np.clip(tq, ttc_floor, ttc_cap))


def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = DEFAULT_BINS) -> float:
    y_true = y_true.astype(np.int64)
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = DEFAULT_BINS) -> pd.DataFrame:
    y_true = y_true.astype(np.int64)
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    rows = []
    for b in range(n_bins):
        m = (idx == b)
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "p_mean": float(p[m].mean()) if np.any(m) else np.nan,
            "y_rate": float(y_true[m].mean()) if np.any(m) else np.nan,
            "p_min": float(bins[b]),
            "p_max": float(bins[b+1]),
        })
    return pd.DataFrame(rows)


def metrics_for_scores(y_true: np.ndarray, score: np.ndarray, name: str) -> dict:
    y_true = y_true.astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    score01 = np.clip(score, 0.0, 1.0)
    out = {"method": name, "n": int(len(y_true)), "pos_rate": float(y_true.mean())}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, score))
    except Exception:
        out["pr_auc"] = float("nan")
    out["brier"] = float(np.mean((score01 - y_true) ** 2))
    out["ece"] = float(ece_score(y_true, score01, DEFAULT_BINS))
    return out


# ---------------- Survival baselines: LogNormal AFT ----------------

def _negloglik_lognormal_aft(params, X, t, cens) -> float:
    beta = params[:-1]
    sigma = float(np.exp(params[-1]))
    mu = X @ beta
    logt = np.log(t + 1e-12)
    z = (logt - mu) / (sigma + 1e-12)
    ll_unc = -np.log(t + 1e-12) - np.log(sigma + 1e-12) + norm.logpdf(z)
    ll_cens = norm.logsf(z)
    ll = np.where(cens, ll_cens, ll_unc)
    l2 = 1e-4 * float(np.sum(beta ** 2))
    return -float(np.sum(ll) - l2)


def fit_lognormal_aft(X, t, cens, max_iter: int = 400):
    D = X.shape[1]
    x0 = np.zeros((D + 1,), dtype=np.float64)
    res = minimize(_negloglik_lognormal_aft, x0, args=(X, t, cens),
                   method="L-BFGS-B", options={"maxiter": max_iter})
    if not res.success:
        log(f"[WARN] LogNormal AFT optimize: {res.message}")
    beta = res.x[:-1]
    sigma = float(np.exp(res.x[-1]))
    return beta, sigma


def predict_p_event_lognormal(beta, sigma, X, t_q):
    mu = X @ beta
    zq = (np.log(t_q + 1e-12) - mu) / (sigma + 1e-12)
    return np.clip(norm.cdf(zq), 0.0, 1.0)


def randomized_pit_lognormal(beta, sigma, X, t_obs, cens, t_cap, rng):
    mu = X @ beta
    z = (np.log(t_obs + 1e-12) - mu) / (sigma + 1e-12)
    F_obs = norm.cdf(z)
    zc = (np.log(t_cap + 1e-12) - mu) / (sigma + 1e-12)
    F_cap = norm.cdf(zc)
    u = np.empty_like(F_obs)
    u[~cens] = F_obs[~cens]
    u[cens] = F_cap[cens] + (1.0 - F_cap[cens]) * rng.random(np.sum(cens))
    return np.clip(u, 0.0, 1.0)


# ---------------- Survival baselines: Weibull AFT ----------------

def _negloglik_weibull_aft(params, X, t, cens) -> float:
    beta = params[:-1]
    k = float(np.exp(params[-1]))  # shape
    lam = np.exp(X @ beta)  # scale
    logf = np.log(k + 1e-12) - k * np.log(lam + 1e-12) + (k - 1.0) * np.log(t + 1e-12) - (t / (lam + 1e-12)) ** k
    logS = - (t / (lam + 1e-12)) ** k
    ll = np.where(cens, logS, logf)
    l2 = 1e-4 * float(np.sum(beta ** 2)) + 1e-4 * float(params[-1] ** 2)
    return -float(np.sum(ll) - l2)


def fit_weibull_aft(X, t, cens, max_iter: int = 600):
    D = X.shape[1]
    x0 = np.zeros((D + 1,), dtype=np.float64)
    res = minimize(_negloglik_weibull_aft, x0, args=(X, t, cens),
                   method="L-BFGS-B", options={"maxiter": max_iter})
    if not res.success:
        log(f"[WARN] Weibull AFT optimize: {res.message}")
    beta = res.x[:-1]
    k = float(np.exp(res.x[-1]))
    return beta, k


def predict_p_event_weibull(beta, k, X, t_q):
    lam = np.exp(X @ beta)
    return np.clip(1.0 - np.exp(- (t_q / (lam + 1e-12)) ** k), 0.0, 1.0)


def randomized_pit_weibull(beta, k, X, t_obs, cens, t_cap, rng):
    lam = np.exp(X @ beta)
    F_obs = 1.0 - np.exp(- (t_obs / (lam + 1e-12)) ** k)
    F_cap = 1.0 - np.exp(- (t_cap / (lam + 1e-12)) ** k)
    u = np.empty_like(F_obs)
    u[~cens] = F_obs[~cens]
    u[cens] = F_cap[cens] + (1.0 - F_cap[cens]) * rng.random(np.sum(cens))
    return np.clip(u, 0.0, 1.0)


# ---------------- feature extraction ----------------

def build_feature_matrix(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if c.startswith(LABEL_COLS_PREFIX):
            continue
        if c.startswith(EXCLUDE_PREFIX):
            continue
        if not (c.startswith("x__") or c.startswith("c__")):
            continue
        cols.append(c)
    if not cols:
        raise RuntimeError("No features found: expected columns starting with x__ or c__")
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.to_numpy(np.float64), cols


def resolve_eval_dir(eval_dir: Path, run_dir: Path, split: str) -> Path:
    """Accept either eval root or run subdir; return directory that contains preds/eval json."""
    cand = []
    if eval_dir.exists():
        cand.append(eval_dir / run_dir.name)
        cand.append(eval_dir)
    for c in cand:
        if c.exists() and (list(c.glob(f"preds_{split}_*.npz")) or list(c.glob(f"eval_{split}_*.json"))):
            return c
    # fallback: recursive search for run_dir.name
    for c in eval_dir.glob(f"**/{run_dir.name}"):
        if c.is_dir() and (list(c.glob(f"preds_{split}_*.npz")) or list(c.glob(f"eval_{split}_*.json"))):
            return c
    return eval_dir

def load_eval_tq(eval_run_dir: Path, split: str) -> float:
    """Load tail TTC threshold (seconds) from eval json. Key is tail_ttc_threshold_s."""
    cand = sorted(eval_run_dir.glob(f"eval_{split}_*.json"))
    if not cand:
        return float("nan")
    p = cand[-1]
    try:
        with open(p, "r") as f:
            j = json.load(f)
        for k in ["tail_ttc_threshold_s","tail_ttc_threshold","tail_ttc_threshold_sec","tail_ttc_threshold_s"]:
            if k in j:
                return float(j[k])
        # older naming
        if "tail_ttc_threshold_s" in j:
            return float(j["tail_ttc_threshold_s"])
        if "tail_ttc_threshold_s" in j:
            return float(j["tail_ttc_threshold_s"])
    except Exception as e:
        log(f"[WARN] failed to parse eval json {p}: {e}")
    return float("nan")

def load_model_preds(eval_dir: Path, run_dir: Path, split: str):
    eval_run_dir = resolve_eval_dir(eval_dir, run_dir, split)
    cand = sorted(eval_run_dir.glob(f"preds_{split}_*.npz"))
    if not cand:
        cand = sorted(eval_run_dir.glob(f"preds_{split}.npz"))
    if not cand:
        return None
    p = cand[-1]
    # Some preds_*.npz bundles may include object-typed arrays (e.g., lists of column names).
    # NumPy refuses to load those unless allow_pickle=True. These files are produced by our
    # own pipeline (trusted), so we enable allow_pickle and then drop non-numeric arrays.
    arr = np.load(p, allow_pickle=True)
    out = {}
    for k in arr.files:
        v = arr[k]
        # Drop object arrays (metadata) – we only need numeric tensors for scoring.
        if isinstance(v, np.ndarray) and v.dtype == object:
            continue
        out[k] = v
    out["_path"] = str(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run", required=True, help="run directory containing segment_split.json")
    ap.add_argument("--eval_dir", default="", help="optional: eval output dir to read preds_*.npz for ablations")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=10.0)
    ap.add_argument("--tail_q", type=float, default=0.10)
    ap.add_argument("--tail_q_source", default="train_candidates", choices=["train_candidates", "train_all"])
    ap.add_argument("--use_eval_tq", action="store_true", help="if set, load t_q from eval json under --eval_dir to exactly match model eval")
    ap.add_argument("--gate_target_recall", type=float, default=None, help="optional: for sweep bookkeeping only (baselines do not use it)")
    ap.add_argument("--require_edges", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    run_dir = Path(args.run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[baselines] csv={csv_path}")
    log(f"[baselines] run={run_dir}")

    split = load_split(run_dir)
    df_all = pd.read_csv(csv_path)

    df_train = pick_split_df(df_all, split.get("train", []))
    df_eval = pick_split_df(df_all, split.get(args.split, []))
    if len(df_train) == 0 or len(df_eval) == 0:
        raise RuntimeError(f"Empty split: train={len(df_train)} {args.split}={len(df_eval)}")

    t_q = compute_tail_threshold_tq(
        df_train,
        args.tail_q,
        args.tail_q_source,
        args.ttc_floor,
        args.ttc_cap,
    )

    tq_source = "train_quantile"


    # Optional: force baselines to use the same t_q that was used during the model eval.
    # This is critical for fair comparisons (otherwise train-quantile recomputation may drift).
    if args.use_eval_tq:
        if not args.eval_dir:
            raise RuntimeError("--use_eval_tq was set but --eval_dir is missing. Provide the eval output directory so baselines use the exact same t_q as model eval.")
        ed = Path(args.eval_dir)
        eval_run_dir = resolve_eval_dir(ed, run_dir, args.split)
        tq_eval = load_eval_tq(eval_run_dir, args.split)
        if not np.isfinite(tq_eval):
            raise RuntimeError(f"--use_eval_tq was set but could not load tail_ttc_threshold_s from eval json under: {eval_run_dir}. Make sure eval_risk_models.py was run with --save_dir pointing to --eval_dir and produced eval_{args.split}_*.json.")
        log(f"[baselines] using eval t_q from {eval_run_dir}: {tq_eval:.6f}s (overrides train quantile)")
        t_q = float(np.clip(tq_eval, args.ttc_floor, args.ttc_cap))
        tq_source = "eval_json"
    log(f"[baselines] tail_q={args.tail_q} source={args.tail_q_source} => t_q={t_q:.6f}s")

    # y_event on eval split
    ttc_eval_raw = extract_ttc_seconds(df_eval)
    ttc_eval, cens_eval = clip_and_censor(ttc_eval_raw, args.ttc_floor, args.ttc_cap)
    y_event = (ttc_eval <= t_q + 1e-12).astype(np.int64)
    if args.require_edges:
        m_edge = build_candidate_mask(df_eval)
        y_event = (y_event.astype(bool) & m_edge).astype(np.int64)

    # If eval preds exist, prefer y_event from preds for *scoring*, to guarantee
    # label-definition consistency with eval_risk_models.py (avoids subtle TTC extraction diffs).
    preds_for_labels = None
    if args.eval_dir:
        try:
            ed0 = Path(args.eval_dir)
            preds_for_labels = load_model_preds(ed0, run_dir, args.split)
            if isinstance(preds_for_labels, dict) and ("y_event" in preds_for_labels):
                y_from = np.asarray(preds_for_labels["y_event"]).reshape(-1).astype(np.int64)
                if len(y_from) == len(y_event):
                    if not np.array_equal(y_from, y_event):
                        log("[WARN] y_event mismatch (CSV-derived vs eval preds). Using eval preds y_event for scoring.")
                    y_event = y_from
                else:
                    log(f"[WARN] eval preds y_event length mismatch: {len(y_from)} vs {len(y_event)}; keeping CSV-derived y_event.")
        except Exception as e:
            log(f"[WARN] failed to load y_event from eval preds: {e}")

    # Features
    X_train_raw, feat_cols = build_feature_matrix(df_train)
    X_eval_raw, _ = build_feature_matrix(df_eval)

    pre = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
    ])
    X_train = pre.fit_transform(X_train_raw)
    X_eval = pre.transform(X_eval_raw)

    # Train ttc for survival
    ttc_train_raw = extract_ttc_seconds(df_train)
    ttc_train, cens_train = clip_and_censor(ttc_train_raw, args.ttc_floor, args.ttc_cap)

    rng = np.random.default_rng(args.seed)

    methods = []
    cal_tables = {}

    # 1) LogNormal AFT
    log("[fit] LogNormal AFT ...")
    beta_ln, sigma_ln = fit_lognormal_aft(X_train, ttc_train, cens_train)
    p_ln = predict_p_event_lognormal(beta_ln, sigma_ln, X_eval, t_q)
    methods.append(metrics_for_scores(y_event, p_ln, "surv_lognormal_aft"))
    cal_tables["surv_lognormal_aft"] = calibration_table(y_event, p_ln, DEFAULT_BINS)
    u_ln = randomized_pit_lognormal(beta_ln, sigma_ln, X_eval, ttc_eval, cens_eval, args.ttc_cap, rng)
    pit_ln = {
        "pit_mean": float(np.mean(u_ln)),
        "pit_std": float(np.std(u_ln)),
        "pit_ks": float(kstest(u_ln, "uniform").statistic),
        "censored_ratio": float(np.mean(cens_eval)),
        "t_q": float(t_q),
        "tail_q": float(args.tail_q),
    }
    with open(out_dir / "pit_surv_lognormal_aft.json", "w") as f:
        json.dump(pit_ln, f, indent=2)

    # 2) Weibull AFT
    log("[fit] Weibull AFT ...")
    beta_wb, k_wb = fit_weibull_aft(X_train, ttc_train, cens_train)
    p_wb = predict_p_event_weibull(beta_wb, k_wb, X_eval, t_q)
    methods.append(metrics_for_scores(y_event, p_wb, "surv_weibull_aft"))
    cal_tables["surv_weibull_aft"] = calibration_table(y_event, p_wb, DEFAULT_BINS)
    u_wb = randomized_pit_weibull(beta_wb, k_wb, X_eval, ttc_eval, cens_eval, args.ttc_cap, rng)
    pit_wb = {
        "pit_mean": float(np.mean(u_wb)),
        "pit_std": float(np.std(u_wb)),
        "pit_ks": float(kstest(u_wb, "uniform").statistic),
        "censored_ratio": float(np.mean(cens_eval)),
        "t_q": float(t_q),
        "tail_q": float(args.tail_q),
    }
    with open(out_dir / "pit_surv_weibull_aft.json", "w") as f:
        json.dump(pit_wb, f, indent=2)

    # 3) Strong classifier baseline
    log("[fit] HistGradientBoostingClassifier ...")
    y_train_event = (ttc_train <= t_q + 1e-12).astype(np.int64)
    if args.require_edges:
        m_edge_train = build_candidate_mask(df_train)
        y_train_event = (y_train_event.astype(bool) & m_edge_train).astype(np.int64)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=50,
        l2_regularization=0.0,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train_event)
    p_clf = clf.predict_proba(X_eval)[:, 1]
    methods.append(metrics_for_scores(y_event, p_clf, "clf_hist_gbdt"))
    cal_tables["clf_hist_gbdt"] = calibration_table(y_event, p_clf, DEFAULT_BINS)

    # 4) Your model ablations from eval preds
    if args.eval_dir:
        ed = Path(args.eval_dir)
        preds = preds_for_labels if preds_for_labels is not None else load_model_preds(ed, run_dir, args.split)
        if preds is None:
            log(f"[WARN] No preds_*.npz found under eval_dir={ed} (resolved={resolve_eval_dir(ed, run_dir, args.split)}); skipping model ablations.")
        else:
            log(f"[ablations] using {preds.get('_path')}")
            if ("p_gate" in preds) and ("p_event" in preds):
                p_gate = np.asarray(preds["p_gate"]).reshape(-1)
                p_event_m = np.asarray(preds["p_event"]).reshape(-1)
                risk = np.asarray(preds.get("risk", p_gate * p_event_m)).reshape(-1)
                y_eval = y_event
                if "y_event" in preds:
                    try:
                        y_eval = np.asarray(preds["y_event"]).reshape(-1).astype(np.int64)
                    except Exception:
                        y_eval = y_event
                methods.append(metrics_for_scores(y_eval, p_gate, "model_gate_only"))
                methods.append(metrics_for_scores(y_eval, p_event_m, "model_expert_only"))
                methods.append(metrics_for_scores(y_eval, risk, "model_2stage_risk"))
                cal_tables["model_gate_only"] = calibration_table(y_eval, np.clip(p_gate, 0, 1), DEFAULT_BINS)
                cal_tables["model_expert_only"] = calibration_table(y_eval, np.clip(p_event_m, 0, 1), DEFAULT_BINS)
                cal_tables["model_2stage_risk"] = calibration_table(y_eval, np.clip(risk, 0, 1), DEFAULT_BINS)
            else:
                log("[WARN] preds npz missing p_gate/p_event; skipping model ablations.")

    df_sum = pd.DataFrame(methods).sort_values(by="pr_auc", ascending=False)
    df_sum["split"] = str(args.split)
    df_sum["tail_q"] = float(args.tail_q)
    df_sum["tail_q_source"] = str(args.tail_q_source)
    df_sum["t_q_seconds"] = float(t_q)
    df_sum["tq_source"] = str(tq_source)
    df_sum.to_csv(out_dir / "summary_metrics.csv", index=False)

    for k, tab in cal_tables.items():
        tab.to_csv(out_dir / f"calibration_{k}.csv", index=False)

    with open(out_dir / "tail_event_config.json", "w") as f:
        json.dump({
            "risk_def": "quantile_tail",
            "tail_q": float(args.tail_q),
            "tail_q_source": str(args.tail_q_source),
            "use_eval_tq": bool(args.use_eval_tq),
            "gate_target_recall": (float(args.gate_target_recall) if args.gate_target_recall is not None else None),
            "tq_source": str(tq_source),
            "t_q_seconds": float(t_q),
            "ttc_floor": float(args.ttc_floor),
            "ttc_cap": float(args.ttc_cap),
            "require_edges": bool(args.require_edges),
            "n_train": int(len(df_train)),
            "n_eval": int(len(df_eval)),
            "pos_rate_eval": float(y_event.mean()),
            "censored_ratio_eval": float(np.mean(cens_eval)),
            "feature_dim": int(X_train.shape[1]),
            "n_features": int(len(feat_cols)),
        }, f, indent=2)

    log(f"[ok] wrote {out_dir/'summary_metrics.csv'}")
    log("[done]")


if __name__ == "__main__":
    main()
