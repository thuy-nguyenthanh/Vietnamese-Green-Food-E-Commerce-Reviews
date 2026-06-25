"""
Bootstrap 95% CI for primary T2 holdout and T3 zero-shot metrics.

Outputs:
  Step07_Aspect/Datas/eval/phase55_final_001_t2_gold_holdout/bootstrap_ci.json
  Step08_ViSoBERT/Datas/eval/phase55_tier23_001_t3_human/bootstrap_ci.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

WORKSPACE = Path(__file__).resolve().parents[2]
STEP07_TOOLS = WORKSPACE / "Step07_Aspect" / "Tools"
if str(STEP07_TOOLS) not in sys.path:
    sys.path.insert(0, str(STEP07_TOOLS))

T2_PRED = (
    WORKSPACE
    / "Step07_Aspect"
    / "Datas"
    / "eval"
    / "phase55_final_001_t2_gold_holdout"
    / "t2_predictions.csv"
)
T2_REPORT = (
    WORKSPACE
    / "Step07_Aspect"
    / "Datas"
    / "eval"
    / "phase55_final_001_t2_gold_holdout"
    / "aspect_eval_T2_holdout.json"
)
T3_PRED = (
    WORKSPACE
    / "Step08_ViSoBERT"
    / "Datas"
    / "eval"
    / "phase55_tier23_001_t3_human"
    / "t3_predictions.csv"
)
T3_REPORT = (
    WORKSPACE
    / "Step08_ViSoBERT"
    / "Datas"
    / "eval"
    / "phase55_tier23_001_t3_human"
    / "t3_eval_report.json"
)


def _ci95(values: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
        "bootstrap_std": round(float(np.std(values)), 4),
    }


def bootstrap_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    labels: list[str] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    macro_f1s: list[float] = []
    per_class_f1: dict[str, list[float]] = {lbl: [] for lbl in labels}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        macro_f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0, labels=labels)))
        for lbl in labels:
            per_class_f1[lbl].append(
                float(f1_score(yt, yp, labels=[lbl], average="macro", zero_division=0))
            )

    macro_arr = np.array(macro_f1s)
    out = {
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "macro_f1": {
            "point": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)), 4),
            **_ci95(macro_arr),
        },
        "per_class_f1": {},
    }
    for lbl in labels:
        arr = np.array(per_class_f1[lbl])
        support = int((y_true == lbl).sum())
        out["per_class_f1"][lbl] = {
            "support": support,
            "point": round(float(f1_score(y_true, y_pred, labels=[lbl], average="macro", zero_division=0)), 4),
            **_ci95(arr),
        }
    return out


def bootstrap_t3_neg_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    neg_label = "NEGATIVE"

    f1_negs: list[float] = []
    recall_negs: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        f1_negs.append(float(f1_score(yt, yp, labels=[neg_label], average="macro", zero_division=0)))
        pos = int((yt == neg_label).sum())
        tp = int(((yt == neg_label) & (yp == neg_label)).sum())
        recall_negs.append(float(tp / pos) if pos else 0.0)

    f1_arr = np.array(f1_negs)
    rec_arr = np.array(recall_negs)
    return {
        "f1_neg": {
            "point": round(float(f1_score(y_true, y_pred, labels=[neg_label], average="macro", zero_division=0)), 4),
            **_ci95(f1_arr),
        },
        "recall_neg": {
            "point": round(
                float(((y_true == neg_label) & (y_pred == neg_label)).sum() / max(int((y_true == neg_label).sum()), 1)),
                4,
            ),
            **_ci95(rec_arr),
        },
    }


def run_t2_bootstrap(n_bootstrap: int, seed: int) -> dict:
    if not T2_PRED.exists():
        raise FileNotFoundError(f"Run T2 holdout eval first: {T2_PRED}")
    df = pd.read_csv(T2_PRED)
    y_true = df["label_gold"].astype(str).to_numpy()
    y_pred = df["label_pred"].astype(str).to_numpy()
    labels = sorted(set(y_true) | set(y_pred))

    metrics = bootstrap_classification_metrics(y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed, labels=labels)
    report = json.loads(T2_REPORT.read_text(encoding="utf-8")) if T2_REPORT.exists() else {}

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_type": "step7_t2_gold_holdout_bootstrap",
        "source_predictions": str(T2_PRED),
        "reference_point_report": str(T2_REPORT),
        "metrics": metrics,
        "reference_macro_f1_from_report": report.get("macro_f1"),
    }
    out_path = T2_PRED.parent / "bootstrap_ci.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_t3_bootstrap(n_bootstrap: int, seed: int) -> dict:
    if not T3_PRED.exists():
        raise FileNotFoundError(f"Run T3 eval first: {T3_PRED}")
    df = pd.read_csv(T3_PRED)
    y_true = df["label_gold"].astype(str).to_numpy()
    y_pred = df["label_pred"].astype(str).to_numpy()
    labels = sorted(set(y_true) | set(y_pred))

    metrics = bootstrap_classification_metrics(y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed, labels=labels)
    neg_metrics = bootstrap_t3_neg_metrics(y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed)
    metrics.update(neg_metrics)

    report = json.loads(T3_REPORT.read_text(encoding="utf-8")) if T3_REPORT.exists() else {}
    primary = report.get("primary_metrics", {})

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_type": "step8_t3_zero_shot_bootstrap",
        "source_predictions": str(T3_PRED),
        "reference_point_report": str(T3_REPORT),
        "metrics": metrics,
        "reference_from_report": primary,
    }
    out_path = T3_PRED.parent / "bootstrap_ci.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for T2/T3 primary metrics.")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lane", choices=("t2", "t3", "both"), default="both")
    args = parser.parse_args()

    if args.lane in ("t2", "both"):
        t2 = run_t2_bootstrap(args.n_bootstrap, args.seed)
        print("T2 macro_f1:", t2["metrics"]["macro_f1"])
    if args.lane in ("t3", "both"):
        t3 = run_t3_bootstrap(args.n_bootstrap, args.seed)
        print("T3 macro_f1:", t3["metrics"]["macro_f1"])
        print("T3 f1_neg:", t3["metrics"]["f1_neg"])


if __name__ == "__main__":
    main()
