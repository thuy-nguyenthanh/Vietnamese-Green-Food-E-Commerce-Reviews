"""
Step 9 primary: review-level non-positive classification + SHAP (Q3 gate).

Target (independent): rating_star <= 3 from Step 3.4 metadata.
Features: aspect shares (Step 7 preds) + sentiment aggregates (Step 8) + review meta.
Includes bootstrap 95% CI for top-5 |SHAP| and spam-weight sensitivity.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


WORKSPACE = Path(__file__).resolve().parents[2]
PRED_BASE = WORKSPACE / "Step08_ViSoBERT" / "Datas" / "predictions"
ASPECT_PRED_BASE = WORKSPACE / "Step07_Aspect" / "Datas" / "predictions"
SPAM_DIR = WORKSPACE / "Step03_33_34" / "Datas" / "34_spam_fake_layer"
OUT_BASE = WORKSPACE / "Step09_SHAP" / "Datas" / "primary"

ASPECT_CLASSES = ["A1_product", "A2_label", "A3_logistics", "A4_service", "A5_price", "OOD"]


def _resolve_run_tag(run_tag: str | None) -> str:
    if run_tag:
        return run_tag
    latest = PRED_BASE / "LATEST.txt"
    if not latest.exists():
        raise FileNotFoundError("Missing Step08 predictions/LATEST.txt")
    return latest.read_text(encoding="utf-8").strip().splitlines()[0]


def _load_predictions(run_tag: str, split_name: str) -> pd.DataFrame:
    path = PRED_BASE / run_tag / f"{split_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    return pd.read_csv(path)


def _load_review_metadata() -> pd.DataFrame:
    rows = []
    for path in sorted(SPAM_DIR.glob("[0-9]*.csv")):
        df = pd.read_csv(path, usecols=["id", "rating_star", "analysis_weight_final"])
        rows.append(df)
    meta = pd.concat(rows, ignore_index=True)
    meta["review_id"] = meta["id"].astype(str)
    meta = meta.drop_duplicates(subset=["review_id"], keep="first")
    return meta[["review_id", "rating_star", "analysis_weight_final"]]


def _aspect_shares(run_tag: str, split_name: str) -> pd.DataFrame:
    path = ASPECT_PRED_BASE / run_tag / f"{split_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing aspect predictions: {path}")
    df = pd.read_csv(path)
    df["review_id"] = df["review_id"].astype(str)
    if "pred_aspect" not in df.columns:
        raise ValueError("Aspect predictions missing pred_aspect column")

    records = []
    for review_id, g in df.groupby("review_id"):
        counts = g["pred_aspect"].astype(str).value_counts()
        total = max(len(g), 1)
        row = {"review_id": review_id, "num_clauses_aspect": int(len(g))}
        for cls in ASPECT_CLASSES:
            row[f"share_{cls}"] = float(counts.get(cls, 0) / total)
        row["ood_rate"] = row["share_OOD"]
        records.append(row)
    return pd.DataFrame(records)


def _build_feature_matrix(
    sentiment_df: pd.DataFrame,
    aspect_shares: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    work = sentiment_df.copy()
    work["review_id"] = work["review_id"].astype(str)
    work = work.merge(aspect_shares, on="review_id", how="left")
    work = work.merge(meta, on="review_id", how="left")
    work["rating_star"] = pd.to_numeric(work["rating_star"], errors="coerce")
    work["analysis_weight_final"] = pd.to_numeric(work["analysis_weight_final"], errors="coerce").fillna(1.0)
    work["target_non_positive"] = (work["rating_star"] <= 3).astype(int)
    work["text_length"] = work.get("review_text_cleaned", "").fillna("").astype(str).str.len()
    return work


def _feature_cols() -> list[str]:
    sentiment_cols = [
        "num_sentences",
        "num_pos",
        "num_neg",
        "num_neu",
        "sentiment_mean_weighted",
        "avg_confidence",
        "has_mixed",
        "prob_pos",
        "prob_neu",
        "prob_neg",
    ]
    aspect_cols = [f"share_{c}" for c in ASPECT_CLASSES] + ["ood_rate", "num_clauses_aspect"]
    meta_cols = ["rating_star", "analysis_weight_final", "text_length"]
    return sentiment_cols + aspect_cols + meta_cols


def _bootstrap_top5_shap_ci(
    shap_arr: np.ndarray,
    feature_cols: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    mean_abs = np.abs(shap_arr).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:5]
    top_features = [feature_cols[i] for i in top_idx]

    results = []
    n = shap_arr.shape[0]
    for feat, idx in zip(top_features, top_idx):
        boots = []
        for _ in range(n_bootstrap):
            sample = rng.integers(0, n, size=n)
            boots.append(float(np.abs(shap_arr[sample, idx]).mean()))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results.append(
            {
                "feature": feat,
                "mean_abs_shap": float(mean_abs[idx]),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
            }
        )
    return results


def _eval_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight: pd.Series | None = None,
) -> dict:
    if sample_weight is not None:
        # Retrain not needed for eval-only; predict with fitted model
        pass
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    y = y_test.astype(int)
    out = {
        "pr_auc": float(average_precision_score(y, prob)),
        "f1_non_positive": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "n_test": int(len(y)),
        "n_positive_class": int(y.sum()),
    }
    if y.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step09 primary with independent target (rating_star<=3).")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    run_tag = _resolve_run_tag(args.run_tag)
    meta = _load_review_metadata()

    train_sent = _load_predictions(run_tag, "train")
    val_sent = _load_predictions(run_tag, "val")
    test_sent = _load_predictions(run_tag, "test")

    train_asp = _aspect_shares(run_tag, "train")
    val_asp = _aspect_shares(run_tag, "val")
    test_asp = _aspect_shares(run_tag, "test")

    train_df = _build_feature_matrix(train_sent, train_asp, meta)
    val_df = _build_feature_matrix(val_sent, val_asp, meta)
    test_df = _build_feature_matrix(test_sent, test_asp, meta)

    full_train = pd.concat([train_df, val_df], ignore_index=True)
    feature_cols = _feature_cols()

    for col in feature_cols + ["target_non_positive"]:
        if col not in full_train.columns:
            raise ValueError(f"Missing column after feature build: {col}")

    X_train = full_train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = full_train["target_non_positive"].astype(int)
    w_train = full_train["analysis_weight_final"].astype(float)

    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_test = test_df["target_non_positive"].astype(int)

    # Exclude rating_star from features when predicting rating-derived target
    train_cols = [c for c in feature_cols if c != "rating_star"]
    X_train_fit = X_train[train_cols]
    X_test_fit = X_test[train_cols]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train_fit, y_train, sample_weight=w_train)

    metrics_default = _eval_model(model, X_test_fit, y_test)

    # Spam sensitivity: uniform weights
    model_unweighted = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    model_unweighted.fit(X_train_fit, y_train)
    metrics_unweighted = _eval_model(model_unweighted, X_test_fit, y_test)

    prob = model.predict_proba(X_test_fit)[:, 1]
    pred = (prob >= 0.5).astype(int)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_fit)
        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[1])
        else:
            shap_arr = np.array(shap_values)
    except Exception:
        explainer = shap.Explainer(model.predict, X_train_fit)
        shap_exp = explainer(X_test_fit)
        shap_arr = np.array(shap_exp.values)

    shap_top5 = _bootstrap_top5_shap_ci(shap_arr, train_cols, n_bootstrap=args.bootstrap)

    out_dir = OUT_BASE / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "review_id": test_df["review_id"],
            "rating_star": test_df["rating_star"],
            "prob_non_positive": prob,
            "pred_non_positive": pred,
            "true_non_positive": y_test,
        }
    ).to_csv(out_dir / "test_predictions_non_positive.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(shap_top5).to_csv(out_dir / "shap_importance_bootstrap.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_definition": "rating_star <= 3 (independent of model predictions)",
        "target_leakage_note": "rating_star excluded from feature matrix",
        "rows_train": int(len(X_train_fit)),
        "rows_test": int(len(X_test_fit)),
        "metrics_primary": metrics_default,
        "sensitivity_unweighted_train": metrics_unweighted,
        "bootstrap_n": args.bootstrap,
        "top5_features_shap_ci": shap_top5,
        "feature_columns": train_cols,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_BASE / "LATEST.txt").write_text(f"{run_tag}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
