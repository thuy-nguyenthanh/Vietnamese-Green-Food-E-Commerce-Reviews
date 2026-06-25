"""
McNemar test: ViSoBERT zero-shot vs silver-trained TF-IDF baseline on T3 human gold (n=120).

Appendix comparison per v3 §8.4 (full ViSoBERT focal fine-tune deferred).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


WORKSPACE = Path(__file__).resolve().parents[2]
EVAL_BASE = WORKSPACE / "Step08_ViSoBERT" / "Datas" / "eval"
INPUT_BASE = WORKSPACE / "Step08_ViSoBERT" / "Datas" / "downstream_inputs"
PHASE55_BASE = WORKSPACE / "Step05_SSOT" / "Datas" / "phase55"
UNION_PATH = WORKSPACE / "Step08_ViSoBERT" / "Datas" / "sentiment_boost" / "phase55_tier2_apply_001" / "train_union_extended_plus_tier2.csv"

LABEL_MAP = {"POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL", "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL"}


def _normalize(label: str) -> str:
    return LABEL_MAP.get(str(label).strip().upper(), "NEUTRAL")


def _text_series(df: pd.DataFrame) -> pd.Series:
    norm = df.get("clause_text_normalized", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    raw = df.get("clause_text", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    return norm.where(norm != "", raw)


def _mcnemar(b_correct: np.ndarray, a_correct: np.ndarray) -> dict:
    # b=ViSoBERT, a=baseline
    b_only = int(np.sum(b_correct & ~a_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    n_disc = b_only + a_only
    if n_disc == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(b_only, n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "visobert_correct_baseline_wrong": b_only,
        "baseline_correct_visobert_wrong": a_only,
        "n_discordant": n_disc,
        "p_value_two_sided": round(p_value, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="McNemar ViSoBERT vs silver TF-IDF baseline on T3.")
    parser.add_argument("--eval-tag", default="phase55_tier23_001_t3_human")
    parser.add_argument("--run-tag", default="phase55_final_001")
    args = parser.parse_args()

    t3_pred_path = EVAL_BASE / args.eval_tag / "t3_predictions.csv"
    t3_preds = pd.read_csv(t3_pred_path)
    t3_preds["clause_id_final"] = t3_preds["clause_id_final"].astype(str)

    train = pd.read_csv(UNION_PATH) if UNION_PATH.exists() else pd.read_csv(INPUT_BASE / args.run_tag / "train.csv")
    x_train = _text_series(train)
    y_train = train["sentiment_polarity_hint"].astype(str).map(_normalize)

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)

    frame_path = PHASE55_BASE / "phase55_tier23_001" / "tier3" / "tier3_mini_iaa_sampling_frame.csv"
    frame = pd.read_csv(frame_path)
    frame["clause_id_final"] = frame["clause_id_final"].astype(str)
    t3_text = frame.merge(t3_preds[["clause_id_final"]], on="clause_id_final")
    x_t3 = _text_series(t3_text)
    baseline_pred = model.predict(x_t3)

    merged = t3_preds.copy()
    merged["baseline_pred"] = baseline_pred
    merged["visobert_correct"] = merged["label_gold"].astype(str) == merged["label_pred"].astype(str)
    merged["baseline_correct"] = merged["label_gold"].astype(str) == merged["baseline_pred"].astype(str)

    mcnemar = _mcnemar(merged["visobert_correct"].values, merged["baseline_correct"].values)
    visobert_acc = float(merged["visobert_correct"].mean())
    baseline_acc = float(merged["baseline_correct"].mean())

    out_dir = EVAL_BASE / args.eval_tag
    merged.to_csv(out_dir / "t3_mcnemar_pairs.csv", index=False, encoding="utf-8-sig")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_tag": args.eval_tag,
        "comparison": "ViSoBERT zero-shot vs TF-IDF+LR (silver clause train)",
        "n_t3": int(len(merged)),
        "visobert_accuracy": round(visobert_acc, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "delta_accuracy_visobert_minus_baseline": round(visobert_acc - baseline_acc, 4),
        "mcnemar": mcnemar,
        "note": "Full ViSoBERT focal fine-tune per v3 §8.2 not run; baseline is appendix proxy.",
    }
    report_path = out_dir / "t3_mcnemar_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Wrote", report_path)


if __name__ == "__main__":
    main()
